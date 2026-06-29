#!/usr/bin/env python3
"""Adaptive brightness ensemble — control vs 약제 brightness 편차가 클 때 brightness
비중을 키우는 gated ensemble 검증.

도메인 인사이트(사용자): control brightness 와 약제 brightness 의 편차를 이용한 G/NG
판단이 잘 working하는 영역이 있다 → 편차가 큰 row에서만 brightness 를 더 신뢰.

  gate = |brightness_ratio_t6 - 1|  (= |drug_brightness/ctl_brightness - 1|, 약제-control 편차)
  편차 큰 row(gate 상위 q%)에만 w_br 부여, 나머지는 image-only.

변형: image_only / 고정 ens_br30(비교용) / gate 상위 q∈{25,33,50}% × w_br∈{0.3,0.5}
각 변형: per-cell isotonic(FDA fit) → threshold 0.65 → drast_gng→MIC→SIR→evalEA.
출력: agent_system/output/brightness_adaptive/
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kptae/project/qnt_algorithm")
sys.path.insert(0, str(ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup        # noqa: E402
from agent_system.brightness_gbm_crossdomain import build_features  # noqa: E402

WORKBOOK = ROOT / "claudeCode/data/F21_GN_Selecetd_model_260123 (3) (1).xlsx"
BASE_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
BRIGHT_OOF = ROOT / "agent_system/output/brightness_gbm/brightness_crossdomain_oof.csv"
FDA_CSV = ("/home/kptae/data/allinfo/fda2023_preprocessedWithFeature_validControlInfo_"
           "useMax_TrueWith_object_area_useMax_TrueWith_object_count_useMax_TrueWith_"
           "brightness_useMax_False.csv")
OUT_DIR = ROOT / "agent_system/output/brightness_adaptive"
VERIFY = ROOT / "claudeCode/verify_method_sir_pipeline.py"
NATIVE_THRESHOLD = 0.65
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]


def brightness_true_cells() -> set:
    wb = pd.read_excel(WORKBOOK, "Model selected", header=0).dropna(axis=1, how="all")
    wb = wb[["organism_group", "antimicrobial", "Brightness"]].dropna(subset=["organism_group"])
    wb["organism_group"] = wb["organism_group"].astype(str).str.strip()
    wb["antimicrobial"] = wb["antimicrobial"].astype(str).str.strip()
    return set(zip(wb.loc[wb["Brightness"] == True, "organism_group"],   # noqa: E712
                   wb.loc[wb["Brightness"] == True, "antimicrobial"]))


def calibrate_variant(base: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    df = base.copy()
    df["dtw_model_pred"] = np.clip(score, 0.0, 1.0)
    iso = PerCellIsotonicLookup.fit(df)
    df = iso.calibrate_dataframe(df, out_col="_cal")
    df["dtw_model_pred"] = df["_cal"]
    return df.drop(columns=["_cal"])


def run_pipeline(per_conc_csv: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(VERIFY), "--per_conc_csv", str(per_conc_csv),
                    "--output_dir", str(out_dir), "--native_threshold", str(NATIVE_THRESHOLD)],
                   check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    return pd.read_csv(out_dir / "method_summary_model_pred.csv")


def pass_vme(summ: pd.DataFrame) -> pd.DataFrame:
    s = summ[["organism_group", "antimicrobial", "FDA_fail_list"]].copy()
    s["pass"] = s["FDA_fail_list"] == "PASS"
    s["vme"] = s["FDA_fail_list"].astype(str).str.contains("VME")
    return s


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = brightness_true_cells()
    base_full = pd.read_csv(BASE_PER_CONC, low_memory=False)
    base = base_full[base_full.apply(
        lambda r: (str(r["organism_group"]).strip(),
                   str(r["antimicrobial"]).strip()) in cells, axis=1)].copy()
    n_cells = base.groupby(["organism_group", "antimicrobial"]).ngroups
    print(f"[base] Brightness=True {n_cells} cells / {len(base)} rows")

    # brightness GBM score
    br = pd.read_csv(BRIGHT_OOF, usecols=KEY + ["brightness_pred"])
    base = base.merge(br, on=KEY, how="left")

    # FDA brightness 편차 feature (control vs drug) 재계산
    print("[load] FDA brightness features (편차 gate 계산)")
    feat = build_features(FDA_CSV, "brightness", label_col=None)
    feat["gate_raw"] = (feat["oa_ratio_t6"] - 1.0).abs()   # |drug/ctl - 1| 마지막 timepoint
    base = base.merge(feat[KEY + ["gate_raw"]], on=KEY, how="left")
    base["gate_raw"] = base["gate_raw"].fillna(0.0)
    base["brightness_pred"] = base["brightness_pred"].fillna(base["dtw_model_pred"])

    img = base["dtw_model_pred"].astype(float).to_numpy()
    brp = base["brightness_pred"].astype(float).to_numpy()
    gate = base["gate_raw"].astype(float).to_numpy()
    print(f"[gate] |drug/ctl-1| 분포: median {np.median(gate):.3f} "
          f"p75 {np.percentile(gate,75):.3f} p90 {np.percentile(gate,90):.3f}")

    variants = {"image_only": img, "fixed_br30": 0.7 * img + 0.3 * brp}
    # adaptive: 편차 상위 q% row에만 w_br 부여
    for q in (50, 33, 25):
        thr = np.percentile(gate, 100 - q)
        for w in (0.3, 0.5):
            mask = gate >= thr
            wvec = np.where(mask, w, 0.0)
            variants[f"adapt_q{q}_w{int(w*100)}"] = (1 - wvec) * img + wvec * brp
            if w == 0.5:
                print(f"  adapt_q{q}: gate>={thr:.3f} → {int(mask.sum())}/{len(mask)} rows weighted")

    summaries = {}
    for name, score in variants.items():
        pc = calibrate_variant(base.drop(columns=["brightness_pred", "gate_raw"]), score)
        pc.to_csv(OUT_DIR / f"per_conc_{name}.csv", index=False)
        summaries[name] = pass_vme(run_pipeline(OUT_DIR / f"per_conc_{name}.csv",
                                                OUT_DIR / f"sir_{name}"))

    merged = None
    for name, s in summaries.items():
        c = s.rename(columns={"pass": f"{name}_pass", "vme": f"{name}_vme",
                              "FDA_fail_list": f"{name}_fail"})
        merged = c if merged is None else merged.merge(
            c, on=["organism_group", "antimicrobial"], how="outer")
    merged.to_csv(OUT_DIR / "brightness_adaptive_compare.csv", index=False)

    img_pass = int(merged["image_only_pass"].sum()); img_vme = int(merged["image_only_vme"].sum())
    print(f"\n===== Brightness=True {len(merged)} cells (image_only: PASS {img_pass}, VME {img_vme}) =====")
    h2h = []
    for name in variants:
        p = int(merged[f"{name}_pass"].sum()); v = int(merged[f"{name}_vme"].sum())
        gain = int(((~merged["image_only_pass"]) & merged[f"{name}_pass"]).sum())
        loss = int((merged["image_only_pass"] & (~merged[f"{name}_pass"])).sum())
        vnew = int(((~merged["image_only_vme"]) & merged[f"{name}_vme"]).sum())
        vfix = int((merged["image_only_vme"] & (~merged[f"{name}_vme"])).sum())
        tag = "" if name == "image_only" else f"  ΔPASS {p-img_pass:+d}(g{gain}/l{loss}) ΔVME {v-img_vme:+d}(fix{vfix}/new{vnew})"
        print(f"  {name:16s} PASS {p:3d}  VME {v:2d}{tag}")
        h2h.append({"variant": name, "pass": p, "vme": v,
                    "net_pass": p - img_pass, "net_vme": v - img_vme,
                    "pass_gain": gain, "pass_loss": loss, "vme_fixed": vfix, "vme_new": vnew})

    ens = [r for r in h2h if r["variant"] != "image_only"]
    best = max(ens, key=lambda r: (r["net_pass"], -r["net_vme"]))
    print(f"\n  최선 변형: {best['variant']}  ΔPASS {best['net_pass']:+d}  ΔVME {best['net_vme']:+d}")
    verdict = ("개선" if best["net_pass"] > 0 and best["net_vme"] <= 0 else
               "VME 증가 동반" if best["net_pass"] > 0 else "개선 없음")
    print(f"  판정: {verdict}")
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "n_cells": int(len(merged)), "image_only_pass": img_pass, "image_only_vme": img_vme,
        "gate": "|brightness_drug/ctl_t6 - 1|", "results": h2h, "best": best, "verdict": verdict,
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'brightness_adaptive_compare.csv'}")


if __name__ == "__main__":
    main()

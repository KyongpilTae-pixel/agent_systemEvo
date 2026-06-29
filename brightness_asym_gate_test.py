#!/usr/bin/env python3
"""비대칭 brightness gating — brightness 를 G 방향(P(NG) 낮추는 쪽)으로만 반영.

근거: 운영 score = P(NG). VME(=거짓 NG, 실제 G인데 NG 예측 → MIC 과소 → S 오보고)는
brightness 가 P(NG)를 *올릴* 때 생긴다. 따라서 brightness 가 image 보다 더 G 라고 말할
때만(brp < img) 점수를 내리고, 더 NG 라고 말할 때(brp >= img)는 image 유지 →
false-NG 신규 발생(VME)을 구조적으로 차단하면서, ME/EA 개선(G 방향)에서 오는 PASS
이득은 보존.

추가로 control-약제 brightness 편차 gate(|drug/ctl_t6-1|) 상위 row 한정 옵션 병행.
대상: Brightness=True 45 cell. recipe: per-cell ISO(FDA fit) → t0.65 → SIR/PASS.
출력: agent_system/output/brightness_asym/
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
OUT_DIR = ROOT / "agent_system/output/brightness_asym"
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


def pass_vme(summ):
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
    print(f"[base] Brightness=True "
          f"{base.groupby(['organism_group','antimicrobial']).ngroups} cells / {len(base)} rows")

    br = pd.read_csv(BRIGHT_OOF, usecols=KEY + ["brightness_pred"])
    base = base.merge(br, on=KEY, how="left")
    feat = build_features(FDA_CSV, "brightness", label_col=None)
    feat["gate_raw"] = (feat["oa_ratio_t6"] - 1.0).abs()
    base = base.merge(feat[KEY + ["gate_raw"]], on=KEY, how="left")
    base["gate_raw"] = base["gate_raw"].fillna(0.0)
    base["brightness_pred"] = base["brightness_pred"].fillna(base["dtw_model_pred"])

    img = base["dtw_model_pred"].astype(float).to_numpy()
    brp = base["brightness_pred"].astype(float).to_numpy()
    gate = base["gate_raw"].astype(float).to_numpy()
    g_dir = brp < img                       # brightness says "more G" (lower P(NG))
    thr33 = np.percentile(gate, 67)         # top-33% deviation

    def asym(w, gate_mask=None):
        m = g_dir if gate_mask is None else (g_dir & gate_mask)
        wv = np.where(m, w, 0.0)
        return (1 - wv) * img + wv * brp

    variants = {
        "image_only": img,
        "sym_q33_w30": None,                # filled below (reference, symmetric best)
        "asym_w30": asym(0.30),
        "asym_w50": asym(0.50),
        "asym_w70": asym(0.70),
        "asym_gate33_w50": asym(0.50, gate >= thr33),
    }
    # symmetric reference (both directions, top-33% gate, w0.3)
    wv = np.where(gate >= thr33, 0.30, 0.0)
    variants["sym_q33_w30"] = (1 - wv) * img + wv * brp

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
    merged.to_csv(OUT_DIR / "brightness_asym_compare.csv", index=False)

    ip = int(merged["image_only_pass"].sum()); iv = int(merged["image_only_vme"].sum())
    print(f"\n===== Brightness=True {len(merged)} cells (image_only PASS {ip} / VME {iv}) =====")
    res = []
    for name in variants:
        p = int(merged[f"{name}_pass"].sum()); v = int(merged[f"{name}_vme"].sum())
        gain = int(((~merged["image_only_pass"]) & merged[f"{name}_pass"]).sum())
        loss = int((merged["image_only_pass"] & (~merged[f"{name}_pass"])).sum())
        vnew = int(((~merged["image_only_vme"]) & merged[f"{name}_vme"]).sum())
        vfix = int((merged["image_only_vme"] & (~merged[f"{name}_vme"])).sum())
        tag = "" if name == "image_only" else \
            f"  ΔPASS {p-ip:+d}(g{gain}/l{loss})  ΔVME {v-iv:+d}(fix{vfix}/new{vnew})"
        print(f"  {name:16s} PASS {p:3d}  VME {v:2d}{tag}")
        res.append({"variant": name, "pass": p, "vme": v, "net_pass": p-ip, "net_vme": v-iv,
                    "pass_gain": gain, "pass_loss": loss, "vme_fixed": vfix, "vme_new": vnew})

    # best with VME constraint (net_vme <= 0)
    safe = [r for r in res if r["variant"] != "image_only" and r["net_vme"] <= 0]
    best_safe = max(safe, key=lambda r: r["net_pass"]) if safe else None
    print(f"\n  VME 비악화(net_vme<=0) 중 최선: "
          f"{best_safe['variant'] if best_safe else '없음'}"
          + (f"  ΔPASS {best_safe['net_pass']:+d} ΔVME {best_safe['net_vme']:+d}" if best_safe else ""))
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "n_cells": int(len(merged)), "image_only_pass": ip, "image_only_vme": iv,
        "results": res, "best_vme_safe": best_safe,
    }, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT_DIR/'brightness_asym_compare.csv'}")


if __name__ == "__main__":
    main()

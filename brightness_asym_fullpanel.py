#!/usr/bin/env python3
"""asym G-방향 brightness gating 을 Brightness=True cell 에만 한정 적용하고
전체 313 cell 에서 운영 routed_iso_t65 (PASS 103) 대비 net PASS/VME 측정.

변형 (Brightness=True cell 만 model_pred 교체, 나머지 cell 은 모두 동일 = routed):
  baseline        : routed model_pred 그대로 (= 운영 routed_iso_t65 재현, PASS≈103)
  asym_br50/br70  : G-방향 blend ((1-w)*img + w*brp) where brp<img, w∈{0.5,0.7}
  brightness_only : Brightness=True cell 을 brightness GBM 단독으로 판단

각 변형: full-panel per-cell isotonic(FDA fit) → t0.65 → drast_gng→MIC→SIR→evalEA.
출력: agent_system/output/brightness_fullpanel/
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kptae/project/qnt_algorithm")
sys.path.insert(0, str(ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

WORKBOOK = ROOT / "claudeCode/data/F21_GN_Selecetd_model_260123 (3) (1).xlsx"
BASE_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
BRIGHT_OOF = ROOT / "agent_system/output/brightness_gbm/brightness_crossdomain_oof.csv"
OUT_DIR = ROOT / "agent_system/output/brightness_fullpanel"
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


def calibrate_full(base: pd.DataFrame, model_pred: np.ndarray) -> pd.DataFrame:
    df = base.copy()
    df["dtw_model_pred"] = np.clip(model_pred, 0.0, 1.0)
    iso = PerCellIsotonicLookup.fit(df)
    df = iso.calibrate_dataframe(df, out_col="_cal")
    df["dtw_model_pred"] = df["_cal"]
    return df.drop(columns=["_cal"])


def run_pipeline(per_conc_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(VERIFY), "--per_conc_csv", str(per_conc_csv),
                    "--output_dir", str(out_dir), "--native_threshold", str(NATIVE_THRESHOLD)],
                   check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    summ = pd.read_csv(out_dir / "method_summary_model_pred.csv")
    feas = pd.read_csv(out_dir / "method_sir_feasibility.csv")
    vme_rate = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    return summ, vme_rate


def main() -> None:
    global BRIGHT_OOF, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--bright_oof", default=str(BRIGHT_OOF))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    BRIGHT_OOF = Path(args.bright_oof)
    OUT_DIR = Path(args.out_dir)
    print(f"[config] bright_oof={BRIGHT_OOF.name}  out_dir={OUT_DIR.name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = brightness_true_cells()
    base = pd.read_csv(BASE_PER_CONC, low_memory=False)
    n_total = base.groupby(["organism_group", "antimicrobial"]).ngroups
    bt_mask = base.apply(lambda r: (str(r["organism_group"]).strip(),
                                    str(r["antimicrobial"]).strip()) in cells, axis=1).to_numpy()
    print(f"[panel] {n_total} cells, {len(base)} rows; Brightness=True rows {int(bt_mask.sum())}")

    br = pd.read_csv(BRIGHT_OOF, usecols=KEY + ["brightness_pred"])
    base = base.merge(br, on=KEY, how="left")
    img = base["dtw_model_pred"].astype(float).to_numpy()
    brp = base["brightness_pred"].fillna(base["dtw_model_pred"]).astype(float).to_numpy()
    g_dir = brp < img    # brightness says more-G (lower P(NG))

    def asym(w):
        mp = img.copy()
        sel = bt_mask & g_dir
        mp[sel] = (1 - w) * img[sel] + w * brp[sel]
        return mp

    def bright_only():
        mp = img.copy()
        mp[bt_mask] = brp[bt_mask]
        return mp

    variants = {
        "baseline": img,
        "asym_br50": asym(0.50),
        "asym_br70": asym(0.70),
        "brightness_only": bright_only(),
    }

    base_for_iso = base.drop(columns=["brightness_pred"])
    results = {}
    for name, mp in variants.items():
        pc = calibrate_full(base_for_iso, mp)
        csv = OUT_DIR / f"per_conc_{name}.csv"
        pc.to_csv(csv, index=False)
        summ, vme_rate = run_pipeline(csv, OUT_DIR / f"sir_{name}")
        results[name] = (summ, vme_rate)
        print(f"[done] {name}")

    # merge per-cell PASS across variants on full panel
    merged = None
    for name, (summ, _) in results.items():
        s = summ[["organism_group", "antimicrobial", "FDA_fail_list"]].copy()
        s[f"{name}_pass"] = s["FDA_fail_list"] == "PASS"
        s[f"{name}_vme"] = s["FDA_fail_list"].astype(str).str.contains("VME")
        s = s.rename(columns={"FDA_fail_list": f"{name}_fail"})
        merged = s if merged is None else merged.merge(
            s, on=["organism_group", "antimicrobial"], how="outer")
    merged.to_csv(OUT_DIR / "fullpanel_compare.csv", index=False)

    bp = int(merged["baseline_pass"].sum()); bv = int(merged["baseline_vme"].sum())
    print(f"\n===== 전체 {len(merged)} cell — Brightness=True 한정 적용 =====")
    print(f"  baseline (운영 routed_iso_t65 재현)  PASS {bp}  VME-cell {bv}  "
          f"VME-rate {results['baseline'][1]:.4f}")
    summary = [{"variant": "baseline", "pass": bp, "vme_cells": bv,
                "vme_rate": round(results['baseline'][1], 4),
                "net_pass": 0, "net_vme_cells": 0}]
    for name in ("asym_br50", "asym_br70", "brightness_only"):
        p = int(merged[f"{name}_pass"].sum()); v = int(merged[f"{name}_vme"].sum())
        vr = results[name][1]
        gain = int(((~merged["baseline_pass"]) & merged[f"{name}_pass"]).sum())
        loss = int((merged["baseline_pass"] & (~merged[f"{name}_pass"])).sum())
        print(f"  {name:16s}  PASS {p}  VME-cell {v}  VME-rate {vr:.4f}   "
              f"ΔPASS {p-bp:+d}(g{gain}/l{loss})  ΔVME-cell {v-bv:+d}")
        summary.append({"variant": name, "pass": p, "vme_cells": v, "vme_rate": round(vr, 4),
                        "net_pass": p - bp, "net_vme_cells": v - bv,
                        "pass_gain": gain, "pass_loss": loss})

    # which cells flipped for asym_br70
    v = "asym_br70"
    gain = merged[(~merged["baseline_pass"]) & merged[f"{v}_pass"]]
    loss = merged[merged["baseline_pass"] & (~merged[f"{v}_pass"])]
    print(f"\n  [{v} PASS gain]")
    print(gain[["organism_group", "antimicrobial", "baseline_fail", f"{v}_fail"]].to_string(index=False))
    if len(loss):
        print(f"  [{v} PASS loss]")
        print(loss[["organism_group", "antimicrobial", "baseline_fail", f"{v}_fail"]].to_string(index=False))

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "n_cells": int(len(merged)), "native_threshold": NATIVE_THRESHOLD,
        "scope": "asym G-direction brightness applied to Brightness=True cells only",
        "baseline_is_routed_iso_t65": True, "summary": summary,
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'fullpanel_compare.csv'}")


if __name__ == "__main__":
    main()

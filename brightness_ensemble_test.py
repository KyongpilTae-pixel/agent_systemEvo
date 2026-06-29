#!/usr/bin/env python3
"""Brightness=True (image sequence 표현력 낮은) cell에서 object_area 앙상블이
image-only 대비 운영 PASS/VME를 개선하는지 검증.

워크북 F21_GN의 Brightness=True 45 cell(FDA에 존재)을 대상으로, 동일 운영 recipe
(per-cell isotonic(FDA fit) → threshold 0.65 → drast_gng→MIC→SIR→evalEA)를
3개 score 변형에 적용:

  image_only   : routed model_pred                     (운영 baseline)
  prob_ens     : 0.5*model_pred + 0.5*objarea_gbm       (확률공간 평균; ShiftedCellEnsemble식)
  objarea_only : objarea_gbm (cross-domain OOF)

확률공간 평균 후 ISO를 fit → rank_avg+ISO scale mismatch 회피.
출력: agent_system/output/brightness_ensemble/
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
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

WORKBOOK = ROOT / "claudeCode/data/F21_GN_Selecetd_model_260123 (3) (1).xlsx"
BASE_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
GBM_OOF = ROOT / "claudeCode/output_dtw_aggregate_full/objarea_crossdomain_oof.csv"
OUT_DIR = ROOT / "agent_system/output/brightness_ensemble"
VERIFY = ROOT / "claudeCode/verify_method_sir_pipeline.py"
NATIVE_THRESHOLD = 0.65

KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]


def brightness_true_cells() -> set[tuple[str, str]]:
    wb = pd.read_excel(WORKBOOK, "Model selected", header=0).dropna(axis=1, how="all")
    wb = wb[["organism_group", "antimicrobial", "Brightness"]].dropna(subset=["organism_group"])
    wb["organism_group"] = wb["organism_group"].astype(str).str.strip()
    wb["antimicrobial"] = wb["antimicrobial"].astype(str).str.strip()
    t = wb[wb["Brightness"] == True]  # noqa: E712
    return set(zip(t["organism_group"], t["antimicrobial"]))


def calibrate_variant(base: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    df = base.copy()
    df["dtw_model_pred"] = np.clip(score, 0.0, 1.0)
    iso = PerCellIsotonicLookup.fit(df)
    df = iso.calibrate_dataframe(df, out_col="_cal")
    df["dtw_model_pred"] = df["_cal"]
    return df.drop(columns=["_cal"])


def run_pipeline(per_conc_csv: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(VERIFY), "--per_conc_csv", str(per_conc_csv),
           "--output_dir", str(out_dir), "--native_threshold", str(NATIVE_THRESHOLD)]
    print("[run]", out_dir.name)
    subprocess.run(cmd, check=True, cwd=str(ROOT),
                   stdout=subprocess.DEVNULL)
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

    gbm = pd.read_csv(GBM_OOF, usecols=KEY + ["objarea_crossdomain_pred"])
    base = base.merge(gbm, on=KEY, how="left")
    n_miss = int(base["objarea_crossdomain_pred"].isna().sum())
    if n_miss:
        print(f"[warn] {n_miss} rows missing objarea_gbm → fallback to model_pred")
        base["objarea_crossdomain_pred"] = base["objarea_crossdomain_pred"].fillna(
            base["dtw_model_pred"])

    img = base["dtw_model_pred"].astype(float).to_numpy()
    obj = base["objarea_crossdomain_pred"].astype(float).to_numpy()
    variants = {
        "image_only": img,
        "prob_ens": 0.5 * img + 0.5 * obj,
        "objarea_only": obj,
    }

    summaries = {}
    for name, score in variants.items():
        pc = calibrate_variant(base.drop(columns=["objarea_crossdomain_pred"]), score)
        csv = OUT_DIR / f"per_conc_{name}.csv"
        pc.to_csv(csv, index=False)
        summaries[name] = pass_vme(run_pipeline(csv, OUT_DIR / f"sir_{name}"))

    # merge per-cell PASS/VME across variants
    merged = None
    for name, s in summaries.items():
        cols = s.rename(columns={"pass": f"{name}_pass", "vme": f"{name}_vme",
                                 "FDA_fail_list": f"{name}_fail"})
        merged = cols if merged is None else merged.merge(
            cols, on=["organism_group", "antimicrobial"], how="outer")
    merged.to_csv(OUT_DIR / "brightness_ensemble_compare.csv", index=False)

    print("\n===== Brightness=True cells: variant PASS / VME =====")
    summ_rows = []
    for name in variants:
        p = int(merged[f"{name}_pass"].sum())
        v = int(merged[f"{name}_vme"].sum())
        print(f"  {name:14s}  PASS {p:3d} / {len(merged)}   VME cells {v}")
        summ_rows.append({"variant": name, "pass": p, "vme_cells": v, "n": len(merged)})

    # head-to-head: ens vs image_only
    flips_gain = merged[(~merged["image_only_pass"]) & merged["prob_ens_pass"]]
    flips_loss = merged[merged["image_only_pass"] & (~merged["prob_ens_pass"])]
    vme_fix = merged[merged["image_only_vme"] & (~merged["prob_ens_vme"])]
    vme_new = merged[(~merged["image_only_vme"]) & merged["prob_ens_vme"]]
    print(f"\n  prob_ens vs image_only:  PASS gain {len(flips_gain)} / loss {len(flips_loss)}")
    print(f"  VME fixed {len(vme_fix)} / newly introduced {len(vme_new)}")
    if len(flips_gain):
        print("  [PASS gain cells]")
        print(flips_gain[["organism_group", "antimicrobial",
                          "image_only_fail", "prob_ens_fail"]].to_string(index=False))
    if len(flips_loss):
        print("  [PASS loss cells]")
        print(flips_loss[["organism_group", "antimicrobial",
                          "image_only_fail", "prob_ens_fail"]].to_string(index=False))

    manifest = {
        "n_cells": int(len(merged)), "native_threshold": NATIVE_THRESHOLD,
        "variants": summ_rows,
        "prob_ens_vs_image": {
            "pass_gain": int(len(flips_gain)), "pass_loss": int(len(flips_loss)),
            "vme_fixed": int(len(vme_fix)), "vme_new": int(len(vme_new)),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'brightness_ensemble_compare.csv'}")
    print(f"[saved] {OUT_DIR/'manifest.json'}")


if __name__ == "__main__":
    main()

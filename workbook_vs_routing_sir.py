#!/usr/bin/env python3
"""Task 1 — 워크북(F21_GN) 선정 모델 vs 우리 AUROC-routing 선정 모델, 실질 충돌
26 cell을 운영 SIR/PASS 파이프라인으로 비교.

두 시스템은 동일 cell에서 서로 다른 모델을 고른다 (gap >= 0.02 인 26 cell).
각 cell에 대해 후보 두 모델의 per-conc 예측을 가져와 운영 recipe와 동일하게
  raw model_pred → per-cell isotonic (FDA full-data fit) → threshold 0.65
  → production drast_gng → determineMIC → interpretSIR → evalEA
를 돌려 FDA PASS / VME / EA / CA 를 cell 단위로 비교한다.

26 cell만 다루므로 두 variant 의 나머지 cell 은 비교 대상이 아니다.
출력: agent_system/output/workbook_vs_routing/
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
ROUTING_LOOKUP = ROOT / "claudeCode/output_subset_eval/cell_model_routing_lookup.csv"
BASE_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
SUBSET_DIR = ROOT / "claudeCode/output_subset_eval"   # <model>/model_pred.csv
OUT_DIR = ROOT / "agent_system/output/workbook_vs_routing"
VERIFY = ROOT / "claudeCode/verify_method_sir_pipeline.py"

GAP_THRESHOLD = 0.02
NATIVE_THRESHOLD = 0.65

NAME_NORM = {
    "deployed_model": "same_mic",
    "object_area_045~08_threshold": "object_area_045~08",
    "object_area_all_threshold": "object_area_all",
}


def load_disagreements() -> pd.DataFrame:
    ours = pd.read_csv(ROUTING_LOOKUP)
    wb = pd.read_excel(WORKBOOK, "Model selected", header=0).dropna(axis=1, how="all").iloc[:, :3]
    wb.columns = ["organism_group", "antimicrobial", "Model"]
    wb = wb.dropna(subset=["organism_group", "Model"])
    wb["wb_model"] = wb["Model"].map(lambda m: NAME_NORM.get(str(m).strip(), str(m).strip()))
    for d in (wb, ours):
        d["organism_group"] = d["organism_group"].astype(str).str.strip()
        d["antimicrobial"] = d["antimicrobial"].astype(str).str.strip()
    both = wb.merge(ours, on=["organism_group", "antimicrobial"], how="inner")
    dis = both[(both["wb_model"] != both["best_model"]) & (both["gap"] >= GAP_THRESHOLD)]
    return dis[["organism_group", "antimicrobial", "wb_model", "best_model",
                "best_auroc", "runner_up", "gap"]].reset_index(drop=True)


def model_pred_lookup(model: str) -> pd.DataFrame:
    """(sample_id, antimicrobial, concentration_idx_0) -> model_pred for one model."""
    csv = SUBSET_DIR / model / "model_pred.csv"
    df = pd.read_csv(csv, usecols=["sample_id", "antimicrobial",
                                    "concentration_idx_0", "model_pred"])
    return df.rename(columns={"model_pred": f"_mp_{model}"})


def build_variant(base: pd.DataFrame, cells: pd.DataFrame, choice_col: str,
                  models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each disagreement cell, overwrite dtw_model_pred with the chosen
    model's raw prediction; then per-cell isotonic-calibrate the whole frame."""
    df = base.copy()
    key = ["sample_id", "antimicrobial", "concentration_idx_0"]
    for _, c in cells.iterrows():
        model = c[choice_col]
        mask = ((df["organism_group"] == c["organism_group"]) &
                (df["antimicrobial"] == c["antimicrobial"]))
        if not mask.any():
            continue
        sub = df.loc[mask, key].merge(models[model], on=key, how="left")
        df.loc[mask, "dtw_model_pred"] = sub[f"_mp_{model}"].to_numpy()
    # per-cell isotonic on full (26-cell) frame, then write calibrated back
    iso = PerCellIsotonicLookup.fit(df)
    df = iso.calibrate_dataframe(df, out_col="_cal")
    df["dtw_model_pred"] = df["_cal"]
    return df.drop(columns=["_cal"]), iso


def run_pipeline(per_conc_csv: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(VERIFY),
           "--per_conc_csv", str(per_conc_csv),
           "--output_dir", str(out_dir),
           "--native_threshold", str(NATIVE_THRESHOLD)]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return pd.read_csv(out_dir / "method_summary_model_pred.csv")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_disagreements()
    print(f"[cells] {len(cells)} real-disagreement cells (gap>={GAP_THRESHOLD})")

    needed = sorted(set(cells["wb_model"]) | set(cells["best_model"]))
    print(f"[models] loading per-conc preds for: {needed}")
    models = {m: model_pred_lookup(m) for m in needed}

    base_full = pd.read_csv(BASE_PER_CONC, low_memory=False)
    pair = set(zip(cells["organism_group"], cells["antimicrobial"]))
    base = base_full[base_full.apply(
        lambda r: (r["organism_group"], r["antimicrobial"]) in pair, axis=1)].copy()
    print(f"[base] restricted to {base.groupby(['organism_group','antimicrobial']).ngroups} "
          f"cells / {len(base)} rows")

    our_pc, _ = build_variant(base, cells, "best_model", models)
    wb_pc, _ = build_variant(base, cells, "wb_model", models)
    our_csv = OUT_DIR / "per_conc_our_iso.csv"
    wb_csv = OUT_DIR / "per_conc_wb_iso.csv"
    our_pc.to_csv(our_csv, index=False)
    wb_pc.to_csv(wb_csv, index=False)

    our_sum = run_pipeline(our_csv, OUT_DIR / "sir_our")
    wb_sum = run_pipeline(wb_csv, OUT_DIR / "sir_wb")

    keep = ["organism_group", "antimicrobial", "FDA_fail_list",
            "EA", "CA", "VME", "ME", "mE", "total", "S", "R"]
    keep_our = [c for c in keep if c in our_sum.columns]
    o = our_sum[keep_our].add_prefix("our_").rename(
        columns={"our_organism_group": "organism_group", "our_antimicrobial": "antimicrobial"})
    w = wb_sum[keep_our].add_prefix("wb_").rename(
        columns={"wb_organism_group": "organism_group", "wb_antimicrobial": "antimicrobial"})
    comp = cells.merge(o, on=["organism_group", "antimicrobial"], how="left") \
                .merge(w, on=["organism_group", "antimicrobial"], how="left")
    comp["our_pass"] = (comp["our_FDA_fail_list"] == "PASS")
    comp["wb_pass"] = (comp["wb_FDA_fail_list"] == "PASS")
    # VME presence per cell from the FDA_fail_list string (robust to NaN VME counts)
    comp["our_has_vme"] = comp["our_FDA_fail_list"].astype(str).str.contains("VME")
    comp["wb_has_vme"] = comp["wb_FDA_fail_list"].astype(str).str.contains("VME")

    def outcome(r):
        if r["our_pass"] and not r["wb_pass"]:
            return "우리 우위 (our PASS, wb FAIL)"
        if r["wb_pass"] and not r["our_pass"]:
            return "워크북 우위 (wb PASS, our FAIL)"
        if r["our_pass"] and r["wb_pass"]:
            return "동률 (둘 다 PASS)"
        return "동률 (둘 다 FAIL)"
    comp["outcome"] = comp.apply(outcome, axis=1)
    comp.to_csv(OUT_DIR / "comparison_26cells.csv", index=False)

    print("\n===== 26-cell SIR/PASS 비교 =====")
    print(comp["outcome"].value_counts().to_string())
    print(f"\nour PASS: {int(comp['our_pass'].sum())} / wb PASS: {int(comp['wb_pass'].sum())}")
    print(f"VME-incurring cells  our={int(comp['our_has_vme'].sum())} "
          f"wb={int(comp['wb_has_vme'].sum())}")
    print(f"  our-VME but wb-clean: {int((comp['our_has_vme'] & ~comp['wb_has_vme']).sum())} | "
          f"wb-VME but our-clean: {int((comp['wb_has_vme'] & ~comp['our_has_vme']).sum())}")
    for col in ("VME", "ME"):
        oc, wc = f"our_{col}", f"wb_{col}"
        if oc in comp.columns:
            print(f"{col} count sum  our={int(pd.to_numeric(comp[oc],errors='coerce').fillna(0).sum())} "
                  f"wb={int(pd.to_numeric(comp[wc],errors='coerce').fillna(0).sum())}")
    pd.set_option("display.width", 220)
    show = comp[["organism_group", "antimicrobial", "best_model", "wb_model", "gap",
                 "our_FDA_fail_list", "wb_FDA_fail_list", "outcome"]]
    print("\n" + show.to_string(index=False))

    manifest = {
        "n_cells": int(len(cells)),
        "gap_threshold": GAP_THRESHOLD,
        "native_threshold": NATIVE_THRESHOLD,
        "our_pass": int(comp["our_pass"].sum()),
        "wb_pass": int(comp["wb_pass"].sum()),
        "outcome_counts": comp["outcome"].value_counts().to_dict(),
        "vme_cells_our": int(comp["our_has_vme"].sum()),
        "vme_cells_wb": int(comp["wb_has_vme"].sum()),
        "vme_our_only": int((comp["our_has_vme"] & ~comp["wb_has_vme"]).sum()),
        "vme_wb_only": int((comp["wb_has_vme"] & ~comp["our_has_vme"]).sum()),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'comparison_26cells.csv'}")
    print(f"[saved] {OUT_DIR/'manifest.json'}")


if __name__ == "__main__":
    main()

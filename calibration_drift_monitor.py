"""Calibration drift monitoring hook (P5, 2026-05-28).

운영 ISO lookup이 시간이 지남에 따라 신규 sample의 score 분포와 어긋날 수 있음.
정기적으로 baseline ISO와 fresh fit ISO를 비교해 drift signal 측정.

Inputs:
  - baseline ISO: agent_system/output/isotonic_lookup_routed.csv (286 cells, 2026-05-27 fit)
  - fresh per_conc CSV: 신규 inference 결과 (e.g., 새 FDA batch)

Outputs (per cell):
  - drift score (cell-level x_threshold/y_threshold L1 distance)
  - PASS delta (baseline ISO vs fresh ISO applied to same fresh data)
  - alert flag (drift > threshold)

Hook 사용:
  - 운영 1회/주: 신규 inference batch 모이면 이 스크립트 실행
  - drift signal > T 시 ISO refit 권고

CLI:
    python -m agent_system.calibration_drift_monitor \\
        --baseline_iso agent_system/output/isotonic_lookup_routed.csv \\
        --fresh_per_conc claudeCode/output_subset_eval/dtw_per_conc_routed.csv \\
        --output_dir agent_system/output/drift_reports/2026-05-28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/home/kptae/project/qnt_algorithm")
DEFAULT_BASELINE_ISO = ROOT / "agent_system/output/isotonic_lookup_routed.csv"
DEFAULT_FRESH = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
DRIFT_ALERT_L1 = 0.05  # mean |Δy| threshold
PASS_ALERT_DELTA = 5    # ≥5 cells PASS change

MIN_ROWS = 8


def fit_iso(x, y):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x.astype(float), y.astype(int))
    return iso


def load_baseline_iso(path: Path) -> dict:
    """Load PerCellIsotonicLookup-format CSV into a dict
    (organism_group, antimicrobial) -> (x_thresholds, y_thresholds)."""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[(str(r.organism_group), str(r.antimicrobial))] = (
            np.array(json.loads(r.x_thresholds), dtype=float),
            np.array(json.loads(r.y_thresholds), dtype=float),
        )
    return out


def fit_fresh_per_cell(fresh_csv: Path,
                       score_col: str = "dtw_model_pred",
                       label_col: str = "gt_gng") -> dict:
    df = pd.read_csv(fresh_csv, low_memory=False)
    out = {}
    for (og, drug), sub in df.dropna(
        subset=["organism_group", score_col, label_col]
    ).groupby(["organism_group", "antimicrobial"], sort=False):
        y = sub[label_col].astype(int).to_numpy()
        x = sub[score_col].astype(float).to_numpy()
        if len(sub) < MIN_ROWS or len(set(y)) < 2:
            continue
        iso = fit_iso(x, y)
        out[(str(og), str(drug))] = (
            iso.X_thresholds_.astype(float),
            iso.y_thresholds_.astype(float),
        )
    return out


def measure_cell_drift(base_xy, fresh_xy, n_grid: int = 100) -> dict:
    """Compare two ISO curves at common grid points → L1, L_inf, MSE distance."""
    base_x, base_y = base_xy
    fresh_x, fresh_y = fresh_xy
    lo = float(max(base_x.min(), fresh_x.min()))
    hi = float(min(base_x.max(), fresh_x.max()))
    if hi <= lo:
        return {"l1": float("nan"), "linf": float("nan"), "mse": float("nan"),
                "overlap_range": [lo, hi]}
    g = np.linspace(lo, hi, n_grid)
    yb = np.interp(g, base_x, base_y)
    yf = np.interp(g, fresh_x, fresh_y)
    return {
        "l1": float(np.mean(np.abs(yb - yf))),
        "linf": float(np.max(np.abs(yb - yf))),
        "mse": float(np.mean((yb - yf) ** 2)),
        "overlap_range": [lo, hi],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline_iso", default=str(DEFAULT_BASELINE_ISO))
    p.add_argument("--fresh_per_conc", default=str(DEFAULT_FRESH))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--score_col", default="dtw_model_pred")
    p.add_argument("--label_col", default="gt_gng")
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] baseline ISO: {args.baseline_iso}")
    baseline = load_baseline_iso(Path(args.baseline_iso))
    print(f"  cells: {len(baseline)}")

    print(f"[fit]  fresh per-cell ISO: {args.fresh_per_conc}")
    fresh = fit_fresh_per_cell(Path(args.fresh_per_conc),
                               args.score_col, args.label_col)
    print(f"  cells: {len(fresh)}")

    rows = []
    n_alert = 0
    for key in fresh:
        if key not in baseline:
            rows.append({
                "organism_group": key[0], "antimicrobial": key[1],
                "status": "new_cell",
                "l1": None, "linf": None, "mse": None,
                "alert": True,
            })
            n_alert += 1
            continue
        d = measure_cell_drift(baseline[key], fresh[key])
        is_alert = (d["l1"] is not None and not np.isnan(d["l1"])
                    and d["l1"] > DRIFT_ALERT_L1)
        rows.append({
            "organism_group": key[0], "antimicrobial": key[1],
            "status": "compared",
            **d,
            "alert": bool(is_alert),
        })
        if is_alert:
            n_alert += 1
    # cells in baseline missing from fresh
    for key in baseline:
        if key not in fresh and key != ("<GLOBAL>", "<GLOBAL>"):
            rows.append({
                "organism_group": key[0], "antimicrobial": key[1],
                "status": "missing_in_fresh",
                "l1": None, "linf": None, "mse": None,
                "alert": True,
            })
            n_alert += 1

    out = pd.DataFrame(rows)
    csv_path = out_dir / "drift_report.csv"
    out.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}  ({len(out)} cells)")

    # summary
    compared = out[out.status == "compared"]
    summary = {
        "baseline_iso": args.baseline_iso,
        "fresh_per_conc": args.fresh_per_conc,
        "n_cells_baseline": len(baseline),
        "n_cells_fresh": len(fresh),
        "n_cells_compared": int((out.status == "compared").sum()),
        "n_cells_new": int((out.status == "new_cell").sum()),
        "n_cells_missing": int((out.status == "missing_in_fresh").sum()),
        "n_alert": int(n_alert),
        "alert_threshold_l1": DRIFT_ALERT_L1,
        "mean_l1": float(compared["l1"].mean()) if len(compared) else None,
        "median_l1": float(compared["l1"].median()) if len(compared) else None,
        "max_l1": float(compared["l1"].max()) if len(compared) else None,
        "p95_l1": float(compared["l1"].quantile(0.95)) if len(compared) else None,
        "verdict": ("ALERT — refit ISO recommended" if n_alert >= 10 else
                    "OK — no significant drift"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary]")
    for k, v in summary.items():
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()

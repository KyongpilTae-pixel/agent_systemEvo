"""Per-cell ISO 재학습 helper — 임의 score column에 isotonic을 LOSO fit.

dtw_per_conc CSV의 dtw_model_pred 컬럼을 입력으로, per-cell GroupKFold(5)
by sample_id로 isotonic regression을 fit + OOF calibrated probability 산출.
calibrated score를 dtw_model_pred로 덮어쓴 새 CSV 출력.

용도: rank_avg / z_score 등 비-probability score를 ISO로 보정해 verify_method_sir_pipeline
(native_prob threshold 0.5)에 맞게 변환.

CLI:
  python -m agent_system.refit_iso_on_per_conc \
      --input  claudeCode/output_subset_eval/dtw_per_conc_shifted_rankavg3.csv \
      --output claudeCode/output_subset_eval/dtw_per_conc_shifted_rankavg3_iso.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

DEFAULT_N_FOLDS = 5
MIN_CELL_ROWS = 8
MIN_CELL_SAMPLES = 3


def refit_iso(input_csv: Path, output_csv: Path,
              score_col: str = "dtw_model_pred",
              label_col: str = "gt_gng",
              n_folds: int = DEFAULT_N_FOLDS) -> dict:
    df = pd.read_csv(input_csv, low_memory=False)
    n0 = len(df)
    print(f"[load] {input_csv}: {n0:,} rows")

    if "organism_group" not in df.columns or "antimicrobial" not in df.columns:
        raise ValueError("organism_group + antimicrobial 컬럼 필요")
    if score_col not in df.columns:
        raise ValueError(f"{score_col} 컬럼 없음")
    if label_col not in df.columns:
        raise ValueError(f"{label_col} 컬럼 없음")

    calibrated = df[score_col].astype(float).copy()  # default fallback = original
    n_cells_fit = 0
    n_cells_skipped = 0
    n_rows_replaced = 0

    for (og, drug), sub in df.dropna(subset=["organism_group", score_col, label_col])\
                              .groupby(["organism_group", "antimicrobial"], sort=False):
        n = len(sub)
        n_samples = sub["sample_id"].nunique()
        y = sub[label_col].astype(int).to_numpy()
        if n < MIN_CELL_ROWS or n_samples < MIN_CELL_SAMPLES or len(set(y)) < 2:
            n_cells_skipped += 1
            continue
        x = sub[score_col].astype(float).to_numpy()
        groups = sub["sample_id"].to_numpy()
        k = min(n_folds, n_samples)
        oof = np.full(len(sub), np.nan)
        for tri, vai in GroupKFold(n_splits=k).split(x, y, groups):
            if len(set(y[tri])) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x[tri], y[tri])
            oof[vai] = iso.predict(x[vai])
        # cell 전체에 fallback fit (val에 한 fold가 못 들어간 경우)
        if np.isnan(oof).any():
            iso_full = IsotonicRegression(out_of_bounds="clip")
            iso_full.fit(x, y)
            mask = np.isnan(oof)
            oof[mask] = iso_full.predict(x[mask])
        calibrated.loc[sub.index] = oof
        n_cells_fit += 1
        n_rows_replaced += int(len(sub))

    df[score_col] = calibrated
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    info = {
        "input": str(input_csv), "output": str(output_csv),
        "score_col": score_col, "n_rows": n0,
        "n_cells_fit": n_cells_fit, "n_cells_skipped": n_cells_skipped,
        "n_rows_replaced": n_rows_replaced,
    }
    print(f"[done] {info}")
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--score_col", default="dtw_model_pred")
    p.add_argument("--label_col", default="gt_gng")
    p.add_argument("--n_folds", default=DEFAULT_N_FOLDS, type=int)
    args = p.parse_args()
    refit_iso(args.input, args.output, args.score_col, args.label_col,
              args.n_folds)


if __name__ == "__main__":
    main()

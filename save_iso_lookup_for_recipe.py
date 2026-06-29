"""Per-cell ISO 곡선을 PerCellIsotonicLookup 포맷으로 저장 (운영 배포용).

`refit_iso_on_per_conc.py`는 OOF 검증용. 운영 배포 시에는 full-data fit ISO가
필요. 이 스크립트는:
  - per_conc CSV의 score_col로부터 cell마다 ISO 적합 (전체 data, no CV)
  - PerCellIsotonicLookup 형식 (x_thresholds + y_thresholds JSON) CSV 저장
  - 런타임: PerCellIsotonicLookup.load(...).calibrate(score, org, drug)

CLI:
  python -m agent_system.save_iso_lookup_for_recipe \\
      --input  claudeCode/output_subset_eval/dtw_per_conc_routed.csv \\
      --output agent_system/output/isotonic_lookup_routed.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

MIN_CELL_ROWS = 8


def build_lookup(input_csv: Path, output_csv: Path,
                 score_col: str = "dtw_model_pred",
                 label_col: str = "gt_gng") -> dict:
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"[load] {input_csv}: {len(df):,} rows")

    rows = []
    n_fit = 0
    n_skip = 0
    for (og, drug), sub in df.dropna(
        subset=["organism_group", score_col, label_col]
    ).groupby(["organism_group", "antimicrobial"], sort=False):
        y = sub[label_col].astype(int).to_numpy()
        x = sub[score_col].astype(float).to_numpy()
        if len(sub) < MIN_CELL_ROWS or len(set(y)) < 2:
            n_skip += 1
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        n_g = int((y == 0).sum())
        n_ng = int((y == 1).sum())
        rows.append({
            "organism_group": str(og), "antimicrobial": str(drug),
            "x_thresholds": json.dumps(iso.X_thresholds_.astype(float).tolist()),
            "y_thresholds": json.dumps(iso.y_thresholds_.astype(float).tolist()),
            "n_train_rows": int(len(sub)), "n_g": n_g, "n_ng": n_ng,
        })
        n_fit += 1

    # Global fallback row (identity)
    rows.append({
        "organism_group": "<GLOBAL>", "antimicrobial": "<GLOBAL>",
        "x_thresholds": json.dumps([0.0, 1.0]),
        "y_thresholds": json.dumps([0.0, 1.0]),
        "n_train_rows": 0, "n_g": 0, "n_ng": 0,
    })

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    info = {"input": str(input_csv), "output": str(output_csv),
            "n_cells_fit": n_fit, "n_cells_skipped": n_skip}
    print(f"[done] {info}")
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--score_col", default="dtw_model_pred")
    p.add_argument("--label_col", default="gt_gng")
    args = p.parse_args()
    build_lookup(args.input, args.output, args.score_col, args.label_col)


if __name__ == "__main__":
    main()

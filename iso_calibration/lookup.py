"""Operational per-cell isotonic calibration lookup.

Companion to `threshold_lookup.py`. Where the threshold lookup stores a single
cut-off τ per cell, this one stores the **full isotonic step function** per
cell so runtime can reproduce `isotonic(model_pred) → P(NG)`. This is the
recipe that gave the strongest VME reduction in our LOSO evaluation
(`calibration_ensemble.py`).

Storage format: one CSV row per cell, with two JSON-encoded list columns
holding sklearn `IsotonicRegression`'s `X_thresholds_` and `y_thresholds_`
(the breakpoints + plateau values of the step function). Runtime
classification uses `np.interp` for piecewise-linear interpolation, which
matches sklearn's internal predict path for monotonic targets (sklearn does
the same linear interp between thresholds).

Like `threshold_lookup.py`, this module fits on the *whole cell* for the
deployable artifact — LOSO is used elsewhere only to estimate honest
generalisation.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


_OG_COL = "organism_group"
_DRUG_COL = "antimicrobial"
_MODEL_COL = "dtw_model_pred"
_LABEL_COL = "gt_gng"


@dc.dataclass
class CellIsotonic:
    organism_group: str
    antimicrobial: str
    x_thresholds: list[float]   # sklearn IsotonicRegression.X_thresholds_
    y_thresholds: list[float]   # sklearn IsotonicRegression.y_thresholds_
    n_train_rows: int
    n_g: int
    n_ng: int


def _fit_iso(s: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, y)
    return iso


class PerCellIsotonicLookup:
    """Stores a fitted per-cell isotonic calibrator. The deployment runtime
    only needs numpy (no sklearn), since prediction is `np.interp` against the
    stored breakpoints.
    """

    def __init__(
        self,
        cells: dict[tuple[str, str], CellIsotonic],
        fallback_threshold: float = 0.5,
    ) -> None:
        self.cells = cells
        self.fallback_threshold = float(fallback_threshold)

    # -------------------------- construction --------------------------------
    @classmethod
    def fit(
        cls,
        per_conc: pd.DataFrame,
        min_rows: int = 30,
        min_per_class: int = 3,
        fallback_threshold: float = 0.5,
    ) -> "PerCellIsotonicLookup":
        df = per_conc.dropna(subset=[_MODEL_COL, _OG_COL, _DRUG_COL, _LABEL_COL])
        cells: dict[tuple[str, str], CellIsotonic] = {}
        for (og, amr), sub in df.groupby([_OG_COL, _DRUG_COL], sort=False):
            y = sub[_LABEL_COL].astype(int).values
            s = sub[_MODEL_COL].astype(float).values
            n_g = int((y == 0).sum()); n_ng = int((y == 1).sum())
            if len(sub) < min_rows or n_g < min_per_class or n_ng < min_per_class:
                continue
            iso = _fit_iso(s, y)
            # sklearn's IsotonicRegression exposes step breakpoints after fit:
            # `X_thresholds_` (sorted x's that define plateaus) and
            # `y_thresholds_` (corresponding y values).
            cells[(str(og), str(amr))] = CellIsotonic(
                organism_group=str(og), antimicrobial=str(amr),
                x_thresholds=iso.X_thresholds_.astype(float).tolist(),
                y_thresholds=iso.y_thresholds_.astype(float).tolist(),
                n_train_rows=int(len(sub)), n_g=n_g, n_ng=n_ng,
            )
        return cls(cells, fallback_threshold)

    # -------------------------- I/O ------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for c in self.cells.values():
            rows.append({
                "organism_group": c.organism_group,
                "antimicrobial": c.antimicrobial,
                "x_thresholds": json.dumps(c.x_thresholds),
                "y_thresholds": json.dumps(c.y_thresholds),
                "n_train_rows": c.n_train_rows,
                "n_g": c.n_g,
                "n_ng": c.n_ng,
            })
        df = pd.DataFrame(rows)
        # GLOBAL sentinel row: identity function as fallback (y = x clipped to [0,1])
        df = pd.concat([
            df,
            pd.DataFrame([{
                "organism_group": "<GLOBAL>",
                "antimicrobial": "<GLOBAL>",
                "x_thresholds": json.dumps([0.0, 1.0]),
                "y_thresholds": json.dumps([0.0, 1.0]),
                "n_train_rows": int(sum(c.n_train_rows for c in self.cells.values())),
                "n_g": 0, "n_ng": 0,
            }])
        ], ignore_index=True)
        return df

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PerCellIsotonicLookup":
        df = pd.read_csv(path)
        if "<GLOBAL>" not in df["organism_group"].values:
            raise ValueError(f"<GLOBAL> sentinel missing in {path}")
        cells: dict[tuple[str, str], CellIsotonic] = {}
        for _, r in df[df["organism_group"] != "<GLOBAL>"].iterrows():
            key = (str(r["organism_group"]), str(r["antimicrobial"]))
            cells[key] = CellIsotonic(
                organism_group=key[0], antimicrobial=key[1],
                x_thresholds=json.loads(r["x_thresholds"]),
                y_thresholds=json.loads(r["y_thresholds"]),
                n_train_rows=int(r["n_train_rows"]),
                n_g=int(r["n_g"]), n_ng=int(r["n_ng"]),
            )
        return cls(cells)

    # -------------------------- runtime ------------------------------------
    def calibrate(
        self,
        model_pred: Union[float, Iterable[float], np.ndarray],
        organism_group: str,
        antimicrobial: str,
    ) -> Union[float, np.ndarray]:
        """Apply the cell's isotonic step function to one or more model_pred
        values. Cells absent from the table fall back to the identity (no
        recalibration — equivalent to using raw model_pred)."""
        x = np.asarray(model_pred, dtype=float)
        cell = self.cells.get((organism_group, antimicrobial))
        if cell is None:
            out = np.clip(x, 0.0, 1.0)
        else:
            # np.interp matches sklearn IsotonicRegression's internal predict
            # (piecewise-linear between thresholds, clipped to [y_thresholds_[0],
            # y_thresholds_[-1]] = [0, 1] in our setup).
            out = np.interp(
                x,
                np.asarray(cell.x_thresholds, dtype=float),
                np.asarray(cell.y_thresholds, dtype=float),
            )
        if out.shape == ():
            return float(out)
        return out

    def classify(
        self,
        model_pred: Union[float, Iterable[float], np.ndarray],
        organism_group: str,
        antimicrobial: str,
        threshold: float | None = None,
    ) -> Union[int, np.ndarray]:
        thr = self.fallback_threshold if threshold is None else float(threshold)
        cal = self.calibrate(model_pred, organism_group, antimicrobial)
        x = np.asarray(cal, dtype=float)
        out = (x >= thr).astype(int)
        if out.shape == ():
            return int(out)
        return out

    def calibrate_dataframe(
        self, df: pd.DataFrame, out_col: str = "model_pred_calibrated",
    ) -> pd.DataFrame:
        result = df.copy()
        result[out_col] = np.clip(result[_MODEL_COL].astype(float), 0.0, 1.0)
        for (og, amr), sub_idx in result.groupby(
            [_OG_COL, _DRUG_COL], sort=False
        ).groups.items():
            cell = self.cells.get((str(og), str(amr)))
            if cell is None:
                continue
            xs = np.asarray(cell.x_thresholds, dtype=float)
            ys = np.asarray(cell.y_thresholds, dtype=float)
            scores = result.loc[sub_idx, _MODEL_COL].astype(float).values
            result.loc[sub_idx, out_col] = np.interp(scores, xs, ys)
        return result

    # -------------------------- diagnostics ---------------------------------
    def summary(self) -> dict:
        if not self.cells:
            return {"n_cells": 0}
        step_counts = [len(c.x_thresholds) for c in self.cells.values()]
        return {
            "n_cells": len(self.cells),
            "fallback_threshold": self.fallback_threshold,
            "median_n_steps_per_cell": float(np.median(step_counts)),
            "min_n_steps": int(np.min(step_counts)),
            "max_n_steps": int(np.max(step_counts)),
        }


# -----------------------------------------------------------------------------
# CLI: fit + sanity-apply to the source per_conc.
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per_conc_csv", default=Path("per_conc.csv"), type=Path,
                   help="입력 CSV (dtw_model_pred, gt_gng, organism_group, antimicrobial)")
    p.add_argument("--out_csv", default=Path("isotonic_lookup.csv"), type=Path,
                   help="동결 lookup 저장 경로")
    p.add_argument("--min_rows", default=30, type=int)
    p.add_argument("--fallback_threshold", default=0.5, type=float)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[load] {args.per_conc_csv}")
    df = pd.read_csv(args.per_conc_csv, low_memory=False)
    print(f"  rows: {len(df)}")

    lookup = PerCellIsotonicLookup.fit(
        df, min_rows=args.min_rows,
        fallback_threshold=args.fallback_threshold,
    )
    print("[summary]")
    for k, v in lookup.summary().items():
        print(f"  {k}: {v}")

    print(f"[save] {args.out_csv}")
    lookup.save(args.out_csv)
    size_kb = args.out_csv.stat().st_size / 1024
    print(f"  size: {size_kb:.1f} KB")

    print("[sanity] load + classify entire per_conc")
    reloaded = PerCellIsotonicLookup.load(args.out_csv)
    applied = reloaded.calibrate_dataframe(df.dropna(subset=[_MODEL_COL]))
    y = applied[_LABEL_COL].astype(int).values
    cal = applied["model_pred_calibrated"].astype(float).values
    raw = applied[_MODEL_COL].astype(float).values

    def cnt(score):
        pred = (score >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        return tp, fn, fp, tn

    print()
    print("== In-sample whole-dataset NG miss / G miss @ 0.5 ==")
    tp, fn, fp, tn = cnt(raw)
    print(f"  raw model_pred       NG miss = {fn:5d}   G miss = {fp:5d}")
    tp, fn, fp, tn = cnt(cal)
    print(f"  isotonic calibrated  NG miss = {fn:5d}   G miss = {fp:5d}")
    print("(LOSO honest estimate: calibration_ensemble_summary.csv.)")


if __name__ == "__main__":
    main()

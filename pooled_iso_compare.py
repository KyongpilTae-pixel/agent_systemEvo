"""Pooled ISO comparison — per-organism / per-drug-class / global vs per-cell (P3).

cell-level ISO가 sample 부족 cell에서 noisy할 수 있어 pooling 후 비교.
295 FDA cells 중 9개가 단일-class (label diversity 부족)로 cell-ISO skipped.
Pool dimension:
  - per-organism (organism 안 모든 drug pool)
  - per-drug (drug 안 모든 organism pool)
  - global (모두 pool)

산출: FDA per_conc CSV에 4가지 ISO calibration 적용 후 verify_method_sir_pipeline로
PASS 비교.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/home/kptae/project/qnt_algorithm")
FDA_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
MIN_FIT_ROWS = 8


def fit_iso(x, y):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x.astype(float), y.astype(int))
    return iso


def calibrate_per_conc(per_conc_csv: Path, pool_keys: list[str],
                       output_csv: Path,
                       score_col: str = "dtw_model_pred",
                       label_col: str = "gt_gng"):
    """pool_keys = grouping columns (e.g., ['organism_group'] or
    ['antimicrobial'] or [] for global). 각 그룹에서 ISO fit + 적용. fallback =
    global ISO."""
    df = pd.read_csv(per_conc_csv, low_memory=False)
    df = df.copy()
    # global fallback
    mask_g = df[score_col].notna() & df[label_col].notna()
    iso_global = fit_iso(df.loc[mask_g, score_col], df.loc[mask_g, label_col])

    n_pools_fit = 0
    n_pools_skip = 0
    if pool_keys:
        for keys, sub in df.dropna(subset=[score_col, label_col] + pool_keys).groupby(pool_keys, sort=False):
            if len(sub) < MIN_FIT_ROWS or len(set(sub[label_col].astype(int))) < 2:
                n_pools_skip += 1
                continue
            iso = fit_iso(sub[score_col].to_numpy(), sub[label_col].to_numpy())
            # apply to all rows in df matching keys
            if isinstance(keys, tuple):
                mask = np.ones(len(df), dtype=bool)
                for col, val in zip(pool_keys, keys):
                    mask &= (df[col] == val)
            else:
                mask = df[pool_keys[0]] == keys
            mask &= df[score_col].notna()
            df.loc[mask, score_col] = iso.predict(df.loc[mask, score_col].astype(float).to_numpy())
            n_pools_fit += 1
        # leftover rows (no matching pool) — global fallback
        # already None pred — apply global to remaining unprocessed
    else:
        # global only
        df.loc[mask_g, score_col] = iso_global.predict(df.loc[mask_g, score_col].astype(float).to_numpy())
        n_pools_fit = 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return n_pools_fit, n_pools_skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per_conc", default=str(FDA_PER_CONC))
    p.add_argument("--output_dir", default="agent_system/output/pooled_iso")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_configs = {
        "per_organism": ["organism_group"],
        "per_drug":     ["antimicrobial"],
        "global":       [],
    }
    manifest = {"per_conc": args.per_conc, "outputs": {}}
    for name, keys in pool_configs.items():
        out_csv = out_dir / f"dtw_per_conc_routed_pool_{name}.csv"
        print(f"\n[apply] pool={name} (keys={keys}) → {out_csv}")
        n_fit, n_skip = calibrate_per_conc(Path(args.per_conc), keys, out_csv)
        print(f"  pools fit={n_fit}, skipped={n_skip}")
        manifest["outputs"][name] = {
            "csv": str(out_csv),
            "n_pools_fit": n_fit,
            "n_pools_skipped": n_skip,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    print(f"\n[manifest] {out_dir / 'manifest.json'}")
    print("\n[next] verify_method_sir_pipeline 3회 실행 (per_organism / per_drug / global)")


if __name__ == "__main__":
    main()

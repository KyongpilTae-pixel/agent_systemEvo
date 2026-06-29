"""Cross-domain ISO transferability test — train→FDA (P2, 2026-05-28).

학습 도메인에서 fit한 isotonic calibration을 FDA evaluation에 적용했을 때
PASS 변화를 측정. 현재 운영은 FDA-fit ISO만 검증됨 (within-FDA LOSO transferable).
진짜 cross-domain transfer는 미검증.

학습 도메인 inference 결과: claudeCode/output_dtw_aggregate_train_domain/
dtw_per_conc.csv (27,128 rows, 558 samples, 7 project_id 중 FDA Clinical 외
다른 도메인 위주).

**제약**: 학습 도메인 sample의 organism_group 매핑 source 미확보 → cell-level
ISO fit 불가. 본 스크립트는 **global ISO** (all cells pooled)로 first-pass
transfer 측정. cell-level은 organism mapping 추가 후 followup.

비교:
  - baseline raw       : 학습 도메인 raw model_pred → FDA → PASS
  - FDA cell ISO       : FDA에서 cell별 ISO fit → FDA → PASS (현재 운영 권고)
  - train global ISO   : 학습 도메인 global ISO → FDA → PASS (cross-domain test)
  - FDA global ISO     : FDA에서 global ISO fit → FDA → PASS (control)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/home/kptae/project/qnt_algorithm")
TRAIN_PER_CONC = ROOT / "claudeCode/output_dtw_aggregate_train_domain/dtw_per_conc.csv"
FDA_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"


def fit_global_iso(per_conc_csv: Path,
                   score_col: str = "dtw_model_pred",
                   label_col: str = "gt_gng"):
    df = pd.read_csv(per_conc_csv, low_memory=False)
    mask = df[score_col].notna() & df[label_col].notna()
    x = df.loc[mask, score_col].astype(float).to_numpy()
    y = df.loc[mask, label_col].astype(int).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    return iso, len(x)


def apply_iso_to_per_conc(per_conc_csv: Path, iso: IsotonicRegression,
                          output_csv: Path,
                          score_col: str = "dtw_model_pred"):
    df = pd.read_csv(per_conc_csv, low_memory=False)
    mask = df[score_col].notna()
    df.loc[mask, score_col] = iso.predict(df.loc[mask, score_col].astype(float).to_numpy())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return len(df)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_per_conc", default=str(TRAIN_PER_CONC))
    p.add_argument("--fda_per_conc", default=str(FDA_PER_CONC))
    p.add_argument("--output_dir", default="agent_system/output/cross_domain_iso")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) train global ISO fit
    print("[fit] train global ISO")
    iso_train, n_train = fit_global_iso(Path(args.train_per_conc))
    print(f"  n_train_rows = {n_train:,}")
    print(f"  ISO X bounds = [{iso_train.X_thresholds_.min():.4f}, "
          f"{iso_train.X_thresholds_.max():.4f}]")
    print(f"  ISO Y bounds = [{iso_train.y_thresholds_.min():.4f}, "
          f"{iso_train.y_thresholds_.max():.4f}]")

    # 2) FDA global ISO fit (control)
    print("\n[fit] FDA global ISO (control)")
    iso_fda, n_fda = fit_global_iso(Path(args.fda_per_conc))
    print(f"  n_fda_rows = {n_fda:,}")

    # 3) apply train ISO to FDA per_conc
    fda_train_iso_csv = out_dir / "dtw_per_conc_routed_train_global_iso.csv"
    print(f"\n[apply] train ISO → FDA → {fda_train_iso_csv}")
    n_rows = apply_iso_to_per_conc(Path(args.fda_per_conc), iso_train, fda_train_iso_csv)
    print(f"  {n_rows:,} rows")

    # 4) apply FDA global ISO to FDA per_conc (control)
    fda_global_iso_csv = out_dir / "dtw_per_conc_routed_fda_global_iso.csv"
    print(f"\n[apply] FDA global ISO → FDA → {fda_global_iso_csv}")
    apply_iso_to_per_conc(Path(args.fda_per_conc), iso_fda, fda_global_iso_csv)

    # manifest
    manifest = {
        "train_per_conc": args.train_per_conc,
        "fda_per_conc": args.fda_per_conc,
        "n_train_rows": n_train,
        "n_fda_rows": n_fda,
        "outputs": {
            "fda_with_train_global_iso": str(fda_train_iso_csv),
            "fda_with_fda_global_iso": str(fda_global_iso_csv),
        },
        "iso_train_x_bounds": [float(iso_train.X_thresholds_.min()),
                               float(iso_train.X_thresholds_.max())],
        "iso_train_y_bounds": [float(iso_train.y_thresholds_.min()),
                               float(iso_train.y_thresholds_.max())],
        "iso_fda_x_bounds": [float(iso_fda.X_thresholds_.min()),
                             float(iso_fda.X_thresholds_.max())],
        "iso_fda_y_bounds": [float(iso_fda.y_thresholds_.min()),
                             float(iso_fda.y_thresholds_.max())],
        "note": ("Global ISO fit (cell pooling) — cell-specific ISO requires "
                 "organism_group mapping for train samples (currently unavailable). "
                 "This run measures global score distribution transferability only."),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    print(f"\n[manifest] {out_dir / 'manifest.json'}")
    print("\n[next] verify_method_sir_pipeline 실행으로 PASS 측정:")
    print(f"  python -m claudeCode.verify_method_sir_pipeline \\")
    print(f"      --per_conc_csv {fda_train_iso_csv.relative_to(ROOT)} \\")
    print(f"      --gbm_oof_csv claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv \\")
    print(f"      --output_dir claudeCode/output_sir_eval_train_global_iso_patched")


if __name__ == "__main__":
    main()

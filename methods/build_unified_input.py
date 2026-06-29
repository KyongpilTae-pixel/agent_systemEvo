"""통합 입력 빌더 — 흩어진 per-row 신호를 단일 per_conc 파일로 합친다.

methods 패키지의 모든 방법론은 외부 CSV 를 각자 읽지 않고, 이 하나의 통합 입력
(data/unified_per_conc.parquet)에서 필요한 source 컬럼만 소비한다.

통합 컬럼 (base.UNIFIED_COLUMNS):
  키            : sample_id, antimicrobial, concentration_idx_0
  메타          : organism_group, gt_gng, bucket(aligned/shifted/fda_only/...)
  활성 점수     : ng_score (= 기본값 src_same_mic; SCORE 방법론이 덮어씀)
  source 점수   : src_same_mic   (baseline same_mic.pt)
                  src_routed      (per-cell routed model_pred)
                  src_objarea_gbm (object_area cross-domain GBM)
                  src_object_area (raw object_area DTW)
                  src_brightness_gbm (brightness GBM 예측; --no_brightness 시 NaN)

사용:
  python -m agent_system.methods.build_unified_input            # brightness 포함
  python -m agent_system.methods.build_unified_input --no_brightness
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from agent_system.methods import base as B

# ---- source 경로 ----
BASE_PER_CONC = B.ROOT / "claudeCode/output_dtw_aggregate_full/dtw_per_conc.csv"
ROUTED_CSV = B.ROOT / "claudeCode/output_subset_eval/routed_model_pred.csv"
GBM_OOF = B.ROOT / "claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv"
CELL_DIFF = B.ROOT / "claudeCode/output_dataset_diff_traintest_normalized/dataset_cell_diff_summary.csv"


def build(with_brightness: bool = True) -> pd.DataFrame:
    print(f"[load] base same_mic per_conc: {BASE_PER_CONC}")
    base = pd.read_csv(BASE_PER_CONC, low_memory=False)
    keep = B.KEY + [B.OG_COL, B.GENUS_COL, B.LABEL_COL, "dtw_model_pred", "dtw_object_area"]
    u = base[[c for c in keep if c in base.columns]].copy()
    # genus 컬럼이 base 에 없으면 organism_group 첫 단어로 유도 (FDA 24 organism_group 100% 일치)
    if B.GENUS_COL not in u.columns:
        u[B.GENUS_COL] = u[B.OG_COL].astype(str).str.split().str[0]
    u = u.rename(columns={"dtw_model_pred": B.SRC_SAME_MIC,
                          "dtw_object_area": B.SRC_OBJECT_AREA})

    print(f"[merge] routed model_pred: {ROUTED_CSV}")
    rt = pd.read_csv(ROUTED_CSV, low_memory=False, usecols=B.KEY + ["model_pred"])
    u = u.merge(rt.rename(columns={"model_pred": B.SRC_ROUTED}), on=B.KEY, how="left")
    # routed 결측 행은 same_mic 으로 폴백 (build_per_conc_for_recipe routed_per_cell 와 동일)
    u[B.SRC_ROUTED] = u[B.SRC_ROUTED].where(u[B.SRC_ROUTED].notna(), u[B.SRC_SAME_MIC])

    print(f"[merge] object_area GBM oof: {GBM_OOF}")
    gb = pd.read_csv(GBM_OOF, low_memory=False, usecols=B.KEY + ["objarea_crossdomain_pred"])
    u = u.merge(gb.rename(columns={"objarea_crossdomain_pred": B.SRC_OBJAREA_GBM}),
                on=B.KEY, how="left")

    print(f"[merge] cell bucket: {CELL_DIFF}")
    cd = pd.read_csv(CELL_DIFF, low_memory=False, usecols=[B.OG_COL, B.DRUG_COL, "bucket"])
    cd = cd.rename(columns={"bucket": B.COL_BUCKET})
    u = u.merge(cd, on=[B.OG_COL, B.DRUG_COL], how="left")
    u[B.COL_BUCKET] = u[B.COL_BUCKET].fillna("unknown")

    # brightness GBM 예측 (선택) — 학습 CSV out-of-domain GBM, FDA brightness trajectory 추론
    if with_brightness:
        try:
            import sys
            sys.path.insert(0, str(B.ROOT / "agent_system/brightness_asym_corrector"))
            from brightness_corrector import BrightnessAsymCorrector  # noqa: E402
            import config as BC  # noqa: E402
            print("[brightness] GBM 추론 (FDA brightness trajectory)…")
            corr = BrightnessAsymCorrector()
            bp = corr.predict_brightness(BC.FDA_CSV, BC.FDA_BRIGHTNESS_COL)
            bp = bp[[c for c in (B.KEY + ["brightness_pred"]) if c in bp.columns]]
            u = u.merge(bp.rename(columns={"brightness_pred": B.SRC_BRIGHTNESS_GBM}),
                        on=B.KEY, how="left")
            print(f"  brightness_pred matched: {int(u[B.SRC_BRIGHTNESS_GBM].notna().sum()):,}")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] brightness 추론 실패 → NaN 컬럼 ({e})")
            u[B.SRC_BRIGHTNESS_GBM] = np.nan
    else:
        u[B.SRC_BRIGHTNESS_GBM] = np.nan

    # 활성 점수 기본값 = same_mic baseline
    u[B.SCORE_COL] = u[B.SRC_SAME_MIC]

    # 컬럼 순서 정렬
    u = u[[c for c in B.UNIFIED_COLUMNS if c in u.columns]]
    return u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no_brightness", action="store_true",
                    help="brightness GBM 추론 생략 (SRC_BRIGHTNESS_GBM = NaN)")
    ap.add_argument("--out", default=str(B.UNIFIED_INPUT))
    args = ap.parse_args()

    u = build(with_brightness=not args.no_brightness)
    B.DATA_DIR.mkdir(parents=True, exist_ok=True)
    u.to_parquet(args.out, index=False)
    print(f"\n[saved] {args.out}")
    print(f"  rows={len(u):,}  cols={list(u.columns)}")
    print(f"  cells={u.groupby([B.OG_COL, B.DRUG_COL]).ngroups}  "
          f"samples={u['sample_id'].nunique()}")
    nn = {c: int(u[c].notna().sum()) for c in
          (B.SRC_SAME_MIC, B.SRC_ROUTED, B.SRC_OBJAREA_GBM, B.SRC_OBJECT_AREA,
           B.SRC_BRIGHTNESS_GBM)}
    print(f"  non-null: {nn}")


if __name__ == "__main__":
    main()

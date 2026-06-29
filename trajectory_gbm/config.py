"""trajectory_gbm 공통 설정 + signal registry.

brightness 와 object_area 는 **입력 trajectory 컬럼만 다르고** feature 추출·GBM 학습·
GroupKFold CV·비대칭 보정 로직이 동일하다. 그 차이를 `SignalConfig` 한 곳에 모았다.

- TRAIN_CSV / FDA_CSV : 두 신호 공통 (한 CSV 안에 brightness·object_area 둘 다 존재).
- per-signal 차이      : train_col / fda_col(원본 trajectory 컬럼), model_path,
                         (선택) 비대칭 보정 적용 cell, 기본 weight.
"""
from __future__ import annotations

import dataclasses as dc
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parents[1]                      # /home/kptae/project/qnt_algorithm
MODELS_DIR = PKG_DIR / "models"

# ---- 공통 데이터 소스 ----
TRAIN_CSV = "/data/dRAST30_prepare_csv/2024_0709_traintest_allInfo.csv"
FDA_CSV = (
    "/home/kptae/data/allinfo/fda2023_preprocessedWithFeature_validControlInfo_"
    "useMax_TrueWith_object_area_useMax_TrueWith_object_count_useMax_TrueWith_"
    "brightness_useMax_False.csv"
)
LABEL_COL = "bmd_gng"                           # 학습 CSV 라벨 (G/NG)

# 운영 image model_pred(cell-routed) per_conc + 라벨 정렬 기준 (cross-domain AUROC 평가용)
ROUTED_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
ALIGN_OOF = ROOT / "claudeCode/output_dtw_aggregate_full/objarea_crossdomain_oof.csv"

# ---- 운영 SIR 파이프라인 (보정 평가용) ----
VERIFY_PIPELINE = ROOT / "claudeCode/verify_method_sir_pipeline.py"
NATIVE_THRESHOLD = 0.65                          # 운영 G/NG threshold (ISO 보정 후)

# ---- LightGBM 학습 하이퍼파라미터 (공통) ----
LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
    min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, verbose=-1,
)
LGB_NUM_ROUNDS = 400


@dc.dataclass(frozen=True)
class SignalConfig:
    """한 trajectory 신호의 입력/모델/보정 설정."""
    name: str
    train_col: str                 # TRAIN_CSV 의 원본 trajectory 컬럼
    fda_col: str                   # FDA_CSV 의 원본 trajectory 컬럼
    model_path: Path               # 학습된 GBM 저장 경로
    apply_cells_csv: Path | None = None   # 비대칭 보정 적용 cell (없으면 보정 미사용)
    asym_weight: float = 0.70      # 비대칭 G-방향 보정 비중


SIGNALS: dict[str, SignalConfig] = {
    "brightness": SignalConfig(
        name="brightness",
        train_col="brightness_list",
        fda_col="brightness",
        model_path=MODELS_DIR / "brightness_gbm.txt",
        # brightness 비대칭 보정은 GroupKFold CV 에서 held-out 미재현(2026-06-04) → 운영 미사용.
        apply_cells_csv=PKG_DIR.parents[0] / "brightness_asym_corrector" / "brightness_apply_cells.csv",
        asym_weight=0.70,
    ),
    "object_area": SignalConfig(
        name="object_area",
        train_col="object_area",
        fda_col="object_area",
        model_path=MODELS_DIR / "object_area_gbm.txt",
        apply_cells_csv=None,       # object_area 는 baseline/ensemble 용도 (보정자 아님)
        asym_weight=0.70,
    ),
}


def get_signal(name: str) -> SignalConfig:
    if name not in SIGNALS:
        raise KeyError(f"unknown signal {name!r}; available: {list(SIGNALS)}")
    return SIGNALS[name]

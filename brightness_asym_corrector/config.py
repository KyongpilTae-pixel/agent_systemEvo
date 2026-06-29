"""brightness asym corrector 설정값 (운영 검증 기준).

최적 방안 (전체 313 cell 검증):
  Brightness=True 45 cell 에 한정해 asym G-방향 brightness 보정(w=0.7) 적용 →
  PASS 103 → 106 (+3), VME-cell 43 → 42 (-1).
"""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parents[1]   # /home/kptae/project/qnt_algorithm

# ---- 데이터 소스 ----
TRAIN_CSV = "/data/dRAST30_prepare_csv/2024_0709_traintest_allInfo.csv"
FDA_CSV = (
    "/home/kptae/data/allinfo/fda2023_preprocessedWithFeature_validControlInfo_"
    "useMax_TrueWith_object_area_useMax_TrueWith_object_count_useMax_TrueWith_"
    "brightness_useMax_False.csv"
)
TRAIN_BRIGHTNESS_COL = "brightness_list"   # 학습 CSV
FDA_BRIGHTNESS_COL = "brightness"          # FDA CSV
LABEL_COL = "bmd_gng"                       # 학습 라벨 (G/NG)

# 운영 image model_pred (cell-routed) 가 들어있는 per_conc + label 정렬용
ROUTED_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
# 라벨/organism_group 정렬 기준 (FDA per-row gt_gng, model_pred)
ALIGN_OOF = ROOT / "claudeCode/output_dtw_aggregate_full/objarea_crossdomain_oof.csv"

# ---- 모델/적용 설정 ----
MODEL_PATH = PKG_DIR / "model" / "brightness_gbm.txt"
APPLY_CELLS_CSV = PKG_DIR / "brightness_apply_cells.csv"   # Brightness=True (organism_group, antimicrobial)
ASYM_WEIGHT = 0.70          # brightness 비중 (G-방향에만 적용)
NATIVE_THRESHOLD = 0.65     # 운영 G/NG threshold (ISO 보정 후)

# ---- 운영 SIR 파이프라인 (검증/평가용) ----
VERIFY_PIPELINE = ROOT / "claudeCode/verify_method_sir_pipeline.py"

# ---- LightGBM 학습 하이퍼파라미터 ----
LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
    min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, verbose=-1,
)
LGB_NUM_ROUNDS = 400

# ---- NOTE ----
# evo_selected_model_and_brightness_thershold_260529.xlsx 의 Brightness_threshold(0.4~0.75)
# 값은 **10시간 readout 기준**이라 현재 데이터(7-timepoint trajectory)에는 사용하지 않는다.
# 본 패키지는 그 고정 threshold 대신, 현재 데이터의 brightness trajectory 로 GBM 을 학습하고
# 현재 FDA 도메인에서 per-cell isotonic + threshold(0.65) 로 보정한다 (데이터-구동).

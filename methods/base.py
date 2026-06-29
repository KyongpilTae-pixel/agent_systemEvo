"""methods 패키지 공통 계약 — Stage / Status / Method ABC / 통합 입력 스키마.

설계(2026-06-05): FDA PASS 개선 방법론 10종은 모두 운영 파이프라인의 4 스테이지 중
하나에 속한다.

  per_conc ─▶ [A.SCORE] ─▶ [B.CALIBRATE] ─▶ [C.THRESHOLD] ─▶ [D.EVALUATE] ─▶ PASS/VME
             점수 선택/변환    score→P(NG)        P(NG)→G/NG        MIC→SIR→EA

모든 SCORE/CALIBRATE 방법론은 동일 계약을 따른다:
  apply(per_conc: DataFrame) -> DataFrame   (활성 점수 컬럼 SCORE_COL 을 갱신해 반환)

입력은 build_unified_input.py 가 만드는 **단일 통합 per_conc**(UNIFIED_INPUT) 하나로 충분.
각 방법론은 외부 CSV 를 따로 읽지 않고 통합 프레임의 source 컬럼만 소비한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import pandas as pd

# --------------------------- 경로 ---------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # /home/kptae/project/qnt_algorithm
PKG_DIR = Path(__file__).resolve().parent           # agent_system/methods
DATA_DIR = PKG_DIR / "data"
OUTPUT_DIR = PKG_DIR / "output"
UNIFIED_INPUT = DATA_DIR / "unified_per_conc.parquet"          # 전체(1507 sample)
UNIFIED_CLEAN = DATA_DIR / "unified_per_conc_clean.parquet"    # TE 제거(1434 sample)
UNIFIED_TE = DATA_DIR / "unified_per_conc_te.parquet"          # Technical error 별도 관리
TE_SAMPLES_CSV = DATA_DIR / "te_samples.csv"                   # 제외 sample 목록
# TE 제거 기준 = QuantaMatrix 수기 선별 파일(all_concentration sheet 의 sample_id = clean set)
CLEAN_SOURCE_XLSX = ROOT / "claudeCode/data/FDA_Clinical_selected_model.xlsx"

# 운영 SIR/PASS 평가 파이프라인 (EA panel-notation fix 내장)
VERIFY_PIPELINE = ROOT / "claudeCode/verify_method_sir_pipeline.py"
# python 인터프리터 (conda qnt_algorithm) — verify subprocess 용
PYTHON = "/home/kptae/miniconda3/envs/qnt_algorithm/bin/python"

# --------------------------- per_conc 계약 컬럼 -------------------------------
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]
OG_COL = "organism_group"
GENUS_COL = "genus"          # 배포 환경에서 실제로 알 수 있는 단위 (organism_group 보다 coarse)
DRUG_COL = "antimicrobial"
LABEL_COL = "gt_gng"
# 활성(현재) NG 점수 — SCORE/CALIBRATE 가 갱신. 'ng_score' = No-Growth 지향 점수
# (routing 이면 모델 sigmoid 확률 P(NG), rank_avg/z-score 면 비확률 점수도 가능).
# 외부 dtw_per_conc.csv 의 'dtw_model_pred' 와 동일 의미지만 이름이 오해를 주어(=DTW 아님,
# sigmoid(gng_logit) 확률) methods 패키지 내부에서는 ng_score 로 부른다.
SCORE_COL = "ng_score"
# 외부 경계 컬럼: verify_method_sir_pipeline(build_scores) 와 PerCellIsotonicLookup 이
# 'dtw_model_pred' 를 하드코딩하므로, 그 경계에서만 ng_score ↔ dtw_model_pred 로 번역한다.
EXTERNAL_SCORE_COL = "dtw_model_pred"

# 통합 입력의 source 컬럼 (불변; 방법론은 여기서 읽어 SCORE_COL 로 씀)
SRC_SAME_MIC = "src_same_mic"          # baseline same_mic.pt 점수
SRC_ROUTED = "src_routed"              # per-cell routed model_pred
SRC_OBJAREA_GBM = "src_objarea_gbm"    # object_area cross-domain GBM 예측
SRC_OBJECT_AREA = "src_object_area"    # raw object_area DTW (z-ensemble 용)
SRC_BRIGHTNESS_GBM = "src_brightness_gbm"  # brightness GBM 예측 (없으면 NaN)
COL_BUCKET = "bucket"                  # aligned / shifted / fda_only (cell domain shift)

UNIFIED_COLUMNS = (KEY + [OG_COL, GENUS_COL, LABEL_COL, SCORE_COL, SRC_SAME_MIC, SRC_ROUTED,
                         SRC_OBJAREA_GBM, SRC_OBJECT_AREA, SRC_BRIGHTNESS_GBM, COL_BUCKET])


# --------------------------- 분류 enum ---------------------------------------
class Stage(Enum):
    SCORE = "score"          # per-conc NG 점수 선택/변환 → SCORE_COL
    CALIBRATE = "calibrate"  # score → P(NG)
    THRESHOLD = "threshold"  # P(NG) → G/NG (이진화 lever)
    EVALUATE = "evaluate"    # MIC→SIR→EA/CA/ME/VME correctness


class Status(Enum):
    ADOPTED = "adopted"          # 운영 채택 (routed_iso_t65)
    AUXILIARY = "auxiliary"      # 보조/탐색 (AUROC 기여 등)
    REJECTED = "rejected"        # 검증(CV/held-out)에서 기각
    ALTERNATIVE = "alternative"  # 대안 (trade-off)


# --------------------------- Method ABC --------------------------------------
class Method(ABC):
    """모든 방법론의 공통 베이스.

    클래스 속성(메타): name / stage / status / pass_impact / summary.
    핵심 계약: apply(per_conc) -> per_conc  (SCORE_COL 갱신).
    EVALUATE/THRESHOLD 처럼 데이터를 변형하지 않는 방법론은 apply 를 identity 로 두고
    각자 고유 메서드(activate / tune 등)를 노출한다.
    """

    name: str = "method"
    stage: Stage = Stage.SCORE
    status: Status = Status.AUXILIARY
    pass_impact: str = ""      # 예: "+30", "+3 (★LNZ)", "OOF −4"
    summary: str = ""

    @abstractmethod
    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        """per_conc 를 받아 활성 점수(SCORE_COL)를 갱신한 새 DataFrame 반환."""

    # 공통 유틸: source 컬럼 존재 확인 (통합 입력 계약 강제)
    @staticmethod
    def _require(per_conc: pd.DataFrame, cols) -> None:
        missing = [c for c in cols if c not in per_conc.columns]
        if missing:
            raise KeyError(
                f"통합 입력에 필요한 컬럼 누락: {missing}. "
                f"build_unified_input.py 로 unified_per_conc 를 먼저 생성하세요.")

    def meta(self) -> dict:
        return {"name": self.name, "stage": self.stage.value,
                "status": self.status.value, "pass_impact": self.pass_impact,
                "summary": self.summary, "class": type(self).__name__}

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} stage={self.stage.value} "
                f"status={self.status.value} impact={self.pass_impact!r}>")

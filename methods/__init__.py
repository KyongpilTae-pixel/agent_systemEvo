"""methods — FDA PASS 개선 방법론 통합 패키지.

운영 파이프라인 4 스테이지(SCORE → CALIBRATE → THRESHOLD → EVALUATE)에 방법론 10종을
공통 인터페이스(base.Method)로 모듈화. 입력은 단일 통합 파일(build_unified_input.py 의
data/unified_per_conc.parquet) 하나로 충분.

빠른 시작:
    from agent_system.methods import operational_recipe, catalog
    print(catalog())                       # 방법론 카탈로그 (중요도 순)
    print(operational_recipe().evaluate()) # routed_iso_t65 → {'pass':103, ...}

방법론 직접 조립:
    from agent_system.methods.recipe import Recipe
    from agent_system.methods.score.model_routing import PerCellModelRouting
    from agent_system.methods.calibrate.isotonic import PerCellIsotonic
    Recipe(score=PerCellModelRouting(), calibrate=PerCellIsotonic(), threshold=0.65).evaluate()
"""
from __future__ import annotations

from agent_system.methods.base import Method, Stage, Status
from agent_system.methods.recipe import Recipe
# SCORE
from agent_system.methods.score.model_routing import PerCellModelRouting, BaselineSameMic
from agent_system.methods.score.rank_avg_ensemble import RankAvgEnsemble
from agent_system.methods.score.objarea_gbm import ObjAreaGBM
from agent_system.methods.score.shifted_ensemble import ShiftedCellEnsemble
from agent_system.methods.score.zscore_ensemble import ZScoreEnsemble
from agent_system.methods.score.brightness_corrector import BrightnessAsymCorrector
from agent_system.methods.score.genus_routing import GenusModelRouting
# CALIBRATE / THRESHOLD / EVALUATE
from agent_system.methods.calibrate.isotonic import PerCellIsotonic
from agent_system.methods.threshold.tuner import ThresholdTuner
from agent_system.methods.evaluate.ea_fix import EAPanelNotationFix

# 방법론(scoring/calibration/threshold) — 중요도(FDA PASS 기여) 순.
# LNZ 통과 방법론(★)은 PerCellModelRouting.
REGISTRY = [
    PerCellIsotonic,          # ① +30  (운영 PASS 주역)
    PerCellModelRouting,      # ② +3   ★LNZ
    ThresholdTuner,           # ③ VME 통제 lever
    RankAvgEnsemble,          # ④ AUROC↑ / PASS 기각
    ObjAreaGBM,               # ⑤ 보조 (이미지 무관)
    ShiftedCellEnsemble,      # ⑥ shifted-only 탐색
    ZScoreEnsemble,           # ⑦ 대안 앙상블
    BrightnessAsymCorrector,  # ⑧ CV 기각
    GenusModelRouting,        # ⑨ 대안 (보수)
    BaselineSameMic,          # (대조군)
]

# 평가기 correctness 수정 (방법론 아님 — PASS 를 올리는 게 아니라 잘못된 FAIL 을 정정).
# 파이프라인 필수 컴포넌트이며 verify_method_sir_pipeline 가 자동 적용.
FIXES = [EAPanelNotationFix]   # EA panel-notation 버그 수정 (+32 = 정정 효과)

__all__ = [c.__name__ for c in REGISTRY] + ["EAPanelNotationFix"] + [
    "Method", "Stage", "Status", "Recipe", "REGISTRY", "FIXES",
    "catalog", "fixes", "operational_recipe"]


def _no_required_args(cls) -> bool:
    import inspect
    sig = inspect.signature(cls.__init__)
    return all(p.default is not inspect.Parameter.empty or p.name == "self"
               for p in sig.parameters.values())


def _meta_static(cls) -> dict:
    return {"name": cls.name, "stage": cls.stage.value, "status": cls.status.value,
            "pass_impact": cls.pass_impact, "summary": cls.summary, "class": cls.__name__}


def catalog() -> list[dict]:
    """방법론 메타(name/stage/status/pass_impact/summary) 리스트 — 중요도 순."""
    return [c().meta() if _no_required_args(c) else _meta_static(c) for c in REGISTRY]


def fixes() -> list[dict]:
    """평가기 correctness 수정 메타 (방법론 아님)."""
    return [c().meta() if _no_required_args(c) else _meta_static(c) for c in FIXES]


def operational_recipe(threshold: float = 0.65, calibrate_mode: str = "insample") -> Recipe:
    """운영 권고 routed_iso_t65 = routing(★LNZ) + isotonic + threshold 0.65."""
    return Recipe(score=PerCellModelRouting(),
                  calibrate=PerCellIsotonic(mode=calibrate_mode),
                  threshold=threshold, name=f"routed_iso_t{int(threshold*100)}")

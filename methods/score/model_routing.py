"""② Per-cell model routing — ★ LNZ 약제 통과 방법론 (routing +3, AUROC +0.031).

각 (organism_group, antimicrobial) cell 마다 AUROC 최고인 subset OrderPickGNG 모델
(same_mic / beta_lactam / gram_negative / early_stop / object_area_045~08 / object_area_all)
을 선택해 그 모델의 model_pred 를 활성 점수로 사용.

★ LNZ(linezolid) gram-positive 5 cell: baseline(same_mic)은 전부 FAIL 이지만 routing 으로
   E. faecium→early_stop, S. Coag-neg→beta_lactam, E. faecalis→object_area_045~08,
   S. lugdunensis→early_stop 선택 → 4/5 PASS. 분기점 E. faecium×LNZ: our=early_stop→PASS
   vs genus=beta_lactam→FAIL.

오프라인 PASS 경로: 통합 입력의 src_routed(사전 계산된 routed model_pred)를 활성 점수로 주입.
온라인 추론(raw image): PerCellModelRouter (claudeCode/per_cell_model_router) 를 그대로 노출.

stage=SCORE, status=ADOPTED.
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status


class PerCellModelRouting(Method):
    name = "per_cell_model_routing"
    stage = Stage.SCORE
    status = Status.ADOPTED
    pass_impact = "+3 (★LNZ)"
    summary = ("cell별 best subset 모델 라우팅. LNZ gram-positive 5 cell 중 4 PASS 전환. "
               "오프라인=src_routed 주입, 온라인=PerCellModelRouter.")

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        self._require(per_conc, [B.SRC_ROUTED])
        out = per_conc.copy()
        out[B.SCORE_COL] = out[B.SRC_ROUTED].astype(float)
        return out

    # ---- 온라인 추론(raw image → 점수) 경로 ----
    @staticmethod
    def online_router():
        """운영 런타임 라우터 (lazy 모델 로딩). raw safetensors/image → 점수."""
        from claudeCode.per_cell_model_router import PerCellModelRouter
        return PerCellModelRouter.from_cell_lookup()


class BaselineSameMic(Method):
    """baseline — same_mic.pt 단일 모델 점수 (routing 대조군)."""
    name = "baseline_same_mic"
    stage = Stage.SCORE
    status = Status.ALTERNATIVE
    pass_impact = "기준(0)"
    summary = "사전훈련 same_mic.pt 점수. routing/calibration 비교의 baseline."

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        self._require(per_conc, [B.SRC_SAME_MIC])
        out = per_conc.copy()
        out[B.SCORE_COL] = out[B.SRC_SAME_MIC].astype(float)
        return out

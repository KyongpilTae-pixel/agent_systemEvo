"""⑥ Object-area GBM — 이미지 무관 LightGBM 점수 (보조).

raw object_area trajectory 만으로 학습한 cross-domain LightGBM 의 P(NG) 예측.
FDA cross-domain AUROC 0.9394 (model_pred 0.9466 의 99%, −0.0072) — 이미지 representation
의 marginal contribution 이 작음을 보이는 핵심 자산. 단독 운영 점수로는 model_pred 보다
약간 낮아 미채택이며, 앙상블(rank-avg / shifted) 구성요소로 활용.

오프라인: 통합 입력의 src_objarea_gbm(사전 계산 OOF/cross-domain 예측)을 활성 점수로 주입.
온라인: agent_system.trajectory_gbm 모듈로 raw object_area → 예측.
stage=SCORE, status=AUXILIARY.
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status


class ObjAreaGBM(Method):
    name = "objarea_gbm"
    stage = Stage.SCORE
    status = Status.AUXILIARY
    pass_impact = "AUROC 0.9394 (이미지 무관)"
    summary = ("object_area trajectory LightGBM. cross-domain AUROC 0.9394. 단독 미채택, "
               "앙상블 구성요소.")

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        self._require(per_conc, [B.SRC_OBJAREA_GBM])
        out = per_conc.copy()
        out[B.SCORE_COL] = out[B.SRC_OBJAREA_GBM].astype(float)
        return out

    @staticmethod
    def online_model():
        """raw object_area trajectory → 예측 (trajectory_gbm 모듈 반환; predict 모드 사용)."""
        from agent_system.trajectory_gbm import model as traj_model
        return traj_model

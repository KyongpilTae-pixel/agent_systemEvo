"""⑧ Z-score ensemble — z(model_pred) + α·z(object_area) (대안 앙상블).

전역 z-score 정규화 후 object_area 를 작은 가중치(α=0.2)로 더하는 global-α 앙상블.
출력은 확률이 아니므로(rank/z 스케일) 반드시 CALIBRATE(isotonic) 가 뒤따라야 운영
threshold 적용이 가능. EnsembleScoreTable(per-cell z+α)의 단순 global 버전이며 미채택(대안).

base = "routed"(기본) 또는 "same_mic". α = alpha. stage=SCORE, status=ALTERNATIVE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status

_BASE_COL = {"routed": B.SRC_ROUTED, "same_mic": B.SRC_SAME_MIC}


def _z(x: np.ndarray) -> np.ndarray:
    mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
    return (x - mu) / (sd if sd > 0 else 1.0)


class ZScoreEnsemble(Method):
    name = "zscore_ensemble"
    stage = Stage.SCORE
    status = Status.ALTERNATIVE
    pass_impact = "대안(미채택)"
    summary = "z(model)+α·z(object_area). 확률 아님 → isotonic 보정 필수. global-α 앙상블."

    def __init__(self, alpha: float = 0.2, base: str = "same_mic"):
        if base not in _BASE_COL:
            raise ValueError(f"base∈{{routed,same_mic}}: {base}")
        self.alpha = float(alpha)
        self.base = base

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        bcol = _BASE_COL[self.base]
        self._require(per_conc, [bcol, B.SRC_OBJECT_AREA])
        out = per_conc.copy()
        out[B.SCORE_COL] = (_z(out[bcol].astype(float).values)
                            + self.alpha * _z(out[B.SRC_OBJECT_AREA].astype(float).values))
        return out

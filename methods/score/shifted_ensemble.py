"""⑦ Shifted-cell ensemble — domain-shift cell 한정 선택적 앙상블 (탐색).

train↔FDA 분포가 어긋난(bucket=="shifted") cell 에만 base 점수와 object_area GBM 을
확률 평균(0.5/0.5)으로 혼합하고, 나머지 cell 은 base 를 유지. shifted-only AUROC
0.9349→0.9455(+0.0105). 운영은 PerCellModelRouting 이 더 효과적이라 미채택(탐색 자산).

base = "routed"(기본) 또는 "same_mic". blend 가중치 w(기본 0.5). stage=SCORE, status=AUXILIARY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status

_BASE_COL = {"routed": B.SRC_ROUTED, "same_mic": B.SRC_SAME_MIC}


class ShiftedCellEnsemble(Method):
    name = "shifted_cell_ensemble"
    stage = Stage.SCORE
    status = Status.AUXILIARY
    pass_impact = "shifted-only AUROC +0.0105"
    summary = ("bucket=shifted cell 에만 0.5·base+0.5·GBM 확률 평균. 나머지는 base 유지.")

    def __init__(self, w: float = 0.5, base: str = "routed"):
        if base not in _BASE_COL:
            raise ValueError(f"base∈{{routed,same_mic}}: {base}")
        self.w = float(w)
        self.base = base

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        bcol = _BASE_COL[self.base]
        self._require(per_conc, [bcol, B.SRC_OBJAREA_GBM, B.COL_BUCKET])
        out = per_conc.copy()
        b = out[bcol].astype(float)
        g = out[B.SRC_OBJAREA_GBM].astype(float)
        blended = (1.0 - self.w) * b + self.w * g
        is_shifted = (out[B.COL_BUCKET].astype(str) == "shifted") & g.notna()
        out[B.SCORE_COL] = np.where(is_shifted, blended, b)
        return out

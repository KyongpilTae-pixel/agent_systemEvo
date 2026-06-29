"""⑤ Rank-average ensemble (model_pred + object_area GBM) — AUROC↑ / PASS 기각.

score = (1-w)·rank_pct(base) + w·rank_pct(objarea_gbm)   (global percentile rank)

AUROC 축에서 최고(macro 0.9619, +0.013)지만 운영 PASS 에는 기여하지 않는다. 2026-06-05
재검토(rankavg_gbm_iso_cv): rank-scale per-cell ISO 재calibrate + held-out OOF 로도
routed 대비 OOF PASS w0.3 −4 / w0.5 −9 → **운영 미투입**(AUROC-only/연구 자산).

base = "routed"(기본, src_routed) 또는 "same_mic"(src_same_mic).
gbm 결측 행은 rank(base) 로 폴백(같은 rank 공간 유지). stage=SCORE, status=REJECTED.
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status

_BASE_COL = {"routed": B.SRC_ROUTED, "same_mic": B.SRC_SAME_MIC}


class RankAvgEnsemble(Method):
    name = "rank_avg_ensemble"
    stage = Stage.SCORE
    status = Status.REJECTED
    pass_impact = "AUROC +0.013 / OOF PASS −4~−9"
    summary = ("0.5·rank(model)+0.5·rank(GBM). AUROC 최고지만 held-out PASS 악화 → 미투입.")

    def __init__(self, w_gbm: float = 0.5, base: str = "routed"):
        if base not in _BASE_COL:
            raise ValueError(f"base∈{{routed,same_mic}}: {base}")
        self.w_gbm = float(w_gbm)
        self.base = base

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        bcol = _BASE_COL[self.base]
        self._require(per_conc, [bcol, B.SRC_OBJAREA_GBM])
        out = per_conc.copy()
        rb = out[bcol].rank(pct=True)
        rg = out[B.SRC_OBJAREA_GBM].rank(pct=True)
        ens = (1.0 - self.w_gbm) * rb + self.w_gbm * rg
        out[B.SCORE_COL] = ens.where(rg.notna(), rb)   # gbm 결측 → base rank
        return out

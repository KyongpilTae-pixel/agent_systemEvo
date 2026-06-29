"""Routing + objarea_gbm 후보 — cell 별 routed vs objarea_gbm 중 AUROC 우위 선택.

기존 PerCellModelRouting(6 subset 모델 중 best)에 **objarea_gbm 을 추가 후보**로 넣는다.
cell 마다 routed 점수와 objarea_gbm 점수의 per-cell AUROC 를 비교해, objarea_gbm 이 min_gain
이상 높으면 그 cell 은 objarea_gbm 으로 라우팅. (라우팅 결정은 기존 routing lookup 과 동일하게
FDA in-sample AUROC 기준.)

stage=SCORE. switched_cells() 로 전환된 cell 확인.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status


def _auroc(y, s) -> float:
    m = np.isfinite(s)
    if m.sum() < 4 or len(set(y[m].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


class RoutingPlusGBM(Method):
    name = "routing_plus_gbm"
    stage = Stage.SCORE
    status = Status.AUXILIARY
    pass_impact = "routing 후보에 objarea_gbm 추가"
    summary = ("cell별 routed vs objarea_gbm AUROC 비교해 우위 선택. borderline cell 랭킹 개선.")

    def __init__(self, min_gain: float = 0.005):
        self.min_gain = float(min_gain)
        self._switched: list[tuple] = []

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        self._require(per_conc, [B.SRC_ROUTED, B.SRC_OBJAREA_GBM, B.LABEL_COL,
                                 B.OG_COL, B.DRUG_COL])
        out = per_conc.copy()
        out[B.SCORE_COL] = out[B.SRC_ROUTED].astype(float)   # 기본 routed
        self._switched = []
        for (og, amr), sub in per_conc.groupby([B.OG_COL, B.DRUG_COL], sort=False):
            y = sub[B.LABEL_COL].astype(int).to_numpy()
            a_r = _auroc(y, sub[B.SRC_ROUTED].astype(float).to_numpy())
            a_g = _auroc(y, sub[B.SRC_OBJAREA_GBM].astype(float).to_numpy())
            if np.isfinite(a_g) and np.isfinite(a_r) and (a_g - a_r) > self.min_gain:
                out.loc[sub.index, B.SCORE_COL] = sub[B.SRC_OBJAREA_GBM].astype(float)
                self._switched.append((str(og), str(amr), round(a_r, 4), round(a_g, 4)))
        return out

    def switched_cells(self) -> pd.DataFrame:
        return pd.DataFrame(self._switched,
                            columns=[B.OG_COL, B.DRUG_COL, "auroc_routed", "auroc_gbm"])

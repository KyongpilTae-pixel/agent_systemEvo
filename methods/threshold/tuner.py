"""④ Global threshold tuning — VME budget 하 PASS 최대 threshold 추천 (조정 lever).

calibrated P(NG) 를 G/NG 로 이진화하는 전역 threshold 를 VME-PASS trade-off lever 로 사용.
운영 3-recipe: t0.50(적극 111/3.56%) / t0.65(균형 ★103/1.99%) / t0.75(보수 91/1.50%).

효율: score+calibrate 체인은 1회만 적용(apply_chain)하고 threshold 만 sweep 한다.
stage=THRESHOLD, status=ADOPTED. apply()는 identity; tune() 이 핵심.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status
from agent_system.methods.recipe import Recipe, run_verify, load_unified

DEFAULT_GRID = (0.50, 0.65, 0.75)


class ThresholdTuner(Method):
    name = "threshold_tuner"
    stage = Stage.THRESHOLD
    status = Status.ADOPTED
    pass_impact = "VME 통제 lever"
    summary = ("calibrated P(NG) 이진화 전역 threshold. VME budget 하 PASS 최대 t 추천. "
               "운영 t0.65(103/1.99%).")

    def __init__(self, grid: Sequence[float] = DEFAULT_GRID, vme_budget: float = 0.02):
        self.grid = list(grid)
        self.vme_budget = vme_budget

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        # threshold 는 verify 단계 lever — per_conc 변형 없음.
        return per_conc

    def tune(self, score: Optional[Method] = None, calibrate: Optional[Method] = None,
             per_conc: Optional[pd.DataFrame] = None,
             out_subdir: str = "threshold_tune") -> dict:
        """주어진 score+calibrate 로 만든 P(NG) 에 대해 grid threshold sweep.

        반환: {"table": [...per threshold...], "recommended": t, "rows": DataFrame}
        """
        if per_conc is None:
            per_conc = load_unified()
        rec = Recipe(score=score, calibrate=calibrate)
        pc = rec.apply_chain(per_conc)            # 1회만 score+calibrate
        rows = []
        for t in self.grid:
            r = run_verify(pc, B.OUTPUT_DIR / out_subdir / f"t{t}", t)
            rows.append({"threshold": t, "pass": r["pass"],
                         "vme_cells": r["vme_cells"], "vme_rate": r["vme_rate"]})
        tbl = pd.DataFrame(rows)
        # VME budget 만족 중 PASS 최대 (없으면 VME 최소)
        ok = tbl[tbl["vme_rate"] <= self.vme_budget]
        if len(ok):
            recommended = float(ok.sort_values(["pass", "vme_rate"],
                                               ascending=[False, True])["threshold"].iloc[0])
        else:
            recommended = float(tbl.sort_values("vme_rate")["threshold"].iloc[0])
        return {"table": rows, "recommended": recommended, "rows": tbl,
                "vme_budget": self.vme_budget}

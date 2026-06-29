"""Hybrid isotonic — genus 기본 + VME 위험 cell 만 organism 폴백.

genus-iso 는 held-out 일반화가 organism 보다 우위(+4)지만 특정 cell 에서 R 균주를 S 로
오판(VME)한다. 그래서 **VME 위험 cell(genus VME > organism VME)만 organism-iso 로 되돌리고
나머지는 genus-iso** 를 쓰는 per-cell calibration 라우팅.

fallback_cells = {(organism_group, antimicrobial)} — 이 cell 은 organism-iso, 그 외 genus-iso.
mode 는 두 backend 에 동일 적용(insample/oof/lookup). stage=CALIBRATE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status
from agent_system.methods.calibrate.isotonic import PerCellIsotonic


class HybridIsotonic(Method):
    name = "hybrid_isotonic"
    stage = Stage.CALIBRATE
    status = Status.ALTERNATIVE
    pass_impact = "genus 일반화 + VME 회귀 차단"
    summary = ("genus-iso 기본 + VME 위험 cell 만 organism-iso 폴백. 일반화 이득 유지하며 "
               "VME 회귀 방지.")

    def __init__(self, mode: str = "insample", n_folds: int = 5,
                 fallback_cells: Optional[Iterable[tuple]] = None,
                 fallback_csv: Optional[str | Path] = None):
        self.mode = mode
        self.org = PerCellIsotonic(mode, n_folds, cell_by="organism_group")
        self.gen = PerCellIsotonic(mode, n_folds, cell_by="genus")
        if fallback_cells is not None:
            self.fallback = {(str(o), str(d)) for o, d in fallback_cells}
        elif fallback_csv is not None:
            df = pd.read_csv(fallback_csv)
            self.fallback = {(str(o), str(d)) for o, d in
                             zip(df[B.OG_COL], df[B.DRUG_COL])}
        else:
            self.fallback = set()

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        self._require(per_conc, [B.SCORE_COL, B.OG_COL, B.DRUG_COL, B.LABEL_COL, B.GENUS_COL])
        gen = self.gen.apply(per_conc)            # 전부 genus 보정
        org = self.org.apply(per_conc)            # 전부 organism 보정
        out = gen.copy()
        keys = list(zip(per_conc[B.OG_COL].astype(str), per_conc[B.DRUG_COL].astype(str)))
        mask = pd.Series([k in self.fallback for k in keys], index=per_conc.index)
        out.loc[mask, B.SCORE_COL] = org.loc[mask, B.SCORE_COL]
        return out

    def stats(self) -> dict:
        return {"fallback_cells": len(self.fallback), "mode": self.mode}

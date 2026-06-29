"""⑨ Brightness asymmetric corrector — G-방향 비대칭 보정 (CV 기각).

Brightness=True apply cell(45개) 한정, brightness GBM 예측(sp)이 image 점수(img)보다 더
G(=낮은 NG)일 때(sp<img)만 점수를 G-방향으로 내림: corrected = (1-w)·img + w·sp.
운영 score=P(NG)이라 G-방향만 보정하면 VME(거짓 NG)만 차단.

in-sample +3 PASS(103→106)였으나 GroupKFold(sample_id) CV held-out 미재현
(baseline 87 → corrected 86, 모든 w 열세, 2026-06-04) → **운영 미투입**. 원인: in-sample
isotonic 이 보정 score↔라벨을 cell 내부에서 암기. base=src_routed, sp=src_brightness_gbm.
stage=SCORE, status=REJECTED.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status

# apply cells (organism_group, antimicrobial) — Brightness=True 45 cell
APPLY_CELLS_CSV = B.ROOT / "agent_system/brightness_asym_corrector/brightness_apply_cells.csv"


def _load_apply_cells() -> set:
    cells = pd.read_csv(APPLY_CELLS_CSV)
    return set(zip(cells["organism_group"].astype(str).str.strip(),
                   cells["antimicrobial"].astype(str).str.strip()))


class BrightnessAsymCorrector(Method):
    name = "brightness_asym_corrector"
    stage = Stage.SCORE
    status = Status.REJECTED
    pass_impact = "in-sample +3 / CV −1"
    summary = ("apply cell 한정 G-방향 brightness 보정(sp<img). in-sample +3 but "
               "GroupKFold CV held-out 미재현 → 미투입.")

    def __init__(self, weight: float = 0.7, base: str = "routed"):
        self.weight = float(weight)
        self.base_col = B.SRC_ROUTED if base == "routed" else B.SRC_SAME_MIC
        self.apply_cells = _load_apply_cells()

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        self._require(per_conc, [self.base_col, B.SRC_BRIGHTNESS_GBM, B.OG_COL, B.DRUG_COL])
        out = per_conc.copy()
        img = out[self.base_col].astype(float).to_numpy()
        sp_raw = out[B.SRC_BRIGHTNESS_GBM].astype(float).to_numpy()
        sp = np.where(np.isfinite(sp_raw), sp_raw, img)     # 미커버 → 보정 안 함
        in_cell = np.array([(str(o).strip(), str(d).strip()) in self.apply_cells
                            for o, d in zip(out[B.OG_COL], out[B.DRUG_COL])])
        mask = in_cell & (sp < img) & np.isfinite(sp_raw)   # G-방향만 (비대칭)
        corrected = img.copy()
        corrected[mask] = (1 - self.weight) * img[mask] + self.weight * sp[mask]
        out[B.SCORE_COL] = np.clip(corrected, 0.0, 1.0)
        return out

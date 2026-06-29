"""① Per-cell isotonic calibration — 운영 PASS 주역 (+30, baseline 72 → 103).

각 (organism_group, antimicrobial) cell 단위로 IsotonicRegression 을 적합해 raw NG 점수를
P(NG) 확률로 단조 보정. 기존 검증 클래스 claudeCode/isotonic_lookup.PerCellIsotonicLookup
을 재사용하는 thin wrapper (로직 재구현 없음).

모드:
  insample : 전체 패널 full-fit + calibrate     ← 운영 배포 방식 (in-sample, 낙관적)
  oof      : GroupKFold(sample_id) OOF isotonic  ← held-out 정직 평가 (누설 0)
  lookup   : frozen isotonic_lookup CSV 로드 적용 ← 실배포(고정 자산) 재현

stage=CALIBRATE, status=ADOPTED.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status
from claudeCode.isotonic_lookup import PerCellIsotonicLookup

GROUP_COL = "sample_id"
# 운영 배포 자산 (full-data fit, routed 점수 기준)
DEFAULT_LOOKUP = B.ROOT / "agent_system/output/isotonic_lookup_routed.csv"


class PerCellIsotonic(Method):
    name = "per_cell_isotonic"
    stage = Stage.CALIBRATE
    status = Status.ADOPTED
    pass_impact = "+30"
    summary = ("cell별 isotonic 으로 raw 점수 → P(NG). 운영 PASS 주역. "
               "in-sample(배포)=103, held-out OOF=87.")

    def __init__(self, mode: str = "insample", n_folds: int = 5,
                 lookup_csv: Optional[str | Path] = None,
                 cell_by: str = "organism_group", method: str = "standard",
                 roc_lambda: float = 0.1):
        if mode not in ("insample", "oof", "lookup"):
            raise ValueError(f"mode∈{{insample,oof,lookup}}: {mode}")
        if cell_by not in ("organism_group", "genus"):
            raise ValueError(f"cell_by∈{{organism_group,genus}}: {cell_by}")
        if method not in ("standard", "cir", "pchip", "roc"):
            raise ValueError(f"method∈{{standard,cir,pchip,roc}}: {method}")
        self.mode = mode
        self.n_folds = n_folds
        self.lookup_csv = Path(lookup_csv) if lookup_csv else DEFAULT_LOOKUP
        self.cell_by = cell_by   # 학습+적용 그룹 단위 (배포 환경은 genus 만 알 수 있음)
        self.method = method     # standard(PAVA 계단)/cir/pchip(연속 단조)/roc(ties 완화)
        self.roc_lambda = roc_lambda

    def _fit(self, work: pd.DataFrame):
        """method 에 따라 표준 또는 smooth/roc lookup 적합."""
        if self.method == "standard":
            return PerCellIsotonicLookup.fit(work)
        from agent_system.methods.calibrate.smooth_isotonic import build_smooth_lookup
        return build_smooth_lookup(work, method=self.method, roc_lambda=self.roc_lambda)

    def _prep(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        """작업 프레임: ng_score→dtw_model_pred 번역 + cell_by 단위를 organism_group 자리에 매핑.

        PerCellIsotonicLookup 은 (organism_group, antimicrobial) 로 그룹하므로, genus 단위
        학습/적용 시 organism_group 컬럼에 genus 값을 넣어 그룹 단위만 genus 로 바꾼다.
        (실제 organism_group 은 호출부 out 프레임이 보존 → summary 는 organism_group 으로 집계.)
        """
        work = per_conc.copy()
        work[B.EXTERNAL_SCORE_COL] = work[B.SCORE_COL]      # ng_score → dtw_model_pred
        if self.cell_by == "genus":
            self._require(per_conc, [B.GENUS_COL])
            work[B.OG_COL] = work[B.GENUS_COL].astype(str)  # 그룹 단위 = genus
        return work

    # ---- calibration backends (method 에 따라 standard/smooth lookup) ----
    def _insample(self, per_conc: pd.DataFrame) -> np.ndarray:
        iso = self._fit(per_conc)
        return iso.calibrate_dataframe(per_conc, out_col="_cal")["_cal"].to_numpy()

    def _oof(self, per_conc: pd.DataFrame) -> np.ndarray:
        df = per_conc.reset_index(drop=True)
        oof = np.full(len(df), np.nan, dtype=float)
        for tr, te in GroupKFold(n_splits=self.n_folds).split(df, groups=df[GROUP_COL].values):
            iso = self._fit(df.iloc[tr])
            oof[te] = iso.calibrate_dataframe(df.iloc[te], out_col="_cal")["_cal"].to_numpy()
        if not np.isfinite(oof).all():
            raise RuntimeError("OOF 미할당 row 존재")
        return oof

    def _lookup(self, per_conc: pd.DataFrame) -> np.ndarray:
        iso = PerCellIsotonicLookup.load(self.lookup_csv)
        return iso.calibrate_dataframe(per_conc, out_col="_cal")["_cal"].to_numpy()

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        self._require(per_conc, [B.SCORE_COL, B.OG_COL, B.DRUG_COL, B.LABEL_COL])
        out = per_conc.copy()                  # 실제 organism_group 보존 (summary 용)
        work = self._prep(out)                 # 경계 번역 + cell_by 그룹 매핑
        if self.mode == "insample":
            out[B.SCORE_COL] = self._insample(work)
        elif self.mode == "oof":
            out[B.SCORE_COL] = self._oof(work)
        else:
            out[B.SCORE_COL] = self._lookup(work)
        return out

    # ---- 학습 전용 (보정/평가 없이 isotonic 적합만) ----
    def fit_lookup(self, per_conc: pd.DataFrame,
                   save_path: str | Path | None = None,
                   min_rows: int = 30, min_per_class: int = 3) -> PerCellIsotonicLookup:
        """ng_score ↔ gt_gng 로 cell별 isotonic 을 **학습만** 하고 lookup 반환(보정 X).

        full-data fit(배포용). save_path 주면 CSV 로 저장(런타임은 sklearn 없이 np.interp).
        조건 미달 cell(min_rows 30 / 클래스당 min_per_class 3)은 학습에서 제외(=identity fallback).
        """
        self._require(per_conc, [B.SCORE_COL, B.OG_COL, B.DRUG_COL, B.LABEL_COL])
        work = self._prep(per_conc)   # 경계 번역 + cell_by(organism_group/genus) 그룹 매핑
        iso = PerCellIsotonicLookup.fit(work, min_rows=min_rows, min_per_class=min_per_class)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            iso.save(save_path)
        return iso

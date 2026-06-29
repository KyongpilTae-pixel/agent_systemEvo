"""Per-cell isotonic calibration — 공유용 독립 구현 (프로젝트 의존 없음).

핵심 엔진은 `lookup.PerCellIsotonicLookup` (numpy/pandas/sklearn 만). 본 모듈은 그 위에
학습/적용 3모드 + genus 단위 + Hybrid 폴백을 컬럼명 설정 가능한 형태로 제공한다.

표준 흐름:
  점수(score_col) ↔ 라벨(label_col) 로 (group, drug) cell 별 isotonic 적합 → 점수를 P(NG) 로 보정.
  단조 함수라 ranking 보존(AUROC 불변), 확률 품질만 개선. 런타임 보정은 sklearn 없이 np.interp.

cell_by:
  "organism_group" : (og_col, drug_col) cell 단위 (기본)
  "genus"          : (genus_col, drug_col) cell 단위 (배포 환경이 genus 만 알 때)
mode:
  "insample" : 전체 데이터 full-fit + 보정 (in-sample, 낙관)
  "oof"      : GroupKFold(group_col) held-out (정직)
  "lookup"   : 동결 CSV 로드 적용 (sklearn 불필요 런타임)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .lookup import PerCellIsotonicLookup

# lookup.PerCellIsotonicLookup 이 내부적으로 기대하는 표준 컬럼명(고정)
_STD_SCORE, _STD_LABEL, _STD_OG, _STD_DRUG = (
    "dtw_model_pred", "gt_gng", "organism_group", "antimicrobial")

ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_LOOKUP = {"organism_group": ASSETS / "isotonic_lookup_routed.csv",
                  "genus": ASSETS / "isotonic_lookup_genus.csv"}


class PerCellIsotonic:
    """Per-cell isotonic calibrator. 사용자 DataFrame 의 컬럼명을 설정으로 받는다."""

    def __init__(self, mode: str = "insample", n_folds: int = 5,
                 lookup_csv: Optional[str | Path] = None, cell_by: str = "organism_group",
                 score_col: str = "dtw_model_pred", label_col: str = "gt_gng",
                 og_col: str = "organism_group", genus_col: str = "genus",
                 drug_col: str = "antimicrobial", group_col: str = "sample_id"):
        if mode not in ("insample", "oof", "lookup"):
            raise ValueError(f"mode∈{{insample,oof,lookup}}: {mode}")
        if cell_by not in ("organism_group", "genus"):
            raise ValueError(f"cell_by∈{{organism_group,genus}}: {cell_by}")
        self.mode, self.n_folds, self.cell_by = mode, n_folds, cell_by
        self.score_col, self.label_col = score_col, label_col
        self.og_col, self.genus_col, self.drug_col, self.group_col = (
            og_col, genus_col, drug_col, group_col)
        self.lookup_csv = Path(lookup_csv) if lookup_csv else DEFAULT_LOOKUP[cell_by]

    # 사용자 컬럼 → lookup 표준 컬럼 작업 프레임 (cell_by=genus 면 og 자리에 genus)
    def _std(self, df: pd.DataFrame, need_label: bool) -> pd.DataFrame:
        grp = self.genus_col if self.cell_by == "genus" else self.og_col
        for c in (self.score_col, grp, self.drug_col):
            if c not in df.columns:
                raise KeyError(f"컬럼 누락: {c}")
        w = pd.DataFrame(index=df.index)
        w[_STD_SCORE] = df[self.score_col].astype(float)
        w[_STD_OG] = df[grp].astype(str)
        w[_STD_DRUG] = df[self.drug_col].astype(str)
        w[self.group_col] = (df[self.group_col] if self.group_col in df.columns
                             else np.arange(len(df)))
        if need_label:
            if self.label_col not in df.columns:
                raise KeyError(f"라벨 컬럼 누락: {self.label_col}")
            w[_STD_LABEL] = df[self.label_col].astype(int)
        return w

    def fit_lookup(self, df: pd.DataFrame, save_path: Optional[str | Path] = None,
                   min_rows: int = 30, min_per_class: int = 3) -> PerCellIsotonicLookup:
        """점수↔라벨로 cell별 isotonic 학습만 (보정 X). save_path 주면 CSV 동결."""
        w = self._std(df, need_label=True)
        iso = PerCellIsotonicLookup.fit(w, min_rows=min_rows, min_per_class=min_per_class)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            iso.save(save_path)
        return iso

    def _oof(self, w: pd.DataFrame) -> np.ndarray:
        from sklearn.model_selection import GroupKFold
        oof = np.full(len(w), np.nan, dtype=float)
        groups = w[self.group_col].to_numpy()
        for tr, te in GroupKFold(n_splits=self.n_folds).split(w, groups=groups):
            iso = PerCellIsotonicLookup.fit(w.iloc[tr])
            oof[te] = iso.calibrate_dataframe(w.iloc[te], out_col="_c")["_c"].to_numpy()
        if not np.isfinite(oof).all():
            raise RuntimeError("OOF 미할당 row")
        return oof

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """score_col 을 P(NG) 보정값으로 갱신한 새 DataFrame 반환."""
        need_label = self.mode in ("insample", "oof")
        w = self._std(df, need_label=need_label)
        if self.mode == "insample":
            iso = PerCellIsotonicLookup.fit(w)
            cal = iso.calibrate_dataframe(w, out_col="_c")["_c"].to_numpy()
        elif self.mode == "oof":
            cal = self._oof(w)
        else:  # lookup
            iso = PerCellIsotonicLookup.load(self.lookup_csv)
            cal = iso.calibrate_dataframe(w, out_col="_c")["_c"].to_numpy()
        out = df.copy()
        out[self.score_col] = cal
        return out


class HybridIsotonic:
    """genus 기본 + VME 위험 cell 만 organism 폴백 (best-of-both calibration)."""

    def __init__(self, mode: str = "insample", n_folds: int = 5,
                 fallback_cells: Optional[Iterable[tuple]] = None,
                 fallback_csv: Optional[str | Path] = None, **colkw):
        self.org = PerCellIsotonic(mode, n_folds, cell_by="organism_group", **colkw)
        self.gen = PerCellIsotonic(mode, n_folds, cell_by="genus", **colkw)
        self.og_col = colkw.get("og_col", "organism_group")
        self.drug_col = colkw.get("drug_col", "antimicrobial")
        self.score_col = colkw.get("score_col", "dtw_model_pred")
        if fallback_cells is not None:
            self.fallback = {(str(o), str(d)) for o, d in fallback_cells}
        elif fallback_csv is not None:
            f = pd.read_csv(fallback_csv)
            self.fallback = {(str(o), str(d)) for o, d in
                             zip(f[self.og_col], f[self.drug_col])}
        else:
            self.fallback = set()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        gen = self.gen.apply(df)
        org = self.org.apply(df)
        out = gen.copy()
        keys = list(zip(df[self.og_col].astype(str), df[self.drug_col].astype(str)))
        mask = pd.Series([k in self.fallback for k in keys], index=df.index)
        out.loc[mask, self.score_col] = org.loc[mask, self.score_col]
        return out

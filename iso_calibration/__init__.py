"""iso_calibration — per-cell isotonic 보정 공유 패키지 (프로젝트 의존 없음).

핵심: 모델 점수를 (organism_group 또는 genus, antimicrobial) cell 단위로 isotonic 보정해
P(NG) 확률로 만든다. 단조라 ranking 보존(AUROC 불변), 확률 품질만 개선. 런타임은 sklearn 없이
np.interp.

빠른 시작:
    from iso_calibration import PerCellIsotonic
    cal = PerCellIsotonic(mode="lookup").apply(df)        # 동결 자산으로 보정 (운영)
    iso = PerCellIsotonic().fit_lookup(df, "my_lookup.csv")  # 직접 학습 + 동결
"""
from .lookup import PerCellIsotonicLookup, CellIsotonic
from .calibrator import PerCellIsotonic, HybridIsotonic

__all__ = ["PerCellIsotonicLookup", "CellIsotonic", "PerCellIsotonic", "HybridIsotonic"]

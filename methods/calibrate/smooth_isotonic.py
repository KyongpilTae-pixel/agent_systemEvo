"""Smooth isotonic calibration — 계단(PAVA) 대신 연속 단조 곡선 (overfit 완화).

표준 per-cell isotonic 은 계단 함수라 표본 적은 cell 에서 score↔label 과적합
(in-sample 103 vs OOF 87 gap). smooth 변형으로 분산을 줄여 held-out 일반화를 노린다.

method:
  "cir"   : Centered Isotonic Regression (Oron&Flournoy) — 각 level-set 의 중심 x 에 적합값을
            놓고 선형 보간. 표본 적은 cell 에서 덜 과적합. 가장 원리적.
  "pchip" : PAVA breakpoint 를 monotone cubic(PCHIP)으로 보간 → 부드러운 단조 곡선.
  "roc"   : ROC-regularized — isotonic flat 구간 ties 가 경계 ranking 해상도 저하 → rank-단조로
            λ(roc_lambda) 만큼 당겨 ties 를 깸. discrimination 보존(둘 다 단조→단조).

산출은 표준과 동일한 CellIsotonic(x_thresholds, y_thresholds) 포맷 → 런타임은 np.interp 그대로
(sklearn-free). PerCellIsotonicLookup.fit 과 동일한 cell grouping/필터(min_rows/min_per_class).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

from claudeCode.isotonic_lookup import (CellIsotonic, PerCellIsotonicLookup,
                                        _OG_COL, _DRUG_COL, _MODEL_COL, _LABEL_COL)

GRID = 100   # pchip 보간 grid


def _fit_cell_smooth(s: np.ndarray, y: np.ndarray, method: str, roc_lambda: float = 0.1):
    """(x_thresholds, y_thresholds) 반환 — 연속 단조."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    yhat = iso.fit_transform(s, y)                      # 점별 적합값(단조)
    xs = np.asarray(iso.X_thresholds_, dtype=float)
    ys = np.asarray(iso.y_thresholds_, dtype=float)

    if method == "roc":
        # ROC-regularized: isotonic 의 flat 구간 ties 가 경계 ranking 해상도를 떨어뜨림.
        # rank-단조(=score 정규화)로 λ 만큼 당겨 ties 를 깨 discrimination 보존(둘 다 단조→단조).
        smin, smax = float(s.min()), float(s.max())
        if smax - smin < 1e-9:
            return xs.tolist(), ys.tolist()
        grid = np.linspace(smin, smax, GRID)
        iso_g = np.interp(grid, xs, ys)
        rank_g = (grid - smin) / (smax - smin)          # ROC 보존 단조 기준
        yg = np.clip((1.0 - roc_lambda) * iso_g + roc_lambda * rank_g, 0.0, 1.0)
        yg = np.maximum.accumulate(yg)
        return grid.tolist(), yg.tolist()

    if method == "cir":
        # level-set(동일 yhat 연속 구간)의 중심 x + 적합값 → 선형보간 점
        order = np.argsort(s)
        s_o, yh_o = np.asarray(s, float)[order], yhat[order]
        cx, cy, i = [], [], 0
        while i < len(yh_o):
            j = i
            while j < len(yh_o) and yh_o[j] == yh_o[i]:
                j += 1
            cx.append(float(s_o[i:j].mean())); cy.append(float(yh_o[i])); i = j
        cx, cy = np.array(cx), np.array(cy)
        # x 중복 제거(증가 보장)
        keep = np.concatenate([[True], np.diff(cx) > 1e-9])
        cx, cy = cx[keep], cy[keep]
        if len(cx) < 2:
            return xs.tolist(), ys.tolist()
        return cx.tolist(), cy.tolist()

    if method == "pchip":
        ux, idx = np.unique(xs, return_index=True)
        uy = ys[idx]
        if len(ux) < 2:
            return xs.tolist(), ys.tolist()
        grid = np.linspace(float(ux.min()), float(ux.max()), GRID)
        gy = np.clip(PchipInterpolator(ux, uy)(grid), 0.0, 1.0)
        gy = np.maximum.accumulate(gy)                  # 단조 강제(수치 안전)
        return grid.tolist(), gy.tolist()

    raise ValueError(f"method∈{{cir,pchip}}: {method}")


def build_smooth_lookup(per_conc, method: str = "cir",
                        min_rows: int = 30, min_per_class: int = 3,
                        roc_lambda: float = 0.1) -> PerCellIsotonicLookup:
    """표준 PerCellIsotonicLookup.fit 과 동일 grouping/필터로 smooth cell 적합.
    method ∈ {cir, pchip, roc}."""
    df = per_conc.dropna(subset=[_MODEL_COL, _OG_COL, _DRUG_COL, _LABEL_COL])
    cells = {}
    for (og, amr), sub in df.groupby([_OG_COL, _DRUG_COL], sort=False):
        y = sub[_LABEL_COL].astype(int).values
        s = sub[_MODEL_COL].astype(float).values
        n_g, n_ng = int((y == 0).sum()), int((y == 1).sum())
        if len(sub) < min_rows or n_g < min_per_class or n_ng < min_per_class:
            continue
        xt, yt = _fit_cell_smooth(s, y, method, roc_lambda=roc_lambda)
        cells[(str(og), str(amr))] = CellIsotonic(
            organism_group=str(og), antimicrobial=str(amr),
            x_thresholds=xt, y_thresholds=yt,
            n_train_rows=int(len(sub)), n_g=n_g, n_ng=n_ng)
    return PerCellIsotonicLookup(cells)

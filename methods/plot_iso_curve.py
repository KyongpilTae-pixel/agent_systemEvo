"""학습된 per-cell isotonic 의 교정 함수(ng_score → 보정 P(NG)) 예시 그래프.

한 cell 을 골라 isotonic 계단 함수(X_thresholds_ ↔ y_thresholds_)를 raw 데이터(ng_score vs
gt_gng)와 함께 그린다. organism_group cell 과 그 상위 genus cell 곡선을 겹쳐 비교.

사용: python -m agent_system.methods.plot_iso_curve --organism "Escherichia coli" --drug MEV
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent_system.methods import base as B
from agent_system.methods.recipe import load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic


def _curve(iso, key):
    cell = iso.cells.get(key)
    if cell is None:
        return None
    return np.asarray(cell.x_thresholds), np.asarray(cell.y_thresholds), cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", default="Escherichia coli")
    ap.add_argument("--drug", default=None, help="미지정 시 해당 organism 에서 행 많은 drug 자동 선택")
    ap.add_argument("--out", default=str(B.PKG_DIR / "output" / "iso_curve_example.png"))
    ap.add_argument("--summary_csv", default=None,
                    help="genus-iso method_summary CSV 주면 organism×drug PASS/FAIL 배지 표시")
    args = ap.parse_args()

    u = load_unified()
    routed = PerCellModelRouting().apply(u)          # ng_score = routed model_pred
    genus = str(routed.loc[routed[B.OG_COL] == args.organism, B.GENUS_COL].iloc[0])

    # drug 자동 선택: 해당 organism 에서 G/NG 둘 다 있고 행 많은 약제
    sub_org = routed[routed[B.OG_COL] == args.organism]
    if args.drug is None:
        cand = (sub_org.groupby(B.DRUG_COL)
                .agg(n=("gt_gng", "size"), npos=("gt_gng", "sum")))
        cand = cand[(cand["npos"] >= 5) & (cand["n"] - cand["npos"] >= 5)]
        args.drug = cand["n"].idxmax()
    cell_key = (args.organism, args.drug)

    # 학습: organism_group cell / genus cell
    iso_org = PerCellIsotonic(cell_by="organism_group").fit_lookup(routed)
    iso_gen = PerCellIsotonic(cell_by="genus").fit_lookup(routed)
    org_c = _curve(iso_org, cell_key)
    gen_c = _curve(iso_gen, (genus, args.drug))

    # raw 데이터 (이 organism×drug 행)
    rows = sub_org[sub_org[B.DRUG_COL] == args.drug]
    x_raw = rows[B.SCORE_COL].to_numpy()
    y_raw = rows[B.LABEL_COL].to_numpy()

    grid = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    # raw 산점 (G=0 아래, NG=1 위) — jitter
    rng_y = y_raw + (np.random.RandomState(0).rand(len(y_raw)) - 0.5) * 0.06
    ax.scatter(x_raw, rng_y, s=14, alpha=0.35, color="#90a4ae",
               label=f"raw rows (n={len(rows)}, NG={int(y_raw.sum())})")
    # isotonic 곡선
    if org_c:
        xo, yo, co = org_c
        ax.plot(grid, np.interp(grid, xo, yo), color="#1565c0", lw=2.4,
                label=f"organism: {args.organism} (n={co.n_train_rows})")
    if gen_c:
        xg, yg, cg = gen_c
        ax.plot(grid, np.interp(grid, xg, yg), color="#e65100", lw=2.4, ls="--",
                label=f"genus: {genus} (n={cg.n_train_rows})")
    ax.axhline(0.65, color="#c62828", lw=1, ls=":", alpha=0.8)
    ax.text(0.01, 0.66, "threshold 0.65", color="#c62828", fontsize=9)
    ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls="-", alpha=0.6, label="identity (no calibration)")

    # PASS/FAIL 배지 (genus-iso summary)
    if args.summary_csv:
        import pandas as pd
        sm = pd.read_csv(args.summary_csv)
        row = sm[(sm["organism_group"] == args.organism) & (sm["antimicrobial"] == args.drug)]
        if len(row):
            verdict = str(row["FDA_fail_list"].iloc[0])
            is_pass = verdict == "PASS"
            # 범례와 같은 쪽(그래프 밖 오른쪽, 범례 아래)에 배치
            ax.text(1.02, 0.30, f"genus-iso eval: {'PASS' if is_pass else verdict}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=11,
                    fontweight="bold", color="#1b5e20" if is_pass else "#b71c1c",
                    bbox=dict(boxstyle="round", fc="#e8f5e9" if is_pass else "#ffebee",
                              ec="#1b5e20" if is_pass else "#b71c1c"))

    ax.set_xlabel("ng_score (raw, routed model P(NG))")
    ax.set_ylabel("calibrated P(NG)  =  isotonic(ng_score)")
    ax.set_title(f"Per-cell isotonic calibration curve  -  {args.organism} x {args.drug}")
    ax.set_xlim(0, 1); ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.25)
    # 범례는 그래프 밖(오른쪽)에 배치
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(f"cell = {args.organism} × {args.drug} (genus={genus})")
    if org_c:
        print(f"  organism breakpoints x={np.round(org_c[0],3).tolist()}")
        print(f"                       y={np.round(org_c[1],3).tolist()}")


if __name__ == "__main__":
    main()

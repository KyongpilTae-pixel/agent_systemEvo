"""한 cell(organism×drug)의 점수 분포 히스토그램 3종 — raw / organism-iso / genus-iso.

각 패널: G(gt_gng=0)와 NG(gt_gng=1)를 색으로 나눠 분포를 그리고 threshold 0.65 표시.
보정이 두 클래스를 0.65 경계 양옆으로 어떻게 분리하는지 시각화.

사용: python -m agent_system.methods.plot_score_histograms --organism "Enterococcus faecium" --drug LNZ
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

THR = 0.65


def _panel(ax, score, y, title):
    g, ng = score[y == 0], score[y == 1]
    bins = np.linspace(0, 1, 26)
    ax.hist(g, bins=bins, color="#90a4ae", alpha=0.75, label=f"G (n={len(g)})")
    ax.hist(ng, bins=bins, color="#c62828", alpha=0.6, label=f"NG (n={len(ng)})")
    # 오분류(threshold 0.65 기준)를 해칭으로 강조
    #   G 인데 ≥0.65 → NG 로 오판(false NG, ME 방향)
    #   NG 인데 <0.65 → G 로 오판(false G, VME 방향)
    g_over = int((g >= THR).sum()); ng_miss = int((ng < THR).sum())
    ax.hist(g[g >= THR], bins=bins, facecolor="none", edgecolor="#263238",
            hatch="xxx", lw=0.0)
    ax.hist(ng[ng < THR], bins=bins, facecolor="none", edgecolor="#4a0000",
            hatch="xxx", lw=0.0)
    ax.axvline(THR, color="#1565c0", lw=1.6, ls="--")
    ax.text(THR + 0.01, ax.get_ylim()[1] * 0.92, "0.65", color="#1565c0", fontsize=9)
    # 오분류 개수 박스
    box = (f"misclassified @0.65\n"
           f"G->NG (false NG): {g_over}\n"
           f"NG->G (false G): {ng_miss}\n"
           f"total: {g_over + ng_miss}")
    ax.text(0.42, 0.96, box, transform=ax.transAxes, va="top", ha="left", fontsize=9.2,
            bbox=dict(boxstyle="round", fc="#fffde7", ec="#999"))
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=8.5, loc="upper left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", default="Enterococcus faecium")
    ap.add_argument("--drug", default="LNZ")
    ap.add_argument("--out", default=str(B.PKG_DIR / "output" / "score_histograms.png"))
    args = ap.parse_args()

    u = load_unified()
    routed = PerCellModelRouting().apply(u)                       # ng_score = routed
    cal_org = PerCellIsotonic("insample", cell_by="organism_group").apply(routed)
    cal_gen = PerCellIsotonic("insample", cell_by="genus").apply(routed)

    mask = (routed[B.OG_COL] == args.organism) & (routed[B.DRUG_COL] == args.drug)
    y = routed.loc[mask, B.LABEL_COL].to_numpy()
    raw = routed.loc[mask, B.SCORE_COL].to_numpy()
    org = cal_org.loc[mask, B.SCORE_COL].to_numpy()
    gen = cal_gen.loc[mask, B.SCORE_COL].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 10.2), sharex=True)
    _panel(axes[0], raw, y, "raw ng_score")
    _panel(axes[1], org, y, "organism-iso calibrated")
    _panel(axes[2], gen, y, "genus-iso calibrated")
    for ax in axes:
        ax.set_ylabel("count")
    axes[2].set_xlabel("score / P(NG)")
    fig.suptitle(f"{args.organism} x {args.drug}  -  score distribution (G vs NG, threshold 0.65)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(f"cell rows: {int(mask.sum())}  (NG={int(y.sum())}, G={int((y==0).sum())})")


if __name__ == "__main__":
    main()

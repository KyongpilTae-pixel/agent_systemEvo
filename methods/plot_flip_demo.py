"""FAIL→PASS 데모 (정직한 OOF 기준) — raw vs OOF isotonic 한 cell.

E. faecium × LNZ: raw = VME(FAIL) → organism-iso OOF = PASS. 72 샘플 중 단 1개(VME)가
verdict 를 가른다(EA 는 68/72 동일). 행 오분류가 아니라 경계 샘플의 SIR 이동이 동인임을 시각화.

세로 2단(raw / OOF-iso) 히스토그램 + verdict·EA/VME/ME 배지. find_flip_cell.py 가 먼저
output/flip_raw, output/flip_org_oof 를 생성해야 한다.

사용: python -m agent_system.methods.plot_flip_demo --organism "Enterococcus faecium" --drug LNZ
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent_system.methods import base as B

THR = 0.65
OUT = B.PKG_DIR / "output"


def _load(dirname, organism, drug):
    pc = pd.read_csv(OUT / dirname / "per_conc.csv", low_memory=False)
    pc = pc[(pc[B.OG_COL] == organism) & (pc[B.DRUG_COL] == drug)]
    sm = pd.read_csv(OUT / dirname / "method_summary_model_pred.csv")
    sm = sm[(sm[B.OG_COL] == organism) & (sm[B.DRUG_COL] == drug)]
    ev = pd.read_csv(OUT / dirname / "method_eval_model_pred.csv")
    ev = ev[(ev[B.OG_COL] == organism) & (ev[B.DRUG_COL] == drug)]
    verdict = str(sm["FDA_fail_list"].iloc[0])
    cnt = {c: int(ev[c].sum()) for c in ("isEA", "isVME", "isME") if c in ev.columns}
    return pc["dtw_model_pred"].to_numpy(), pc["gt_gng"].to_numpy(), verdict, cnt, len(ev)


def _panel(ax, score, y, title, verdict, cnt, n_samp):
    g, ng = score[y == 0], score[y == 1]
    bins = np.linspace(0, 1, 26)
    ax.hist(g, bins=bins, color="#90a4ae", alpha=0.75, label=f"G (n={len(g)})")
    ax.hist(ng, bins=bins, color="#c62828", alpha=0.6, label=f"NG (n={len(ng)})")
    ax.axvline(THR, color="#1565c0", lw=1.6, ls="--")
    ax.text(THR + 0.01, ax.get_ylim()[1] * 0.9, "0.65", color="#1565c0", fontsize=9)
    is_pass = verdict == "PASS"
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylabel("count")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8.5, loc="upper left")
    badge = (f"verdict: {'PASS' if is_pass else 'FAIL ('+verdict+')'}\n"
             f"samples: {n_samp}\n"
             f"EA pass: {cnt.get('isEA')}   VME: {cnt.get('isVME')}   ME: {cnt.get('isME')}")
    ax.text(0.98, 0.95, badge, transform=ax.transAxes, va="top", ha="right", fontsize=9.5,
            fontweight="bold", color="#1b5e20" if is_pass else "#b71c1c",
            bbox=dict(boxstyle="round", fc="#e8f5e9" if is_pass else "#ffebee",
                      ec="#1b5e20" if is_pass else "#b71c1c"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", default="Enterococcus faecium")
    ap.add_argument("--drug", default="LNZ")
    ap.add_argument("--out", default=str(OUT / "flip_demo.png"))
    args = ap.parse_args()

    raw = _load("flip_raw", args.organism, args.drug)
    iso = _load("flip_org_oof", args.organism, args.drug)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8.4), sharex=True)
    _panel(axes[0], raw[0], raw[1], "raw ng_score (no calibration)", raw[2], raw[3], raw[4])
    _panel(axes[1], iso[0], iso[1], "organism-iso  OOF (held-out, GroupKFold sample_id)",
           iso[2], iso[3], iso[4])
    axes[1].set_xlabel("score / P(NG)")
    # 실패 모드(raw verdict)와 변화량을 동적으로 제목에 표기
    deltas = []
    for k, lbl in [("isVME", "VME"), ("isME", "ME"), ("isEA", "EA pass")]:
        d = iso[3].get(k, 0) - raw[3].get(k, 0)
        if d:
            deltas.append(f"{lbl} {raw[3].get(k)}->{iso[3].get(k)}")
    fig.suptitle(f"FAIL ({raw[2]}) -> PASS  -  {args.organism} x {args.drug}  "
                 f"(n={raw[4]}; {', '.join(deltas)})", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(f"raw: {raw[2]} (VME={raw[3].get('isVME')})  ->  OOF-iso: {iso[2]} (VME={iso[3].get('isVME')})")


if __name__ == "__main__":
    main()

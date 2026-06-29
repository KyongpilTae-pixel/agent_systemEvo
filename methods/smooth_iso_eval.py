"""4 보정 기법 비교 — standard / cir / pchip / roc (clean, in-sample + OOF).

score=routed, threshold 0.65. 핵심 가설: 계단(standard)의 overfit(in-sample vs OOF gap)을
smooth(cir/pchip)·roc 가 줄여 held-out(OOF) PASS 를 올리는가. PASS·총 VME·gap 비교.

사용: python -m agent_system.methods.smooth_iso_eval
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

METHODS = ["standard", "cir", "pchip", "roc"]


def _total_vme(name):
    ev = pd.read_csv(B.OUTPUT_DIR / name / "method_eval_model_pred.csv")
    return int(ev["isVME"].sum())


def main() -> None:
    u = load_unified(clean=True)
    res = {}
    for meth in METHODS:
        for mode in ("insample", "oof"):
            name = f"smiso_{meth}_{mode}"
            r = Recipe(PerCellModelRouting(),
                       PerCellIsotonic(mode, method=meth), 0.65, name=name).evaluate(u)
            res[(meth, mode)] = (r["pass"], r["vme_cells"], _total_vme(name))

    print(f"\n{'method':10s} {'in-PASS':>8s} {'OOF-PASS':>9s} {'gap':>5s} "
          f"{'OOF-VMEcell':>11s} {'OOF-총VME':>9s}")
    for meth in METHODS:
        ip = res[(meth, "insample")][0]
        op, ovc, otv = res[(meth, "oof")]
        print(f"{meth:10s} {ip:>8d} {op:>9d} {ip-op:>5d} {ovc:>11d} {otv:>9d}")
    print("\n[해석] gap 작을수록 overfit 적음. OOF-PASS 가 standard(95) 대비 ↑면 smooth 효과.")


if __name__ == "__main__":
    main()

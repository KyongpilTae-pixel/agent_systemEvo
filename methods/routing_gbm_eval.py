"""RoutingPlusGBM 평가 (clean 기준) — routed 대비 borderline cell 개선 측정.

routed(운영) vs routing+gbm 을 isotonic+0.65 로 평가(clean=TE 제거 1434). 전환 cell 목록과
per-cell verdict 변화(FAIL→PASS / PASS→FAIL / 지표 호전)를 보고.

사용: python -m agent_system.methods.routing_gbm_eval
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.score.routing_gbm import RoutingPlusGBM
from agent_system.methods.calibrate.isotonic import PerCellIsotonic
from agent_system.methods import base as B

KEY = ["organism_group", "antimicrobial"]
GAINS = [0.005, 0.01]


def _iso():
    return PerCellIsotonic("insample", cell_by="organism_group")


def _summ(name):
    return pd.read_csv(B.OUTPUT_DIR / name / "method_summary_model_pred.csv")[
        KEY + ["FDA_fail_list"]]


def main() -> None:
    u = load_unified(clean=True)
    base_r = Recipe(PerCellModelRouting(), _iso(), 0.65, name="rg_routed").evaluate(u)
    sr = _summ("rg_routed").rename(columns={"FDA_fail_list": "routed"})
    print(f"[clean] routed(운영) PASS {base_r['pass']} / VME-cell {base_r['vme_cells']}\n")

    for gain in GAINS:
        m = RoutingPlusGBM(min_gain=gain)
        name = f"rg_gbm_{int(gain*1000)}"
        r = Recipe(m, _iso(), 0.65, name=name).evaluate(u)
        sw = m.switched_cells()
        sg = _summ(name).rename(columns={"FDA_fail_list": "gbm"})
        cmp = sr.merge(sg, on=KEY)
        gain_cells = cmp[(cmp.routed != "PASS") & (cmp.gbm == "PASS")]
        loss_cells = cmp[(cmp.routed == "PASS") & (cmp.gbm != "PASS")]
        print(f"=== min_gain {gain} ===")
        print(f"  PASS {base_r['pass']} → {r['pass']} (Δ{r['pass']-base_r['pass']:+d}) | "
              f"VME-cell {base_r['vme_cells']} → {r['vme_cells']} | 전환 cell {len(sw)}")
        print(f"  FAIL→PASS {len(gain_cells)} / PASS→FAIL {len(loss_cells)}")
        if len(gain_cells):
            print("  [FAIL→PASS]")
            print(gain_cells.merge(sw, on=KEY, how="left").to_string(index=False))
        if len(loss_cells):
            print("  [PASS→FAIL]")
            print(loss_cells.merge(sw, on=KEY, how="left").to_string(index=False))
        # 전환 cell 중 verdict 호전(토큰 감소)
        sw.to_csv(B.OUTPUT_DIR / f"{name}_switched.csv", index=False)
        print(f"  전환 cell 저장 → output/{name}_switched.csv\n")


if __name__ == "__main__":
    main()

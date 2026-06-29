"""(a)+(b) 결합 평가 — RoutingPlusGBM(score) + {organism/genus/hybrid} isotonic, clean 기준.

score 축(routing 후보에 objarea_gbm 추가)과 calibration 축(genus+VME폴백 hybrid)이 직교라
stack 했을 때 더 오르는지 확인. routed baseline(hybc_* 산출, 별도)과 비교.

matrix: score∈{routed, routing_gbm} × calib∈{organism,genus,hybrid} × mode∈{insample,oof}.
(routed 행은 hybc_* 재사용 가능하나 일관성 위해 여기서 routing_gbm 만 신규 산출.)

사용: python -m agent_system.methods.combined_eval
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.routing_gbm import RoutingPlusGBM
from agent_system.methods.calibrate.isotonic import PerCellIsotonic
from agent_system.methods.calibrate.hybrid_iso import HybridIsotonic

OUT = B.OUTPUT_DIR
KEY = ["organism_group", "antimicrobial"]


def _percell_vme(name: str) -> pd.DataFrame:
    ev = pd.read_csv(OUT / name / "method_eval_model_pred.csv")
    return ev.groupby(KEY)["isVME"].sum().rename("VME").reset_index()


def _total_vme(name: str) -> int:
    return int(pd.read_csv(OUT / name / "method_eval_model_pred.csv")["isVME"].sum())


def main() -> None:
    u = load_unified(clean=True)
    score = RoutingPlusGBM(min_gain=0.005)
    sw = score.apply(u)  # 전환 cell 미리 (로그)
    n_sw = len(score.switched_cells())
    print(f"[데이터셋] clean 1434 | RoutingPlusGBM 전환 cell {n_sw}\n")

    pfx = "comb"

    def ev(calib_tag, mode, calib):
        name = f"{pfx}_{calib_tag}_{mode}"
        r = Recipe(RoutingPlusGBM(min_gain=0.005), calib, 0.65, name=name).evaluate(u)
        return r["pass"], r["vme_cells"], _total_vme(name)

    res = {}
    # organism/genus insample → fallback 도출 (routing_gbm score 기준)
    res[("organism", "insample")] = ev("organism", "insample",
                                       PerCellIsotonic("insample", cell_by="organism_group"))
    res[("genus", "insample")] = ev("genus", "insample",
                                    PerCellIsotonic("insample", cell_by="genus"))
    o = _percell_vme(f"{pfx}_organism_insample").rename(columns={"VME": "o"})
    g = _percell_vme(f"{pfx}_genus_insample").rename(columns={"VME": "g"})
    fb = o.merge(g, on=KEY, how="outer").fillna(0)
    fb = fb[fb["g"] > fb["o"]][KEY].reset_index(drop=True)
    fb.to_csv(OUT / f"{pfx}_fallback_cells.csv", index=False)
    fbset = set(zip(fb[B.OG_COL], fb[B.DRUG_COL]))
    print(f"[fallback] genus→organism {len(fb)}개 (routing_gbm score 기준 in-sample VME 악화)")

    res[("hybrid", "insample")] = ev("hybrid", "insample",
                                     HybridIsotonic("insample", fallback_cells=fbset))
    for v, c in [("organism", PerCellIsotonic("oof", cell_by="organism_group")),
                 ("genus", PerCellIsotonic("oof", cell_by="genus")),
                 ("hybrid", HybridIsotonic("oof", fallback_cells=fbset))]:
        res[(v, "oof")] = ev(v, "oof", c)

    print("\n=== routing_gbm score × isotonic (clean) ===")
    print(f"  {'calib':9s} {'mode':9s} {'PASS':>5s} {'VME-cell':>9s} {'총VME':>6s}")
    for mode in ("insample", "oof"):
        for v in ("organism", "genus", "hybrid"):
            p, vc, tv = res[(v, mode)]
            print(f"  {v:9s} {mode:9s} {p:>5d} {vc:>9d} {tv:>6d}")

    print("\n=== routed score 대비 (앞 hybc_* 결과) ===")
    print("  routed   organism oof 95 / genus oof 97 / hybrid oof 97 (VME 81/81/73)")
    print("  routed   *_insample      106 / 106 / 108           (VME 76/75/64)")
    for mode in ("insample", "oof"):
        for v in ("organism", "genus", "hybrid"):
            p, vc, tv = res[(v, mode)]
            print(f"  routing_gbm {v:9s} {mode:9s} PASS {p} VME {tv}")


if __name__ == "__main__":
    main()

"""Hybrid isotonic 평가 — genus 기본 + VME 위험 cell 만 organism 폴백.

fallback set = (해당 데이터셋) in-sample 에서 genus VME > organism VME 인 cell.
organism / genus / hybrid 를 in-sample + OOF 로 평가: PASS, VME-cell, 총 VME isolate.

--clean: TE 제거 평가셋(1434) 사용. 산출 dir prefix 와 fallback 을 그 데이터셋에서 재도출.
사용: python -m agent_system.methods.hybrid_iso_eval [--clean]
"""
from __future__ import annotations

import argparse

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic
from agent_system.methods.calibrate.hybrid_iso import HybridIsotonic

OUT = B.PKG_DIR / "output"
KEY = ["organism_group", "antimicrobial"]


def _percell_vme(dirname: str) -> pd.DataFrame:
    ev = pd.read_csv(OUT / dirname / "method_eval_model_pred.csv")
    return ev.groupby(KEY)["isVME"].sum().rename("VME").reset_index()


def _total_vme(name: str) -> int:
    return int(pd.read_csv(OUT / name / "method_eval_model_pred.csv")["isVME"].sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    u = load_unified(clean=args.clean)
    pfx = "hybc" if args.clean else "hyb"
    tag = "clean(TE 제거)" if args.clean else "full"
    print(f"[데이터셋] {tag} — sample {u['sample_id'].nunique()}")

    def ev(variant, mode, calib):
        name = f"{pfx}_{variant}_{mode}"
        r = Recipe(PerCellModelRouting(), calib, 0.65, name=name).evaluate(u)
        return r["pass"], r["vme_cells"], _total_vme(name)

    res = {}
    # 1) organism/genus in-sample 먼저 → fallback 도출
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
    print(f"[fallback] genus→organism {len(fb)}개 (in-sample genus VME 악화)")

    # 2) hybrid in-sample + 전부 OOF
    res[("hybrid", "insample")] = ev("hybrid", "insample",
                                     HybridIsotonic("insample", fallback_cells=fbset))
    for v, c in [("organism", PerCellIsotonic("oof", cell_by="organism_group")),
                 ("genus", PerCellIsotonic("oof", cell_by="genus")),
                 ("hybrid", HybridIsotonic("oof", fallback_cells=fbset))]:
        res[(v, "oof")] = ev(v, "oof", c)

    print(f"\n  {'variant':9s} {'mode':9s} {'PASS':>5s} {'VME-cell':>9s} {'총VME':>6s}")
    for mode in ("insample", "oof"):
        for v in ("organism", "genus", "hybrid"):
            p, vc, tv = res[(v, mode)]
            print(f"  {v:9s} {mode:9s} {p:>5d} {vc:>9d} {tv:>6d}")
    print("\n[요약]")
    for mode in ("insample", "oof"):
        o_, g_, h_ = res[("organism", mode)], res[("genus", mode)], res[("hybrid", mode)]
        print(f"  {mode:9s}: PASS org {o_[0]} / gen {g_[0]} / hyb {h_[0]} | "
              f"총VME org {o_[2]} / gen {g_[2]} / hyb {h_[2]}")


if __name__ == "__main__":
    main()

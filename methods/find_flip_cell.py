"""raw → isotonic 으로 FAIL→PASS 뒤집히는 cell 탐색 (정직한 OOF 기준 포함).

비교(모두 score=routing, threshold 0.65):
  raw          : ISO 없음 (ng_score 직접 threshold)
  org_oof      : organism-iso, GroupKFold(sample_id) OOF  ← 학습/평가 분리(정직)
  gen_oof      : genus-iso,    GroupKFold(sample_id) OOF
출력: 각 비교에서 FAIL→PASS cell 목록 + 첫 후보(샘플 수 많은) 선정.
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

KEY = ["organism_group", "antimicrobial"]


def _summary(recipe: Recipe, u) -> pd.DataFrame:
    r = recipe.evaluate(u)
    return pd.read_csv(r["summary_csv"]), r["pass"]


def main() -> None:
    u = load_unified()
    routed = PerCellModelRouting().apply(u)
    n_samp = (routed.groupby(KEY)["sample_id"].nunique().rename("n_samples").reset_index())

    sr, p_raw = _summary(Recipe(PerCellModelRouting(), None, 0.65, name="flip_raw"), u)
    so, p_oof_o = _summary(Recipe(PerCellModelRouting(), PerCellIsotonic("oof", cell_by="organism_group"),
                                  0.65, name="flip_org_oof"), u)
    sg, p_oof_g = _summary(Recipe(PerCellModelRouting(), PerCellIsotonic("oof", cell_by="genus"),
                                  0.65, name="flip_gen_oof"), u)
    print(f"PASS  raw {p_raw} | organism-iso OOF {p_oof_o} | genus-iso OOF {p_oof_g}")

    base = sr[KEY + ["FDA_fail_list"]].rename(columns={"FDA_fail_list": "raw"})
    for tag, s in [("org_oof", so), ("gen_oof", sg)]:
        base = base.merge(s[KEY + ["FDA_fail_list"]].rename(columns={"FDA_fail_list": tag}),
                          on=KEY, how="outer")
    base = base.merge(n_samp, on=KEY, how="left")

    for tag in ("org_oof", "gen_oof"):
        flip = base[(base["raw"] != "PASS") & (base[tag] == "PASS")].sort_values(
            "n_samples", ascending=False)
        print(f"\n=== raw FAIL → {tag} PASS : {len(flip)} cells ===")
        print(flip[KEY + ["n_samples", "raw", tag]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()

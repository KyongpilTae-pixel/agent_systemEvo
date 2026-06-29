"""genus vs organism_group 단위 isotonic 보정 비교 — summary 는 organism_group 으로.

배포 환경은 species(organism_group)가 아니라 **genus 만** 알 수 있다. 그래서 isotonic 학습+적용
을 (genus, antimicrobial) 단위로 했을 때, (organism_group, antimicrobial) 단위 대비 FDA PASS/VME
가 달라지는지 확인한다. **평가(summary)는 두 경우 모두 organism_group×antimicrobial** (verify 고정).

score=PerCellModelRouting, threshold=0.65. insample(배포) + oof(held-out) 둘 다.

사용: python -m agent_system.methods.genus_vs_organism_iso
"""
from __future__ import annotations

from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic


def main() -> None:
    u = load_unified()
    routed = PerCellModelRouting().apply(u)

    # 학습 곡선 개수 (cell 단위 차이 확인)
    n_org = len(PerCellIsotonic(cell_by="organism_group").fit_lookup(routed).cells)
    n_gen = len(PerCellIsotonic(cell_by="genus").fit_lookup(routed).cells)
    print(f"[학습 곡선 수] organism_group cell {n_org}  vs  genus cell {n_gen}")
    print(f"  organism_group 고유 {u['organism_group'].nunique()} / genus 고유 {u['genus'].nunique()}\n")

    cases = [
        ("organism", "insample"), ("genus", "insample"),
        ("organism", "oof"),      ("genus", "oof"),
    ]
    print(f"  {'cell_by':9s} {'mode':9s} {'PASS':>5s} {'VME-cell':>9s} {'VME-rate':>9s}")
    res = {}
    for cell_by_short, mode in cases:
        cell_by = "organism_group" if cell_by_short == "organism" else "genus"
        r = Recipe(score=PerCellModelRouting(),
                   calibrate=PerCellIsotonic(mode=mode, cell_by=cell_by),
                   threshold=0.65, name=f"{cell_by_short}_{mode}").evaluate(u)
        res[(cell_by_short, mode)] = r
        print(f"  {cell_by_short:9s} {mode:9s} {r['pass']:>5d} {r['vme_cells']:>9d} "
              f"{r['vme_rate']:>9.4f}")

    print("\n[차이 — summary 는 둘 다 organism_group]")
    for mode in ("insample", "oof"):
        o, g = res[("organism", mode)]["pass"], res[("genus", mode)]["pass"]
        print(f"  {mode:9s}: organism {o}  →  genus {g}   (Δ {g-o:+d})")


if __name__ == "__main__":
    main()

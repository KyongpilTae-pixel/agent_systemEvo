"""genus 단위 per-cell isotonic 학습 + 배포 자산(CSV) 저장.

배포 환경은 species(organism_group)가 아니라 **genus 만** 알 수 있다. 그래서 isotonic 을
(genus, antimicrobial) 단위로 학습해 동결한다. 런타임은 sklearn 없이 np.interp 로 동작.

학습 입력 = routed ng_score(=routed model_pred) ↔ gt_gng, genus 단위.
저장 CSV 규약: PerCellIsotonicLookup 은 organism_group 컬럼으로 그룹하므로, **organism_group
컬럼에 genus 값**을 담아 저장한다(키=(genus, antimicrobial)). 런타임은 반드시
`PerCellIsotonic(mode="lookup", cell_by="genus", lookup_csv=...)` 로 로드(내부에서 행의 genus 를
organism_group 자리에 매핑해 매칭).

산출: agent_system/output/isotonic_lookup_genus.csv
사용:  python -m agent_system.methods.save_genus_iso_lookup
"""
from __future__ import annotations

import argparse

from agent_system.methods import base as B
from agent_system.methods.recipe import load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

DEFAULT_OUT = B.ROOT / "agent_system/output/isotonic_lookup_genus.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min_rows", type=int, default=30)
    ap.add_argument("--min_per_class", type=int, default=3)
    args = ap.parse_args()

    u = load_unified()
    routed = PerCellModelRouting().apply(u)          # ng_score = routed model_pred
    cal = PerCellIsotonic(cell_by="genus")
    iso = cal.fit_lookup(routed, save_path=args.out,
                         min_rows=args.min_rows, min_per_class=args.min_per_class)

    genera = sorted({g for g, _ in iso.cells})
    print(f"[saved] {args.out}")
    print(f"  genus-cell 곡선 {len(iso.cells)}개  (genus {len(genera)}종)")
    print(f"  genus: {genera}")
    print("  주의: 저장 CSV 의 organism_group 컬럼 = genus 값. 런타임은 "
          "PerCellIsotonic(mode='lookup', cell_by='genus') 로 로드.")


if __name__ == "__main__":
    main()

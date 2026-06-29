"""모듈 패키지 교차검증 — 알려진 PASS 수치 재현 확인.

각 레시피를 통합 입력 + 모듈 조립으로 평가해 독립 기준과 대조:
  routed + iso(insample) @0.65   → 103  (운영 routed_iso_t65)
  routed + iso(oof)      @0.65   → 87   (held-out, 2026-06-04 brightness CV baseline)
  rankavg_w0.3 + iso(oof)@0.65   → 83   (2026-06-05 rankavg_gbm_iso_cv)
  brightness + iso(insample)@0.65→ 106  (in-sample +3, 2026-06-02)
"""
from __future__ import annotations

from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.score.rank_avg_ensemble import RankAvgEnsemble
from agent_system.methods.score.brightness_corrector import BrightnessAsymCorrector
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

CASES = [
    ("routed + iso(insample) @0.65", 103,
     PerCellModelRouting(), PerCellIsotonic("insample"), 0.65),
    ("routed + iso(oof) @0.65", 87,
     PerCellModelRouting(), PerCellIsotonic("oof"), 0.65),
    ("rankavg_w0.3 + iso(oof) @0.65", 83,
     RankAvgEnsemble(w_gbm=0.3), PerCellIsotonic("oof"), 0.65),
    ("brightness + iso(insample) @0.65", 106,
     BrightnessAsymCorrector(0.7), PerCellIsotonic("insample"), 0.65),
]


def main() -> None:
    u = load_unified()
    print(f"{'recipe':36s} {'expect':>6s} {'got':>4s} {'vme':>4s} {'ok'}")
    all_ok = True
    for label, expect, score, cal, thr in CASES:
        r = Recipe(score=score, calibrate=cal, threshold=thr,
                   name=label.split()[0]).evaluate(u)
        ok = r["pass"] == expect
        all_ok &= ok
        print(f"{label:36s} {expect:>6d} {r['pass']:>4d} {r['vme_cells']:>4d} "
              f"{'✓' if ok else '✗ MISMATCH'}")
    print("\n결과:", "전부 재현 ✓" if all_ok else "불일치 존재 ✗")


if __name__ == "__main__":
    main()

"""모델 구조 4종(+lora)의 per-conc 예측을 ng_score 로 주입 → 기존 검증 verify 파이프라인으로
FDA PASS/VME 산출, baseline(same_mic) 대비 개선 확인.

- 입력: claudeCode/output/arch_pred_{model}.parquet (sample_id·antimicrobial·conc·pred).
- 평가셋: unified_per_conc_clean (TE 제거 1434 sample). arch_pred 와 키·gt 완전 정합(검증됨).
- 두 recipe: (raw) 보정 없음 threshold 0.65, (iso) PerCellIsotonic OOF + 0.65 — 모든 모델 동일 적용.
- evaluator 는 methods.Recipe/run_verify(검증된 harness) 그대로 재사용 — 신규 PASS 로직 작성 없음.

사용: python -m agent_system.methods.eval_arch_fda_pass [--models baseline complex objarea temporal]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

ARCH = B.ROOT / "claudeCode/output"


def inject(unified: pd.DataFrame, model: str) -> pd.DataFrame:
    """unified 의 ng_score 를 해당 모델 arch_pred 예측으로 교체."""
    pred = pd.read_parquet(ARCH / f"arch_pred_{model}.parquet")[B.KEY + ["pred"]]
    pc = unified.copy()
    pc = pc.merge(pred, on=B.KEY, how="left")
    miss = int(pc["pred"].isna().sum())
    if miss:
        raise SystemExit(f"[{model}] ng_score 매칭 실패 {miss} rows — 키 불일치")
    pc[B.SCORE_COL] = pc["pred"].astype(float)
    return pc.drop(columns=["pred"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+",
                   default=["baseline", "complex", "objarea", "temporal"])
    p.add_argument("--threshold", type=float, default=0.65)
    args = p.parse_args()

    u = load_unified(clean=True)
    print(f"[eval] unified_clean {len(u):,} rows / {u['sample_id'].nunique()} samples\n")

    rows = []
    for m in args.models:
        pc = inject(u, m)
        raw = Recipe(None, None, args.threshold, name=f"archpass_{m}_raw").evaluate(pc)
        iso = Recipe(None, PerCellIsotonic("oof"), args.threshold,
                     name=f"archpass_{m}_iso").evaluate(pc)
        rows.append((m, raw["pass"], raw["vme_cells"], iso["pass"], iso["vme_cells"],
                     iso["n_cells"]))
        print(f"  {m:10s} raw PASS {raw['pass']:>3d} (VME {raw['vme_cells']:>2d}) | "
              f"iso PASS {iso['pass']:>3d} (VME {iso['vme_cells']:>2d})")

    base = next((r for r in rows if r[0] == "baseline"), None)
    print(f"\n========== FDA PASS/FAIL (clean, threshold {args.threshold}, n_cells {rows[0][5]}) ==========")
    print(f"{'model':10s} {'raw PASS':>9s} {'ΔrawP':>6s} {'rawVME':>7s} "
          f"{'iso PASS':>9s} {'ΔisoP':>6s} {'isoVME':>7s}")
    for (m, rp, rv, ip, iv, nc) in rows:
        drp = f"{rp-base[1]:+d}" if base else "—"
        dip = f"{ip-base[3]:+d}" if base else "—"
        print(f"{m:10s} {rp:>9d} {drp:>6s} {rv:>7d} {ip:>9d} {dip:>6s} {iv:>7d}")
    print("\n[해석] iso = PerCellIsotonic OOF(honest) 보정 후. Δ>0 면 baseline(same_mic) 대비 PASS 개선.")

    out = pd.DataFrame(rows, columns=["model", "raw_pass", "raw_vme",
                                      "iso_pass", "iso_vme", "n_cells"])
    out["threshold"] = args.threshold
    csv = ARCH / "arch_fda_pass.csv"
    if csv.exists():        # 누적 갱신(이전 모델 보존, 동일 모델은 교체)
        prev = pd.read_csv(csv)
        out = pd.concat([prev[~prev["model"].isin(out["model"])], out], ignore_index=True)
    out.to_csv(csv, index=False)
    print(f"[saved] {csv}")


if __name__ == "__main__":
    main()

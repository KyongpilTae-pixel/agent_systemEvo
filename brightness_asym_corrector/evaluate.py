#!/usr/bin/env python3
"""검증 — 전체 313 cell 운영 SIR/PASS 로 baseline vs corrector 비교.

  baseline  : routed model_pred → per-cell isotonic → t0.65 → SIR/PASS  (= 운영 routed_iso_t65, PASS 103)
  corrected : Brightness=True 45 cell 에 asym G-방향 brightness 보정 후 동일 파이프라인

검증 결과(2026-06-02): PASS 103 → 106 (+3), VME-cell 43 → 42 (-1).

사용:  python evaluate.py            (전체 파이프라인 실행, ~15-20분)
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                       # noqa: E402
from brightness_corrector import BrightnessAsymCorrector  # noqa: E402
sys.path.insert(0, str(C.ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

OUT_DIR = C.PKG_DIR / "artifacts" / "eval"


def _calibrate(per_conc: pd.DataFrame) -> pd.DataFrame:
    """full-panel per-cell isotonic fit + calibrate (운영 방식과 동일)."""
    iso = PerCellIsotonicLookup.fit(per_conc)
    out = iso.calibrate_dataframe(per_conc, out_col="_cal")
    out["dtw_model_pred"] = out["_cal"]
    return out.drop(columns=["_cal"])


def _run_sir(per_conc_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(C.VERIFY_PIPELINE),
                    "--per_conc_csv", str(per_conc_csv), "--output_dir", str(out_dir),
                    "--native_threshold", str(C.NATIVE_THRESHOLD)],
                   check=True, cwd=str(C.ROOT), stdout=subprocess.DEVNULL)
    summ = pd.read_csv(out_dir / "method_summary_model_pred.csv")
    feas = pd.read_csv(out_dir / "method_sir_feasibility.csv")
    vme_rate = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    return summ, vme_rate


def _pass_vme(summ: pd.DataFrame, tag: str) -> pd.DataFrame:
    s = summ[["organism_group", "antimicrobial", "FDA_fail_list"]].copy()
    s[f"{tag}_pass"] = s["FDA_fail_list"] == "PASS"
    s[f"{tag}_vme"] = s["FDA_fail_list"].astype(str).str.contains("VME")
    return s.rename(columns={"FDA_fail_list": f"{tag}_fail"})


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply_cells", default=str(C.APPLY_CELLS_CSV),
                    help="적용 대상 cell CSV (organism_group, antimicrobial)")
    ap.add_argument("--out_subdir", default="eval", help="artifacts 하위 출력 폴더명")
    args = ap.parse_args()
    out_dir = C.PKG_DIR / "artifacts" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    global OUT_DIR
    OUT_DIR = out_dir
    base = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)

    # baseline (보정 없음)
    base_pc = _calibrate(base.copy())
    base_csv = OUT_DIR / "per_conc_baseline.csv"
    base_pc.to_csv(base_csv, index=False)
    base_summ, base_vr = _run_sir(base_csv, OUT_DIR / "sir_baseline")

    # corrected (asym G-방향, 지정 cell 한정)
    corr = BrightnessAsymCorrector(apply_cells_csv=args.apply_cells)
    corr_raw = corr.apply_to_per_conc(base.copy(), C.FDA_CSV)
    corr_pc = _calibrate(corr_raw.drop(columns=["_brightness_pred", "_corrected"]))
    corr_csv = OUT_DIR / "per_conc_corrected.csv"
    corr_pc.to_csv(corr_csv, index=False)
    corr_summ, corr_vr = _run_sir(corr_csv, OUT_DIR / "sir_corrected")

    m = _pass_vme(base_summ, "baseline").merge(
        _pass_vme(corr_summ, "corrected"),
        on=["organism_group", "antimicrobial"], how="outer")
    m.to_csv(OUT_DIR / "eval_compare.csv", index=False)

    bp, bv = int(m["baseline_pass"].sum()), int(m["baseline_vme"].sum())
    cp, cv = int(m["corrected_pass"].sum()), int(m["corrected_vme"].sum())
    gain = m[(~m["baseline_pass"]) & m["corrected_pass"]]
    loss = m[m["baseline_pass"] & (~m["corrected_pass"])]
    print("\n===== 전체 313 cell — baseline vs brightness asym corrector =====")
    print(f"  baseline   PASS {bp}  VME-cell {bv}  VME-rate {base_vr:.4f}")
    print(f"  corrected  PASS {cp}  VME-cell {cv}  VME-rate {corr_vr:.4f}   "
          f"ΔPASS {cp-bp:+d}(gain {len(gain)}/loss {len(loss)})  ΔVME-cell {cv-bv:+d}")
    print("\n  [PASS gain]")
    print(gain[["organism_group", "antimicrobial", "baseline_fail", "corrected_fail"]]
          .to_string(index=False))
    if len(loss):
        print("  [PASS loss]")
        print(loss[["organism_group", "antimicrobial", "baseline_fail", "corrected_fail"]]
              .to_string(index=False))

    (OUT_DIR / "eval_manifest.json").write_text(json.dumps({
        "baseline": {"pass": bp, "vme_cells": bv, "vme_rate": round(base_vr, 4)},
        "corrected": {"pass": cp, "vme_cells": cv, "vme_rate": round(corr_vr, 4)},
        "delta_pass": cp - bp, "delta_vme_cells": cv - bv,
        "pass_gain": int(len(gain)), "pass_loss": int(len(loss)),
        "weight": C.ASYM_WEIGHT, "threshold": C.NATIVE_THRESHOLD,
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'eval_compare.csv'}")


if __name__ == "__main__":
    main()

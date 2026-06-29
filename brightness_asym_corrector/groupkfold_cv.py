#!/usr/bin/env python3
"""GroupKFold(sample_id) CV — brightness asym corrector 의 +3 PASS held-out 검증.

배경: 운영 검증(evaluate.py)은 per-cell isotonic 을 **전체 FDA 패널에 in-sample fit**
하고, brightness 비중 w=0.7 도 FDA PASS 를 보고 선택했다. 따라서 +3 PASS 가 in-sample
과적합인지 held-out 에서 유지되는지 확인이 필요 (RESUME/daily 2026-06-02 의 '운영 투입 전 필수').

설계:
  유일한 FDA in-sample 요소 = per-cell isotonic (brightness GBM 은 학습 CSV로 out-of-domain
  학습 → FDA 에 in-sample 아님). 그래서 GroupKFold(sample_id) **OOF isotonic** 으로 검증:
    각 fold: train sample 들로 isotonic fit → held-out sample 만 calibrate.
    5 fold 의 held-out 예측을 이어붙여 **전체 패널 OOF 예측** 재구성 (각 row 는 자신을 못 본
    isotonic 으로만 보정됨 = 누설 0) → 운영 SIR/PASS 파이프라인 1회 실행.
  brightness 보정(w)은 isotonic 이전 단계라 OOF 와 무관하게 결정적으로 적용.

산출 (artifacts/cv/):
  1) baseline  in-sample  (full isotonic)            ← 운영 reference (≈103)
  2) corrected in-sample  (full isotonic, w=0.7)     ← 운영 reference (≈106)
  3) baseline  OOF        (GroupKFold isotonic)       ← held-out baseline
  4) corrected OOF        (GroupKFold isotonic, w)    ← held-out corrected  [w-sweep]
  → cv_manifest.json + cv_compare.csv + 콘솔 요약.

사용:
  python groupkfold_cv.py                 # 기본 5-fold, w-sweep {0.5,0.7,1.0}
  python groupkfold_cv.py --folds 5 --weights 0.7
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                       # noqa: E402
from brightness_corrector import BrightnessAsymCorrector  # noqa: E402
sys.path.insert(0, str(C.ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

GROUP_COL = "sample_id"


# ----------------------------- calibration -----------------------------------
def calibrate_insample(per_conc: pd.DataFrame) -> pd.DataFrame:
    """운영 방식: 전체 패널 per-cell isotonic fit + calibrate (evaluate.py 와 동일)."""
    iso = PerCellIsotonicLookup.fit(per_conc)
    out = iso.calibrate_dataframe(per_conc, out_col="_cal")
    out["dtw_model_pred"] = out["_cal"]
    return out.drop(columns=["_cal"])


def calibrate_oof(per_conc: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """GroupKFold(sample_id) OOF isotonic: held-out row 는 자신을 못 본 isotonic 으로만 보정.

    isotonic 이 skip 한 cell(train fold 에서 min_rows/min_per_class 미달) → identity(raw),
    이는 운영 fallback 과 동일.
    """
    df = per_conc.reset_index(drop=True).copy()
    oof = np.full(len(df), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in gkf.split(df, groups=df[GROUP_COL].values):
        iso = PerCellIsotonicLookup.fit(df.iloc[tr_idx])
        te_cal = iso.calibrate_dataframe(df.iloc[te_idx], out_col="_cal")
        oof[te_idx] = te_cal["_cal"].to_numpy()
    assert np.isfinite(oof).all(), "OOF 미할당 row 존재"
    out = df.copy()
    out["dtw_model_pred"] = oof
    return out


# ----------------------------- SIR runner ------------------------------------
def run_sir(per_conc: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "per_conc.csv"
    per_conc.to_csv(csv, index=False)
    subprocess.run([sys.executable, str(C.VERIFY_PIPELINE),
                    "--per_conc_csv", str(csv), "--output_dir", str(out_dir),
                    "--native_threshold", str(C.NATIVE_THRESHOLD)],
                   check=True, cwd=str(C.ROOT), stdout=subprocess.DEVNULL)
    summ = pd.read_csv(out_dir / "method_summary_model_pred.csv")
    feas = pd.read_csv(out_dir / "method_sir_feasibility.csv")
    vme_rate = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    return summ, vme_rate


def pass_vme(summ: pd.DataFrame, tag: str) -> pd.DataFrame:
    s = summ[["organism_group", "antimicrobial", "FDA_fail_list"]].copy()
    s[f"{tag}_pass"] = s["FDA_fail_list"] == "PASS"
    s[f"{tag}_vme"] = s["FDA_fail_list"].astype(str).str.contains("VME")
    return s.rename(columns={"FDA_fail_list": f"{tag}_fail"})


def counts(summ: pd.DataFrame) -> tuple[int, int]:
    pas = int((summ["FDA_fail_list"] == "PASS").sum())
    vme = int(summ["FDA_fail_list"].astype(str).str.contains("VME").sum())
    return pas, vme


# ----------------------------- main ------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.7, 1.0],
                    help="corrected OOF 에서 sweep 할 brightness 비중 w")
    ap.add_argument("--out_subdir", default="cv")
    args = ap.parse_args()

    out_dir = C.PKG_DIR / "artifacts" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)
    print(f"[load] routed per_conc rows={len(base)} samples={base[GROUP_COL].nunique()} "
          f"cells={base.groupby(['organism_group','antimicrobial']).ngroups}")

    results = {}      # tag -> {"pass":, "vme":, "vme_rate":}
    summ_store = {}   # tag -> summ df

    # brightness 보정값 (w 별) 미리 계산 — GBM 추론 1회 후 재사용
    corr_cache = {}
    for w in args.weights:
        corr = BrightnessAsymCorrector(weight=w)
        c = corr.apply_to_per_conc(base.copy(), C.FDA_CSV)
        corr_cache[w] = c.drop(columns=["_brightness_pred", "_corrected"])

    # ---- 1) baseline in-sample (운영 reference) ----
    print("\n[1/?] baseline in-sample (full isotonic)")
    s, vr = run_sir(calibrate_insample(base.copy()), out_dir / "baseline_insample")
    p, v = counts(s); results["baseline_insample"] = dict(pass_=p, vme=v, vme_rate=vr)
    summ_store["baseline_insample"] = s
    print(f"      PASS {p}  VME-cell {v}  VME-rate {vr:.4f}")

    # ---- 2) corrected in-sample (운영 reference, w=0.7 우선) ----
    w_ref = 0.7 if 0.7 in args.weights else args.weights[0]
    print(f"[2/?] corrected in-sample (full isotonic, w={w_ref})")
    s, vr = run_sir(calibrate_insample(corr_cache[w_ref].copy()),
                    out_dir / "corrected_insample")
    p, v = counts(s); results[f"corrected_insample_w{w_ref}"] = dict(pass_=p, vme=v, vme_rate=vr)
    summ_store["corrected_insample"] = s
    print(f"      PASS {p}  VME-cell {v}  VME-rate {vr:.4f}")

    # ---- 3) baseline OOF (held-out) ----
    print(f"[3/?] baseline OOF (GroupKFold {args.folds})")
    base_oof_summ, vr = run_sir(calibrate_oof(base.copy(), args.folds),
                                out_dir / "baseline_oof")
    p, v = counts(base_oof_summ); results["baseline_oof"] = dict(pass_=p, vme=v, vme_rate=vr)
    summ_store["baseline_oof"] = base_oof_summ
    print(f"      PASS {p}  VME-cell {v}  VME-rate {vr:.4f}")

    # ---- 4) corrected OOF (held-out) — w sweep ----
    for i, w in enumerate(args.weights):
        print(f"[4.{i+1}] corrected OOF (GroupKFold {args.folds}, w={w})")
        s, vr = run_sir(calibrate_oof(corr_cache[w].copy(), args.folds),
                        out_dir / f"corrected_oof_w{w}")
        p, v = counts(s); results[f"corrected_oof_w{w}"] = dict(pass_=p, vme=v, vme_rate=vr)
        summ_store[f"corrected_oof_w{w}"] = s
        print(f"      PASS {p}  VME-cell {v}  VME-rate {vr:.4f}")

    # ---- cell-level diff: baseline_oof vs corrected_oof(w_ref) ----
    cmp = pass_vme(base_oof_summ, "base_oof").merge(
        pass_vme(summ_store[f"corrected_oof_w{w_ref}"], "corr_oof"),
        on=["organism_group", "antimicrobial"], how="outer")
    cmp.to_csv(out_dir / "cv_compare.csv", index=False)
    gain = cmp[(~cmp["base_oof_pass"]) & cmp["corr_oof_pass"]]
    loss = cmp[cmp["base_oof_pass"] & (~cmp["corr_oof_pass"])]

    # ---- summary ----
    bo = results["baseline_oof"]
    co = results[f"corrected_oof_w{w_ref}"]
    bi = results["baseline_insample"]
    ci = results[f"corrected_insample_w{w_ref}"]
    print("\n" + "=" * 70)
    print("GroupKFold(sample_id) CV 요약 — brightness asym corrector")
    print("=" * 70)
    print(f"  in-sample  : baseline {bi['pass_']}  → corrected {ci['pass_']} "
          f"(Δ{ci['pass_']-bi['pass_']:+d})   [운영 reference]")
    print(f"  OOF (held) : baseline {bo['pass_']}  → corrected {co['pass_']} "
          f"(Δ{co['pass_']-bo['pass_']:+d})   [held-out, w={w_ref}]")
    print(f"  OOF gain cells {len(gain)} / loss cells {len(loss)} ;  "
          f"VME-cell {bo['vme']}→{co['vme']}")
    print("\n  [OOF PASS gain]")
    print(gain[["organism_group", "antimicrobial",
                "base_oof_fail", "corr_oof_fail"]].to_string(index=False) or "  (none)")
    if len(loss):
        print("  [OOF PASS loss]")
        print(loss[["organism_group", "antimicrobial",
                    "base_oof_fail", "corr_oof_fail"]].to_string(index=False))
    print("\n  [w-sensitivity, OOF]")
    for w in args.weights:
        r = results[f"corrected_oof_w{w}"]
        print(f"    w={w}:  PASS {r['pass_']}  (Δ vs baseline_oof {r['pass_']-bo['pass_']:+d})  "
              f"VME-cell {r['vme']}  VME-rate {r['vme_rate']:.4f}")

    manifest = {
        "design": "GroupKFold(sample_id) OOF isotonic; brightness GBM is out-of-domain "
                  "(train CSV), so OOF targets the only FDA in-sample step (per-cell isotonic).",
        "folds": args.folds,
        "w_reference": w_ref,
        "results": {k: {"pass": v["pass_"], "vme_cells": v["vme"],
                        "vme_rate": round(v["vme_rate"], 4)} for k, v in results.items()},
        "insample_delta_pass": ci["pass_"] - bi["pass_"],
        "oof_delta_pass": co["pass_"] - bo["pass_"],
        "oof_gain_cells": int(len(gain)),
        "oof_loss_cells": int(len(loss)),
        "oof_gain_list": gain[["organism_group", "antimicrobial"]].to_dict("records"),
        "oof_loss_list": loss[["organism_group", "antimicrobial"]].to_dict("records"),
    }
    (out_dir / "cv_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out_dir/'cv_manifest.json'}")
    print(f"[saved] {out_dir/'cv_compare.csv'}")


if __name__ == "__main__":
    main()

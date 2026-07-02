#!/usr/bin/env python3
"""brightness GBM in-sample 학습(OOF 아님) → 보정 → 평가. brightness 사용 cell(45) 대상.

요청(2026-07-02):
  - OOF 하지 말고 GBM 을 FDA2023 로 in-sample 학습하고 같은 셋에서 평가(과적합 포함, 상한 확인).
  - 우선 brightness 사용 cell '전체(combined)' 로 GBM 1개 학습 → 판정. (per-cell 은 --mode percell)
  - PASS 변화가 없어도 EA/CA 상승 여부도 확인.

파이프라인:
  1) FDA features(brightness trajectory) + gt_gng(ALIGN_OOF), brightness=True 45 cell 로 한정
  2) --mode combined: 45 cell rows 전체로 GBM 1개 in-sample 학습 → 예측
     --mode percell : cell 마다 그 cell rows 로 GBM in-sample 학습 → 예측
  3) AsymCorrector.correct(routed per_conc, sig_pred, w=0.7) — G-방향 비대칭 보정
  4) per-cell isotonic(full) → verify SIR → PASS/VME + EAp/CAp
  5) baseline vs corrected: PASS gain/loss + EA/CA 상승 cell(PASS 무변화 포함)

산출: artifacts/eval_insample_<mode>/  (eval_compare.csv, eval_manifest.json, percell_auroc.csv)
사용: python train_and_eval_brightcells_insample.py [--mode combined|percell]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                                # noqa: E402
from brightness_features import build_features, feature_columns   # noqa: E402
from agent_system.trajectory_gbm.corrector import AsymCorrector, KEY  # noqa: E402
from agent_system.trajectory_gbm.config import get_signal         # noqa: E402
from evaluate import _calibrate, _run_sir                         # noqa: E402

# combined/percell 은 rows 가 적어 전역 param(min_data_in_leaf=50)은 과소적합 → 완화
INSAMPLE_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=15,
                       min_data_in_leaf=10, feature_fraction=0.9, bagging_fraction=0.9,
                       bagging_freq=1, verbose=-1)
ROUNDS = 300


def _labeled_fda():
    fda = build_features(C.FDA_CSV, C.FDA_BRIGHTNESS_COL)
    fcols = feature_columns(fda)
    align = pd.read_csv(C.ALIGN_OOF)
    feat_only = [c for c in fcols if c not in KEY]        # concentration_idx_0 은 KEY 로만(중복 방지)
    lab = align[KEY + ["gt_gng", "organism_group"]].merge(
        fda[KEY + feat_only], on=KEY, how="inner").dropna(subset=feat_only + ["gt_gng"])
    return lab, fcols                                     # 학습엔 full fcols(concentration_idx_0 포함) 사용


def sig_pred_combined(lab, fcols, apply_cells):
    sub = lab[[(str(o).strip(), str(a).strip()) in apply_cells
               for o, a in zip(lab.organism_group, lab.antimicrobial)]].copy()
    y = sub["gt_gng"].astype(int).values
    b = lgb.train(INSAMPLE_PARAMS, lgb.Dataset(sub[fcols].values, label=y, feature_name=fcols),
                  num_boost_round=ROUNDS)
    p = b.predict(sub[fcols].values)
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    print(f"[combined] brightness cell rows {len(sub):,}  in-sample AUROC {auc:.4f}")
    out = sub[KEY].copy(); out["sig_pred"] = p
    return out, pd.DataFrame([("__combined__", "", len(sub), auc)],
                             columns=["organism_group", "antimicrobial", "n", "insample_auroc"])


def sig_pred_percell(lab, fcols, apply_cells):
    rows, aucs = [], []
    for (org, drug), g in lab.groupby(["organism_group", "antimicrobial"]):
        if (str(org).strip(), str(drug).strip()) not in apply_cells:
            continue
        y = g["gt_gng"].astype(int).values
        if y.sum() == 0 or y.sum() == len(y) or len(g) < 30:
            continue
        b = lgb.train(INSAMPLE_PARAMS, lgb.Dataset(g[fcols].values, label=y, feature_name=fcols),
                      num_boost_round=ROUNDS)
        p = b.predict(g[fcols].values)
        aucs.append((org, drug, len(g), roc_auc_score(y, p)))
        s = g[KEY].copy(); s["sig_pred"] = p; rows.append(s)
    sig = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEY + ["sig_pred"])
    adf = pd.DataFrame(aucs, columns=["organism_group", "antimicrobial", "n", "insample_auroc"])
    print(f"[percell] 학습 cell {len(adf)}  평균 in-sample AUROC {adf.insample_auroc.mean():.4f}  rows {len(sig):,}")
    return sig, adf


def _metrics(summ, tag):
    s = summ[["organism_group", "antimicrobial", "EAp", "CAp", "VMEp", "MEp", "FDA_fail_list"]].copy()
    return s.rename(columns={c: f"{tag}_{c}" for c in ["EAp", "CAp", "VMEp", "MEp", "FDA_fail_list"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["combined", "percell"], default="combined")
    args = ap.parse_args()
    os.environ["PYTHONPATH"] = str(C.ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")  # SIR 서브프로세스 import
    out_dir = C.PKG_DIR / "artifacts" / f"eval_insample_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ac = pd.read_csv(C.APPLY_CELLS_CSV)
    apply_cells = set(zip(ac["organism_group"].astype(str).str.strip(),
                          ac["antimicrobial"].astype(str).str.strip()))
    lab, fcols = _labeled_fda()
    sig, adf = (sig_pred_combined if args.mode == "combined" else sig_pred_percell)(lab, fcols, apply_cells)
    adf.to_csv(out_dir / "percell_auroc.csv", index=False)

    base = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)
    base_pc = _calibrate(base.copy())
    base_csv = out_dir / "per_conc_baseline.csv"; base_pc.to_csv(base_csv, index=False)
    base_summ, base_vr = _run_sir(base_csv, out_dir / "sir_baseline")

    corr = AsymCorrector(get_signal("brightness"), weight=C.ASYM_WEIGHT,
                         apply_cells_csv=str(C.APPLY_CELLS_CSV))
    corr_raw = corr.correct(base.copy(), sig, model_pred_col="dtw_model_pred")
    corr_pc = _calibrate(corr_raw.drop(columns=["_sig_pred", "_corrected"]))
    corr_csv = out_dir / "per_conc_corrected.csv"; corr_pc.to_csv(corr_csv, index=False)
    corr_summ, corr_vr = _run_sir(corr_csv, out_dir / "sir_corrected")

    m = _metrics(base_summ, "b").merge(_metrics(corr_summ, "c"),
                                       on=["organism_group", "antimicrobial"], how="outer")
    m["is_bright"] = [(str(o).strip(), str(a).strip()) in apply_cells
                      for o, a in zip(m.organism_group, m.antimicrobial)]
    m["b_pass"] = m["b_FDA_fail_list"] == "PASS"
    m["c_pass"] = m["c_FDA_fail_list"] == "PASS"
    m["dEAp"] = (m["c_EAp"] - m["b_EAp"]).round(4)
    m["dCAp"] = (m["c_CAp"] - m["b_CAp"]).round(4)
    m["dVMEp"] = (m["c_VMEp"] - m["b_VMEp"]).round(4)
    m.to_csv(out_dir / "eval_compare.csv", index=False)

    bp, cp = int(m.b_pass.sum()), int(m.c_pass.sum())
    bm = m[m.is_bright].copy()
    gain = bm[(~bm.b_pass) & bm.c_pass]; loss = bm[bm.b_pass & (~bm.c_pass)]
    ea_up = bm[bm.dEAp > 1e-9]; ca_up = bm[bm.dCAp > 1e-9]
    ea_dn = bm[bm.dEAp < -1e-9]; ca_dn = bm[bm.dCAp < -1e-9]
    print(f"\n===== brightness in-sample GBM ({args.mode}) — 전체 313 & brightness cell({len(bm)}) =====")
    print(f"  전체 PASS: baseline {bp} → corrected {cp} ({cp-bp:+d})")
    print(f"  [brightness cell] PASS gain {len(gain)} / loss {len(loss)}")
    print(f"  [brightness cell] EA↑ {len(ea_up)} (EA↓ {len(ea_dn)}) · CA↑ {len(ca_up)} (CA↓ {len(ca_dn)})")
    print(f"  [brightness cell] 평균 EAp {bm.b_EAp.mean():.4f}→{bm.c_EAp.mean():.4f} ({bm.dEAp.mean():+.4f}) · "
          f"CAp {bm.b_CAp.mean():.4f}→{bm.c_CAp.mean():.4f} ({bm.dCAp.mean():+.4f}) · "
          f"VMEp {bm.b_VMEp.mean():.4f}→{bm.c_VMEp.mean():.4f}")
    show = bm[(bm.dEAp.abs() > 1e-9) | (bm.dCAp.abs() > 1e-9) | (bm.b_pass != bm.c_pass)]
    print("\n  [변화 cell 상세 (EA/CA/VME/PASS)]")
    if len(show):
        print(show[["organism_group", "antimicrobial", "b_EAp", "c_EAp", "dEAp",
                    "b_CAp", "c_CAp", "dCAp", "b_VMEp", "c_VMEp", "b_pass", "c_pass"]]
              .round(3).to_string(index=False))
    else:
        print("   (변화 없음)")
    (out_dir / "eval_manifest.json").write_text(json.dumps({
        "mode": args.mode, "params": INSAMPLE_PARAMS, "rounds": ROUNDS,
        "all_pass": {"baseline": bp, "corrected": cp, "delta": cp - bp},
        "bright_cells": int(len(bm)),
        "bright_pass_gain": int(len(gain)), "bright_pass_loss": int(len(loss)),
        "bright_EA_up": int(len(ea_up)), "bright_EA_down": int(len(ea_dn)),
        "bright_CA_up": int(len(ca_up)), "bright_CA_down": int(len(ca_dn)),
        "bright_mean_dEAp": round(float(bm.dEAp.mean()), 4),
        "bright_mean_dCAp": round(float(bm.dCAp.mean()), 4),
        "bright_mean_dVMEp": round(float((bm.c_VMEp - bm.b_VMEp).mean()), 4),
        "mean_insample_auroc": round(float(adf.insample_auroc.mean()), 4),
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()

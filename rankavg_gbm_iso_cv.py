#!/usr/bin/env python3
"""ens_rankavg(GBM 포함) 의 운영 경로 재검토 — per-cell ISO refit 후 held-out PASS 비교.

배경(2026-06-05): `ens_rankavg` = 0.5·rank(model_pred)+0.5·rank(objarea_gbm) 는 AUROC 축
에서 최고(0.9619)지만, Multi-Agent Cycle 2(2026-05-27)에서 운영 PASS 기여 0 으로 측정됐다
(단, ① EA bug fix 이전, ② shifted cell 한정, ③ threshold 0.5, ④ held-out 미검증).
사용자 요청 (b): rank_avg 점수를 rank-scale per-cell isotonic 으로 재calibrate 한 뒤,
EA-fix 운영 파이프라인 + threshold 0.65 로 routed_iso_t65(PASS 103) 와 정면 비교.

방법론(brightness GroupKFold CV 와 동일):
  - 후보 score 를 routed per_conc 의 dtw_model_pred 컬럼에 주입 (isotonic 이전 단계).
  - in-sample : 전체 패널 per-cell isotonic fit+calibrate (운영 배포 방식, 낙관적).
  - OOF (held): GroupKFold(sample_id) per-cell isotonic — 자기 자신을 못 본 isotonic 만 적용
                (누설 0). ★ 정직한 일반화 판정 기준.
  - 각 calibrated per_conc → verify_method_sir_pipeline (EA fix, native_threshold=0.65)
    → method_summary_model_pred.csv 의 FDA_fail_list 로 PASS/VME 집계.

후보:
  routed       : 운영 routed model_pred (control = routed_iso_t65). in 103 / OOF 87 재현 기대.
  rankavg_w*   : (1-w)·rank(routed) + w·rank(objarea_gbm)  [GBM 비중 w sweep]
  gbm_only     : rank(objarea_gbm) 단독 (참고)

사용:
  python -m agent_system.rankavg_gbm_iso_cv               # 기본 w sweep {0.3,0.5}
  python -m agent_system.rankavg_gbm_iso_cv --weights 0.3 0.5 0.7 --folds 5
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

PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR / "brightness_asym_corrector"))
import config as C  # noqa: E402  (ROUTED_PER_CONC, NATIVE_THRESHOLD, VERIFY_PIPELINE, ROOT)
sys.path.insert(0, str(C.ROOT))
from claudeCode.isotonic_lookup import PerCellIsotonicLookup  # noqa: E402

GROUP_COL = "sample_id"
GBM_OOF = C.ROOT / "claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv"
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]


# --------------------------- candidate scores --------------------------------
def load_base() -> pd.DataFrame:
    """routed per_conc + objarea GBM OOF 점수 병합. dtw_model_pred = routed model_pred."""
    base = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)
    gbm = pd.read_csv(GBM_OOF, low_memory=False,
                      usecols=KEY + ["objarea_crossdomain_pred"])
    base = base.merge(gbm, on=KEY, how="left")
    return base


def make_score(base: pd.DataFrame, name: str, w: float = 0.5) -> pd.DataFrame:
    """후보 score 를 dtw_model_pred 에 주입한 per_conc 반환 (isotonic 이전).

    rank 는 build_scores(compare_methods_per_cell) 와 동일하게 global pct-rank.
    gbm 결측 행은 rank(routed) 로 폴백(같은 rank 공간 유지).
    """
    df = base.copy()
    if name == "routed":
        return df  # control: dtw_model_pred 그대로
    rr = df["dtw_model_pred"].rank(pct=True)
    if name == "gbm_only":
        rg = df["objarea_crossdomain_pred"].rank(pct=True)
        df["dtw_model_pred"] = rg.fillna(rr)
        return df
    if name == "rankavg":
        rg = df["objarea_crossdomain_pred"].rank(pct=True)
        ens = (1.0 - w) * rr + w * rg
        df["dtw_model_pred"] = ens.where(rg.notna(), rr)  # gbm 결측 → routed rank
        return df
    raise ValueError(name)


# --------------------------- calibration (brightness CV 와 동일) ---------------
def calibrate_insample(per_conc: pd.DataFrame) -> pd.DataFrame:
    iso = PerCellIsotonicLookup.fit(per_conc)
    out = iso.calibrate_dataframe(per_conc, out_col="_cal")
    out["dtw_model_pred"] = out["_cal"]
    return out.drop(columns=["_cal"])


def calibrate_oof(per_conc: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    df = per_conc.reset_index(drop=True).copy()
    oof = np.full(len(df), np.nan, dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=df[GROUP_COL].values):
        iso = PerCellIsotonicLookup.fit(df.iloc[tr])
        oof[te] = iso.calibrate_dataframe(df.iloc[te], out_col="_cal")["_cal"].to_numpy()
    assert np.isfinite(oof).all(), "OOF 미할당 row"
    out = df.copy()
    out["dtw_model_pred"] = oof
    return out


# --------------------------- SIR runner --------------------------------------
def run_sir(per_conc: pd.DataFrame, out_dir: Path, threshold: float) -> tuple[pd.DataFrame, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "per_conc.csv"
    # verify 내부에서 objarea_crossdomain_pred 를 다시 merge 하므로 헬퍼 컬럼 제거(컬럼 충돌 방지)
    per_conc.drop(columns=["objarea_crossdomain_pred"], errors="ignore").to_csv(csv, index=False)
    subprocess.run([sys.executable, str(C.VERIFY_PIPELINE),
                    "--per_conc_csv", str(csv), "--output_dir", str(out_dir),
                    "--native_threshold", str(threshold)],
                   check=True, cwd=str(C.ROOT), stdout=subprocess.DEVNULL)
    summ = pd.read_csv(out_dir / "method_summary_model_pred.csv")
    feas = pd.read_csv(out_dir / "method_sir_feasibility.csv")
    vme_rate = float(feas.loc[feas["method"] == "model_pred", "vme_rate"].iloc[0])
    return summ, vme_rate


def counts(summ: pd.DataFrame) -> tuple[int, int]:
    pas = int((summ["FDA_fail_list"] == "PASS").sum())
    vme = int(summ["FDA_fail_list"].astype(str).str.contains("VME").sum())
    return pas, vme


def pass_flags(summ: pd.DataFrame, tag: str) -> pd.DataFrame:
    s = summ[["organism_group", "antimicrobial", "FDA_fail_list"]].copy()
    s[f"{tag}_pass"] = s["FDA_fail_list"] == "PASS"
    return s.rename(columns={"FDA_fail_list": f"{tag}_fail"})


# --------------------------- main --------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.3, 0.5],
                    help="GBM 비중 w: score=(1-w)·rank(routed)+w·rank(gbm)")
    ap.add_argument("--threshold", type=float, default=C.NATIVE_THRESHOLD)
    ap.add_argument("--out_subdir", default="rankavg_gbm_iso", type=str)
    args = ap.parse_args()

    out_root = PKG_DIR / "output" / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    base = load_base()
    print(f"[load] routed per_conc rows={len(base)} samples={base[GROUP_COL].nunique()} "
          f"cells={base.groupby(['organism_group','antimicrobial']).ngroups} "
          f"gbm_matched={int(base['objarea_crossdomain_pred'].notna().sum())}")

    # 후보 정의: (tag, score_name, w)
    cands = [("routed", "routed", None)]
    for w in args.weights:
        cands.append((f"rankavg_w{w}", "rankavg", w))
    cands.append(("gbm_only", "gbm_only", None))

    results = {}     # tag -> {"in":(p,v,vr), "oof":(p,v,vr)}
    summ_store = {}  # (tag, mode) -> summ
    for tag, name, w in cands:
        sc = make_score(base, name, w if w is not None else 0.5)
        # in-sample
        s_in, vr_in = run_sir(calibrate_insample(sc.copy()),
                              out_root / f"{tag}_insample", args.threshold)
        p_in, v_in = counts(s_in)
        # OOF
        s_oof, vr_oof = run_sir(calibrate_oof(sc.copy(), args.folds),
                                out_root / f"{tag}_oof", args.threshold)
        p_oof, v_oof = counts(s_oof)
        results[tag] = dict(in_pass=p_in, in_vme=v_in, in_vme_rate=vr_in,
                            oof_pass=p_oof, oof_vme=v_oof, oof_vme_rate=vr_oof)
        summ_store[(tag, "oof")] = s_oof
        print(f"  [{tag:14s}] in-sample PASS {p_in:3d} (VME {v_in}, {vr_in:.4f})  |  "
              f"OOF PASS {p_oof:3d} (VME {v_oof}, {vr_oof:.4f})")

    # ---- control 대비 cell-level gain/loss (OOF) ----
    ctrl = summ_store[("routed", "oof")]
    diffs = {}
    for tag, _, _ in cands:
        if tag == "routed":
            continue
        cmp = pass_flags(ctrl, "routed").merge(
            pass_flags(summ_store[(tag, "oof")], tag),
            on=["organism_group", "antimicrobial"], how="outer")
        gain = cmp[(~cmp["routed_pass"]) & cmp[f"{tag}_pass"]]
        loss = cmp[cmp["routed_pass"] & (~cmp[f"{tag}_pass"])]
        cmp.to_csv(out_root / f"cv_compare_{tag}.csv", index=False)
        diffs[tag] = dict(gain=len(gain), loss=len(loss),
                          gain_list=gain[["organism_group", "antimicrobial"]].to_dict("records"),
                          loss_list=loss[["organism_group", "antimicrobial"]].to_dict("records"))

    # ---- 요약 출력 ----
    r0 = results["routed"]
    print("\n" + "=" * 74)
    print(f"ens_rankavg(GBM) + per-cell ISO — held-out PASS 비교  (threshold={args.threshold})")
    print("=" * 74)
    print(f"  {'후보':16s} {'in-sample':>10s} {'OOF(held)':>10s} {'ΔOOF vs routed':>16s} "
          f"{'OOF VME-rate':>12s}")
    for tag, _, _ in cands:
        r = results[tag]
        d = r["oof_pass"] - r0["oof_pass"]
        dstr = "—" if tag == "routed" else f"{d:+d}"
        print(f"  {tag:16s} {r['in_pass']:10d} {r['oof_pass']:10d} {dstr:>16s} "
              f"{r['oof_vme_rate']:12.4f}")
    for tag in diffs:
        print(f"\n  [{tag}] OOF gain cells {diffs[tag]['gain']} / loss cells {diffs[tag]['loss']}")
        if diffs[tag]["gain_list"]:
            print("    gain:", diffs[tag]["gain_list"])
        if diffs[tag]["loss_list"]:
            print("    loss:", diffs[tag]["loss_list"])

    manifest = {
        "design": "candidate score → per-cell isotonic (in-sample full-fit & GroupKFold OOF) "
                  "→ EA-fix verify pipeline → FDA PASS. Honest verdict = OOF.",
        "threshold": args.threshold, "folds": args.folds, "weights": args.weights,
        "results": results,
        "oof_diff_vs_routed": {t: {"gain": d["gain"], "loss": d["loss"]} for t, d in diffs.items()},
        "oof_gain_loss_detail": diffs,
        "verdict_note": "routed = control (운영 routed_iso_t65). OOF PASS 가 control 을 "
                        "유의미하게 넘으면 GBM rank-avg 운영 재투입 후보.",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out_root/'manifest.json'}")


if __name__ == "__main__":
    main()

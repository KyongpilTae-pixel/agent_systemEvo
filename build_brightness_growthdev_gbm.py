#!/usr/bin/env python3
"""성장편차(growth_dev) 기반 brightness GBM — 사용자 도메인 지식 직접 반영.

핵심 판별 feature (알려진 도메인 규칙):
  drug_growth = brightness_drug[last] - brightness_drug[1]
  ctl_growth  = brightness_ctl [last] - brightness_ctl [1]
  growth_dev  = ctl_growth - drug_growth     # control 대비 약제가 덜 자란 정도 → 클수록 NG(억제)
* index 0 은 noise 가 많아 사용하지 않음 (성장은 index 1 부터 측정).

보조 feature 도 모두 t1.. 기반(t0 제외): 마지막 index 편차, t1.. per-timepoint diff/ratio.
Train: 2024_0709_traintest_allInfo Train split / Eval: train-Test + FDA cross-domain.
출력: agent_system/output/brightness_gd_gbm/brightness_gd_crossdomain_oof.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/kptae/project/qnt_algorithm")
sys.path.insert(0, str(ROOT))
from claudeCode.add_object_area_dtw import parse_csv_list  # noqa: E402
from claudeCode.objarea_raw_gbm import MODEL_TIME_LEN, CTL_PICK_INDEX  # noqa: E402

TRAIN_CSV = "/data/dRAST30_prepare_csv/2024_0709_traintest_allInfo.csv"
FDA_CSV = ("/home/kptae/data/allinfo/fda2023_preprocessedWithFeature_validControlInfo_"
           "useMax_TrueWith_object_area_useMax_TrueWith_object_count_useMax_TrueWith_"
           "brightness_useMax_False.csv")
OBJAREA_OOF = ROOT / "claudeCode/output_dtw_aggregate_full/objarea_crossdomain_oof.csv"
OUT_DIR = ROOT / "agent_system/output/brightness_gd_gbm"
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]
EPS = 1e-6
T1 = 1  # 성장 측정 시작 index (index 0 = noise, 제외)


def _gd_features(sid, drug, rank, traj, ctl) -> dict:
    """index 0 제외, t1..last 기반 control-거리/성장편차 feature."""
    last = MODEL_TIME_LEN - 1                      # = 6
    drug_growth = float(traj[last] - traj[T1])     # 약제 t1→last 성장
    ctl_growth = float(ctl[last] - ctl[T1])        # control t1→last 성장
    feat = {
        "sample_id": sid, "antimicrobial": drug, "concentration_idx_0": rank,
        # ---- 핵심 ----
        "growth_dev": ctl_growth - drug_growth,                       # control-약제 성장 차이
        "growth_dev_norm": (ctl_growth - drug_growth) / (abs(ctl_growth) + EPS),
        "drug_growth_1L": drug_growth,
        "ctl_growth_1L": ctl_growth,
        # ---- 마지막 index 편차 (drug vs ctl) ----
        "dev_last": float(traj[last] - ctl[last]),
        "dev_last_norm": float((traj[last] - ctl[last]) / (ctl[last] + EPS)),
        # ---- 약제 후반 성장 (t3→last) ----
        "drug_late_1L": float(traj[last] - traj[3]),
    }
    # t1..last per-timepoint control 거리/비율 (t0 제외)
    for t in range(T1, MODEL_TIME_LEN):
        feat[f"diff_t{t}"] = float(traj[t] - ctl[t])
        feat[f"ratio_t{t}"] = float(traj[t] / (ctl[t] + EPS))
    return feat


def build_features_gd(csv_path: str, bright_col: str, label_col: str | None,
                      extra_cols: tuple = ()) -> pd.DataFrame:
    use = {"sample_id", "antimicrobial", "concentration", bright_col}
    if label_col:
        use.add(label_col)
    use |= set(extra_cols)
    df = pd.read_csv(csv_path, usecols=lambda c: c in use, low_memory=False)
    df["br"] = df[bright_col].apply(parse_csv_list)
    df = df.dropna(subset=["br"]).reset_index(drop=True)

    ctl_map = {}
    for sid, sub in df[df["antimicrobial"] == "Cont"].groupby("sample_id", sort=False):
        max_len = int(sub["br"].apply(len).max())
        if max_len < MODEL_TIME_LEN:
            continue
        padded = np.full((len(sub), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(sub["br"].values):
            padded[i, :len(arr)] = arr
        pick = padded[:, CTL_PICK_INDEX]
        if np.all(~np.isfinite(pick)):
            continue
        best = int(np.nanargmax(pick))
        ct = padded[best, :MODEL_TIME_LEN]
        if np.all(np.isfinite(ct)):
            ctl_map[sid] = ct.astype(np.float64)

    rows = []
    for (sid, drug), sub in df[df["antimicrobial"] != "Cont"].groupby(
            ["sample_id", "antimicrobial"], sort=False):
        if sid not in ctl_map:
            continue
        ctl = ctl_map[sid]
        for rank, srow in sub.sort_values("concentration").reset_index(drop=True).iterrows():
            traj = np.asarray(srow["br"], dtype=np.float64)
            if len(traj) < MODEL_TIME_LEN:
                continue
            traj = traj[:MODEL_TIME_LEN]
            if not np.all(np.isfinite(traj)):
                continue
            f = _gd_features(sid, drug, rank, traj, ctl)
            if label_col:
                f[label_col] = srow[label_col]
            for c in extra_cols:
                f[c] = srow[c]
            rows.append(f)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] TRAIN: {TRAIN_CSV}")
    tr_feat = build_features_gd(TRAIN_CSV, "brightness_list", "bmd_gng", ("train_or_test",))
    tr_feat["y"] = (tr_feat["bmd_gng"].astype(str) == "NG").astype(int)
    fcols = [c for c in tr_feat.columns
             if c not in ("sample_id", "antimicrobial", "bmd_gng", "train_or_test", "y")]
    print(f"[features] {len(fcols)}: {fcols}")
    tr = tr_feat[tr_feat["train_or_test"] == "Train"].reset_index(drop=True)
    te = tr_feat[tr_feat["train_or_test"] == "Test"].reset_index(drop=True)
    print(f"  Train {len(tr)}  Test {len(te)}")

    # growth_dev 와 NG 의 단변량 상관 (방향 확인)
    corr = np.corrcoef(tr["growth_dev"].values, tr["y"].values)[0, 1]
    print(f"[sanity] corr(growth_dev, NG) = {corr:+.3f}  (양수 기대: 약제가 덜 자람→NG)")

    dtrain = lgb.Dataset(tr[fcols].values, label=tr["y"].values, feature_name=fcols)
    params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1)
    booster = lgb.train(params, dtrain, num_boost_round=400)
    auc_in = roc_auc_score(te["y"].values, booster.predict(te[fcols].values))
    print(f"[in-domain] train-Test AUROC = {auc_in:.4f}")

    print(f"[load] FDA: {FDA_CSV}")
    fda = build_features_gd(FDA_CSV, "brightness", None)
    fda["brightness_pred"] = booster.predict(fda[fcols].values)
    oof = pd.read_csv(OBJAREA_OOF)
    keep = oof[KEY + ["gt_gng", "dtw_model_pred", "organism_group"]].rename(
        columns={"dtw_model_pred": "model_pred"})
    out = keep.merge(fda[KEY + ["brightness_pred"]], on=KEY, how="left")
    sub = out.dropna(subset=["brightness_pred", "gt_gng"])
    auc_cross = roc_auc_score(sub["gt_gng"], sub["brightness_pred"])
    auc_model = roc_auc_score(sub["gt_gng"], sub["model_pred"])
    # growth_dev 단독 AUROC (FDA)
    fda_m = fda.merge(keep[KEY + ["gt_gng"]], on=KEY, how="inner").dropna(subset=["gt_gng"])
    auc_gd = roc_auc_score(fda_m["gt_gng"], fda_m["growth_dev"])
    print(f"[cross-domain FDA] gd-GBM AUROC {auc_cross:.4f} | growth_dev 단독 {auc_gd:.4f} | "
          f"model_pred {auc_model:.4f} (n={len(sub)})")

    out.to_csv(OUT_DIR / "brightness_gd_crossdomain_oof.csv", index=False)
    booster.save_model(str(OUT_DIR / "brightness_gd_gbm.txt"))
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain")})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False)
    print("\n[feature importance] top 10")
    print(imp.head(10).to_string(index=False))

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "feature_set": "growth_dev (ctl/drug last-t1 성장차) + t1.. 거리, index0 제외",
        "n_features": len(fcols), "corr_growthdev_NG": round(float(corr), 3),
        "auroc_in_domain": round(float(auc_in), 4),
        "auroc_cross_domain_fda": round(float(auc_cross), 4),
        "auroc_growthdev_alone_fda": round(float(auc_gd), 4),
        "auroc_model_pred_fda": round(float(auc_model), 4),
        "top_features": imp.head(8).to_dict("records"),
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'brightness_gd_crossdomain_oof.csv'}")


if __name__ == "__main__":
    main()

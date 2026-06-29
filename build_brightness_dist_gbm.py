#!/usr/bin/env python3
"""거리 기반 brightness GBM — control-약제 brightness 편차(oa_diff/oa_ratio)에만
명시적으로 의존하는 판단기. 사용자 임상 메커니즘("control 과의 거리로 G/NG") 직접 반영.

기존 brightness GBM 은 절대 trajectory(drug 41% + ctl 34%)에 주로 의존, 명시적 거리
feature 는 3.8% 뿐이었다. 본 모델은 distance/relative feature 만 사용.

  distance features: oa_ratio_t0..t6, oa_diff_t0..t6, oa_ratio_mean, oa_auc_ratio,
                     oa_max_ratio  (+ concentration_idx_0 = 농도 위치)

Train: 2024_0709_traintest_allInfo Train split / Eval: train-Test + FDA cross-domain.
출력: agent_system/output/brightness_dist_gbm/brightness_dist_crossdomain_oof.csv
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
from agent_system.brightness_gbm_crossdomain import build_features, TRAIN_CSV, FDA_CSV  # noqa: E402

OBJAREA_OOF = ROOT / "claudeCode/output_dtw_aggregate_full/objarea_crossdomain_oof.csv"
OUT_DIR = ROOT / "agent_system/output/brightness_dist_gbm"
KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]

DIST_FEATURES = (
    [f"oa_ratio_t{t}" for t in range(7)] +
    [f"oa_diff_t{t}" for t in range(7)] +
    ["oa_ratio_mean", "oa_auc_ratio", "oa_max_ratio", "concentration_idx_0"]
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] TRAIN brightness: {TRAIN_CSV}")
    tr_feat = build_features(TRAIN_CSV, "brightness_list", label_col="bmd_gng",
                             extra_cols=("train_or_test",))
    tr_feat["y"] = (tr_feat["bmd_gng"].astype(str) == "NG").astype(int)
    fcols = [c for c in DIST_FEATURES if c in tr_feat.columns]
    print(f"[features] distance-only: {len(fcols)} → {fcols}")
    tr = tr_feat[tr_feat["train_or_test"] == "Train"].reset_index(drop=True)
    te = tr_feat[tr_feat["train_or_test"] == "Test"].reset_index(drop=True)
    print(f"  Train rows {len(tr)}  Test rows {len(te)}")

    dtrain = lgb.Dataset(tr[fcols].values, label=tr["y"].values, feature_name=fcols)
    params = dict(objective="binary", metric="auc", learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    booster = lgb.train(params, dtrain, num_boost_round=400)

    auc_in = roc_auc_score(te["y"].values, booster.predict(te[fcols].values))
    print(f"[in-domain] train-Test AUROC = {auc_in:.4f}")

    print(f"[load] FDA brightness: {FDA_CSV}")
    fda_feat = build_features(FDA_CSV, "brightness", label_col=None)
    fda_feat["brightness_pred"] = booster.predict(fda_feat[fcols].values)

    oof = pd.read_csv(OBJAREA_OOF)
    keep = oof[KEY + ["gt_gng", "dtw_model_pred", "organism_group"]].rename(
        columns={"dtw_model_pred": "model_pred"})
    out = keep.merge(fda_feat[KEY + ["brightness_pred"]], on=KEY, how="left")
    sub = out.dropna(subset=["brightness_pred", "gt_gng"])
    auc_cross = roc_auc_score(sub["gt_gng"], sub["brightness_pred"])
    auc_model = roc_auc_score(sub["gt_gng"], sub["model_pred"])
    print(f"[cross-domain FDA] brightness_dist AUROC = {auc_cross:.4f} | "
          f"model_pred {auc_model:.4f}  (n={len(sub)})")

    out.to_csv(OUT_DIR / "brightness_dist_crossdomain_oof.csv", index=False)
    booster.save_model(str(OUT_DIR / "brightness_dist_gbm.txt"))

    # feature importance confirm (이제 거리 feature가 지배해야 함)
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain")})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False)
    print("\n[feature importance] top 10")
    print(imp.head(10).to_string(index=False))

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "feature_set": "distance-only (oa_ratio/oa_diff + conc_idx)",
        "n_features": len(fcols),
        "auroc_in_domain": round(float(auc_in), 4),
        "auroc_cross_domain_fda": round(float(auc_cross), 4),
        "auroc_model_pred_fda": round(float(auc_model), 4),
        "top_features": imp.head(8).to_dict("records"),
    }, ensure_ascii=False, indent=2))
    print(f"\n[saved] {OUT_DIR/'brightness_dist_crossdomain_oof.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-domain brightness GBM — objarea_gbm_crossdomain.py의 brightness 판.

object_area 와 달리 brightness 는 학습 CSV(`brightness_list`)·FDA(`brightness`)
양쪽에 존재 → in-domain 학습 가능.

  - Train: 2024_0709_traintest_allInfo.csv 의 Train split (brightness_list, bmd_gng)
  - Eval in-domain : 같은 CSV 의 Test split
  - Eval cross-domain: FDA2023 (brightness 컬럼)

출력: agent_system/output/brightness_gbm/brightness_crossdomain_oof.csv
  (sample_id, antimicrobial, concentration_idx_0, gt_gng, model_pred, brightness_pred)
"""
from __future__ import annotations
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/kptae/project/qnt_algorithm")
sys.path.insert(0, str(ROOT))
from claudeCode.add_object_area_dtw import parse_csv_list           # noqa: E402
from claudeCode.objarea_raw_gbm import _features_for_row, MODEL_TIME_LEN, CTL_PICK_INDEX  # noqa: E402

TRAIN_CSV = "/data/dRAST30_prepare_csv/2024_0709_traintest_allInfo.csv"
FDA_CSV = ("/home/kptae/data/allinfo/fda2023_preprocessedWithFeature_validControlInfo_"
           "useMax_TrueWith_object_area_useMax_TrueWith_object_count_useMax_TrueWith_"
           "brightness_useMax_False.csv")
OBJAREA_OOF = ROOT / "claudeCode/output_dtw_aggregate_full/objarea_crossdomain_oof.csv"
OUT_DIR = ROOT / "agent_system/output/brightness_gbm"


def build_features(csv_path: str, bright_col: str, label_col: str | None,
                   extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per-(sample,drug,conc) brightness trajectory features (mirror objarea)."""
    use = {"sample_id", "antimicrobial", "concentration", bright_col}
    if label_col:
        use.add(label_col)
    use |= set(extra_cols)
    df = pd.read_csv(csv_path, usecols=lambda c: c in use, low_memory=False)
    df["br"] = df[bright_col].apply(parse_csv_list)
    df = df.dropna(subset=["br"]).reset_index(drop=True)

    ctl_df = df[df["antimicrobial"] == "Cont"]
    ctl_map: dict[str, np.ndarray] = {}
    for sid, sub in ctl_df.groupby("sample_id", sort=False):
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
        ctl_traj = padded[best, :MODEL_TIME_LEN]
        if np.all(np.isfinite(ctl_traj)):
            ctl_map[sid] = ctl_traj

    drug_df = df[df["antimicrobial"] != "Cont"].copy()
    rows = []
    for (sid, drug), sub in drug_df.groupby(["sample_id", "antimicrobial"], sort=False):
        if sid not in ctl_map:
            continue
        ctl = ctl_map[sid].astype(np.float64)
        sub_sorted = sub.sort_values("concentration").reset_index(drop=True)
        for rank, srow in sub_sorted.iterrows():
            traj = np.asarray(srow["br"], dtype=np.float64)
            if len(traj) < MODEL_TIME_LEN:
                continue
            traj = traj[:MODEL_TIME_LEN]
            if not np.all(np.isfinite(traj)):
                continue
            feat = _features_for_row(sid, drug, rank, traj, ctl)
            if label_col:
                feat[label_col] = srow[label_col]
            for c in extra_cols:
                feat[c] = srow[c]
            rows.append(feat)
    return pd.DataFrame(rows)


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("oa_")]
    cols.append("concentration_idx_0")
    return [c for c in cols if c in df.columns]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] TRAIN brightness: {TRAIN_CSV}")
    tr_feat = build_features(TRAIN_CSV, "brightness_list", label_col="bmd_gng",
                             extra_cols=("train_or_test",))
    tr_feat["y"] = (tr_feat["bmd_gng"].astype(str) == "NG").astype(int)
    fcols = feature_cols(tr_feat)
    tr = tr_feat[tr_feat["train_or_test"] == "Train"].reset_index(drop=True)
    te = tr_feat[tr_feat["train_or_test"] == "Test"].reset_index(drop=True)
    print(f"  Train rows {len(tr)}  Test rows {len(te)}  features {len(fcols)}")

    dtrain = lgb.Dataset(tr[fcols].values, label=tr["y"].values, feature_name=fcols)
    params = dict(objective="binary", metric="auc", learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    booster = lgb.train(params, dtrain, num_boost_round=400)

    te_pred = booster.predict(te[fcols].values)
    auc_in = roc_auc_score(te["y"].values, te_pred)
    print(f"[in-domain] train-Test AUROC = {auc_in:.4f}")

    print(f"[load] FDA brightness: {FDA_CSV}")
    fda_feat = build_features(FDA_CSV, "brightness", label_col=None)
    fda_pred = booster.predict(fda_feat[fcols].values)
    fda_feat["brightness_pred"] = fda_pred

    # align labels + model_pred from operational objarea OOF
    oof = pd.read_csv(OBJAREA_OOF)
    key = ["sample_id", "antimicrobial", "concentration_idx_0"]
    keep = oof[key + ["gt_gng", "dtw_model_pred", "organism_group"]].rename(
        columns={"dtw_model_pred": "model_pred"})
    out = keep.merge(fda_feat[key + ["brightness_pred"]], on=key, how="left")
    cov = int(out["brightness_pred"].notna().sum())
    print(f"[merge] FDA rows {len(out)}  brightness covered {cov} "
          f"({100*cov/len(out):.1f}%)")

    sub = out.dropna(subset=["brightness_pred", "gt_gng"])
    auc_cross = roc_auc_score(sub["gt_gng"], sub["brightness_pred"])
    auc_model = roc_auc_score(sub["gt_gng"], sub["model_pred"])
    print(f"[cross-domain FDA] brightness AUROC = {auc_cross:.4f}  "
          f"| model_pred AUROC = {auc_model:.4f}  (n={len(sub)})")

    out_csv = OUT_DIR / "brightness_crossdomain_oof.csv"
    out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    booster.save_model(str(OUT_DIR / "brightness_gbm.txt"))

    import json
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "auroc_in_domain_trainTest": round(float(auc_in), 4),
        "auroc_cross_domain_fda": round(float(auc_cross), 4),
        "auroc_model_pred_fda": round(float(auc_model), 4),
        "n_fda_rows": int(len(out)), "n_fda_brightness_covered": cov,
        "n_features": len(fcols),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

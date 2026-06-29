"""TrajectoryGBM — signal 별 GBM 학습/추론/CV (brightness·object_area 공통 코어).

세 가지 모드 (모두 입력 컬럼만 SignalConfig 로 바뀜):
  1) train_crossdomain : TRAIN_CSV(train split) 로 in-domain 학습 → 저장,
                         in-domain(test split) + cross-domain(FDA) AUROC 보고.
                         (구 train_brightness_gbm.py + objarea_gbm_crossdomain.py 통합)
  2) cv_auroc_indomain : FDA raw-trajectory feature 로 GroupKFold(sample_id) OOF AUROC.
                         (구 objarea_only_gbm.py 의 핵심 — honest in-domain baseline)
  3) predict           : 저장된 GBM 으로 임의 CSV 추론 → per-(sample,drug,conc) pred.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config as C
from .features import build_features, feature_columns

KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]


# --------------------------------------------------------------------------
# 1) cross-domain 학습 (TRAIN_CSV → FDA)
# --------------------------------------------------------------------------
def train_crossdomain(signal: C.SignalConfig, save: bool = True) -> dict:
    """TRAIN_CSV train split 으로 학습, test split(in) + FDA(cross) AUROC 평가."""
    feat = build_features(C.TRAIN_CSV, signal.train_col, label_col=C.LABEL_COL,
                          extra_cols=("train_or_test",))
    feat["y"] = (feat[C.LABEL_COL].astype(str) == "NG").astype(int)
    fcols = feature_columns(feat)
    tr = feat[feat["train_or_test"] == "Train"].reset_index(drop=True)
    te = feat[feat["train_or_test"] == "Test"].reset_index(drop=True)

    dtrain = lgb.Dataset(tr[fcols].values, label=tr["y"].values, feature_name=fcols)
    booster = lgb.train(C.LGB_PARAMS, dtrain, num_boost_round=C.LGB_NUM_ROUNDS)
    auc_in = float(roc_auc_score(te["y"].values, booster.predict(te[fcols].values)))

    # cross-domain FDA: 라벨은 ALIGN_OOF(gt_gng) 와 key join
    fda = build_features(C.FDA_CSV, signal.fda_col)
    fda["pred"] = booster.predict(fda[fcols].values)
    align = pd.read_csv(C.ALIGN_OOF)
    merged = align[KEY + ["gt_gng"]].merge(fda[KEY + ["pred"]], on=KEY, how="inner").dropna()
    auc_cross = float(roc_auc_score(merged["gt_gng"], merged["pred"]))

    meta = {"signal": signal.name, "auroc_in_domain": round(auc_in, 4),
            "auroc_cross_domain_fda": round(auc_cross, 4),
            "n_features": len(fcols), "n_train": int(len(tr)), "n_cross": int(len(merged))}
    if save:
        signal.model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(signal.model_path))
        signal.model_path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


# --------------------------------------------------------------------------
# 2) in-domain GroupKFold OOF AUROC (honest baseline)
# --------------------------------------------------------------------------
def cv_auroc_indomain(signal: C.SignalConfig, n_splits: int = 5,
                      seed: int = 12345) -> dict:
    """FDA raw-trajectory feature 로 GroupKFold(sample_id) OOF AUROC.

    라벨/organism 정렬은 ALIGN_OOF(gt_gng, organism_group). model_pred 비교 포함.
    """
    feat = build_features(C.FDA_CSV, signal.fda_col)
    fcols = feature_columns(feat)
    align = pd.read_csv(C.ALIGN_OOF)
    cols = KEY + ["gt_gng"] + [c for c in ("organism_group", "dtw_model_pred")
                               if c in align.columns]
    df = align[cols].merge(feat[KEY + fcols], on=KEY, how="inner").dropna(
        subset=fcols + ["gt_gng"]).reset_index(drop=True)

    X, y = df[fcols].values, df["gt_gng"].astype(int).values
    groups = df["sample_id"].values
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(df))
    fold_aurocs = []
    for fold, (trn, tst) in enumerate(gkf.split(X, y, groups)):
        params = dict(C.LGB_PARAMS, seed=seed + fold)
        m = lgb.train(params, lgb.Dataset(X[trn], label=y[trn], feature_name=fcols),
                      num_boost_round=C.LGB_NUM_ROUNDS)
        oof[tst] = m.predict(X[tst])
        fold_aurocs.append(float(roc_auc_score(y[tst], oof[tst])))
    df["gbm_oof"] = oof
    overall = float(roc_auc_score(y, oof))
    out = {"signal": signal.name, "n_rows": int(len(df)),
           "n_samples": int(df["sample_id"].nunique()),
           "oof_auroc": round(overall, 4),
           "mean_fold_auroc": round(float(np.mean(fold_aurocs)), 4),
           "fold_aurocs": [round(a, 4) for a in fold_aurocs]}
    if "dtw_model_pred" in df.columns:
        out["model_pred_auroc"] = round(float(roc_auc_score(y, df["dtw_model_pred"])), 4)
        out["gap_model_minus_gbm"] = round(out["model_pred_auroc"] - overall, 4)
    return out, df


# --------------------------------------------------------------------------
# 3) 추론
# --------------------------------------------------------------------------
def predict(signal: C.SignalConfig, csv_path: str | None = None,
            traj_col: str | None = None) -> pd.DataFrame:
    """저장된 GBM 으로 추론 → KEY + '{signal}_pred'(=P(NG))."""
    booster = lgb.Booster(model_file=str(signal.model_path))
    feat = build_features(csv_path or C.FDA_CSV, traj_col or signal.fda_col)
    fcols = [f for f in booster.feature_name() if f in feat.columns]
    feat[f"{signal.name}_pred"] = booster.predict(feat[fcols].values)
    return feat[KEY + [f"{signal.name}_pred"]]

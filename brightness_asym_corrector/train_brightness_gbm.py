#!/usr/bin/env python3
"""학습 — brightness GBM 을 학습 CSV(Train split)로 in-domain 학습.

  Train : 2024_0709_traintest_allInfo.csv  Train split (brightness_list, bmd_gng)
  Eval  : 같은 CSV Test split (in-domain) + FDA2023 (cross-domain)

object_area 와 달리 brightness 는 학습 CSV·FDA 양쪽에 존재 → in-domain 학습 가능.
검증 수치: in-domain AUROC ≈ 0.915, cross-domain FDA AUROC ≈ 0.918 (전이 안정).

사용:
  python train_brightness_gbm.py                 # 모델 재학습 → model/brightness_gbm.txt
  python train_brightness_gbm.py --no-save       # 학습만 (저장 안 함)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                    # noqa: E402
from brightness_features import build_features, feature_columns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_csv", default=C.TRAIN_CSV)
    ap.add_argument("--fda_csv", default=C.FDA_CSV)
    ap.add_argument("--model_out", default=str(C.MODEL_PATH))
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    print(f"[load] TRAIN: {args.train_csv}")
    tr_feat = build_features(args.train_csv, C.TRAIN_BRIGHTNESS_COL,
                             label_col=C.LABEL_COL, extra_cols=("train_or_test",))
    tr_feat["y"] = (tr_feat[C.LABEL_COL].astype(str) == "NG").astype(int)
    fcols = feature_columns(tr_feat)
    tr = tr_feat[tr_feat["train_or_test"] == "Train"].reset_index(drop=True)
    te = tr_feat[tr_feat["train_or_test"] == "Test"].reset_index(drop=True)
    print(f"  Train {len(tr)}  Test {len(te)}  features {len(fcols)}")

    dtrain = lgb.Dataset(tr[fcols].values, label=tr["y"].values, feature_name=fcols)
    booster = lgb.train(C.LGB_PARAMS, dtrain, num_boost_round=C.LGB_NUM_ROUNDS)

    auc_in = roc_auc_score(te["y"].values, booster.predict(te[fcols].values))
    print(f"[in-domain] train-Test AUROC = {auc_in:.4f}")

    print(f"[load] FDA: {args.fda_csv}")
    fda = build_features(args.fda_csv, C.FDA_BRIGHTNESS_COL)
    fda["brightness_pred"] = booster.predict(fda[fcols].values)
    align = pd.read_csv(C.ALIGN_OOF)
    key = ["sample_id", "antimicrobial", "concentration_idx_0"]
    merged = align[key + ["gt_gng"]].merge(fda[key + ["brightness_pred"]],
                                           on=key, how="inner").dropna()
    auc_cross = roc_auc_score(merged["gt_gng"], merged["brightness_pred"])
    print(f"[cross-domain FDA] AUROC = {auc_cross:.4f}  (n={len(merged)})")

    if not args.no_save:
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(args.model_out)
        meta = {"auroc_in_domain": round(float(auc_in), 4),
                "auroc_cross_domain_fda": round(float(auc_cross), 4),
                "n_features": len(fcols), "n_train": int(len(tr))}
        Path(args.model_out).with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[saved] {args.model_out}")


if __name__ == "__main__":
    main()

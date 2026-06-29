"""AsymCorrector — trajectory GBM 으로 image model_pred 를 비대칭 G-방향 보정 (signal-agnostic).

운영 score = P(NG). VME(거짓 NG = 실제 G인데 NG → MIC 과소 → S 오보고)는 P(NG)를 *올릴* 때
생긴다. 따라서 신호 GBM 이 image 보다 더 G 라고 말할 때(sig_pred < img)만 점수를 내리고
(G 방향), 더 NG 라고 말할 때는 image 를 유지 → false-NG(VME) 차단.
  corrected = (1-w)*img + w*sig_pred   (단, sig_pred < img 이고 apply cell 인 row)
            = img                       (그 외)

구 brightness_asym_corrector/brightness_corrector.py 를 일반화 (입력 신호만 SignalConfig 로 교체).

⚠ brightness 신호의 비대칭 보정은 GroupKFold(sample_id) CV 에서 held-out 미재현(2026-06-04) →
운영 미투입. 본 클래스는 그 검증 인프라(또는 다른 신호 실험)용 공통 도구.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import build_features

KEY = ["sample_id", "antimicrobial", "concentration_idx_0"]


class AsymCorrector:
    def __init__(self, signal: C.SignalConfig, weight: float | None = None,
                 apply_cells_csv=None):
        self.signal = signal
        self.booster = lgb.Booster(model_file=str(signal.model_path))
        self.weight = signal.asym_weight if weight is None else float(weight)
        cells_csv = apply_cells_csv or signal.apply_cells_csv
        if cells_csv is None:
            self.apply_cells: set = set()
        else:
            cells = pd.read_csv(cells_csv)
            self.apply_cells = set(zip(cells["organism_group"].astype(str).str.strip(),
                                       cells["antimicrobial"].astype(str).str.strip()))

    def predict_signal(self, csv_path: str, traj_col: str | None = None) -> pd.DataFrame:
        feat = build_features(csv_path, traj_col or self.signal.fda_col)
        fcols = [f for f in self.booster.feature_name() if f in feat.columns]
        feat["sig_pred"] = self.booster.predict(feat[fcols].values)
        return feat[KEY + ["sig_pred"]]

    def correct(self, per_conc: pd.DataFrame, sig_pred: pd.DataFrame,
                model_pred_col: str = "dtw_model_pred") -> pd.DataFrame:
        df = per_conc.merge(sig_pred, on=KEY, how="left")
        img = df[model_pred_col].astype(float).to_numpy()
        sp = df["sig_pred"].astype(float).to_numpy()
        sp = np.where(np.isfinite(sp), sp, img)        # 미커버 row → 보정 안 함
        in_cell = df.apply(lambda r: (str(r["organism_group"]).strip(),
                                      str(r["antimicrobial"]).strip()) in self.apply_cells,
                           axis=1).to_numpy()
        g_dir = sp < img
        mask = in_cell & g_dir & np.isfinite(df["sig_pred"].to_numpy())
        corrected = img.copy()
        corrected[mask] = (1 - self.weight) * img[mask] + self.weight * sp[mask]
        out = df.copy()
        out["_sig_pred"] = sp
        out["_corrected"] = mask
        out[model_pred_col] = np.clip(corrected, 0.0, 1.0)
        return out.drop(columns=["sig_pred"])

    def apply_to_per_conc(self, per_conc: pd.DataFrame, csv_path: str,
                          traj_col: str | None = None,
                          model_pred_col: str = "dtw_model_pred") -> pd.DataFrame:
        sp = self.predict_signal(csv_path, traj_col)
        out = self.correct(per_conc, sp, model_pred_col)
        n = int(out["_corrected"].sum())
        print(f"[{self.signal.name} corrector] G-방향 보정 row {n:,} / {len(out):,} "
              f"(apply cells {len(self.apply_cells)}, w={self.weight})")
        return out

"""[SHIM] BrightnessAsymCorrector → trajectory_gbm.corrector.AsymCorrector 로 통합.

2026-06-04: 비대칭 G-방향 보정 로직을 `agent_system/trajectory_gbm/corrector.py` 로 일반화
(brightness·object_area 공통). 이 파일은 기존 인터페이스(predict_brightness / apply_to_per_conc /
`_brightness_pred` 컬럼)를 보존하는 thin 래퍼 — evaluate.py / groupkfold_cv.py 가 그대로 동작한다.

⚠ brightness 비대칭 보정의 +3 PASS 는 GroupKFold(sample_id) CV 에서 held-out 미재현(2026-06-04)
→ 운영 미투입. 상세 = artifacts/cv/cv_report.html.
"""
from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C                                                    # noqa: E402
from agent_system.trajectory_gbm.corrector import AsymCorrector, KEY  # noqa: E402,F401
from agent_system.trajectory_gbm.config import get_signal            # noqa: E402


class BrightnessAsymCorrector(AsymCorrector):
    """brightness 신호 전용 AsymCorrector (로컬 패키지 자산 + 기존 컬럼명 보존)."""

    def __init__(self, model_path=None, apply_cells_csv=None, weight=None):
        sig = dc.replace(
            get_signal("brightness"),
            model_path=Path(model_path or C.MODEL_PATH),
            apply_cells_csv=Path(apply_cells_csv or C.APPLY_CELLS_CSV),
            asym_weight=C.ASYM_WEIGHT,
        )
        super().__init__(sig, weight=weight)

    # ---- 기존 인터페이스 별칭 ----
    def predict_brightness(self, brightness_csv: str, brightness_col: str) -> pd.DataFrame:
        return self.predict_signal(brightness_csv, brightness_col).rename(
            columns={"sig_pred": "brightness_pred"})

    def apply_to_per_conc(self, per_conc, brightness_csv, brightness_col=None,
                          model_pred_col: str = "dtw_model_pred"):
        out = super().apply_to_per_conc(per_conc, brightness_csv,
                                        traj_col=brightness_col or C.FDA_BRIGHTNESS_COL,
                                        model_pred_col=model_pred_col)
        return out.rename(columns={"_sig_pred": "_brightness_pred"})


if __name__ == "__main__":
    pc = pd.read_csv(C.ROUTED_PER_CONC, low_memory=False)
    corr = BrightnessAsymCorrector()
    out = corr.apply_to_per_conc(pc, C.FDA_CSV)
    print(out.loc[out["_corrected"], ["organism_group", "antimicrobial"]]
          .drop_duplicates().to_string(index=False))

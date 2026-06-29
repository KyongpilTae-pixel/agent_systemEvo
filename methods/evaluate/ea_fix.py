"""EA panel-notation bug fix — 평가기 correctness 수정 (★방법론 아님).

⚠ 이것은 scoring/calibration **방법론이 아니라 버그 수정**이다. PASS 를 새로 끌어올리는
개선책이 아니라, 잘못 FAIL 처리되던 cell 을 **정정**하는 평가기 패치다(정정 효과 71→103, +32).

운영 drastModules/mics.py 의 evalEA 가 BMD doubling notation(0.015625/0.03125/0.0625/
0.125)과 dRAST half-dilution notation(0.016/0.03/0.06/0.12)의 mismatch 를 ±1 dilution 으로
인식하지 못해 EA 가 부당하게 FAIL 되던 버그를 수정. (A) log2-ratio fallback patch +
(B) bmd_mic notation 정규화. 검증: 원본 True 보존(strict superset), PASS→FAIL = 0.

verify_method_sir_pipeline 가 import 시 자동 적용하므로 Recipe.evaluate 경로에선 항상 켜져 있다.
REGISTRY(방법론 카탈로그)에는 포함되지 않고 FIXES 로 분리. stage=EVALUATE.
apply()는 데이터를 변형하지 않는 identity; 평가기에 작용하는 activate()/정규화 유틸을 노출.
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods.base import Method, Stage, Status
from claudeCode.patched_mics import (apply_patch, normalize_bmd_mic_series,
                                      normalize_bmd_mic_notation)


class EAPanelNotationFix(Method):
    name = "ea_panel_notation_fix"
    stage = Stage.EVALUATE
    status = Status.ADOPTED
    pass_impact = "정정 +32 (개선 아님)"
    summary = ("⚠버그 수정(방법론 아님). evalEA BMD↔dRAST notation mismatch 정정 "
               "(log2-ratio fallback + bmd_mic 정규화). 71→103. verify 에 자동 내장.")

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        # 평가기 correctness 패치이므로 per_conc 변형 없음 (identity).
        return per_conc

    # ---- 평가기에 작용하는 진짜 동작 ----
    @staticmethod
    def activate() -> bool:
        """drastModules.mics.Mic.evalEA → log2-ratio patch 적용 (idempotent)."""
        return apply_patch()

    @staticmethod
    def normalize_series(series: pd.Series) -> pd.Series:
        """bmd_mic notation 정규화 (BMD doubling → dRAST half-dilution)."""
        return normalize_bmd_mic_series(series)

    @staticmethod
    def normalize(value):
        return normalize_bmd_mic_notation(value)

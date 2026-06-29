"""⑩ Genus model mapping — evo 워크북 (Genus, drug) → 모델 라우팅 (대안).

evo_selected_model_and_brightness_thershold_260529.xlsx 의 genus 단위 모델 맵핑을 그 genus
소속 organism 전부에 적용하는 라우팅. our(AUROC) per-organism routing 대비 보수적:
PASS 95 / VME-rate 1.55% (vs our 103 / 1.77%) — PASS↔VME trade-off. LNZ 분기점에서
genus=beta_lactam→FAIL(our=early_stop→PASS)이라 LNZ gram-positive 를 놓침.

검증된 산출(evaluate_genus_routing.py 의 per_conc_genus_routing.csv)의 점수를 활성 점수로
주입하는 thin wrapper. 산출이 없으면 안내. stage=SCORE, status=ALTERNATIVE.
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B
from agent_system.methods.base import Method, Stage, Status

GENUS_PER_CONC = (B.ROOT / "agent_system/brightness_asym_corrector/artifacts/"
                  "genus_routing/per_conc_genus_routing.csv")


class GenusModelRouting(Method):
    name = "genus_model_routing"
    stage = Stage.SCORE
    status = Status.ALTERNATIVE
    pass_impact = "PASS 95 / VME 1.55% (보수)"
    summary = ("evo genus 모델 맵핑 라우팅. our 103 대비 PASS 95(보수적, VME-rate 낮음). "
               "LNZ 특이 cell 놓침.")

    def __init__(self, per_conc_csv=None):
        self.per_conc_csv = per_conc_csv or GENUS_PER_CONC

    def apply(self, per_conc: pd.DataFrame) -> pd.DataFrame:
        if not self.per_conc_csv.exists():
            raise FileNotFoundError(
                f"{self.per_conc_csv} 없음. 먼저 evaluate_genus_routing.py 로 생성하세요.")
        g = pd.read_csv(self.per_conc_csv, low_memory=False,
                        usecols=B.KEY + ["dtw_model_pred"])
        g = g.rename(columns={"dtw_model_pred": "_genus_score"})
        out = per_conc.merge(g, on=B.KEY, how="left")
        # genus 미지정 행은 기존 활성 점수(routed) fallback
        out[B.SCORE_COL] = out["_genus_score"].where(out["_genus_score"].notna(),
                                                     out[B.SCORE_COL])
        return out.drop(columns=["_genus_score"])

"""Agent H — Training Data Optimizer.

Phase 1 stub: 기존 큐레이션 스크립트 wrapping. 실제 retrain은 사용자 confirm 후.
"""
from __future__ import annotations

from typing import Optional

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import AgentResult, CurationRecipe, SharedState


class TrainingDataOptimizer(BaseAgent):
    CODE = "H"
    NAME = "TrainingDataOptimizer"
    ROLE = "학습 데이터 큐레이션 — filter/rebalance/clean + retrain 제안"

    @tool("후보 curation recipe 제안 (object_area 필터, 재균형, normalize 등)")
    def propose_curations(self, *, state: SharedState,
                          target_drug: Optional[str] = None,
                          target_organism: Optional[str] = None) -> AgentResult:
        # Phase 1: 알려진 효과 있는 recipe 후보 정적 목록.
        recipes = [
            CurationRecipe(
                recipe_id="oa_band_045_08",
                strategy="object_area_filter",
                parameters={"band": [0.45, 0.80]},
                description="control object_area 0.45-0.80 band 통과 row만 사용"),
            CurationRecipe(
                recipe_id="rebalance_R_pbc_aminoglycoside",
                strategy="rebalance_R",
                parameters={"target_drug_class": "Aminoglycoside",
                            "scenario": "pbc_down_sample"},
                target_drug="GEN/AMI/...",
                description="PBC R-spike 보정 — Scenario 2 (2026-05-15)"),
            CurationRecipe(
                recipe_id="organism_normalize_cons",
                strategy="organism_normalize",
                parameters={"map": "CoNS species → 'Coagulase-negative'"},
                description="CoNS species 통합 (fda_only 121→57)"),
        ]
        return self._ok("propose_curations",
                        proposals=[r.__dict__ for r in recipes])

    @tool("curation recipe 시뮬레이션 (실제 parquet 생성 없이 shift gap 측정)")
    def simulate(self, *, state: SharedState, recipe_id: str) -> AgentResult:
        # Phase 1 stub — 실제로는 analyze_training_reconstruction 호출
        return self._ok("simulate", audit={
            "recipe_id": recipe_id, "expected_shift_gap": "TBD",
            "note": "stub — Phase 2에서 analyze_training_reconstruction wiring"})

    @tool("curation 적용한 parquet 빌드 (재학습용)")
    def build_parquet(self, *, state: SharedState, recipe_id: str,
                      output_path: str) -> AgentResult:
        return self._ok("build_parquet", audit={
            "recipe_id": recipe_id, "output": output_path,
            "note": "stub — Phase 2에서 build_rebalanced_parquet wiring"})

    @tool("retrain 권고 (어느 모델을 어떤 curated dataset으로 재학습?)")
    def recommend_retrain(self, *, state: SharedState) -> AgentResult:
        return self._ok("recommend_retrain", audit={
            "note": "stub — Phase 2에서 model 학습 cost 분석 + 우선순위 산출"})

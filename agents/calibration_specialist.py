"""Agent C — Calibration Specialist."""
from __future__ import annotations

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import AgentResult, SharedState


class CalibrationSpecialist(BaseAgent):
    CODE = "C"
    NAME = "CalibrationSpecialist"
    ROLE = "isotonic per-cell / pooled-organism / pooled-drug-class + threshold tuning"

    @tool("기존 per-cell isotonic 사용 권고 (within-FDA 검증 우세)")
    def recommend_calibration(self, *, state: SharedState,
                              recipe_id: str) -> AgentResult:
        return self._ok("recommend_calibration",
                        proposals=[{
                            "recipe_id": recipe_id,
                            "calibration": "isotonic_per_cell",
                            "threshold": 0.75,
                            "asset": "claudeCode/output_dtw_aggregate_full/isotonic_lookup.csv",
                            "rationale": "within-FDA per-cell ISO 우세, pooled는 손해 (2026-05-19 Task 4)",
                        }])

    @tool("threshold sweep (global thresholds: 0.5, 0.65, 0.75, 0.80)")
    def tune_threshold(self, *, state: SharedState,
                       recipe_id: str) -> AgentResult:
        # Phase 1 stub
        return self._ok("tune_threshold", proposals=[
            {"threshold": t, "note": "0.75는 운영 default"}
            for t in [0.50, 0.65, 0.75, 0.80]
        ])

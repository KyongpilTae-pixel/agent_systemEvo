"""Agent F — Data Quality Auditor."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import AgentResult, SharedState

DIFF_CSV = Path("claudeCode/output_dataset_diff_traintest_normalized/dataset_cell_diff_summary.csv")


class DataQualityAuditor(BaseAgent):
    CODE = "F"
    NAME = "DataQualityAuditor"
    ROLE = "dataset diff + shifted set 갱신 + 이상치 감지"

    @tool("baseline state audit (shifted set + bucket 분포)")
    def audit(self, *, state: SharedState) -> AgentResult:
        if not DIFF_CSV.exists():
            return self._fail("audit", f"{DIFF_CSV} 없음")
        diff = pd.read_csv(DIFF_CSV)
        bucket_counts = diff["bucket"].value_counts().to_dict()
        shifted = [[str(r.organism_group), str(r.antimicrobial)]
                   for _, r in diff[diff.bucket == "shifted"].iterrows()]
        state.shifted_set = shifted
        return self._ok("audit",
                        audit={"n_total": len(diff), "bucket_counts": bucket_counts,
                               "n_shifted": len(shifted)})

    @tool("shifted set 새로고침 (compare_datasets_full_traintest 재실행 제안)")
    def refresh_shifted_set(self, *, state: SharedState) -> AgentResult:
        # Phase 1 stub
        return self._ok("refresh_shifted_set", audit={
            "note": "Phase 2에서 compare_datasets_full_traintest subprocess wiring",
            "current_n_shifted": len(state.shifted_set)})

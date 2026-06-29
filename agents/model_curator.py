"""Agent A — Model Curator."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import AgentResult, SharedState

DEFAULT_AUROC_CSV = "claudeCode/output_subset_eval/subset_models_per_cell_auroc.csv"
DEFAULT_ROUTING_CSV = "claudeCode/output_subset_eval/cell_model_routing_lookup.csv"
THREE_O_IDEA = Path("/home/gyuyoung/project/lab/3.0_idea")


class ModelCurator(BaseAgent):
    CODE = "A"
    NAME = "ModelCurator"
    ROLE = "subset checkpoint 카탈로그 + per-cell AUROC + routing 후보 제안"

    @tool("학습된 OrderPickGNG checkpoint scan (신규 모델 자동 감지)")
    def audit(self, *, state: SharedState) -> AgentResult:
        if not THREE_O_IDEA.exists():
            return self._fail("audit", f"{THREE_O_IDEA} 없음")
        models = []
        for d in sorted(THREE_O_IDEA.iterdir()):
            if not d.is_dir():
                continue
            yml = d / "OrderPickGNG_train_info.yaml"
            if yml.exists():
                models.append(d.name)
        return self._ok("audit", audit={"n_models": len(models),
                                        "models": models})

    @tool("per-cell best model 또는 drug_class routing 제안")
    def propose_routing(self, *, state: SharedState,
                        strategy: str = "best_per_cell") -> AgentResult:
        # Phase 1: 기존 cell_model_routing_lookup.csv 사용
        p = Path(DEFAULT_ROUTING_CSV)
        if not p.exists():
            return self._fail("propose_routing", f"{p} 없음 — per_cell_model_routing.py 먼저 실행 필요")
        df = pd.read_csv(p)
        return self._ok("propose_routing",
                        audit={"strategy": strategy, "n_cells": len(df),
                               "lookup_path": str(p)},
                        proposals=[{"strategy": strategy,
                                    "lookup_csv": str(p),
                                    "summary": df["best_model"].value_counts().to_dict()}])

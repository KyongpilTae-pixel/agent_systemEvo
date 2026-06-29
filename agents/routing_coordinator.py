"""Agent G — Routing Coordinator (Orchestrator의 의사결정 helper).

cell-greedy → consolidation 2-stage 흐름 관리.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import (
    AgentResult, OperationalManifest, SharedState,
)


class RoutingCoordinator(BaseAgent):
    CODE = "G"
    NAME = "RoutingCoordinator"
    ROLE = "agent 후보 종합 + Pareto + 운영 manifest commit"

    @tool("특정 cell에 시도할 후보 변형 목록 (Stage 1 cell-greedy 용)")
    def candidates_for_cell(self, *, state: SharedState,
                            organism_group: str, antimicrobial: str,
                            model_pool: list[str]) -> AgentResult:
        # Phase 1: model_pool 단일 모델 + size 2~3 rank-avg ensemble만
        cands = [{"recipe_id": f"single_{m}", "type": "single",
                  "components": [m]} for m in model_pool]
        for i, m1 in enumerate(model_pool):
            for m2 in model_pool[i+1:]:
                cands.append({"recipe_id": f"rankavg_{m1}_{m2}",
                              "type": "rank_average",
                              "components": [m1, m2]})
        return self._ok("candidates_for_cell",
                        audit={"cell": f"{organism_group}||{antimicrobial}",
                               "n_candidates": len(cands)},
                        proposals=cands)

    @tool("Stage 2: cell-local winner들을 클러스터 → 최소 recipe set으로 통합")
    def consolidate(self, *, state: SharedState,
                    per_cell_winners: dict,
                    max_recipes: int = 5) -> AgentResult:
        # cell → recipe_id 매핑에서 recipe 빈도 count
        recipe_freq = Counter([w["recipe_id"] for w in per_cell_winners.values()])
        top = recipe_freq.most_common(max_recipes)
        coverage = sum(c for _, c in top)
        total = len(per_cell_winners)
        return self._ok("consolidate",
                        audit={"n_cells": total, "n_unique_recipes": len(recipe_freq),
                               "top_recipes": top, "coverage_top_k": coverage,
                               "coverage_pct": (coverage/total*100) if total else 0},
                        proposals=[{"recipe_id": r, "covers_n_cells": c}
                                   for r, c in top])

    @tool("사이클 완료 후 manifest commit (사용자 approve 필요)")
    def commit(self, *, state: SharedState, recipe_id: str,
               cycle_id: int, agent_lineage: list[str],
               validated_metrics: dict, operational_assets: dict,
               user_approved: bool = True) -> AgentResult:
        m = OperationalManifest(
            cycle_id=cycle_id,
            committed_at=__import__("datetime").datetime.now().isoformat(timespec="seconds"),
            recipe_id=recipe_id,
            agent_lineage=agent_lineage,
            validated_metrics=validated_metrics,
            operational_assets=operational_assets,
            user_approved=user_approved,
        )
        p = state.save_manifest(m)
        state.save()
        return self._ok("commit", artifacts=[str(p)],
                        audit={"cycle_id": cycle_id, "recipe_id": recipe_id})

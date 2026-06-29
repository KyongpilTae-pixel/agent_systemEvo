"""Agent B — Ensemble Designer."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import AgentResult, Recipe, SharedState


class EnsembleDesigner(BaseAgent):
    CODE = "B"
    NAME = "EnsembleDesigner"
    ROLE = "N-model 조합 + ensemble 방법(rank/z/prob) 탐색"

    @tool("model_pool에서 size 2~max_size 조합 생성 (rank average default)")
    def search(self, *, state: SharedState, model_pool: list[str],
               max_size: int = 4, top_k: int = 20,
               method: str = "rank_average") -> AgentResult:
        recipes: list[Recipe] = []
        for size in range(2, max_size + 1):
            for combo in combinations(model_pool, size):
                rid = f"{method}__{'_'.join(combo)}"
                recipes.append(Recipe(
                    recipe_id=rid, recipe_type="ensemble",
                    model_pool=list(combo), ensemble_method=method,
                    weights=[1.0] * len(combo),
                    source_agent=self.CODE,
                ))
        # Phase 1: 전체 반환, AUROC ranking은 SIR Evaluator 호출 후
        return self._ok("search",
                        audit={"n_candidates": len(recipes), "method": method,
                               "max_size": max_size},
                        proposals=[r.to_dict() for r in recipes[:top_k * 3]])

    @tool("recipe를 per-row df에 적용 → score 컬럼 추가")
    def apply(self, *, state: SharedState, recipe: dict,
              per_row_csv: str) -> AgentResult:
        df = pd.read_csv(per_row_csv, low_memory=False)
        cols = [f"pred_{m}" for m in recipe["model_pool"]
                if f"pred_{m}" in df.columns]
        if not cols:
            return self._fail("apply", "model_pool에 해당하는 pred_* 컬럼 없음")
        if recipe["ensemble_method"] == "rank_average":
            score = df[cols].rank(pct=True).mean(axis=1)
        elif recipe["ensemble_method"] == "prob_mean":
            score = df[cols].mean(axis=1)
        else:
            return self._fail("apply", f"method 미구현: {recipe['ensemble_method']}")
        out_path = Path(per_row_csv).with_suffix(f".{recipe['recipe_id']}.csv")
        df_out = df[["sample_id", "antimicrobial", "concentration_idx_0"]].copy()
        df_out["model_pred"] = score
        df_out.to_csv(out_path, index=False)
        return self._ok("apply", artifacts=[str(out_path)],
                        audit={"recipe_id": recipe["recipe_id"], "n_rows": len(df_out)})

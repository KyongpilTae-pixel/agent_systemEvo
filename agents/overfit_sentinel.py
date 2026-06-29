"""Agent E — Overfit Sentinel.

Phase 2 wiring: 실제 GroupKFold(5) CV on per-row scores → mean/std/CI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import AgentResult, SharedState, ValidationResult

# Known fragile patterns from prior validation
KNOWN_FRAGILE = [
    {"pattern_id": "per_cell_threshold_tuning",
     "reason": "train↔FDA per-cell optimal threshold Pearson +0.157 (2026-05-19)"},
    {"pattern_id": "per_cell_isotonic_small_n",
     "reason": "n_cell < 20 시 isotonic overfit"},
    {"pattern_id": "per_cell_alpha_small_n",
     "reason": "n_cell < 50 시 α-fit overfit"},
]


class OverfitSentinel(BaseAgent):
    CODE = "E"
    NAME = "OverfitSentinel"
    ROLE = "recipe transferability 검증 + fragile blacklist"

    @tool("recipe validation — known-fragile 매칭 (빠른 sanity check)")
    def validate(self, *, state: SharedState, recipe_id: str,
                 n_folds: int = 5) -> AgentResult:
        for kf in KNOWN_FRAGILE:
            if kf["pattern_id"] in recipe_id:
                vr = ValidationResult(
                    recipe_id=recipe_id, status="fragile",
                    method=f"GroupKFold({n_folds}) (pattern match)",
                    mean_auroc=float("nan"), std=float("nan"), ci_95=[0.0, 0.0],
                    reason=kf["reason"])
                return self._ok("validate", proposals=[vr.__dict__])
        vr = ValidationResult(
            recipe_id=recipe_id, status="unknown",
            method="known-fragile pattern match only",
            mean_auroc=float("nan"), std=float("nan"), ci_95=[0.0, 0.0],
            reason="패턴 매칭 통과. 정확 검증은 validate_cv() 호출.")
        return self._ok("validate", proposals=[vr.__dict__])

    @tool("실제 5-fold GroupKFold CV — per-row score + gt_gng 필요")
    def validate_cv(self, *, state: SharedState, recipe_id: str,
                    per_row_csv: str, score_col: str = "dtw_model_pred",
                    n_folds: int = 5) -> AgentResult:
        p = Path(per_row_csv)
        if not p.exists():
            return self._fail("validate_cv", f"{p} 없음")
        df = pd.read_csv(p, low_memory=False,
                         usecols=["sample_id", "gt_gng", score_col])
        df = df.dropna(subset=["sample_id", "gt_gng", score_col])
        if len(df) < 50:
            return self._fail("validate_cv", f"표본 부족 ({len(df)})")
        gkf = GroupKFold(n_splits=n_folds)
        aucs = []
        for tri, vai in gkf.split(df, df["gt_gng"], df["sample_id"]):
            y = df["gt_gng"].iloc[vai].astype(int).to_numpy()
            s = df[score_col].iloc[vai].astype(float).to_numpy()
            if len(set(y.tolist())) < 2:
                continue
            aucs.append(float(roc_auc_score(y, s)))
        if not aucs:
            return self._fail("validate_cv", "fold 평가 가능 없음")
        mean = float(np.mean(aucs))
        std = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        # 95% CI (정규 근사)
        ci_low = mean - 1.96 * std / np.sqrt(len(aucs))
        ci_high = mean + 1.96 * std / np.sqrt(len(aucs))
        status = "transferable" if mean >= 0.85 and std < 0.05 else (
            "fragile" if std >= 0.05 else "unknown")
        vr = ValidationResult(
            recipe_id=recipe_id, status=status,
            method=f"GroupKFold({len(aucs)})",
            mean_auroc=round(mean, 4), std=round(std, 4),
            ci_95=[round(ci_low, 4), round(ci_high, 4)],
            reason=f"per-fold AUROCs: {[round(a, 4) for a in aucs]}")
        return self._ok("validate_cv", proposals=[vr.__dict__],
                        audit={"recipe_id": recipe_id, "mean": mean,
                               "std": std, "n_folds_eval": len(aucs)})

    @tool("known-fragile 패턴 blacklist 추가")
    def blacklist_pattern(self, *, state: SharedState,
                          pattern_id: str, reason: str) -> AgentResult:
        state.add_to_blacklist(pattern_id, reason)
        return self._ok("blacklist_pattern",
                        audit={"pattern_id": pattern_id, "reason": reason,
                               "n_blacklist": len(state.blacklist)})

    @tool("현재 blacklist 조회")
    def list_blacklist(self, *, state: SharedState) -> AgentResult:
        return self._ok("list_blacklist", audit={"blacklist": state.blacklist or KNOWN_FRAGILE})

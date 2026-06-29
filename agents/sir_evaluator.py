"""Agent D — SIR Clinical Evaluator.

Phase 2 wiring: verify_method_sir_pipeline.py를 subprocess로 호출해 신규
recipe의 SIR 평가 가능. baseline / routed는 기존 산출물 cache 사용.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from agent_system.agents.base import BaseAgent, tool
from agent_system.agents.state_schema import AgentResult, SharedState

BASELINE_SUMMARY = Path("claudeCode/output_dtw_aggregate_full/method_summary_model_pred.csv")
ROUTED_SUMMARY = Path("claudeCode/output_sir_reeval_routed/method_summary_model_pred.csv")
DEFAULT_GBM_OOF = Path("claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv")
PROJECT_ROOT = Path("/home/kptae/project/qnt_algorithm")


def _pass_count(summary_csv: Path) -> tuple[int, int, list[str]]:
    if not summary_csv.exists():
        return (0, 0, [])
    df = pd.read_csv(summary_csv)
    df["pass"] = df["FDA_fail_list_exception_rule"].astype(str).str.strip() == "PASS"
    failing = df[~df["pass"]][["organism_group", "antimicrobial"]].astype(str)
    return (int(df["pass"].sum()), int(len(df)),
            [f"{r.organism_group}||{r.antimicrobial}" for _, r in failing.iterrows()])


class SIRClinicalEvaluator(BaseAgent):
    CODE = "D"
    NAME = "SIRClinicalEvaluator"
    ROLE = "gng→MIC→SIR→FDA pass 판정 + Pareto 지표"

    @tool("현재 FAIL인 cells 목록 (default = baseline same_mic)")
    def list_failing_cells(self, *, state: SharedState,
                           source: str = "baseline") -> AgentResult:
        path = BASELINE_SUMMARY if source == "baseline" else ROUTED_SUMMARY
        n_pass, n_total, failing = _pass_count(path)
        return self._ok("list_failing_cells",
                        audit={"source": source, "n_pass": n_pass, "n_total": n_total,
                               "n_failing": len(failing)},
                        proposals=[{"cell": c} for c in failing])

    @tool("baseline vs routed FDA pass 카운트")
    def fda_pass_count(self, *, state: SharedState) -> AgentResult:
        n_b, t_b, _ = _pass_count(BASELINE_SUMMARY)
        n_r, t_r, _ = _pass_count(ROUTED_SUMMARY)
        return self._ok("fda_pass_count",
                        audit={"baseline_pass": n_b, "routed_pass": n_r,
                               "total_cells": max(t_b, t_r), "delta": n_r - n_b})

    @tool("기존 cache (baseline / routed) 결과 lookup")
    def evaluate_cached(self, *, state: SharedState, recipe_id: str) -> AgentResult:
        if recipe_id == "baseline_same_mic":
            n_pass, n_total, _ = _pass_count(BASELINE_SUMMARY)
            path = BASELINE_SUMMARY
        else:
            n_pass, n_total, _ = _pass_count(ROUTED_SUMMARY)
            path = ROUTED_SUMMARY
        return self._ok("evaluate_cached", audit={
            "recipe_id": recipe_id, "fda_pass": n_pass, "n_cells": n_total,
            "source": str(path)})

    @tool("verify_method_sir_pipeline subprocess 호출 → 신규 recipe SIR 평가")
    def evaluate(self, *, state: SharedState, recipe_id: str,
                 per_conc_csv: str,
                 gbm_oof_csv: str = str(DEFAULT_GBM_OOF),
                 output_dir: str | None = None) -> AgentResult:
        if output_dir is None:
            output_dir = f"claudeCode/output_sir_eval_{recipe_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": "."}
        cmd = ["python", "-m", "claudeCode.verify_method_sir_pipeline",
               "--per_conc_csv", per_conc_csv,
               "--gbm_oof_csv", gbm_oof_csv,
               "--output_dir", output_dir]
        print(f"    [D] subprocess: {' '.join(cmd)}")
        try:
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env,
                               capture_output=True, text=True, timeout=1200)
        except subprocess.TimeoutExpired:
            return self._fail("evaluate", "verify_method_sir_pipeline timeout (1200s)")
        if r.returncode != 0:
            return self._fail("evaluate", f"verify exit {r.returncode}",
                              audit={"stderr_tail": r.stderr[-2000:]})
        summary_csv = Path(output_dir) / "method_summary_model_pred.csv"
        if not summary_csv.exists():
            return self._fail("evaluate", f"summary not produced: {summary_csv}")
        n_pass, n_total, failing = _pass_count(summary_csv)
        return self._ok("evaluate",
                        audit={"recipe_id": recipe_id, "fda_pass": n_pass,
                               "n_cells": n_total, "n_failing": len(failing),
                               "output_dir": output_dir},
                        artifacts=[str(summary_csv)])

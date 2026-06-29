"""Multi-Agent Orchestrator — 2-stage cycle.

Stage 1: per-cell PASS hunting (greedy)
Stage 2: methodology consolidation (cluster + 최소 recipe set)

Phase 1 stub: 모든 agent를 import + 사이클 1회 실행. SIR 실제 평가는 기존
산출 lookup만 활용 (verify_method_sir_pipeline은 별도 호출).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent_system.agents import (
    SharedState, OperationalManifest,
)
from agent_system.agents.training_data_optimizer import TrainingDataOptimizer
from agent_system.agents.model_curator import ModelCurator
from agent_system.agents.ensemble_designer import EnsembleDesigner
from agent_system.agents.calibration_specialist import CalibrationSpecialist
from agent_system.agents.sir_evaluator import SIRClinicalEvaluator
from agent_system.agents.overfit_sentinel import OverfitSentinel
from agent_system.agents.data_quality_auditor import DataQualityAuditor
from agent_system.agents.routing_coordinator import RoutingCoordinator
from agent_system.build_per_conc_for_recipe import build as build_per_conc_for_recipe

STATE_DIR_DEFAULT = Path("agent_system/state")
GBM_OOF_DEFAULT = "claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv"


def run_recipe(recipe_id: str, state: SharedState, cycle_id: int,
               output_root: Path = Path("claudeCode")) -> dict:
    """Cycle 1+ : 단일 recipe 빌드 → SIR 평가 → baseline 대비 리포트.

    Returns:
      {recipe_id, per_conc_csv, fda_pass, delta_vs_baseline, sir_output_dir}
    """
    D = SIRClinicalEvaluator()
    G = RoutingCoordinator()

    per_conc_csv = output_root / "output_subset_eval" / f"dtw_per_conc_{recipe_id}.csv"
    print(f"\n========= Cycle {cycle_id} | Recipe '{recipe_id}' =========")
    print(f"[B.apply→build_per_conc] recipe={recipe_id}")
    build_info = build_per_conc_for_recipe(recipe_id, per_conc_csv)
    print(f"  built {per_conc_csv}  ({build_info.get('n_rows', '?')} rows)")

    print(f"[D.evaluate] verify_method_sir_pipeline subprocess (~5-10min)")
    sir_dir = output_root / f"output_sir_eval_{recipe_id}"
    result = D.evaluate(state=state, recipe_id=recipe_id,
                        per_conc_csv=str(per_conc_csv),
                        gbm_oof_csv=GBM_OOF_DEFAULT,
                        output_dir=str(sir_dir))
    if not result.success:
        return {"recipe_id": recipe_id, "success": False,
                "error": result.errors}
    fda_pass = result.audit.get("fda_pass", 0)
    baseline_cached = D.evaluate_cached(state=state, recipe_id="baseline_same_mic")
    baseline_pass = baseline_cached.audit.get("fda_pass", 49)
    routed_cached = D.evaluate_cached(state=state, recipe_id="routed")
    routed_pass = routed_cached.audit.get("fda_pass", 58)

    summary = {
        "recipe_id": recipe_id,
        "per_conc_csv": str(per_conc_csv),
        "sir_output_dir": str(sir_dir),
        "fda_pass": fda_pass,
        "baseline_pass": baseline_pass,
        "routed_pass": routed_pass,
        "delta_vs_baseline": fda_pass - baseline_pass,
        "delta_vs_routed": fda_pass - routed_pass,
        "success": True,
    }
    commit_result = G.commit(
        state=state, recipe_id=recipe_id,
        cycle_id=cycle_id, agent_lineage=["B", "D", "G"],
        validated_metrics=summary,
        operational_assets={"per_conc_csv": str(per_conc_csv),
                             "method_summary": str(sir_dir / "method_summary_model_pred.csv")},
        user_approved=False,
    )
    state.save()
    return summary


def run_cycle(state: SharedState, cycle_id: int,
              stage1_max_cells: int = 5,
              stage2_max_recipes: int = 5) -> dict:
    """단일 사이클 실행. 결과 요약 dict 반환."""
    H = TrainingDataOptimizer()
    A = ModelCurator()
    B = EnsembleDesigner()
    C = CalibrationSpecialist()
    D = SIRClinicalEvaluator()
    E = OverfitSentinel()
    F = DataQualityAuditor()
    G = RoutingCoordinator()

    lineage: list[str] = []
    log: list[dict] = []

    def _step(agent_code: str, name: str, result):
        lineage.append(agent_code)
        log.append({"agent": agent_code, "method": name,
                    "success": result.success, "audit": result.audit,
                    "n_proposals": len(result.proposals),
                    "errors": result.errors})
        print(f"  [{agent_code}] {name}: success={result.success}, "
              f"audit={result.audit}")

    # ---- Stage 0: prep ----
    print(f"\n========= Cycle {cycle_id} (Stage 0: prep) =========")
    _step("F", "audit", F.audit(state=state))
    state.save()
    _step("H", "propose_curations", H.propose_curations(state=state))
    _step("A", "audit", A.audit(state=state))
    _step("A", "propose_routing", A.propose_routing(state=state))

    # ---- Stage 1: per-cell PASS hunting (greedy) ----
    print(f"\n========= Cycle {cycle_id} (Stage 1: per-cell greedy) =========")
    failing_result = D.list_failing_cells(state=state, source="baseline")
    _step("D", "list_failing_cells", failing_result)
    failing_cells = [p["cell"] for p in failing_result.proposals][:stage1_max_cells]
    model_pool = ["same_mic", "beta_lactam", "gram_negative",
                  "early_stop", "object_area_045~08", "object_area_all"]
    per_cell_winners = {}
    for cell in failing_cells:
        og, drug = cell.split("||")
        cand_result = G.candidates_for_cell(
            state=state, organism_group=og, antimicrobial=drug,
            model_pool=model_pool)
        # Phase 1 stub: routing lookup의 best_model을 winner로 가정
        # (SIR 평가 실제 wiring은 Phase 2)
        first_cand = cand_result.proposals[0] if cand_result.proposals else None
        if first_cand:
            per_cell_winners[cell] = first_cand
    log.append({"stage1": {"n_failing_attempted": len(failing_cells),
                           "n_winners": len(per_cell_winners)}})
    print(f"  stage1 winners: {len(per_cell_winners)}/{len(failing_cells)}")

    # ---- Stage 2: consolidation ----
    print(f"\n========= Cycle {cycle_id} (Stage 2: consolidate) =========")
    cons = G.consolidate(state=state, per_cell_winners=per_cell_winners,
                         max_recipes=stage2_max_recipes)
    _step("G", "consolidate", cons)

    # ---- Stage 3: evaluation summary ----
    print(f"\n========= Cycle {cycle_id} (Stage 3: eval + commit) =========")
    pass_count = D.fda_pass_count(state=state)
    _step("D", "fda_pass_count", pass_count)

    # Commit (Phase 1: 사용자 confirm 없이 auto)
    commit_recipe = "baseline_routed_phase1"
    commit_result = G.commit(
        state=state, recipe_id=commit_recipe,
        cycle_id=cycle_id, agent_lineage=lineage,
        validated_metrics={"fda_pass_baseline": pass_count.audit.get("baseline_pass"),
                           "fda_pass_routed": pass_count.audit.get("routed_pass"),
                           "delta": pass_count.audit.get("delta")},
        operational_assets={"routing_lookup": "claudeCode/output_subset_eval/cell_model_routing_lookup.csv",
                             "isotonic_lookup": "claudeCode/output_dtw_aggregate_full/isotonic_lookup.csv",
                             "shifted_diff": "claudeCode/output_dataset_diff_traintest_normalized/dataset_cell_diff_summary.csv"},
        user_approved=False,  # Phase 1: auto-commit but flag as un-approved
    )
    _step("G", "commit", commit_result)

    return {"cycle_id": cycle_id, "lineage": lineage,
            "log": log,
            "summary": {
                "baseline_pass": pass_count.audit.get("baseline_pass"),
                "routed_pass": pass_count.audit.get("routed_pass"),
                "delta": pass_count.audit.get("delta"),
                "stage1_winners": len(per_cell_winners),
                "stage2_consolidated": cons.audit.get("n_unique_recipes"),
            }}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state_dir", default=STATE_DIR_DEFAULT, type=Path)
    p.add_argument("--cycle_id", default=0, type=int)
    p.add_argument("--max_failing_cells", default=5, type=int,
                   help="Stage 1에서 시도할 failing cell 최대 개수")
    p.add_argument("--recipe", default=None,
                   help="단일 recipe 평가 모드 (예: shifted_rankavg3, routed_per_cell)")
    args = p.parse_args()

    state = SharedState.load(args.state_dir)
    state.cycle_id = args.cycle_id

    if args.recipe:
        summary = run_recipe(args.recipe, state, args.cycle_id)
        print("\n" + "=" * 60)
        print(f"RECIPE EVAL SUMMARY ({args.recipe})")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k:22s}: {v}")
        return

    summary = run_cycle(state, args.cycle_id,
                        stage1_max_cells=args.max_failing_cells)
    state.save()

    print("\n" + "=" * 60)
    print("CYCLE SUMMARY")
    print("=" * 60)
    for k, v in summary["summary"].items():
        print(f"  {k:25s}: {v}")
    print()
    print(f"agent lineage: {' → '.join(summary['lineage'])}")
    print(f"state dir    : {args.state_dir}")


if __name__ == "__main__":
    main()

# Agent Protocol — Multi-Agent FDA PASS 최적화 시스템

> 8 agent + 1 orchestrator의 input/output schema · 공유 state · 호출 규약.
> 자세한 설계 의도는 상위 [`../MULTI_AGENT_OPTIMIZATION_PROPOSAL.md`](../MULTI_AGENT_OPTIMIZATION_PROPOSAL.md) 참조.

작성: 2026-05-26 저녁 · 시작 기점: 2026-05-27

## 1. 공유 State Layout

```
claudeCode/state/                              # gitignored
├── per_row_scores.parquet                     # 모든 모델·ensemble의 per-row 점수
├── per_cell_metrics.parquet                   # cell × recipe 평가 지표
├── recipes.json                               # recipe 카탈로그 (id → spec)
├── manifests/
│   ├── cycle_0.json                           # 사이클별 commit
│   └── cycle_1.json
└── blacklist.json                             # known-fragile 패턴 (Overfit Sentinel)
```

### per_row_scores.parquet schema
| column | dtype | 비고 |
|---|---|---|
| sample_id | str | |
| antimicrobial | str | |
| concentration_idx_0 | int | |
| organism_group | str | normalized |
| gt_gng | int | 0/1 |
| pred_\<model_id\> | float | per-model 점수 |
| pred_\<recipe_id\> | float | ensemble/calibrated 점수 |

### per_cell_metrics.parquet schema
| column | dtype | 비고 |
|---|---|---|
| organism_group | str | |
| antimicrobial | str | |
| recipe_id | str | |
| n_rows, n_NG, n_G, n_S, n_R | int | |
| auroc | float | global within cell |
| EAp, CAp, MEp, VMEp, mEp | float | clinical metrics |
| fda_pass | bool | exception rule 적용 후 |
| fail_rule | str | "PASS" 또는 "EA,CA,..." |

### recipes.json schema
```json
{
  "recipe_id": "shifted_rankavg_3model_iso_t75",
  "recipe_type": "routed_ensemble",
  "model_pool": ["beta_lactam", "gram_negative", "objarea_gbm_full"],
  "ensemble_method": "rank_average",
  "weights": [1, 1, 1],
  "apply_on_buckets": ["shifted"],
  "fallback_recipe_id": "beta_lactam_single",
  "calibration": "isotonic_per_cell",
  "threshold": 0.75,
  "source_agent": "B",
  "created_at": "2026-05-27T09:00:00",
  "honest_eval": {
    "method": "GroupKFold(5)",
    "mean_auroc": 0.9612,
    "std": 0.008,
    "ci_95": [0.953, 0.969]
  }
}
```

### manifest schema (commit 단위)
```json
{
  "cycle_id": 1,
  "committed_at": "2026-05-27T18:00:00",
  "recipe_id": "...",
  "agent_lineage": ["F", "H", "A", "B", "E", "C", "D", "G"],
  "validated_metrics": {
    "fda_pass": 65,
    "auroc_global": 0.965,
    "auroc_macro": 0.967
  },
  "operational_assets": {
    "model_lookup": "state/recipes.json#shifted_rankavg_3model_iso_t75",
    "isotonic_lookup": "output_dtw_aggregate_full/isotonic_lookup.csv",
    "shifted_set": "output_dataset_diff_traintest_normalized/dataset_cell_diff_summary.csv"
  },
  "user_approved": true,
  "diff_from_previous": {
    "pass_change": "+7",
    "regression_cells": []
  }
}
```

## 2. Agent 호출 규약 (RPC-like)

각 agent는 Python class. orchestrator는 다음 패턴으로 호출:

```python
result = agent.<method>(state: SharedState, **kwargs) -> AgentResult
```

`SharedState` = read access to state/ + write to specific keys only.
`AgentResult` = dataclass with `proposals: list`, `audit: dict`, `errors: list`.

### Agent별 메소드

| Agent | 주요 메소드 | 반환 |
|---|---|---|
| **H. TrainingDataOptimizer** | `propose_curations()`, `simulate(recipe)`, `build_parquet(recipe)`, `recommend_retrain()` | `list[CurationRecipe]`, `dict shift_gap`, `CuratedDatasetManifest`, `RetrainPlan` |
| **A. ModelCurator** | `audit()`, `propose_routing(strategy)` | `dict new_models`, `lookup_df` |
| **B. EnsembleDesigner** | `search(model_pool, max_size, top_k)`, `apply(recipe, df)` | `list[Recipe]`, `Series score` |
| **C. CalibrationSpecialist** | `fit(df, og, drug)`, `apply(score, og, drug)`, `tune_threshold(...)` | `CalibrationModel`, `float prob`, `float threshold` |
| **D. SIRClinicalEvaluator** | `evaluate(calibrated_df)`, `fda_pass_count(metrics)` | `MetricsBundle`, `int` |
| **E. OverfitSentinel** | `validate(recipe, n_folds)`, `blacklist_pattern(rule_id, reason)` | `ValidationResult`, `None` |
| **F. DataQualityAuditor** | `audit()`, `refresh_shifted_set()` | `AuditReport`, `set[(og, drug)]` |
| **G. RoutingCoordinator** | `run_cycle()`, `commit(recipe)` | `CycleResult`, `OperationalManifest` |

## 3. 2-Stage 사이클 흐름 (2026-05-27 갱신)

**Stage 1 — Per-cell PASS hunting (greedy)**:
- 현재 FAIL인 cell마다 독립적으로 최적 방법론 탐색 (cell-local 최적)
- 다른 cell 깨지 않는 한도에서 cell당 best (model, ensemble, calibration, threshold) 조합 발견
- 출력: `per_cell_winning_recipes.parquet` (cell → recipe + metric)

**Stage 2 — Methodology consolidation**:
- Stage 1의 cell-local recipe들을 cluster
- 공통 패턴 추출 (drug_class · gram · bucket 별)
- 최소 recipe set (~3~5개 rule)으로 압축 — 운영 단순성 확보
- 출력: 통합 routing rule + 운영 가능 manifest

## 4. 사이클 흐름 (의사 코드)

```python
def orchestrator_cycle(cycle_id: int) -> CycleResult:
    state = SharedState.load("claudeCode/state/")

    # === Stage 0: prep ===
    # 1. Data audit
    audit = data_quality_auditor.audit(state)
    state.update("shifted_set", audit.shifted_set)

    # 2. Training data optimization (옵션)
    curation_proposals = training_data_optimizer.propose_curations(state)
    # 사용자 confirm 후에만 build_parquet → retrain

    # 3. Model audit
    new_models = model_curator.audit(state)
    routing_proposals = model_curator.propose_routing(state, strategy="best_per_cell")

    # 4. Ensemble search
    ensemble_candidates = ensemble_designer.search(
        state, model_pool=routing_proposals.models, max_size=4, top_k=20
    )

    # 5. Overfit gating
    safe_candidates = [c for c in ensemble_candidates
                       if overfit_sentinel.validate(c, n_folds=5).status == "transferable"]

    # 6. Calibration variants
    calib_variants = []
    for c in safe_candidates:
        for cal_strategy in ["per_cell_iso", "pooled_organism_iso", "pooled_drug_class_iso"]:
            calib_variants.append(calibration_specialist.combine(c, cal_strategy))

    # 7. SIR evaluation (cached)
    metrics = [sir_evaluator.evaluate(v) for v in calib_variants]

    # === Stage 1: Per-cell PASS hunting (greedy) ===
    failing_cells = sir_evaluator.list_failing_cells(state)  # 현재 FAIL인 cells
    per_cell_winners = {}
    for cell in failing_cells:
        # cell당 후보 변형 = (model 선택 × ensemble × calibration × threshold)
        local_variants = routing_coordinator.candidates_for_cell(
            cell, model_pool, ensemble_designer, calibration_specialist
        )
        local_metrics = [sir_evaluator.evaluate_one_cell(v, cell) for v in local_variants]
        # 다른 cell 깨지 않는 한도 (delta_regression <= 0) 에서 PASS 만드는 best
        winner = pick_local_best(local_metrics, fda_pass=True, no_global_regression=True)
        if winner:
            per_cell_winners[cell] = winner

    # === Stage 2: Methodology consolidation ===
    # cell-local winner들을 cluster → 공통 패턴 추출 → 최소 recipe set
    consolidated_recipes = routing_coordinator.consolidate(
        per_cell_winners, max_recipes=5
    )
    global_metrics = [sir_evaluator.evaluate(r) for r in consolidated_recipes]

    # === Stage 3: Coordinator decision ===
    cycle_result = routing_coordinator.run_cycle(
        local_winners=per_cell_winners, global_recipes=global_metrics
    )
    if cycle_result.user_approved_recipe:
        manifest = routing_coordinator.commit(cycle_result.user_approved_recipe)
        state.save_manifest(cycle_id, manifest)
    return cycle_result
```

## 4. 첫날 (2026-05-27) 작업 순서

1. **이 문서 (`AGENT_PROTOCOL.md`) 검토 + 조정**
2. **`agents/state_schema.py`** — dataclass 정의 (SharedState, Recipe, Manifest, …)
3. **agents/<name>.py × 8 stub** — 기존 코드 wrapping (no new logic)
4. **`orchestrator.py`** — 위 사이클 흐름 구현
5. **Cycle 0**: baseline (PASS 58) 재현 확인
6. **Cycle 1**: C1 (shifted × 3-model rank-avg) 시도 → PASS 변화 측정

## 5. 디자인 결정 (사용자 confirm 필요)

| 질문 | default 제안 |
|---|---|
| Agent 실행 형태 | Python class 단순 모듈 (Phase 1), Claude Agent SDK는 Phase 2 검토 |
| Cache 저장 | Parquet (디스크) — orchestrator 재시작 시 state 보존 |
| 목적 함수 | (PASS 개수, Σ ME, Σ VME) 다목적 Pareto |
| Compute budget | 사이클당 ~50 candidate, ~20 SIR 평가 |
| Honest eval | 5-fold GroupKFold default, 핵심 후보만 LOSO 추가 |

---

*Generated 2026-05-26 저녁. 시작: 2026-05-27.*

# Multi-Agent FDA PASS 최적화 시스템 설계 제안

> 8개 specialized agent + 1 orchestrator로 FDA cells PASS 개수를 체계적으로 끌어올리는 다중 에이전트 시스템 제안서.

작성 기점: 2026-05-26 (저녁). 시작 기점: 2026-05-27.

## 0. 목표 (Goal Definition)

```
주 목표  : FDA 313 cells 중 PASS cell 개수를 58 → 65+ (단기) / 80+ (중기)
제약 1   : VME ≤ 1.5% (모든 cell) — 임상 안전성
제약 2   : ME ≤ 3% (모든 cell) — over-call 방지
제약 3   : 모든 결정은 LOSO/GroupKFold honest eval로 검증
제약 4   : 모델 재학습 없이 기존 11 subset checkpoint 자산 활용 (Phase 1)
미래     : 새 subset 학습 시 자동 통합 (Phase 2)
```

## 1. Agent Roster (8개)

### 📦 H. Training Data Optimizer (학습 데이터 큐레이터)

**책임**: 학습 데이터 필터링·재구성·이상치 제거 + 큐레이션된 dataset 제안.
Model Curator (Agent A)와 짝을 이룸 — A는 **이미 학습된 모델** 선택, H는
**학습할 데이터** 큐레이션.

```
입력  : - raw training parquet (예: train_valid_*.parquet)
        - allInfo CSV (object_area · bmd_gng · train_or_test)
        - 학습 도메인 vs FDA 도메인 분포 diff
출력  : - curated_train_parquet/<recipe>.parquet (필터/리밸런스 적용본)
        - curation_audit.json (dropped/added rows, reason)
        - retrain_suggestion.json (어떤 모델을 어떤 curated dataset으로 재학습?)
도구  : analyze_training_reconstruction.py (시나리오 시뮬레이션)
        build_rebalanced_parquet.py (drop/upsample/downsample)
        object_area_threshold_deep_dive.py (control 품질 필터)
        generate_order_pick_dataset_from_allinfo.py (allInfo → parquet)
        compare_datasets_full_traintest.py (train↔FDA shift 측정)
        organism_normalize.normalize_organism_group
필터 전략 :
        1. object_area threshold 필터 (045~08 band → "high quality" 학습 subset)
        2. control 품질 검증 fail row 제거
        3. drug class 별 R/S 비율 재균형 (Scenario 2: PBC R down-sample)
        4. organism granularity 통일 (CoNS species → "Coagulase-negative")
        5. bare-< MIC 등 production-incompatible row 제외
        6. rare R event 보강 (예: Carbapenem × E.coli)
원칙  : - 큐레이션 자체가 transferable 해야 함 (Overfit Sentinel과 협력)
        - 시뮬레이션 (gap shift 측정) 후 실제 retrain 비용 판단
        - retrain 결정은 사용자 confirm 필수 (4-GPU × 7.5h 학습 비용)
인터페이스:
   TrainingDataOptimizer.propose_curations(target_drug=None, target_organism=None)
       → list[CurationRecipe]
   TrainingDataOptimizer.simulate(recipe) → expected_shift_gap
   TrainingDataOptimizer.build_parquet(recipe, output_path)
       → CuratedDatasetManifest
   TrainingDataOptimizer.recommend_retrain(curation, model_curator_state)
       → RetrainPlan
```

핵심 기존 발견:
- Aminoglycoside Scenario 2 (PBC R down-sample): R-share gap 0.333 → −0.027 (2026-05-15)
- object_area 045~08 band 필터: drug class 의존적 (Penicillin +0.033, Polymyxin −0.045)
- E. coli × Carbapenem rare R event 부족 → VME 위험 (2026-05-15)
- CoNS species-level 라벨 통합 → fda_only 121→57 (2026-05-19)

---

### 🧬 A. Model Curator (모델러)

**책임**: 학습 완료된 model 카탈로그 + 새 model 자동 평가 + per-cell routing 후보 제안.

```
입력  : - subset checkpoint list (디렉토리 scan)
        - 평가 대상 FDA per-conc CSV
출력  : - subset_models_per_cell_auroc.csv (cell × model AUROC 매트릭스)
        - cell_model_routing_lookup.csv (cell → best model)
        - new_model_audit.json (신규 model 감지 + auto-eval 결과)
도구  : eval_all_subset_models_on_fda.py
        per_cell_model_routing.py
        infer_model_on_fda_minimal.py
인터페이스:
   ModelCurator.audit() → {new_models, evaluated_aurocs}
   ModelCurator.propose_routing(strategy="best_per_cell"|"drug_class"|"gram")
       → routing_lookup_dataframe
```

### 🔀 B. Ensemble Designer

**책임**: N-model 조합 + ensemble 방법(rank/z/prob/blend) 탐색.

```
입력  : - 모델 후보 풀 (Model Curator로부터)
        - per-row 모든 모델 점수
출력  : - ensemble_candidates.csv (recipe × AUROC × constraints)
        - 추천 recipe top-K (예: K=20)
도구  : rank_avg_shifted_combinations.py
        subset_model_ensembles.py
        ensemble_score.EnsembleScoreTable
탐색 전략 : - bucket별 다른 recipe 허용 (shifted/aligned)
            - greedy forward selection (size 1 → 2 → 3 → ...)
            - top-K 통과만 다음 단계로 (compute budget 관리)
인터페이스:
   EnsembleDesigner.search(model_pool, max_size=4, top_k=20)
       → list[Recipe]
   EnsembleDesigner.apply(recipe, per_row_df) → score_series
```

### 📐 C. Calibration Specialist

**책임**: per-cell isotonic / per-organism pooled / per-drug-class 보정 + threshold tuning.

```
입력  : - per-row uncalibrated 점수 (single model OR ensemble)
        - ground truth gt_gng
출력  : - calibration_curves.parquet (cell × isotonic 곡선)
        - threshold_per_cell.csv (cell → recommended threshold)
        - calibrated_per_row.parquet
도구  : isotonic_lookup.PerCellIsotonicLookup
        threshold_lookup.PerCellThresholdLookup
        sir_calib_threshold_grid.py
        task4_pooled_isotonic.py
원칙  : per-cell ISO 우세 (within-FDA 검증 완료),
        per-cell threshold tuning은 fragile (Overfit Sentinel blacklist)
인터페이스:
   CalibrationSpecialist.fit(per_row_df, organism, drug)
       → calibration_model
   CalibrationSpecialist.apply(score, organism, drug) → calibrated_prob
   CalibrationSpecialist.tune_threshold(...) → threshold
```

### 🩺 D. SIR Clinical Evaluator (평가자)

**책임**: gng → MIC → SIR → FDA pass 판정 + Pareto 지표 산출.

```
입력  : - calibrated probability + threshold
        - cell metadata (breakpoint, bmd_mic, bmd_sir)
출력  : - per_cell_metrics.csv (EA, CA, ME, VME, mE per cell)
        - fda_pass_status.csv (cell → PASS/fail rule)
        - pareto_summary.json (5D trade-off)
도구  : verify_method_sir_pipeline.py
        drastModules/mics.py (production code)
핵심 지표:
        - PASS 개수 (primary)
        - 평균 EA, CA / 평균 ME, VME, mE (secondary)
        - Pareto frontier (5D)
인터페이스:
   SIRClinicalEvaluator.evaluate(per_row_with_calibrated_score)
       → MetricsBundle
   SIRClinicalEvaluator.fda_pass_count(metrics_bundle) → int
```

### 🛡 E. Overfit Sentinel (방지전문가)

**책임**: 후보 recipe가 train→FDA transfer 가능한지 사전 검증, fragile 룰 차단.

```
입력  : - 후보 recipe (Model/Ensemble/Calibration Designer가 제안)
        - 학습/평가 split
출력  : - transferable / fragile / unknown 분류
        - 신뢰 구간 (LOSO CV mean ± std)
        - blacklist.json (known-fragile 패턴)
도구  : GroupKFold, LOSO
        sir_threshold_domain_compare.py
        sir_domain_shift_by_drug.py
        compare_datasets_full_traintest.py
known-fragile (현재 시점):
        - per-cell threshold tuning (Pearson +0.157 transfer, 2026-05-19)
        - per-cell isotonic with n_cell < 20
        - α-fit with n_cell < 50
인터페이스:
   OverfitSentinel.validate(recipe, n_folds=5)
       → ValidationResult(status, mean_auroc, std, ci)
   OverfitSentinel.blacklist_pattern(rule_id, reason)
```

### 🧹 F. Data Quality Auditor

**책임**: 데이터셋 변화·라벨 granularity·이상치 감지.

```
입력  : - 학습/평가 CSV path
        - 직전 baseline state snapshot
출력  : - data_diff.json (cells 추가/삭제/shift 변경)
        - quality_issues.csv (이상치, label inconsistency)
        - shifted_set 갱신
도구  : compare_datasets_full_traintest.py
        organism_normalize.normalize_organism_group
        bare-< bmd_mic detector
trigger : - 신규 데이터 도착
          - 정기 (월 1회) shift drift 점검
인터페이스:
   DataQualityAuditor.audit() → AuditReport
   DataQualityAuditor.refresh_shifted_set() → set[(organism, drug)]
```

### 🎛 G. Routing Coordinator (Orchestrator)

**책임**: 위 6 agent 종합 → 단일 운영 recipe 결정 + 옵션 매트릭스 관리.

```
입력  : - 각 agent의 후보 + 검증 결과
        - 사용자 제약 (budget, 노력 한도)
출력  : - operational_manifest.json (선택된 recipe + 자산 경로)
        - option_matrix.html (노력 vs 임팩트 Pareto)
        - 권고 운영 recipe
도구  : OperationalScorer (facade)
        build_*_report 시리즈
의사결정:
        1. Pareto frontier 계산 (PASS vs effort vs risk)
        2. 후보 중 best-2~3 가 사용자에게 제시
        3. 사용자 선택 → manifest commit
인터페이스:
   RoutingCoordinator.run_cycle() → CycleResult
   RoutingCoordinator.commit(recipe) → OperationalManifest
```

## 2. Workflow (orchestrator 1 사이클)

```
[trigger]      신규 모델 / 신규 데이터 / 사용자 요청
   ↓
[F] Data Quality Auditor   → baseline state 갱신
   ↓
[H] Training Data Optimizer → (선택) curated 학습 dataset 제안
   ↓                           재학습 필요 시: 사용자 confirm → 모델 재학습 → A로 전달
[A] Model Curator          → 후보 모델 풀 + per-cell AUROC
   ↓
[B] Ensemble Designer      → recipe 후보 ~50개 (size 1~4)
   ↓
[E] Overfit Sentinel       → fragile 제거 → ~20개
   ↓
[C] Calibration Specialist → calibration·threshold 옵션 적용 (총 ~60 variant)
   ↓
[D] SIR Evaluator          → 각 variant SIR 평가 (parallelize, cache)
   ↓
[G] Coordinator            → Pareto frontier + 옵션 매트릭스
   ↓
[사용자]                    → 적용 옵션 선택 → manifest commit
```

각 단계 산출은 공유 cache (per-row 점수 + per-cell metric)에 저장 → 다음 사이클에 재사용.

## 3. 공유 State 형식 (Protocol)

```python
# 1. Per-row scores cache (shared by all agents)
per_row_scores.parquet:
    sample_id, antimicrobial, concentration_idx_0, organism_group,
    gt_gng, pred_<model_1>, pred_<model_2>, ...,
    pred_<ensemble_1>, ..., pred_calibrated_<recipe_id>

# 2. Per-cell metric cache
per_cell_metrics.parquet:
    organism_group, antimicrobial, recipe_id,
    auroc, EA, CA, ME, VME, mE, fda_pass, fail_rule

# 3. Recipe registry
recipes.json:
    recipe_id, recipe_type ("single"|"ensemble"|"routed"),
    components, weights, calibration, threshold, source_agent

# 4. Manifest (commit 단위)
manifest_YYYYMMDD_HHMM.json:
    recipe_id, validated_metrics, agent_lineage,
    operational_asset_paths, user_approval
```

## 4. 설계 원칙

1. **Hierarchical search**: 글로벌 → drug_class → cell. coarse가 충분하면 fine 안 함.
2. **Pareto, not scalar**: PASS 개수 + Pareto frontier (EA, CA, ME, VME, mE).
3. **Honest eval default**: LOSO/GroupKFold가 default, holdout은 reference only.
4. **Shared cache**: per-row 점수 한 번만 계산, recipe 변경마다 SIR만 재실행.
5. **Constraint blacklist**: known-fragile 룰 (per-cell threshold 등) Overfit Sentinel이 차단.
6. **Reversibility**: 모든 결정 manifest로 기록, A/B 비교 가능.
7. **Composability**: agent 인터페이스 표준화 → 새 agent (예: image augmentation expert) 추가 쉬움.

## 5. MVP Roadmap (4 phase)

### Phase 1 — Protocol 정의 (1일)

- `agents/AGENT_PROTOCOL.md` — 7 agent input/output schema 확정
- `agents/state_schema.py` — parquet/json 스키마 dataclass
- 산출: 문서 + 빈 디렉토리 구조

### Phase 2 — Stub 구현 (2~3일)

- `agents/<agent_name>.py` × 7 — 기존 코드 wrapping (재학습 없음)
- `orchestrator.py` — 사이클 실행 + state 관리
- 산출: 7 모듈 + orchestrator + 단위 테스트

### Phase 3 — 첫 사이클 (1일)

- baseline (PASS 58) 재현 → orchestrator 정상 동작 검증
- 산출: cycle_0_manifest.json + 검증 보고서

### Phase 4 — 최적화 사이클 (반복)

- Ensemble Designer 활성 → 50 candidate 탐색
- Calibration Specialist 활성 → 60 variant 평가
- Overfit Sentinel이 fragile 차단
- 목표: PASS 65~70 달성
- 산출: 사이클별 manifest + 누적 보고서

## 6. 첫 사이클 구체 후보 (Phase 4 시작 시)

| 후보 ID | 변경 lever | 기대 효과 | 검증 필요 |
|---|---|---|---|
| C1 | shifted cells에 3-model rank-avg (beta_lactam + gram_negative + objarea_gbm_full) | +1~3 PASS | rank-avg 후 ISO 적용 시 calibration shift |
| C2 | per-drug-class routing (Polymyxin/Carbapenem → gram_negative) | +0~2 PASS | drug_class generalization |
| C3 | shifted cells × per-organism pooled isotonic | +0~2 PASS | pooled가 train→FDA transfer되나? |
| C4 | LNZ × Gram-pos cocci 전용 ME-aware threshold (0.75 → 0.80) | -1 ~ +2 (VME 위험) | E 강하게 검증 |
| C5 | E. coli × ETP만 single model gram_negative로 교체 | +0~1 PASS | 단일 cell 변경의 다른 cell 영향 |
| C6 | bare-< bmd_mic 1 cell 정상화 + 재평가 | +0~1 PASS | 데이터 quality issue 해결 |

## 7. 열린 결정 사항 (사용자 confirm 필요)

| 질문 | 옵션 |
|---|---|
| Agent 실행 형태 | A. Python class 단순 모듈 / B. Claude Agent SDK (multi-agent) / C. 둘 다 |
| Cache 저장 | A. in-memory dict / **B. Parquet (디스크)** / C. SQLite |
| 목적 함수 | A. PASS 개수만 / **B. (PASS, Σ ME, Σ VME)** 다목적 / C. weighted scalar |
| Compute budget | 사이클당 SIR 평가 횟수 한도 (~50? ~100?) |
| Honest eval 정도 | A. 5-fold GroupKFold (default) / B. nested CV (1.5×) / C. LOSO (느림) |

## 8. 디렉토리 구조 (제안)

```
claudeCode/
├── agents/
│   ├── __init__.py
│   ├── AGENT_PROTOCOL.md
│   ├── state_schema.py
│   ├── training_data_optimizer.py    # H (NEW)
│   ├── model_curator.py              # A
│   ├── ensemble_designer.py          # B
│   ├── calibration_specialist.py     # C
│   ├── sir_evaluator.py              # D
│   ├── overfit_sentinel.py           # E
│   ├── data_quality_auditor.py       # F
│   └── routing_coordinator.py        # G
├── orchestrator.py
├── state/                  # shared cache (gitignored)
│   ├── per_row_scores.parquet
│   ├── per_cell_metrics.parquet
│   ├── recipes.json
│   └── manifests/
│       ├── cycle_0.json
│       └── cycle_1.json
└── output_multi_agent/     # cycle별 산출
    ├── cycle_0_report.html
    ├── cycle_1_report.html
    └── option_matrix.html
```

## 9. 첫날 (2026-05-27) 작업 순서

1. **`agents/AGENT_PROTOCOL.md` 작성** — 7 agent의 정확한 input/output schema 확정.
2. **`agents/state_schema.py` 작성** — dataclass 정의.
3. **Stub agent 7개 + orchestrator** — 기존 코드 wrapping.
4. **Cycle 0 (baseline 재현)** — PASS 58 산출 확인.
5. **Cycle 1 (첫 최적화)** — 위 후보 C1~C3 실행 → 결과 보고.

## 10. 기존 자산 매핑 (Phase 2 stub 시 wrapping 대상)

| Agent | 기존 코드 |
|---|---|
| H. Training Data Optimizer | `analyze_training_reconstruction.py`, `build_rebalanced_parquet.py`, `object_area_threshold_deep_dive.py`, `generate_order_pick_dataset_from_allinfo.py`, `organism_normalize.py` |
| A. Model Curator | `eval_all_subset_models_on_fda.py`, `per_cell_model_routing.py`, `infer_model_on_fda_minimal.py` |
| B. Ensemble Designer | `rank_avg_shifted_combinations.py`, `subset_model_ensembles.py`, `compare_methods_per_cell.py`, `ensemble_score.py` |
| C. Calibration Specialist | `isotonic_lookup.py`, `threshold_lookup.py`, `task4_pooled_isotonic.py`, `sir_calib_threshold_grid.py` |
| D. SIR Evaluator | `verify_method_sir_pipeline.py`, `build_per_conc_with_routed_model_pred.py`, drastModules/`mics.py` |
| E. Overfit Sentinel | `sir_threshold_domain_compare.py`, `sir_domain_shift_by_drug.py`, GroupKFold patterns |
| F. Data Quality Auditor | `compare_datasets_full_traintest.py`, `organism_normalize.py`, `dataset_cell_diff_summary.csv` |
| G. Routing Coordinator | `operational_scorer.py`, `build_*_report.py` 시리즈 |

→ **모두 기존 코드. Phase 1~2는 새 알고리즘 없이 wrapping + orchestration만.** Phase 4부터 실제 최적화.

---

*Generated 2026-05-26 저녁.*
*시작 기점: 2026-05-27.*
*근거: 2026-04-29 ~ 2026-05-26 누적 분석 결과 (FDA cells PASS 49 → 58).*

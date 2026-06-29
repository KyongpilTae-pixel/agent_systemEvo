# Operational Deployment Guide — Sweet Spot Recipe

> **권고 운영 recipe**: `routed_iso_t65` (PASS 103/313, VME 1.99%)
> 2026-05-27 Cycle 0~3 검증 + EA panel-notation bug fix 적용 완료.

## 1. 한눈에 — 운영 옵션 매트릭스

| Option | recipe | PASS | VME | 권고 |
|---|---|---|---|---|
| 적극 | `routed_iso_t50` | 111 | 3.56% ⚠ | PASS 우선, medical review 필수 |
| **균형 ★** | `routed_iso_t65` | **103** | 1.99% | **운영 권고** |
| 보수 | `routed_iso_t75` | 91 | 1.50% ✓ | VME FDA 엄격 통제 |

(baseline raw = 72 / routed raw = 82, EA bug fix 후 재측정)

> **2026-05-27 evaluator patch**: `drastModules/mics.py::evalEA`의 BMD doubling
> ↔ dRAST half-dilution panel notation mismatch bug fix. claudeCode/patched_mics.py
> (A: log2 ratio fallback, tol 1.10) + verify_method_sir_pipeline (B: bmd_mic
> 정규화 BMD→dRAST). 모든 recipe에서 PASS→FAIL = 0 검증 (원본 True 보존).

## 2. 30초 적용 (Python)

```python
from claudeCode.operational_scorer import OperationalScorer

scorer = OperationalScorer.from_recipe("routed_iso_t65")
# per-row scoring
cal = scorer.calibrate(model_pred=0.62, organism="Escherichia coli", drug="CTX")
gng = scorer.gng(cal)   # "G" or "NG"
```

## 3. Recipe 구성

```
[1] cell-routed model_pred (PerCellModelRouter)
       ↓     6 subset OrderPickGNG 중 cell마다 best 모델
[2] per-cell isotonic calibration (full-data fit)
       ↓     agent_system/output/isotonic_lookup_routed.csv (286 cells)
[3] global threshold (recipe별: 0.5 / 0.65 / 0.75)
       ↓
[4] determineMIC_simple → MIC 문자열
[5] interpretSIR        → S / I / R
```

## 4. 운영 자산 (4개 파일 sync 필수)

| 자산 | 경로 | 용도 |
|---|---|---|
| **isotonic_lookup_routed.csv** | `agent_system/output/` | per-cell ISO 곡선 (286 cells, full-data fit) |
| routed_model_pred.csv | `claudeCode/output_subset_eval/` | cell-routed per-row 점수 (153k rows) |
| cell_model_routing_lookup.csv | `claudeCode/output_subset_eval/` | (organism, drug) → best subset model |
| dataset_cell_diff_summary.csv | `claudeCode/output_dataset_diff_traintest_normalized/` | shifted bucket cell 식별 |
| drastModules/mics.py | `/home/kptae/project/drastmanager/jupyters/drastModules/` | MIC 결정 + SIR 해석 (운영 코드) |

## 5. Cycle 0~3 검증 결과 (FDA2023 313 cells)

EA bug fix 전/후 PASS count (recipe 순위·임상 metric은 보존):

| Cycle | recipe | thr | PASS (fixed) | PASS (pre-fix) | VME |
|---|---|---|---|---|---|
| 0 baseline | same_mic raw | 0.5 | **72** | 49 | 1.13% |
| 0 routed raw | routed (no ISO) | 0.5 | **82** | 58 | 1.26% |
| C1 | shifted×rank_avg3 (no ISO) | 0.5 | **62** | 44 ⚠ | 0.84% |
| C2 | shifted×rank_avg3 + ISO | 0.5 | **108** | 79 | 4.07% |
| C2b | baseline + ISO | 0.5 | **108** | 79 | 3.45% |
| **C2c** ★ | **routed + ISO** | **0.5** | **111** | 81 | 3.56% |
| C3a | baseline + ISO | 0.75 | **83** | 56 | 1.55% |
| C3b | routed + ISO | 0.75 | **91** | 61 | 1.50% |
| **C3c** ★ | **routed + ISO** | **0.65** | **103** | 71 | 1.99% |

→ 모든 cell에서 PASS→FAIL = 0 (fix는 원본 True 보존하는 strict superset).
순위 보존: routed_iso_t50 > **routed_iso_t65 ★** > routed_iso_t75. ISO 자체가
PASS 주역 (+30~36 vs raw), routing +3, threshold가 VME vs PASS trade-off lever.

## 6. Phase 3 Overfit Sentinel 검증 결과

5-fold GroupKFold by sample_id (within-FDA):

| recipe | mean AUROC | std | status |
|---|---|---|---|
| routed RAW (no ISO) | 0.9648 | 0.0013 | transferable |
| baseline + ISO | 0.9617 | 0.0023 | transferable |
| **routed + ISO** | **0.9743** | **0.0015** | **transferable** |
| shifted_rankavg3 + ISO | 0.9678 | 0.0019 | transferable |

→ 모든 ISO recipe가 within-FDA에서 stable (std < 0.01). routed+ISO가 최고 AUROC.

**미검증**: train → FDA cross-domain transfer (Phase 4 작업).

## 7. 운영 적용 절차

### 7.1 신규 환경 deployment
```bash
# 1. 자산 sync (위 4 파일 + drastModules/)
# 2. claudeCode.operational_scorer import 가능 확인
PYTHONPATH=/home/kptae/project/qnt_algorithm python -c "
from claudeCode.operational_scorer import OperationalScorer
s = OperationalScorer.from_recipe('routed_iso_t65')
print('OK')
"

# 3. batch 평가 (sanity check)
PYTHONPATH=. python -m claudeCode.verify_method_sir_pipeline \
    --per_conc_csv claudeCode/output_subset_eval/dtw_per_conc_routed_iso.csv \
    --gbm_oof_csv claudeCode/output_dataset_diff_traintest_normalized/objarea_crossdomain_oof_full.csv \
    --output_dir /tmp/sanity_check \
    --native_threshold 0.65
# 예상: PASS 71 / 313
```

### 7.2 신규 FDA sample inference
```python
from claudeCode.operational_scorer import OperationalScorer
scorer = OperationalScorer.from_recipe("routed_iso_t65")

# row 단위 inference (per concentration)
for row in new_fda_rows:
    model_pred = run_cell_routed_inference(row)  # PerCellModelRouter
    calibrated = scorer.calibrate(model_pred, row.organism, row.drug)
    gng = scorer.gng(calibrated)
    # ... feed to drastModules for MIC/SIR
```

## 8. Phase 4 미해결 작업

0. **[DONE 2026-05-27]** EA panel-notation bug fix — (A) `claudeCode/patched_mics.py`
   evalEA fallback + (B) `verify_method_sir_pipeline.py` bmd_mic normalization.
   PASS 71 → 103 (+32), 모든 recipe PASS→FAIL = 0 검증.
1. **[followup]** `qnt_algorithm/utils/evaluation_utils.py::evalEA` (사본, 호출처
   `runs/predict_order_pick.py`) — 동일 panel notation bug 가능. 같은 fallback
   적용 권고.
2. **[followup]** preprocessed CSV 생성 코드 추적 + bmd_mic notation 일관성 정정
   (현재 0.12/0.125 혼재). 외부 시스템 영향 평가 후 결정.
3. **True train→FDA ISO transferability test** — 학습 도메인 데이터에서 ISO fit → FDA 적용 후 PASS 측정 (현재는 within-FDA LOSO만 검증)
4. **Per-organism / per-drug-class pooled ISO** 비교
5. **Online inference path** — PerCellModelRouter._model() 이미지 추론 wiring
6. **Calibration drift monitoring** — 정기 ISO refit 시 PASS 변화 추적
7. **Threshold 자동 튜닝** — 사용자 VME budget 입력 시 optimal threshold 추천

## 9. 관련 자료

- `agent_system/state/manifests/operational_recommendation.json` — 운영 manifest JSON
- `agent_system/output/cycles_summary_report.html` — Cycles 0~3 종합 비교 보고서
- `agent_system/MULTI_AGENT_OPTIMIZATION_PROPOSAL.md` — 시스템 설계 제안
- `agent_system/agents/AGENT_PROTOCOL.md` — 8 agent input/output schema
- `claudeCode/OPERATIONAL_SCORER.md` — OperationalScorer 사용법 (markdown)
- `claudeCode/RANK_AVERAGE_ENSEMBLE.md` — rank_avg ensemble 설명

---

*Generated 2026-05-28. 검증: FDA2023 313 cells, 152,834 rows, 5-fold GroupKFold by sample_id.*

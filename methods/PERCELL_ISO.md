# Per-cell Isotonic Calibration — 구현·학습·운영 설명서

> 운영 PASS 의 주역 (**+30**, baseline 72 → 103). methods 파이프라인의 **CALIBRATE** 스테이지.
> 코드 `agent_system/methods/calibrate/isotonic.py::PerCellIsotonic`.

## 1. 무엇 / 왜

모델이 낸 raw NG 점수(`ng_score` = sigmoid(gng_logit), routing 후엔 routed model_pred)는 **확률로서
보정(calibration)이 안 돼 있다**. 같은 0.6 이라도 cell(organism×drug)마다 실제 NG 비율이 다르다.

**Per-cell isotonic** 은 각 `(organism_group, antimicrobial)` cell 단위로 **단조 비감소(monotone
non-decreasing) 계단 함수**를 적합해 `ng_score → P(NG)` 로 보정한다. 이후 전역 threshold(0.65)로
G/NG 이진화 → MIC → SIR → PASS.

핵심 성질:
- **단조** → cell 내 ranking(순서) 보존 ⇒ **AUROC 불변**, 확률 품질(Brier)만 개선.
- **per-cell** → cell 마다 다른 곡선 ⇒ organism×drug 별 모델 신뢰도 차이를 cell-local 하게 흡수.
- isotonic + 전역 threshold 0.65 = **cell 별 실효(raw) threshold** 와 동치(단조라 cut 위치만 이동).
  → per-cell ISO 가 PASS 를 올리는 메커니즘은 "cell 마다 결정 경계를 그 cell 분포에 맞춰 재배치".

## 2. 구현

`PerCellIsotonic(mode, n_folds, lookup_csv, cell_by)` — 기존 검증 클래스
`claudeCode/isotonic_lookup.PerCellIsotonicLookup` 을 재사용하는 thin wrapper(로직 재구현 없음).

```python
PerCellIsotonic(mode="insample")                    # 전체 패널 full-fit + 보정 (배포 방식)
PerCellIsotonic(mode="oof", n_folds=5)              # GroupKFold(sample_id) held-out (정직 평가)
PerCellIsotonic(mode="lookup", lookup_csv=...)      # 동결 자산 로드 (sklearn-free 런타임)
PerCellIsotonic(cell_by="genus")                    # 그룹 단위 genus (배포 현실, → GENUS_ISO.md)
```

내부: sklearn `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` 을 cell 별 적합해
`X_thresholds_`(raw 점수 breakpoint) / `y_thresholds_`(보정 확률)을 저장. 런타임 보정은 sklearn 없이
`np.interp(score, x_thresholds, y_thresholds)`.

조건 미달 cell(`min_rows<30` 또는 클래스당 `<3`)은 학습 제외 → **identity fallback**(raw 유지).

## 3. 학습

### 3.1 보정과 함께 (apply)
```python
from agent_system.methods.recipe import load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

routed = PerCellModelRouting().apply(load_unified())   # ng_score = routed model_pred
cal = PerCellIsotonic("insample").apply(routed)        # cell별 적합 + 보정 → ng_score=P(NG)∈[0,1]
```

### 3.2 학습만 + 배포 자산 동결 (fit_lookup)
보정/평가 없이 **isotonic 적합만** 한다. `ng_score ↔ gt_gng`(0/1 라벨)로 cell 별 학습.
```python
iso = PerCellIsotonic().fit_lookup(routed,
        save_path="agent_system/output/isotonic_lookup_routed.csv")
```
- 운영 배포 자산 = `agent_system/output/isotonic_lookup_routed.csv` (**286 cell + 1 GLOBAL fallback**,
  full-data fit). CLI: `python -m agent_system.save_iso_lookup_for_recipe`.
- CSV 컬럼: `organism_group, antimicrobial, x_thresholds(JSON), y_thresholds(JSON), n_train_rows, n_g, n_ng`.

## 4. 운영(런타임) 보정

```python
from agent_system.methods.recipe import Recipe, load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

rec = Recipe(
    score     = PerCellModelRouting(),                 # cell-routed model_pred
    calibrate = PerCellIsotonic(mode="lookup",
                  lookup_csv="agent_system/output/isotonic_lookup_routed.csv"),
    threshold = 0.65)
rec.evaluate(load_unified())     # → routed_iso_t65, {'pass': 103, ...}
```
런타임은 sklearn 불필요(`np.interp`). 미커버 cell 은 identity. 단일 row 보정은
`OperationalScorer.from_recipe("routed_iso_t65").calibrate(model_pred, organism, drug)`.

## 5. 핵심 결과 · in-sample vs OOF

| | in-sample (배포 방식·낙관) | **OOF (held-out·정직)** |
|---|---|---|
| routed + ISO @0.65 (full 1507) | **103** | 87 |
| routed + ISO @0.65 (clean 1434) | 106 | 95 |

- **ISO 자체가 PASS 주역**: baseline raw 72 → +ISO 103 (routing 기여는 +3, ISO 가 +30).
- **in-sample 은 낙관**: 같은 FDA 로 fit+평가 → cell 내부에서 score↔label 암기. 그 차이(103 vs OOF 87 =
  −16)가 overfit 분. **정직한 일반화 추정은 OOF**.
- threshold 는 VME↔PASS lever: t0.50(적극)/**t0.65(균형★)**/t0.75(보수). → OPERATIONAL_DEPLOYMENT.md.

## 6. 파일

| 파일 | 역할 |
|---|---|
| `calibrate/isotonic.py::PerCellIsotonic` | wrapper (insample/oof/lookup, cell_by, fit_lookup) |
| `claudeCode/isotonic_lookup.py::PerCellIsotonicLookup` | 하부 구현 (fit/save/load/calibrate, np.interp 런타임) |
| `agent_system/save_iso_lookup_for_recipe.py` | 배포 자산 학습·저장 CLI |
| `agent_system/output/isotonic_lookup_routed.csv` | 동결 배포 자산 (286 cell + GLOBAL) |
| `claudeCode/output_dtw_aggregate_full/isotonic_calibration_explainer.html` | 교육 자료(PAVA 원리·toy 예시) |

## 7. 주의 · 한계

- **반드시 FDA 도메인에서 fit**: 학습 도메인(train)에서 fit 한 ISO 를 FDA 로 transfer 하면 실패
  (2026-06-01, in-sample sharpness 탓). cross-domain 비전이 확인됨.
- **per-cell threshold 튜닝과 다름**: ISO 는 cell 내부에서 score↔label 을 PAVA 로 적합(파라미터 1개
  튜닝 아님). 과거 per-cell threshold 직접 최적화는 표본 noise·domain 차이로 실패, global threshold 가 우월.
- **cell 판정 ≠ 행 정확도**: PASS 는 sample 별 MIC→SIR 일치율(EA/VME/ME, FDA 비율 기준 + ±1 dilution)로
  정해진다. ISO 의 진짜 효과는 경계 sample 의 MIC 를 합격선 안으로 옮기는 것 (자세히는
  `flip_report_*.html`).
- **확장**: 그룹 단위를 genus 로 바꾼 변형 = `cell_by="genus"` → [GENUS_ISO.md](GENUS_ISO.md),
  VME 위험 cell 만 organism 폴백 = Hybrid → `hybrid_iso_report.html`.

---
*운영 권고: `routed_iso_t65` = PerCellModelRouting + PerCellIsotonic(insample) + threshold 0.65.*

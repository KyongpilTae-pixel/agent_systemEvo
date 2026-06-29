# methods — FDA PASS 개선 방법론 통합 패키지

dRAST 3.0 MIC 예측의 **FDA2023 임상 평가(PASS/VME)**를 개선하기 위해 개발된 방법론 9종을
운영 파이프라인 **4 스테이지** 공통 인터페이스로 모듈화(+ 평가기 버그수정 1종은 별도). 흩어져
있던 구현(claudeCode/, agent_system/)을 **검증된 기존 클래스를 thin-wrap** 하여 한곳에 모았다
(로직 재구현·회귀 없음).

```
per_conc ─▶ [A.SCORE] ─▶ [B.CALIBRATE] ─▶ [C.THRESHOLD] ─▶ [D.EVALUATE] ─▶ PASS/VME
           점수 선택/변환    score→P(NG)        P(NG)→G/NG        MIC→SIR→EA
```

## 통합 입력 (단일 파일)

모든 방법론은 외부 CSV 를 각자 읽지 않고 **하나의 통합 per_conc**(`data/unified_per_conc.parquet`)
의 source 컬럼만 소비한다.

```bash
python -m agent_system.methods.build_unified_input          # brightness 포함
```

| 컬럼 | 의미 |
|---|---|
| `src_same_mic` | baseline same_mic.pt 점수 |
| `src_routed` | per-cell routed model_pred (★LNZ) |
| `src_objarea_gbm` | object_area cross-domain GBM 예측 |
| `src_object_area` | raw object_area DTW |
| `src_brightness_gbm` | brightness GBM 예측 |
| `bucket` | aligned / shifted / fda_only (domain shift) |
| `ng_score` | **활성 점수** (SCORE/CALIBRATE 가 갱신; No-Growth 지향 점수). 외부 `dtw_model_pred` 와 동일 의미이나 DTW 가 아닌 모델 sigmoid 확률이라 명확히 개명 |

## 빠른 시작

```python
from agent_system.methods import operational_recipe, catalog

print(catalog())                        # 방법론 카탈로그 (중요도 순)
print(operational_recipe().evaluate())  # routed_iso_t65 → {'pass':103, ...}
```

직접 조립:

```python
from agent_system.methods.recipe import Recipe
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

Recipe(score=PerCellModelRouting(),          # ★LNZ 통과
       calibrate=PerCellIsotonic("insample"),
       threshold=0.65).evaluate()            # → PASS 103
```

## 사용 예

### 0. 통합 입력 빌드 (최초 1회 / source 갱신 시)

```bash
python -m agent_system.methods.build_unified_input
# → data/unified_per_conc.parquet (152,834행, 6 source 신호)
```

### 1. 운영 권고 한 줄 평가

```python
from agent_system.methods import operational_recipe

operational_recipe().evaluate()
# {'pass': 103, 'vme_cells': 43, 'vme_rate': 0.0177, 'recipe': 'routed_iso_t65', ...}
```

### 2. PerCellIsotonic — 학습(적합) + 적용 3모드

> 전용 설명서: [PERCELL_ISO.md](PERCELL_ISO.md) (구현·학습·운영·in-sample vs OOF·한계).

`apply(per_conc)`는 활성 점수(`ng_score`)를 cell별 isotonic 으로 P(NG)로 보정한다.
점수↔라벨(`gt_gng`)로 cell마다 `IsotonicRegression`을 적합한다 (경계에서 `dtw_model_pred` 로 번역).

```python
from agent_system.methods.recipe import load_unified
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

per_conc = load_unified()
routed   = PerCellModelRouting().apply(per_conc)          # 활성 점수 = routed model_pred

cal_in  = PerCellIsotonic("insample").apply(routed)        # 전체 패널 적합(배포 방식)
cal_oof = PerCellIsotonic("oof", n_folds=5).apply(routed)  # GroupKFold held-out(정직 평가)
cal_lk  = PerCellIsotonic("lookup").apply(routed)          # 사전 학습 CSV 로드(학습 X)
print(cal_in["ng_score"].describe())                       # → [0,1] P(NG)
```

### 3. iso만 학습 + 배포용 고정 lookup 저장 (sklearn-free 런타임 자산)

보정/평가 없이 **isotonic 적합만** 하려면 `fit_lookup()` 을 쓴다. `ng_score ↔ gt_gng` 로 cell별
학습하고(경계에서 `dtw_model_pred` 로 자동 번역), CSV로 동결하면 런타임은 `np.interp`로만 동작.

```python
cal = PerCellIsotonic()                                   # 기본 insample
iso = cal.fit_lookup(routed)                              # iso만 학습 (278 cells), 보정 X
iso = cal.fit_lookup(routed,                              # 학습 + 배포 자산 저장
                     save_path="agent_system/output/isotonic_lookup_routed.csv")

# 저장한 lookup 으로 보정 (재학습 없이):
PerCellIsotonic(mode="lookup",
                lookup_csv="agent_system/output/isotonic_lookup_routed.csv").apply(routed)
```

> 학습 입력 = `ng_score`(점수) ↔ `gt_gng`(0/1 라벨), cell 단위. 조건 미달 cell(min_rows 30 /
> 클래스당 3)은 학습 제외 → identity fallback. 운영 자산 일괄 생성 CLI:
> `python -m agent_system.save_iso_lookup_for_recipe`.

### 4. 직접 조립 + threshold lever

```python
from agent_system.methods.recipe import Recipe
from agent_system.methods.score.model_routing import PerCellModelRouting
from agent_system.methods.calibrate.isotonic import PerCellIsotonic

for t in (0.50, 0.65, 0.75):                      # 적극 / 균형★ / 보수
    r = Recipe(score=PerCellModelRouting(), calibrate=PerCellIsotonic(), threshold=t).evaluate()
    print(t, r["pass"], round(r["vme_rate"], 4))  # 0.65 → 103 / 0.0177
```

### 5. VME budget 하 threshold 자동 추천 (ThresholdTuner)

```python
from agent_system.methods.threshold.tuner import ThresholdTuner

tt = ThresholdTuner(grid=(0.50, 0.65, 0.75), vme_budget=0.02)
res = tt.tune(score=PerCellModelRouting(), calibrate=PerCellIsotonic())
print(res["recommended"], res["table"])           # VME ≤ 2% 중 PASS 최대 threshold
```

### 6. 방법론 교체 비교 (예: GBM rank-avg는 PASS 기각)

```python
from agent_system.methods.score.rank_avg_ensemble import RankAvgEnsemble

Recipe(score=RankAvgEnsemble(w_gbm=0.3),          # routed + object_area GBM rank 평균
       calibrate=PerCellIsotonic("oof"), threshold=0.65).evaluate()
# → pass 83 (routed OOF 87 대비 −4; AUROC↑이나 held-out PASS 악화 → 미투입)
```

### 7. 카탈로그 / 버그수정 조회

```python
from agent_system.methods import catalog, fixes

for m in catalog():                # 방법론 9종 + 대조군 (중요도 순)
    print(m["name"], m["stage"], m["status"], m["pass_impact"])
print(fixes())                     # EAPanelNotationFix (방법론 아님)
```

### 8. 온라인 추론 (raw image → 점수)

```python
router = PerCellModelRouting.online_router()       # PerCellModelRouter (lazy 모델 로딩)
# router.predict_from_safetensors(path, organism_group, antimicrobial) → 점수
```

## 방법론 카탈로그 (중요도 = FDA PASS 기여 순)

| # | 방법론 | 스테이지 | 상태 | PASS 기여 | 모듈 |
|---|---|---|---|---|---|
| ① | `PerCellIsotonic` | calibrate | ✅채택 | **+30** (주역) | calibrate/isotonic.py |
| ② | `PerCellModelRouting` | score | ✅채택 | **+3 ★LNZ** | score/model_routing.py |
| ③ | `ThresholdTuner` | threshold | ✅채택 | VME lever | threshold/tuner.py |
| ④ | `RankAvgEnsemble` | score | ❌기각 | AUROC+0.013 / OOF −4~−9 | score/rank_avg_ensemble.py |
| ⑤ | `ObjAreaGBM` | score | 🟡보조 | AUROC 0.9394 | score/objarea_gbm.py |
| ⑥ | `ShiftedCellEnsemble` | score | 🟡탐색 | shifted-only +0.0105 | score/shifted_ensemble.py |
| ⑦ | `ZScoreEnsemble` | score | 🟡대안 | — | score/zscore_ensemble.py |
| ⑧ | `BrightnessAsymCorrector` | score | 🔴CV기각 | in +3 / CV −1 | score/brightness_corrector.py |
| ⑨ | `GenusModelRouting` | score | 🔴대안 | PASS 95 / VME 1.55% | score/genus_routing.py |

**평가기 버그 수정 (방법론 아님, `FIXES`)**

| 버그 수정 | 스테이지 | 효과 | 모듈 |
|---|---|---|---|
| `EAPanelNotationFix` | evaluate | 정정 +32 (71→103, 개선 아닌 정정) | evaluate/ea_fix.py |

> EA fix 는 BMD↔dRAST notation mismatch 로 **잘못 FAIL 되던 cell 을 정정**하는 평가기 패치다.
> PASS 를 새로 끌어올리는 방법론이 아니라 correctness 복원이며, verify 파이프라인이 자동 적용한다.

### ★ LNZ(linezolid) 통과 방법론 = `PerCellModelRouting`

baseline(same_mic)은 LNZ gram-positive 5 cell 전부 FAIL. **per-cell routing** 이 cell 별 최적
subset 모델 선택 → E. faecium=`early_stop`, S. Coag-neg=`beta_lactam`,
E. faecalis=`object_area_045~08`, S. lugdunensis=`early_stop` → **4/5 PASS**. 분기점
E. faecium×LNZ: our=early_stop→PASS vs genus=beta_lactam→FAIL.

## 운영 권고 = `routed_iso_t65` (PASS 103 / VME 1.99% / AUROC 0.9743)

```python
operational_recipe(threshold=0.65)   # routing(★LNZ) + isotonic + threshold 0.65
```

3-recipe lever: t0.50(적극 111/3.56%) · **t0.65(균형 ★103/1.99%)** · t0.75(보수 91/1.50%).

### genus 단위 보정 (배포 현실) → [GENUS_ISO.md](GENUS_ISO.md)

배포 환경은 genus 만 알 수 있다. `PerCellIsotonic(cell_by="genus")` 로 학습·적용하고 summary 는
organism_group 으로 평가하면 **held-out OOF PASS 가 organism 87 → genus 91(+4)**. 학습·동결:
`python -m agent_system.methods.save_genus_iso_lookup` → `isotonic_lookup_genus.csv`.
운영 채택은 threshold×VME budget 재검증 후(보류). 상세 = [GENUS_ISO.md](GENUS_ISO.md).

## 교차검증

```bash
python -m agent_system.methods.validate_recipes
```

모듈 조립이 독립 기준을 재현: routed iso(insample)=**103**, iso(oof)=**87**,
rankavg_w0.3 iso(oof)=**83**, brightness iso(insample)=**106**.

## 디렉토리

```
agent_system/methods/
  base.py                 # Method ABC · Stage/Status · 통합 입력 스키마
  build_unified_input.py  # 단일 통합 per_conc 빌더
  recipe.py               # Recipe 조립 + 운영 verify(EA fix) → PASS/VME
  validate_recipes.py     # 교차검증 드라이버
  __init__.py             # REGISTRY · catalog() · operational_recipe()
  score/      model_routing · rank_avg_ensemble · objarea_gbm · shifted_ensemble
              · zscore_ensemble · brightness_corrector · genus_routing
  calibrate/  isotonic
  threshold/  tuner
  evaluate/   ea_fix
  data/       unified_per_conc.parquet   (build_unified_input 산출)
  output/     <recipe>/  per_conc.csv + method_summary_*.csv (verify 산출)
```

# Brightness Asymmetric Corrector

Brightness=True (image sequence 표현력이 낮은) cell 에 한정해, control–약제 brightness
신호로 image model_pred 를 **비대칭 G-방향 보정**하여 운영 FDA PASS 를 올리는 모듈.

> ## ⛔ 2026-06-04 GroupKFold(sample_id) CV 결과: **운영 미투입**
> in-sample +3 PASS(103→106)는 **held-out 에서 재현되지 않음**. OOF baseline 87 → corrected 86 (Δ−1),
> 모든 w 에서 corrected ≤ baseline, held-out PASS gain 0 / loss 1. +3 은 isotonic 재적합 + w 선택의
> **in-sample 과적합** 산물. 운영 권고는 `routed_iso_t65` 유지. 본 모듈은 in-sample only.
> 상세: [`artifacts/cv/cv_report.html`](artifacts/cv/cv_report.html) · 재현: `python groupkfold_cv.py`.

## 핵심 결과 (전체 313 cell, 2026-06-02 검증)

| | PASS | VME-cell | VME-rate |
|---|---|---|---|
| baseline (운영 `routed_iso_t65`) | 103 | 43 | 1.77% |
| **+ brightness asym corrector** | **106 (+3)** | **42 (−1)** | 1.81% |

- 적용 범위: **Brightness=True 45 cell 한정** (genus = Proteus/Serratia/Morganella). 나머지 cell 불변.
- 개선 cell: Proteus mirabilis×CZA (VME→PASS), Proteus vulgaris×CPM (ME→PASS),
  Serratia marcescens×CAZ (EA,ME→PASS) — 모두 cephalosporin 과대 R 호출 교정.
- **PASS 손실 0** (비대칭 G-방향이라 false-NG 신규 발생 차단).

## 방법

1. **brightness GBM** (`train_brightness_gbm.py`): 학습 CSV(`brightness_list`, 7-timepoint)로
   in-domain 학습 → FDA 추론. in-domain AUROC ≈ 0.915 ≈ cross-domain 0.918 (도메인 전이 안정).
2. **비대칭 G-방향 보정** (`brightness_corrector.py`): 운영 score = P(NG).
   VME(거짓 NG)는 P(NG)를 *올릴* 때 생기므로, brightness 가 image 보다 **더 G** 라고 말할 때
   (`brp < img`)만 점수를 내린다(G 방향). 더 NG 라고 말할 땐 image 유지 → VME 차단.
   ```
   corrected = (1-w)·img + w·brp     (brp < img 이고 Brightness=True cell)
             = img                    (그 외)        w = 0.7
   ```
3. 이후 per-cell isotonic → threshold 0.65 → drast_gng→MIC→SIR→evalEA (기존 운영 파이프라인).

## 왜 비대칭/절대값 GBM 인가 (검증된 비교)

- 대칭(양방향) 보정: PASS +1 에 VME **+1** 동반 → 비대칭(G-방향)이 VME 신규 0.
- brightness **단독** 판단: PASS 95 (−8) — 단독 판단 불가, 보정자로만 유효.
- **거리(growth_dev) 전용 GBM**: standalone AUROC 0.9265 로 더 높지만 보정자로는 PASS +0/VME +3 로 열세.
  → 절대 trajectory feature GBM(`oa_drug_late_slope` 등) 의 오차 구조가 G-방향 비대칭과 더 잘 맞음.

## 파일

```
config.py                  설정(경로/weight 0.7/threshold 0.65/하이퍼파라미터)
brightness_features.py     self-contained 시계열 feature 추출 (control-pick + hand-crafted)
train_brightness_gbm.py    학습: 학습 CSV → model/brightness_gbm.txt
brightness_corrector.py    추론: BrightnessAsymCorrector (predict_brightness + asym correct)
evaluate.py                검증: 전체 313 cell baseline vs corrected SIR/PASS
brightness_apply_cells.csv 적용 대상 Brightness=True (organism_group, antimicrobial)
model/brightness_gbm.txt   학습된 LightGBM
artifacts/                 OOF, 비교 CSV, manifest, eval 결과
```

## 사용법

```bash
conda activate qnt_algorithm
export PYTHONPATH=/home/kptae/project/qnt_algorithm

# (1) 학습 — 모델 재생성 (선택; 이미 model/ 에 포함)
python train_brightness_gbm.py

# (2) 추론 데모 — routed per_conc 에 보정 적용, 보정된 cell 출력
python brightness_corrector.py

# (3) 검증 — 전체 313 cell baseline vs corrected PASS/VME (~15-20분)
python evaluate.py
```

운영 코드에서 직접 사용:
```python
from brightness_corrector import BrightnessAsymCorrector
corr = BrightnessAsymCorrector()                         # w=0.7, Brightness=True 45 cell
out = corr.apply_to_per_conc(per_conc_df, FDA_CSV)       # out["dtw_model_pred"] = 보정값
# 이후 기존 운영: per-cell isotonic → threshold 0.65 → mics.py
```

## genus 단위 모델 맵핑 평가 (evaluate_genus_routing.py)

evo 워크북의 (Genus, 약제)→Model 맵핑을 organism cell 에 부여하고 **동일 단위(organism×약제)·
동일 파이프라인**으로 평가 (모델 맵핑만 교체). `python evaluate_genus_routing.py`.

| routing | PASS | VME-cell | VME-rate |
|---|---|---|---|
| our_routing (AUROC, per-organism) | 103 | 43 | 1.77% |
| genus_routing (evo 맵핑) | **95 (−8)** | 42 | **1.55%** |

→ **PASS↔VME 트레이드오프**. evo genus 맵핑은 PASS 8 낮지만 VME-rate 1.77%→1.55% (FDA strict 1.5%
근접) — gram_negative 제외한 보수적 메뉴. gain 8 = carbapenem/cephalosporin VME 제거, loss 16 =
R 과대호출(ME). 우리 AUROC routing 은 PASS 최대, evo genus 는 VME 안전.

워크북 모델 선정 일치율(모델 맵핑 비교): F21 organism 19.1%, evo genus 17.3% — 둘 다 임상 기준이라
AUROC 기준 우리 routing 과 표면 일치율 낮음(대부분 near-tie, 실질충돌은 carbapenem/colistin).

## ⚠ 주의

- **evo 파일의 `Brightness_threshold`(0.4~0.75) 는 10시간 readout 기준** → 현재 데이터
  (7-timepoint trajectory)에는 사용하지 않음. 본 모듈은 그 고정 threshold 대신 현재
  데이터로 GBM 학습 + 현재 FDA per-cell isotonic 보정(데이터-구동).
- **in-sample 한계**: w=0.7 을 FDA 패널에서 선택. 운영 투입 전 GroupKFold(sample_id) CV 로
  +3 PASS 가 held-out 에서 유지되는지 확인 권장.
- 모델 라우팅 자체를 genus 단위로 묶는 작업은 별도 (본 모듈은 brightness 보정만 담당).

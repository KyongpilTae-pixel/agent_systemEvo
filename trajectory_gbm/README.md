# trajectory_gbm

trajectory 신호(brightness, object_area, …) GBM 을 **하나의 코어**로 통합한 패키지.
brightness 와 object_area 는 입력 trajectory 컬럼만 다르고 feature 추출·GBM 학습·
GroupKFold CV·비대칭 보정 로직이 **동일**하다 — 그 공통 코어를 여기 모았다.

## 왜 통합되나 (확인 결과)

| | brightness | object_area |
|---|---|---|
| TRAIN_CSV 컬럼 | `brightness_list` | `object_area` |
| FDA_CSV 컬럼 | `brightness` | `object_area` |
| 값 형식 | 콤마 구분 7-timepoint trajectory | 〃 |
| feature 추출 | `features.build_features` (control-pick + hand-crafted) | **동일 코드** |
| GBM 학습/CV | cross-domain train→FDA, GroupKFold(sample_id) | **동일 코드** |

→ 차이는 `config.SignalConfig` 의 `train_col`/`fda_col` **두 줄뿐**. 나머지는 전부 공유.

## 구조

```
config.py      공통 경로 + SIGNALS 레지스트리(SignalConfig: train_col/fda_col/model/weight)
features.py    signal-agnostic trajectory feature 추출 (구 brightness_features.py 일반화)
model.py       train_crossdomain / cv_auroc_indomain / predict
corrector.py   AsymCorrector — 비대칭 G-방향 보정 (구 brightness_corrector.py 일반화)
__main__.py    CLI (train / cv)
models/        학습된 GBM (brightness_gbm.txt, object_area_gbm.txt)
```

## 사용

```bash
conda activate qnt_algorithm
export PYTHONPATH=/home/kptae/project/qnt_algorithm

# cross-domain 학습 (TRAIN→FDA): 모델 저장 + in/cross AUROC
python -m agent_system.trajectory_gbm train --signal object_area
python -m agent_system.trajectory_gbm train --signal brightness

# in-domain GroupKFold OOF AUROC (honest baseline)
python -m agent_system.trajectory_gbm cv --signal object_area --folds 5
```

```python
from agent_system.trajectory_gbm import config, model, corrector
sig = config.get_signal("object_area")
meta = model.train_crossdomain(sig)          # {'auroc_in_domain':..., 'auroc_cross_domain_fda':...}
out, df = model.cv_auroc_indomain(sig)       # {'oof_auroc':..., 'model_pred_auroc':..., 'gap_...':...}

# (실험) 비대칭 보정 — signal 만 교체
corr = corrector.AsymCorrector(config.get_signal("brightness"))
pc = corr.apply_to_per_conc(per_conc_df, config.FDA_CSV)
```

## 신호별 메모

- **object_area**: 이미지 없이 trajectory 만으로 G/NG — in-domain OOF AUROC ≈ 0.93,
  cross-domain ≈ 0.93. `ShiftedCellEnsemble`(운영) 이 이 GBM 자산을 사용. baseline/ensemble 용도.
- **brightness**: cross-domain 전이 안정(in 0.915 ≈ cross 0.918). 단 **비대칭 보정의 +3 PASS 는
  GroupKFold(sample_id) CV 에서 held-out 미재현(2026-06-04) → 운영 미투입** (in-sample 과적합).
  상세 = `../brightness_asym_corrector/artifacts/cv/cv_report.html`.

## 통합 이력 (2026-06-04)

흩어져 있던 objarea_gbm 스크립트(`claudeCode/objarea_*.py`)와 brightness 패키지의 공통 로직을
이 패키지로 통합. 두 신호가 입력 컬럼만 다름을 확인 후 단일 코어로 합침.

**대체된 구 스크립트** (claudeCode/): `objarea_only_gbm.py`(→ `cv`),
`objarea_gbm_crossdomain.py`(→ `train`), `task3_objarea_gbm_full.py`, `task_fda_object_area_gbm.py`,
`regenerate_objarea_oof_full.py`. brightness 패키지의 `brightness_features.py`/`brightness_corrector.py`/
`train_brightness_gbm.py` 는 이 패키지로 re-export(shim) — 기존 CLI·CV 자산 그대로 동작.
(구 스크립트의 HTML 보고서 빌더·per-cell 비교·ensemble 최적화는 분석 전용이라 잔존.)

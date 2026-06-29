# iso_calibration — Per-cell Isotonic Calibration (공유 패키지)

모델의 G/NG 점수를 **(organism_group 또는 genus) × antimicrobial cell 단위로 isotonic 보정**해
P(NG) 확률로 만드는 독립 패키지. **프로젝트 의존 없음** (numpy/pandas/sklearn 만; 런타임 보정은
sklearn 없이 np.interp).

## 무엇 / 왜
- 모델 raw 점수는 cell 마다 실제 NG 비율이 달라 확률로서 보정이 안 돼 있다.
- per-cell isotonic = cell 별 **단조 비감소 계단 함수**로 `score → P(NG)` 보정.
- 단조 → cell 내 ranking 보존(**AUROC 불변**), 확률 품질(Brier)만 개선.
- 배포 환경이 species 를 모르고 **genus 만** 알면 `cell_by="genus"`.

## 설치/요구
- Python ≥ 3.9, `numpy pandas scikit-learn`.
- 폴더 통째로 복사해 `import iso_calibration` (또는 상위 경로를 sys.path 에).

## 입력 DataFrame 규약 (컬럼명은 설정 가능)
| 역할 | 기본 컬럼명 | 비고 |
|---|---|---|
| 점수 | `dtw_model_pred` | 보정 대상 (모델 P(NG) 등) |
| 라벨 | `gt_gng` | 0/1, 학습(insample/oof)에만 필요 |
| cell 키1 | `organism_group` | 또는 `genus` (cell_by) |
| cell 키2 | `antimicrobial` | 약제 |
| 그룹 | `sample_id` | OOF GroupKFold 단위 |

컬럼명이 다르면 생성자 인자로: `PerCellIsotonic(score_col="model_pred", label_col="y", ...)`.

## 사용

### 1) 운영 — 동결 자산으로 보정 (sklearn 불필요 런타임)
```python
from iso_calibration import PerCellIsotonic
cal = PerCellIsotonic(mode="lookup").apply(df)          # assets/isotonic_lookup_routed.csv
cal = PerCellIsotonic(mode="lookup", cell_by="genus").apply(df)  # assets/isotonic_lookup_genus.csv
# 또는 직접 CSV 지정: PerCellIsotonic(mode="lookup", lookup_csv="my_lookup.csv")
```

### 2) 학습 — 직접 fit + 동결
```python
iso = PerCellIsotonic().fit_lookup(df, save_path="my_lookup.csv")   # cell별 isotonic 학습만
# 이후 운영: PerCellIsotonic(mode="lookup", lookup_csv="my_lookup.csv").apply(new_df)
```

### 3) 학습+보정 한 번에 (3 모드)
```python
PerCellIsotonic("insample").apply(df)        # 전체 fit+보정 (in-sample, 낙관)
PerCellIsotonic("oof", n_folds=5).apply(df)  # GroupKFold held-out (정직한 일반화)
PerCellIsotonic("lookup").apply(df)          # 동결 자산 로드
```

### 4) Hybrid (genus 기본 + VME 위험 cell 만 organism)
```python
from iso_calibration import HybridIsotonic
h = HybridIsotonic("insample",
        fallback_cells=[("Staphylococcus aureus","LNZ"), ("Proteus mirabilis","CRO")])
h.apply(df)   # 일반 cell=genus, fallback cell=organism
```

## 동결 자산 (assets/)
| 파일 | 단위 | cell 수 |
|---|---|---|
| `isotonic_lookup_routed.csv` | organism_group×drug | 286 (+GLOBAL) |
| `isotonic_lookup_genus.csv` | genus×drug | 170 |

CSV 컬럼: `organism_group, antimicrobial, x_thresholds(JSON), y_thresholds(JSON), n_train_rows, n_g, n_ng`.
(genus 자산은 organism_group 컬럼에 genus 값을 담음 — 반드시 `cell_by="genus"` 로 로드.)

## 핵심 주의 (검증된 결과 기반)
- **in-sample vs OOF**: `insample` 은 같은 데이터로 fit+평가 → 낙관. 정직한 일반화는 `oof`.
- **도메인 일치 필수**: 보정은 **평가/배포 도메인 데이터로 fit** 해야 한다. 다른 도메인(예: 학습셋)에서
  fit 한 isotonic 을 전이하면 효과가 거의 없음(검증됨) — 모델 점수 분포가 다르면 곡선이 안 맞는다.
- **단조성**: cell 내 순서를 안 바꾼다 → AUROC 동일, 확률/threshold 판정만 개선.

## 구성
```
iso_calibration/
  lookup.py       # PerCellIsotonicLookup — fit/save/load/calibrate (self-contained 엔진)
  calibrator.py   # PerCellIsotonic(3모드/genus) + HybridIsotonic
  assets/         # 동결 lookup CSV 2종
  __init__.py
```

# technical_error (우리쪽 관리본)

농도(well) 단위로 **촬영상의 기술적 오류**를 탐지하는 이미지 모델. FilmModel(ResNet18 + BiLSTM)
체크포인트 4개(`film_qc`, `bubble_qc`, `film_drast`, `bubble_drast`)로 FDA reproducibility 데이터셋의
각 well에 대해 4개 예측 컬럼(`predict_film_qc`, `predict_bubble_qc`, `predict_film_drast`,
`predict_bubble_drast`)을 채워 CSV로 저장한다.

- **film** = 액면 막/필름 아티팩트, **bubble** = 기포. 각각 **qc / drast** 두 도메인 체크포인트.
- 원저자: junhyeok (`/home/junhyeok/technical_error`). **2026-07-27 우리쪽 이관**(코드=여기, 모델/입력/출력=`/home/kptae/data/technical_error`).

## 구성 (코드 = 이 디렉토리, git 관리)

```
predict.py                  실행 스크립트 (엔트리)
run.sh                      conda 활성화 + 실행 (우리 miniconda 경로)
model/anomaly_model.py      FilmModel / AnomalyModel
resnet/                     ResNet18 backbone (1채널 입력)
dataset/anomaly_dataset.py  SafetensorsWellDataset
configs/test_anomaly.yaml   time_len=6, DDP port
```

## 데이터/모델 (git 밖, `/home/kptae/data/technical_error`)

```
checkpoints/{film_qc,bubble_qc,film_drast,bubble_drast}/15.pt   각 147M (md5 검증 복사)
backbone/simclr_epoch10.pth                                     100M SimCLR 자기지도 백본
input/FDA_Analytical_reproducibility_exceptGN26_df_260624.parquet  추론 입력(19,008 drugbug)
output/Reproducibility_technical_error.csv                      결과(실행 후 생성)
```

경로 override: 환경변수 `TECH_ERR_DATA`(데이터 루트), `TECH_ERR_BACKBONE`(백본).

## 실행

```bash
bash agent_system/technical_error/run.sh
```

- **4-GPU DDP**(world_size=4 하드코딩). ★ 4장 모두 사용 → **학습 job과 충돌**하므로 GPU 여유 시 실행.
- conda env `qnt_algorithm`. server 192.168.0.207.

## 동작

1. 원본 parquet은 drugbug당 1행에 control+전체 농도 이미지가 한 `.safetensors`로 뭉쳐 있음.
   `explode_to_wells()`가 well(control 또는 농도 1개) 단위로 펼침(control은 sample당 1행).
2. `SafetensorsWellDataset`이 각 행 `concentration_index` 블록(t0~t6=7프레임)만 잘라 읽음.
3. 체크포인트 4개를 순서대로 돌며 각 컬럼(boolean, logit>0)을 채우고 중간 저장(`output/temp.csv`).
4. 완료 시 `output/Reproducibility_technical_error.csv`.

## 학습셋 · 재학습 (이관 완료 2026-07-27)

**라벨 CSV**(human 기술오류 주석 + train/test split) = `/home/kptae/data/technical_error/trainset/` (md5 검증 복사):
- `0909_train_test_qc.csv` (qc 도메인) · `0910_train_test_drast.csv` (drast 도메인)
- `0915_bubble_qc_train.csv` · `0917_bubble_drast_train.csv` (bubble 도메인별)
- `0908_train_test.csv` · `0728_traintest.csv` (4-label anomaly)
- 추정 매핑: film/bubble_qc ← qc CSV, film/bubble_drast ← drast CSV. **정확 매핑은 원저자 확인 필요.**

**이미지**: 라벨 CSV 의 `safetensors` 컬럼이 공용 중앙 저장소 `/data/dRAST30_safetensors/`(우리 MIC 파이프라인과 동일 원본)를
가리킴 → **복사 안 함, 제자리 참조**.

**재학습 코드**: `train/` (train_film.py · train_anomaly.py[4-label] · train_anomaly_4.py + configs/).
config 는 로컬 `train/configs/` 에서 읽도록 수정, df_path/output_path 는 우리 관리 경로로 교체.
⚠ 이 스크립트들은 junhyeok **실험 중간 상태 스냅샷**(resume·per-epoch load 포함) — 외부 체크포인트 로드는 주석 처리했고,
clean 재학습은 df_path/에폭 범위 조정 후 검증 필요(4개 체크포인트 정확 재현은 별도 검증 TODO).

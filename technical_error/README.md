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

## 학습셋

이 추론 패키지엔 미포함. 원본 학습 데이터는 `/home/junhyeok/anomaly/`(~14GB, 구/중간본 혼재).
실제 4개 체크포인트를 만든 학습셋 특정 후 별도 이관 예정(TODO).

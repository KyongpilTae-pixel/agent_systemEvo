"""predict.py 의 CPU 단일프로세스 변형 (2026-07-30, T-0730-1b).

원본 predict.py 는 DDP+CUDA 4-GPU 전용. GPU 를 학습이 점유 중이라 **CPU 단일프로세스**로 임의 SOURCE
(기본 fda2023)에 TE 4-헤드(film/bubble×qc/drast) 추론. 로직·전처리·체크포인트는 원본과 동일
(explode_to_wells·SafetensorsWellDataset·FilmModel). map_location=cpu, DDP 미사용.

사용:
  cd agent_system/technical_error
  TE_SOURCE=/home/kptae/data/allinfo/fda2023_df.parquet \
  TE_OUTPUT=/home/kptae/data/technical_error/output/fda2023_technical_error.csv \
  python predict_cpu.py
"""
import os

import pandas as pd
import torch
import yaml

# FilmModel.__init__ 등 내부 torch.load 가 map_location 없이 CUDA 텐서를 로드 → CPU 에서 실패.
# 모델 코드 무변경 원칙 유지 위해 torch.load 기본 map_location='cpu' 로 몽키패치(CPU 추론 전용).
_orig_load = torch.load
def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu")
    return _orig_load(*a, **k)
torch.load = _cpu_load
from dataset.anomaly_dataset import SafetensorsWellDataset
from model.anomaly_model import FilmModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from predict import CHECKPOINTS, explode_to_wells  # 원본 정의 재사용(로직 동일)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("TECH_ERR_DATA", "/home/kptae/data/technical_error")
SOURCE_DF_PATH = os.environ.get("TE_SOURCE", "/home/kptae/data/allinfo/fda2023_df.parquet")
OUTPUT_DF_PATH = os.environ.get("TE_OUTPUT", f"{DATA_DIR}/output/fda2023_technical_error.csv")


def get_config(file_name: str) -> dict:
    with open(f"{CODE_DIR}/configs/{file_name}.yaml") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_checkpoint(model, dataloader, df, checkpoint_path, column):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    desc = os.path.basename(os.path.dirname(checkpoint_path))
    for imgs, idx in tqdm(dataloader, desc=desc, ncols=100):
        pred = model(imgs).cpu().numpy()
        df.loc[idx.cpu().numpy(), column] = pred > 0


def main():
    config = get_config("test_anomaly")
    img_len = config["time_len"]
    torch.set_num_threads(os.cpu_count() or 8)

    df = explode_to_wells(pd.read_parquet(SOURCE_DF_PATH))
    print(f"[TE-cpu] source={SOURCE_DF_PATH} · wells={len(df):,} · img_len={img_len}", flush=True)

    model = FilmModel()  # CPU
    dataset = SafetensorsWellDataset(df=df, img_len=img_len)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=8, prefetch_factor=2, persistent_workers=True)

    for checkpoint_path, column in CHECKPOINTS:
        predict_checkpoint(model, dataloader, df, checkpoint_path, column)
        df.to_csv(OUTPUT_DF_PATH, index=False)  # 각 헤드 후 중간저장(안전)
        print(f"[TE-cpu] {column} done → {OUTPUT_DF_PATH}", flush=True)

    print(f"[TE-cpu] saved {OUTPUT_DF_PATH} ({len(df):,} wells)", flush=True)


if __name__ == "__main__":
    main()

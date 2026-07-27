"""FilmModel(ResNet18 + BiLSTM)로 4개 체크포인트(film_qc, bubble_qc, film_drast, bubble_drast)를
순서대로 돌려서 각각의 예측 컬럼을 df에 채워넣는 스크립트.

원본 SOURCE_DF_PATH(parquet)는 drugbug(약물-균 조합)당 1행에 control+전체 농도 이미지가
하나의 .safetensors((concentration_len+1)*(time_len+1) 프레임)로 뭉쳐있다. 예측은 기존 CSV 포맷과
동일하게 well(=control 또는 농도 하나) 단위로 나와야 하므로, explode_to_wells()로 먼저 well 단위
행으로 펼친 뒤 SafetensorsWellDataset이 각 행의 concentration_index번째 블록만 잘라서 읽는다.

temp.csv가 있으면 (이미 well 단위로 펼쳐진) 이전 실행 결과에서 이어서 진행하고, 없으면 원본에서
새로 시작한다.
"""

import os

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from dataset.anomaly_dataset import SafetensorsWellDataset
from model.anomaly_model import FilmModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

# 2026-07-27 우리쪽 관리 이관: 코드=이 파일 위치, 모델/입력/출력=/home/kptae/data/technical_error.
#   (junhyeok 원본 절대경로 → 우리 관리 경로로 교체. 원본 대비 로직 무변경.)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("TECH_ERR_DATA", "/home/kptae/data/technical_error")
CKPT_DIR = f"{DATA_DIR}/checkpoints"
SOURCE_DF_PATH = f"{DATA_DIR}/input/FDA_Analytical_reproducibility_exceptGN26_df_260624.parquet"
os.makedirs(f"{DATA_DIR}/output", exist_ok=True)
TEMP_DF_PATH = f"{DATA_DIR}/output/temp.csv"
OUTPUT_DF_PATH = f"{DATA_DIR}/output/Reproducibility_technical_error.csv"

# (체크포인트 경로, 예측 결과를 채워넣을 컬럼명) - qc/drast x film/bubble 4종
CHECKPOINTS = [
    (f"{CKPT_DIR}/film_qc/15.pt", "predict_film_qc"),
    (f"{CKPT_DIR}/bubble_qc/15.pt", "predict_bubble_qc"),
    (f"{CKPT_DIR}/film_drast/15.pt", "predict_film_drast"),
    (f"{CKPT_DIR}/bubble_drast/15.pt", "predict_bubble_drast"),
]


def get_config(file_name: str) -> dict:
    with open(f"{CODE_DIR}/configs/{file_name}.yaml") as f:
        return yaml.safe_load(f)


def setup(rank: int, world_size: int, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def explode_to_wells(df: pd.DataFrame) -> pd.DataFrame:
    """drugbug당 1행(control+전체 농도가 한 .safetensors에 뭉친 행)을 well 단위 행으로 펼침.

    control은 같은 (project_id, sample_id)의 모든 항생제 행에 동일한 이미지가 중복 저장돼있어서
    (실측 확인함: 항생제가 달라도 control 블록이 byte-identical), 그대로 펼치면 항생제 개수만큼
    같은 control을 중복 예측하게 된다. 그래서 control은 (project_id, sample_id)당 대표 행 하나에서만
    뽑고, 농도는 항생제별로 실제로 다르므로 기존처럼 다 펼친다.

    concentration_index: 0=control, 1..concentration_len=concentration_list 순서 그대로.
    concentration_value: 사람이 보기 편하게 실제 농도값(문자열), control은 "control".
    """
    concentration_len = df["concentration_len"].to_numpy()
    concentration_lists = df["concentration_list"].str.split(",")

    conc_rows = df.loc[df.index.repeat(concentration_len)].reset_index(drop=True)
    conc_rows["concentration_index"] = pd.Series([i for n in concentration_len for i in range(1, n + 1)], dtype="int64")
    conc_rows["concentration_value"] = (
        pd.Series(list(values) for values in concentration_lists).explode().reset_index(drop=True)
    )

    control_rows = df.drop_duplicates(subset=["project_id", "sample_id"]).reset_index(drop=True).copy()
    control_rows["concentration_index"] = 0
    control_rows["concentration_value"] = "control"
    # 이 control은 특정 항생제 것이 아니라 sample 전체가 공유하는 것이라 오해하지 않도록 비워둠
    control_rows["antimicrobial"] = None

    return pd.concat([control_rows, conc_rows], ignore_index=True)


def load_dataframe() -> pd.DataFrame:
    """temp.csv(이전 실행에서 이어서 할, 이미 well 단위로 펼쳐진 결과)가 있으면 그걸,
    없으면 원본 parquet을 well 단위로 펼쳐서 새로 시작."""
    # if os.path.exists(TEMP_DF_PATH):
    #     return pd.read_csv(TEMP_DF_PATH)
    df = pd.read_parquet(SOURCE_DF_PATH)
    return explode_to_wells(df)


@torch.no_grad()
def predict_checkpoint(
    model: DDP, dataloader: DataLoader, df: pd.DataFrame, checkpoint_path: str, column: str, rank: int
) -> None:
    checkpoint = torch.load(checkpoint_path)
    model.module.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    desc = os.path.basename(os.path.dirname(checkpoint_path))
    for imgs, idx in tqdm(dataloader, desc=desc, ncols=100, disable=(rank != 0)):
        predict = model(imgs.to(rank)).cpu().numpy()
        df.loc[idx.cpu().numpy(), column] = predict > 0

    if rank == 0:
        df.to_csv(TEMP_DF_PATH, index=False)


def run(rank: int, world_size: int, config_name: str) -> None:
    config = get_config(config_name)
    img_len = config["time_len"]

    setup(rank, world_size, config["port"])
    torch.cuda.set_device(rank)

    model = FilmModel().to(rank)
    model = DDP(model, device_ids=[rank])

    df = load_dataframe()
    dataset = SafetensorsWellDataset(df=df, img_len=img_len)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

    for checkpoint_path, column in CHECKPOINTS:
        predict_checkpoint(model, dataloader, df, checkpoint_path, column, rank)

    if rank == 0:
        df.to_csv(OUTPUT_DF_PATH, index=False)


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size, "test_anomaly"), nprocs=world_size, join=True)

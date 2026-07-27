import argparse
import ast
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from model.dataset.anomaly_dataset import AnomalyDataset
from code.model.anomaly_model import FilmModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, SubsetRandomSampler
from tqdm import tqdm

mp.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config file name")
    return parser.parse_args()


def get_config(file_name: str):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "{}.yaml".format(file_name)), "r") as f:
        config = yaml.safe_load(f)

    return config


def setup(rank: int, world_size: int, master_port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train(rank: int, world_size: int, config_path: str):
    config = get_config(config_path)

    setup(rank, world_size, config["port"])
    torch.cuda.set_device(rank)

    model = FilmModel()
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    df = pd.read_csv(config["df_path"])
    df_train = df[df["train_or_test"] == "train"].reset_index(drop=True)
    train_dataset = AnomalyDataset(df_train, train=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=config["batch_size"], num_workers=2, pin_memory=True
    )

    df = df[df["train_or_test"] != "train"].reset_index(drop=True)
    valid_dataset = AnomalyDataset(df, train=False)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoader(
        valid_dataset, sampler=valid_sampler, batch_size=config["batch_size"], num_workers=2, pin_memory=True
    )

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    # checkpoint = torch.load("/home/junhyeok/anomaly/model/film_qc/15.pt")  # 저장된 파일 경로
    # model.module.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler"])

    # 훈련 루프
    for epoch in range(config["num_epoch"]):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        train_total = 0.0
        val_total = 0.0

        for img, label in tqdm(train_dataloader, desc=f"{epoch}Epoch Training", ncols=100, disable=(rank != 0)):
            img = img.to(rank)
            label = label.to(rank)
            label = label.to(dtype=torch.float32)

            predict = model(img)
            loss = criterion(predict, label)
            train_acc += ((predict > 0) == label).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_total += img.shape[0]

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for img, label in tqdm(
                    valid_dataloader, desc=f"{epoch}Epoch validation", ncols=100, disable=(rank != 0)
                ):
                    img = img.to(rank)
                    label = label.to(rank)
                    predict = model(img)
                    label = label.to(dtype=torch.float32)

                    val_acc += ((predict > 0) == label).sum()

                    loss = criterion(predict, label)
                    val_loss += loss
                    val_total += img.shape[0]

            scheduler.step(val_loss)

            if rank == 0:
                torch.save(
                    {
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    "{}/{}.pt".format(config["output_path"], epoch),
                )

                train_acc /= train_total
                val_acc /= val_total

                log_path = "train_log.txt"

                with open(log_path, "a") as f:  # append 모드
                    f.write(
                        f"Epoch {epoch} train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f} "
                        f"val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}\n"
                    )

                print(
                    f"Epoch {epoch} train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f} val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}"
                )


def run_training():
    # args = parse_args()
    world_size = torch.cuda.device_count()
    # mp.spawn(train, args=(world_size, args.config), nprocs=world_size, join=True)
    mp.spawn(train, args=(world_size, "train_film"), nprocs=world_size, join=True)


if __name__ == "__main__":
    run_training()

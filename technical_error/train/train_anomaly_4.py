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
from code.model.anomaly_model import AnomalyModel
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

    model = AnomalyModel()
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    df = pd.read_csv(config["df_path"])
    df_train = df[df["train_or_test"] == "train"]
    # df_normal = df_train[
    #     (df_train["bubble"] == 0)
    #     & (df_train["film_dispense"] == 0)
    #     & (df_train["artifact"] == 0)
    #     & (df_train["shade_broth"] == 0)
    # ]
    # df_bubble = df_train[df_train["bubble"] == 1]
    # df_film = df_train[df_train["film_dispense"] == 1]
    # df_artifact = df_train[df_train["artifact"] == 1]
    # df_shade = df_train[df_train["shade_broth"] == 1]
    df = df[df["train_or_test"] != "train"].reset_index(drop=True)

    valid_dataset = AnomalyDataset(df, train=False)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoader(
        valid_dataset, sampler=valid_sampler, batch_size=config["batch_size"], num_workers=32, pin_memory=True
    )

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    # checkpoint = torch.load("/home/junhyeok/anomaly/model/bubble_model/best_model.pt")  # 저장된 파일 경로
    # model.module.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler"])

    # 훈련 루프
    for epoch in range(34, config["num_epoch"], 5):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        bubble_acc = 0.0
        film_acc = 0.0
        artifact_acc = 0.0
        shade_acc = 0.0
        bubble_val = 0.0
        film_val = 0.0
        artifact_val = 0.0
        shade_val = 0.0
        train_total = 0
        val_total = 0
        # ★이관 정리(2026-07-27): junhyeok 외부 체크포인트 per-epoch 로드 — clean 재학습에선 비활성.
        # model.module.load_state_dict(torch.load(f"/home/junhyeok/anomaly/model/bfas/{epoch}.pt")["model_state_dict"])

        df_sampled = (
            pd.concat(
                [
                    df_normal.sample(n=(len(df_normal) // 16)),
                    df_bubble.sample(n=(len(df_bubble) // 16)),
                    df_film.sample(n=(len(df_film) // 16)),
                    df_artifact.sample(n=(len(df_artifact) // 16)),
                    df_shade.sample(n=(len(df_shade) // 16)),
                ]
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )
        train_dataset = AnomalyDataset(df_sampled, train=True)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=config["batch_size"], num_workers=32, pin_memory=True
        )

        for img, label in tqdm(train_dataloader, desc=f"{epoch}Epoch Training", ncols=100, disable=(rank != 0)):
            img = img.to(rank)
            label = label.to(rank)

            predict = model(img)
            loss = criterion(predict, label)
            bubble_acc += ((predict[:, 0] > 0) == label[:, 0]).sum()
            film_acc += ((predict[:, 1] > 0) == label[:, 1]).sum()
            artifact_acc += ((predict[:, 2] > 0) == label[:, 2]).sum()
            shade_acc += ((predict[:, 3] > 0) == label[:, 3]).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_total += img.shape[0]

        scheduler.step()

        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            "{}/{}.pt".format(config["output_path"], epoch),
        )
        model.eval()
        with torch.no_grad():
            for img, label in tqdm(valid_dataloader, desc=f"{epoch}Epoch validation", ncols=100, disable=(rank != 0)):
                img = img.to(rank)
                label = label.to(rank)
                predict = model(img)
                bubble_val += ((predict[:, 0] > 0) == label[:, 0]).sum()
                film_val += ((predict[:, 1] > 0) == label[:, 1]).sum()
                artifact_val += ((predict[:, 2] > 0) == label[:, 2]).sum()
                shade_val += ((predict[:, 3] > 0) == label[:, 3]).sum()
                label = label.to(dtype=torch.float32)

                loss = criterion(predict, label)
                val_loss += loss
                val_total += img.shape[0]

        # del valid_dataloader, valid_sampler
        # gc.collect()

        # scheduler.step(val_loss)

        if rank == 0:
            # bubble_acc /= train_total
            # film_acc /= train_total
            # artifact_acc /= train_total
            # shade_acc /= train_total
            bubble_val /= val_total
            film_val /= val_total
            artifact_val /= val_total
            shade_val /= val_total

            print(
                # train_acc: {bubble_acc:.4f} {film_acc:.4f} {artifact_acc:.4f} {shade_acc:.4f}, train_loss: {train_loss:.4f}
                f"Epoch {epoch} val_acc: {bubble_val:.4f} {film_val:.4f} {artifact_val:.4f} {shade_val:.4f}, val_loss: {val_loss:.4f}"
            )


def run_training():
    # args = parse_args()
    world_size = torch.cuda.device_count()
    # mp.spawn(train, args=(world_size, args.config), nprocs=world_size, join=True)
    mp.spawn(train, args=(world_size, "train_anomaly"), nprocs=world_size, join=True)


if __name__ == "__main__":
    run_training()

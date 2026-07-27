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
    df_train = df[df["train_or_test"] == "train"].reset_index(drop=True)
    df_normal = df_train[
        (df_train["BUBBLE"] == 0)
        & (df_train["FILM_DISPENSE"] == 0)
        & (df_train["ARTIFACT"] == 0)
        & (df_train["NO_AGAR"] == 0)
    ]
    df_noagar = df_train[(df_train["NO_AGAR"] == 1)]
    df_bubble = df_train[(df_train["BUBBLE"] == 1) & (df_train["NO_AGAR"] == 0)]
    df_film = df_train[(df_train["FILM_DISPENSE"] == 1) & (df_train["BUBBLE"] == 0) & (df_train["NO_AGAR"] == 0)]
    df_artifact = df_train[
        (df_train["ARTIFACT"] == 1)
        & (df_train["FILM_DISPENSE"] == 0)
        & (df_train["BUBBLE"] == 0)
        & (df_train["NO_AGAR"] == 0)
    ]
    # train_dataset = AnomalyDataset(df_train, train=True)
    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # train_dataloader = DataLoader(
    #     train_dataset, sampler=train_sampler, batch_size=config["batch_size"], num_workers=2, pin_memory=True
    # )

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    # 총 샘플 수
    # total = 11000

    # # 각 클래스의 positive 샘플 수
    # positive_counts = torch.tensor([190, 1539, 189, 457], dtype=torch.float32)

    # # pos_weight = negative / positive
    # # 음성 샘플 수 = total - positive
    # neg_counts = total - positive_counts
    # pos_weight = neg_counts / positive_counts

    # # BCEWithLogitsLoss에 적용
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(rank))
    criterion = nn.BCEWithLogitsLoss()

    # ★이관 정리(2026-07-27): 아래는 junhyeok 실험 resume(외부 체크포인트 bfas/499.pt) — 우리쪽 clean
    #   재학습에선 비활성. 이어학습하려면 우리 관리 체크포인트 경로로 바꿔 주석 해제.
    # checkpoint = torch.load("/home/junhyeok/anomaly/model/bfas/499.pt")
    # model.module.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler"])

    # 훈련 루프 (원본은 resume 위해 range(88,500). clean 재학습은 range(0, num_epoch))
    for epoch in range(0, config.get("num_epoch", 100)):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        bubble_acc = 0.0
        film_acc = 0.0
        artifact_acc = 0.0
        shade_acc = 0.0
        no_agar_acc = 0.0
        bubble_val = 0.0
        film_val = 0.0
        artifact_val = 0.0
        shade_val = 0.0
        no_agar_val = 0.0
        train_total = 0
        val_total = 0
        # model.module.load_state_dict(torch.load(f"/home/junhyeok/anomaly/model/bfas/{epoch}.pt")["model_state_dict"])

        df_sampled = (
            pd.concat(
                [
                    df_normal.sample(n=(len(df_normal) // 16)),
                    df_bubble.sample(n=(len(df_bubble) // 16)),
                    df_film.sample(n=(len(df_film) // 16)),
                    df_artifact.sample(n=(len(df_artifact) // 16)),
                    df_noagar.sample(n=(len(df_noagar) // 16)),
                ]
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )
        train_dataset = AnomalyDataset(df_sampled, train=True)
        train_sampler = DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=config["batch_size"], num_workers=32, pin_memory=True
        )

        for img, label in tqdm(train_dataloader, desc=f"{epoch}Epoch Training", ncols=100, disable=(rank != 0)):
            img = img.to(rank)
            label = label.to(rank)
            label = label.to(dtype=torch.float32)

            predict = model(img)
            loss = criterion(predict, label)
            bubble_acc += ((predict[:, 0] > 0) == label[:, 0]).sum()
            film_acc += ((predict[:, 1] > 0) == label[:, 1]).sum()
            artifact_acc += ((predict[:, 2] > 0) == label[:, 2]).sum()
            no_agar_acc += ((predict[:, 3] > 0) == label[:, 3]).sum()

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
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            "{}/{}.pt".format(config["output_path"], epoch),
        )

        # del valid_dataloader, valid_sampler
        # gc.collect()

        # scheduler.step(val_loss)

        if rank == 0:
            model.eval()
            with torch.no_grad():
                for img, label in tqdm(
                    valid_dataloader, desc=f"{epoch}Epoch validation", ncols=100, disable=(rank != 0)
                ):
                    img = img.to(rank)
                    label = label.to(rank)
                    predict = model(img)
                    label = label.to(dtype=torch.float32)

                    bubble_val += ((predict[:, 0] > 0) == label[:, 0]).sum()
                    film_val += ((predict[:, 1] > 0) == label[:, 1]).sum()
                    artifact_val += ((predict[:, 2] > 0) == label[:, 2]).sum()
                    no_agar_val += ((predict[:, 3] > 0) == label[:, 3]).sum()

                    loss = criterion(predict, label)
                    val_loss += loss
                    val_total += img.shape[0]
            bubble_acc /= train_total
            film_acc /= train_total
            artifact_acc /= train_total
            no_agar_acc /= train_total
            bubble_val /= val_total
            film_val /= val_total
            artifact_val /= val_total
            no_agar_val /= val_total

            print(
                f"Epoch {epoch} train_acc: {bubble_acc:.4f} {film_acc:.4f} {artifact_acc:.4f} {no_agar_acc:.4f}, train_loss: {train_loss:.4f} val_acc: {bubble_val:.4f} {film_val:.4f} {artifact_val:.4f} {no_agar_val:.4f}, val_loss: {val_loss:.4f}"
            )
            # bubble_acc /= train_total
            # film_acc /= train_total
            # artifact_acc /= train_total
            # shade_acc /= train_total
            # bubble_val /= val_total
            # film_val /= val_total
            # artifact_val /= val_total
            # shade_val /= val_total

            # print(
            #     f"Epoch {epoch} train_acc: {bubble_acc:.4f} {film_acc:.4f} {artifact_acc:.4f} {shade_acc:.4f}, train_loss: {train_loss:.4f}val_acc: {bubble_val:.4f} {film_val:.4f} {artifact_val:.4f} {shade_val:.4f}, val_loss: {val_loss:.4f}"
            # )


def run_training():
    # args = parse_args()
    world_size = torch.cuda.device_count()
    # mp.spawn(train, args=(world_size, args.config), nprocs=world_size, join=True)
    mp.spawn(train, args=(world_size, "train_anomaly"), nprocs=world_size, join=True)


if __name__ == "__main__":
    run_training()

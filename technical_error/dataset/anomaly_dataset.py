import ast

import kornia.augmentation as K
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision import transforms as T


class AnomalyDataset(Dataset):
    def __init__(self, df, train: bool):
        self.train = train
        self.image_paths = df["safetensors"]  # List of file paths
        self.labels = df["bubble"]
        if train:
            self.train_augment = torch.nn.Sequential(
                K.RandomResizedCrop((224, 224), scale=(0.9, 1.0), ratio=(0.95, 1.05), p=1.0, same_on_batch=True),
                K.RandomHorizontalFlip(p=0.5, same_on_batch=True),
                K.RandomRotation(degrees=5.0, p=1.0, same_on_batch=True),
                K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])),
            )
        else:
            self.valid_augment = torch.nn.Sequential(
                K.Resize((224, 224), p=1.0), K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5]))
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        imgs = load_file(img_path)["images"]

        if self.train:
            imgs = self.train_augment(imgs)
        else:
            imgs = self.valid_augment(imgs)

        label = torch.tensor(self.labels[idx], dtype=torch.bool)
        return imgs, label


class TestDataset(Dataset):
    def __init__(self, df, img_len):
        self.df = df
        self.image_paths = df["image_list"].str.split(",")
        self.test_augment = torch.nn.Sequential(K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])))
        self.to_tensor = T.ToTensor()
        self.img_len = img_len

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        imgs = [self.to_tensor(Image.open(p).convert("L")) for p in img_path[: (self.img_len + 1)]]
        img = torch.stack(imgs)

        img = self.test_augment(img)

        return img, torch.tensor(idx)

    # def __init__(self, df):
    #     self.df = df
    #     self.image_paths = df["image_safetensors_path"]
    #     self.test_augment = torch.nn.Sequential(K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])))
    #     self.to_tensor = T.ToTensor()

    # def __len__(self):
    #     return len(self.image_paths)

    # def __getitem__(self, idx: int):
    #     img_path = self.image_paths[idx]
    #     imgs = load_file(img_path)["image"]

    #     imgs = self.test_augment(imgs / 255.0)

    #     return imgs, torch.tensor(idx)

    # def __init__(self, df):
    #     self.df = df
    #     self.image_paths = df["safetensors"]
    #     self.test_augment = torch.nn.Sequential(
    #         K.Resize((224, 224), p=1.0), K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5]))
    #     )

    # def __len__(self):
    #     return len(self.image_paths)

    # def __getitem__(self, idx: int):
    #     img_path = self.image_paths[idx]
    #     img = load_file(img_path)["images"]
    #     img = self.test_augment(img)

    #     return img, torch.tensor(idx)


class SafetensorsWellDataset(Dataset):
    """drugbug당 하나의 .safetensors(shape=(concentration_len+1)*(img_len+1), 1, 224, 224)에서,
    df가 well 단위로 explode돼있다는 전제 하에 각 row의 concentration_index번째 블록(img_len+1 프레임)만 골라 반환.

    TestDataset과 동일하게 img_len(=config의 time_len)을 받고 내부에서 +1 해서 블록 크기를 잡는다 —
    호출부에서 두 Dataset을 그대로 맞바꿔 쓸 수 있게 하기 위함.

    concentration_index: 0 = control, 1..concentration_len = concentration_list 순서.
    predict_film_temp.explode_to_wells()로 만든 df와 짝을 이뤄서 쓴다.
    """

    def __init__(self, df, img_len: int):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df["image_safetensors_path"]
        self.concentration_index = self.df["concentration_index"]
        self.block_size = img_len + 1
        self.test_augment = torch.nn.Sequential(K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        well_idx = int(self.concentration_index[idx])
        start = well_idx * self.block_size
        imgs = load_file(self.image_paths[idx])["image"][start : start + self.block_size]

        imgs = imgs.float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        imgs = self.test_augment(imgs)

        return imgs, torch.tensor(idx)

import os

import torch
from resnet.resnet import resnet18
from torch import nn


class AnomalyModel(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=1, bidirectional=True):
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        state_dict = torch.load(os.environ.get("TECH_ERR_BACKBONE", "/home/kptae/data/technical_error/backbone/simclr_epoch10.pth"))["model_state_dict"]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(state_dict)

        self.lstm = nn.LSTM(
            input_size=feature_dim,  # 각 시점의 입력 차원 (예: 512)
            hidden_size=hidden_dim,  # LSTM이 내부적으로 사용하는 hidden state 차원
            num_layers=num_layers,  # LSTM 층을 몇 개 쌓을지 (예: 1~2 추천)
            batch_first=True,  # 입력이 (batch, seq_len, feature) 순서
            bidirectional=bidirectional,  # 양방향 LSTM 여부
        )
        self.feature_dim = feature_dim
        direction_factor = 2 if bidirectional else 1
        # self.bubble_classifier = nn.Linear(512, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * direction_factor),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        features = self.backbone(x.reshape(b * t, c, h, w))
        lstm_out, (hn, cn) = self.lstm(features.reshape(b, t, self.feature_dim))
        forward_hidden = hn[0]
        backward_hidden = hn[1]
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        # bubble = self.bubble_classifier(features).squeeze()
        predict = self.classifier(combined_hidden).squeeze()
        return predict


class FilmModel(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=1, bidirectional=True):
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        state_dict = torch.load(os.environ.get("TECH_ERR_BACKBONE", "/home/kptae/data/technical_error/backbone/simclr_epoch10.pth"))["model_state_dict"]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(state_dict)

        self.lstm = nn.LSTM(
            input_size=feature_dim,  # 각 시점의 입력 차원 (예: 512)
            hidden_size=hidden_dim,  # LSTM이 내부적으로 사용하는 hidden state 차원
            num_layers=num_layers,  # LSTM 층을 몇 개 쌓을지 (예: 1~2 추천)
            batch_first=True,  # 입력이 (batch, seq_len, feature) 순서
            bidirectional=bidirectional,  # 양방향 LSTM 여부
        )
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        features = self.backbone(x.reshape(b * t, c, h, w))
        lstm_out, (hn, cn) = self.lstm(features.reshape(b, t, self.feature_dim))
        forward_hidden = hn[0]
        backward_hidden = hn[1]
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        predict = self.classifier(combined_hidden).squeeze()
        return predict

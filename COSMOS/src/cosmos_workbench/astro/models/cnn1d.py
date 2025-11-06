
from __future__ import annotations
import torch
from torch import nn

class CNN1D(nn.Module):
    def __init__(self, length: int, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)

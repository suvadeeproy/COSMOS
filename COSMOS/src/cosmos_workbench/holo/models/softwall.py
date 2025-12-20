
from __future__ import annotations
import torch
from torch import nn

class SoftWallParam(nn.Module):
    def __init__(self, init_k: float = 0.2):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([init_k], dtype=torch.float32))

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        return 4.0 * torch.relu(self.k) * (n + 1.0)

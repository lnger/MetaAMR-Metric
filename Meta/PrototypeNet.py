import torch.nn as nn
import torch
import numpy as np

class ImprovedPrototypeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, hidden_dim//4))
        self.prototype_std = nn.Parameter(torch.ones(n_classes, 1))
        
    def forward(self, x):
        encoded = self.encoder(x)
        # 计算马氏距离 (考虑类别方差)
        diff = encoded.unsqueeze(1) - self.prototypes.unsqueeze(0)
        inv_cov = 1.0 / (self.prototype_std.unsqueeze(0) + 1e-6)
        distances = torch.sum(diff.pow(2) * inv_cov, dim=-1)
        return -distances
"""
State Value Critic Baseline Network for PPO.
"""

import numpy as np
import torch
import torch.nn as nn
from ppo.policy import layer_init


class ValueNetwork(nn.Module):
    """Maps observation (83-D) to scalar state-value estimate V(s)."""
    def __init__(self, obs_dim: int = 83, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

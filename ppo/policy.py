"""
Gaussian Actor Policy Network for Continuous CPG Modulation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class GaussianPolicy(nn.Module):
    """Maps observation (83-D) to mean CPG action modulations (8-D) with trainable exploration std."""
    def __init__(self, obs_dim: int = 83, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def forward(self, obs: torch.Tensor, action: torch.Tensor = None):
        mean = self.net(obs)
        std = torch.exp(self.log_std.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(obs)

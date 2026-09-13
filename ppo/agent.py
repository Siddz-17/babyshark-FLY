"""
Unified PPO Actor-Critic Agent Module.
Supports Gaussian continuous action sampling, deterministic evaluation,
value function baseline estimation, and seamless checkpoint loading.
"""

from typing import Tuple, Optional
import os
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight initialization with constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """
    Two-headed Actor-Critic architecture:
    - Actor: Maps observation -> Continuous CPG action modulations
    - Critic: Maps observation -> Scalar state value V(s)
    """

    def __init__(self, obs_dim: int = 83, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Critic (Value Network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor (Policy Mean Network)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        # Trainable log standard deviation for continuous Gaussian exploration
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar state value estimate V(s)."""
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples action from Gaussian policy, evaluates log_prob, entropy, and state value.
        """
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic policy action (mean without exploration noise)."""
        with torch.no_grad():
            return self.actor_mean(obs)

    def load_checkpoint(self, checkpoint_path: str, map_location="cpu"):
        """Safely loads weights from checkpoint dict or raw state_dict."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        data = torch.load(checkpoint_path, map_location=map_location)
        state_dict = data.get("model_state_dict", data)
        self.load_state_dict(state_dict)

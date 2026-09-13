"""
Combined ActorCritic Agent Module.
"""

from typing import Tuple
import torch
import torch.nn as nn
from ppo.policy import GaussianPolicy
from ppo.value import ValueNetwork


class PPOAgent(nn.Module):
    def __init__(self, obs_dim: int = 83, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.actor = GaussianPolicy(obs_dim, action_dim, hidden_dim)
        self.critic = ValueNetwork(obs_dim, hidden_dim)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy = self.actor(obs, action)
        value = self.critic(obs)
        return action, log_prob, entropy, value

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(obs)

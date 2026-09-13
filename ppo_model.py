"""
Actor-Critic Neural Network Architecture for PPO Drosophila Dance Policy.
Implements continuous Gaussian policy and value baseline network.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Applies orthogonal weight initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Two-headed Actor-Critic architecture:
    - Actor: Maps (Proprioception + DNs + Audio State) -> CPG Action Modulations
    - Critic: Maps (Proprioception + DNs + Audio State) -> State Value V(s)
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
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples or evaluates an action given observation obs.
        
        Returns:
            action: Sampled action tensor
            log_prob: Log probability of the action
            entropy: Policy entropy for exploration bonus
            value: State value estimate V(s)
        """
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the mean action without stochastic noise (for evaluation)."""
        with torch.no_grad():
            return self.actor_mean(obs)


if __name__ == "__main__":
    net = ActorCritic()
    dummy_obs = torch.randn(4, 83)
    action, log_prob, entropy, value = net.get_action_and_value(dummy_obs)
    print("ActorCritic Network verified:")
    print(f"  - Action Shape:   {action.shape}")
    print(f"  - Log Prob Shape: {log_prob.shape}")
    print(f"  - Value Shape:    {value.shape}")
    print("PPO Model operational!")

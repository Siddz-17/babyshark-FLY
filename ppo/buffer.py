"""
Rollout Buffer and Generalized Advantage Estimation (GAE) for Vectorized PPO.
"""

import numpy as np
import torch


class RolloutBuffer:
    """Stores transitions from vectorized environments and computes GAE advantages."""
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)

    def insert(self, step: int, obs: torch.Tensor, action: torch.Tensor, logprob: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor):
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value

    def compute_returns_and_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + self.values
        return returns, advantages

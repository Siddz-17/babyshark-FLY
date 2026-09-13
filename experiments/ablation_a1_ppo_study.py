"""
Experiment A1: Controlled PPO-Trained Connectome Ablation Study.
Eliminates all confounds from Experiment A0:
1. Strict sensory equality: Every condition receives identical raw acoustic features
   [raw_amp, onset, low_band, mid_band, high_band] - NO beat_phase, NO beat_pulse, NO BPM!
2. Neutral motor baseline: CPG baseline stepping frequency is fixed at 2.0 Hz (does NOT know music tempo).
3. Equal training budget: All conditions are trained via PPO under identical hyperparameters and seeds.
4. Statistical replication: Evaluated across 3 seeds and 3 distinct tempo test tracks (110, 120, 130 BPM).

Conditions:
  1. Biological Connectome + PPO (FlyWire Tonotopic Topology)
  2. Degree-Preserving Bipartite Rewired Connectome + PPO (Matched in/out degrees & weights)
  3. Direct MLP + PPO (Trained without connectome intermediate)
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dance_gym_env import DrosophilaDanceGymEnv
from connectome_auditory import DrosophilaAuditoryCircuit
from ppo.agent import PPOAgent
from ppo.buffer import RolloutBuffer


def degree_preserving_bipartite_shuffle(W: np.ndarray, n_swaps_factor: int = 15, seed: int = 42) -> np.ndarray:
    """
    Applies degree-preserving bipartite edge swapping (Maslov-Sneppen style).
    Preserves exact row in-degrees, column out-degrees, and edge weight distribution,
    while destroying specific tonotopic wiring patterns.
    """
    rng = np.random.RandomState(seed)
    W_shuf = W.copy()
    rows, cols = np.nonzero(W_shuf)
    n_edges = len(rows)
    if n_edges < 4:
        return W_shuf

    total_swaps = n_edges * n_swaps_factor
    for _ in range(total_swaps):
        i1, i2 = rng.choice(n_edges, size=2, replace=False)
        u1, v1 = rows[i1], cols[i1]
        u2, v2 = rows[i2], cols[i2]

        if u1 == u2 or v1 == v2:
            continue
        if W_shuf[u1, v2] == 0 and W_shuf[u2, v1] == 0:
            w1 = W_shuf[u1, v1]
            w2 = W_shuf[u2, v2]
            W_shuf[u1, v1] = 0
            W_shuf[u2, v2] = 0
            W_shuf[u1, v2] = w1
            W_shuf[u2, v1] = w2
            cols[i1] = v2
            cols[i2] = v1

    return W_shuf


class DirectMLPAgent(nn.Module):
    """Matched Direct MLP Actor-Critic (59 obs: 54 proprio + 5 audio -> 8 actions)."""
    def __init__(self, obs_dim: int = 59, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor_mean(obs)


def train_condition_ppo(
    condition_name: str,
    circuit_override: DrosophilaAuditoryCircuit = None,
    is_direct_mlp: bool = False,
    total_timesteps: int = 12000,
    seed: int = 42,
    device: str = "cpu",
) -> nn.Module:
    """Trains a condition under identical PPO budget and hyperparameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_steps = 256
    num_envs = 2
    batch_size = num_steps * num_envs
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    update_epochs = 4
    minibatch_size = 64

    envs = [
        DrosophilaDanceGymEnv(
            duration=3.0, dt=0.002, default_bpm=120.0,
            randomize_tempo=True, backend="simple",
            blind_to_tempo=True,
            custom_circuit=circuit_override,
        )
        for _ in range(num_envs)
    ]

    obs_dim = 59 if is_direct_mlp else 83
    action_dim = 8

    if is_direct_mlp:
        agent = DirectMLPAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    else:
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    buffer = RolloutBuffer(num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim, device=device)

    next_obs_list = []
    for env in envs:
        o, _ = env.reset(seed=seed)
        if is_direct_mlp:
            # Drop DN dimensions (54:78) -> keep proprio (0:54) and audio (78:83)
            o = np.concatenate([o[:54], o[78:]])
        next_obs_list.append(o)

    next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)

    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            step_actions = action.cpu().numpy()
            step_rewards = []
            step_dones = []
            step_next_obs = []

            for env_idx, env in enumerate(envs):
                o, r, term, trunc, info = env.step(step_actions[env_idx])
                d = term or trunc
                if is_direct_mlp:
                    o = np.concatenate([o[:54], o[78:]])
                if d:
                    o, _ = env.reset()
                    if is_direct_mlp:
                        o = np.concatenate([o[:54], o[78:]])
                step_rewards.append(r)
                step_dones.append(d)
                step_next_obs.append(o)

            rewards_tensor = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(step_dones, dtype=torch.float32).to(device)

            buffer.insert(step, next_obs, action, logprob, rewards_tensor, dones_tensor, value)
            next_obs = torch.tensor(np.array(step_next_obs), dtype=torch.float32).to(device)
            next_done = dones_tensor

        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            returns, advantages = buffer.compute_returns_and_advantages(next_value, next_done, gamma=gamma, gae_lambda=gae_lambda)

        b_obs = buffer.obs.reshape((-1, obs_dim))
        b_actions = buffer.actions.reshape((-1, action_dim))
        b_logprobs = buffer.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[minds], b_actions[minds])
                logratio = newlogprob - b_logprobs[minds]
                ratio = torch.exp(logratio)

                mb_advantages = b_advantages[minds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[minds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    for env in envs:
        env.close()

    return agent


def evaluate_policy_condition(
    agent: nn.Module,
    circuit_override: DrosophilaAuditoryCircuit = None,
    is_direct_mlp: bool = False,
    tempos: List[float] = [110.0, 120.0, 130.0],
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluates a trained policy across multiple test tempos."""
    agent.eval()
    rewards, syncs, stabilities, energies = [], [], [], []

    for bpm in tempos:
        env = DrosophilaDanceGymEnv(
            duration=3.0, dt=0.002, default_bpm=bpm,
            randomize_tempo=False, backend="simple",
            blind_to_tempo=True,
            custom_circuit=circuit_override,
        )
        obs, _ = env.reset(seed=seed)
        if is_direct_mlp:
            obs = np.concatenate([obs[:54], obs[78:]])

        ep_rew, ep_sync, ep_stab, ep_eng = 0.0, [], [], []
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = agent.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if is_direct_mlp:
                obs = np.concatenate([obs[:54], obs[78:]])

            ep_rew += reward
            ep_sync.append(info.get("beat_sync", 0.0))
            ep_stab.append(info.get("stability", 0.0))
            ep_eng.append(info.get("energy_penalty", 0.0))

        env.close()
        rewards.append(ep_rew)
        syncs.append(np.mean(ep_sync))
        stabilities.append(np.mean(ep_stab))
        energies.append(np.mean(ep_eng))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_footfall_sync": float(np.mean(syncs)),
        "std_footfall_sync": float(np.std(syncs)),
        "mean_stability": float(np.mean(stabilities)),
        "mean_energy": float(np.mean(energies)),
    }


def run_experiment_a1():
    print("=" * 85)
    print("EXPERIMENT A1: CONTROLLED PPO-TRAINED CONNECTOME ABLATION STUDY")
    print("No tempo leakage | Equal sensory input | Degree-preserving bipartite control | PPO trained")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware compute device: {device}")

    # Load original biological matrices
    W_ja = np.load("data/flywire/W_jon_ammc.npy")
    W_dn = np.load("data/flywire/W_ammc_dn.npy")

    # Generate degree-preserving bipartite rewired matrices
    W_ja_rewired = degree_preserving_bipartite_shuffle(W_ja, n_swaps_factor=20, seed=42)
    W_dn_rewired = degree_preserving_bipartite_shuffle(W_dn, n_swaps_factor=20, seed=42)

    seeds = [42, 101, 777]
    tempos = [110.0, 120.0, 130.0]
    total_steps = 10000  # 10k steps per seed per condition for clean statistical comparison

    conditions = [
        {
            "name": "1. Biological Connectome + PPO",
            "circuit": DrosophilaAuditoryCircuit(custom_W_jon_ammc=W_ja, custom_W_ammc_dn=W_dn),
            "is_direct_mlp": False,
        },
        {
            "name": "2. Degree-Preserving Rewired + PPO",
            "circuit": DrosophilaAuditoryCircuit(custom_W_jon_ammc=W_ja_rewired, custom_W_ammc_dn=W_dn_rewired),
            "is_direct_mlp": False,
        },
        {
            "name": "3. Direct MLP + PPO",
            "circuit": None,
            "is_direct_mlp": True,
        },
    ]

    all_results = {}

    for cond in conditions:
        name = cond["name"]
        print(f"\n>>> Running Condition: {name}")
        seed_metrics = []

        for s_idx, seed in enumerate(seeds):
            t0 = time.time()
            print(f"  [Seed {seed} ({s_idx+1}/{len(seeds)})] Training PPO ({total_steps} steps)...", end="", flush=True)
            trained_agent = train_condition_ppo(
                condition_name=name,
                circuit_override=cond["circuit"],
                is_direct_mlp=cond["is_direct_mlp"],
                total_timesteps=total_steps,
                seed=seed,
                device=device,
            )
            eval_metrics = evaluate_policy_condition(
                agent=trained_agent,
                circuit_override=cond["circuit"],
                is_direct_mlp=cond["is_direct_mlp"],
                tempos=tempos,
                seed=seed,
                device=device,
            )
            elapsed = time.time() - t0
            print(f" Done ({elapsed:.1f}s) | Reward: {eval_metrics['mean_reward']:.2f} | Sync: {eval_metrics['mean_footfall_sync']:.4f}")
            seed_metrics.append(eval_metrics)

        all_results[name] = {
            "mean_reward": float(np.mean([m["mean_reward"] for m in seed_metrics])),
            "std_reward": float(np.std([m["mean_reward"] for m in seed_metrics])),
            "mean_footfall_sync": float(np.mean([m["mean_footfall_sync"] for m in seed_metrics])),
            "std_footfall_sync": float(np.std([m["mean_footfall_sync"] for m in seed_metrics])),
            "mean_stability": float(np.mean([m["mean_stability"] for m in seed_metrics])),
            "mean_energy": float(np.mean([m["mean_energy"] for m in seed_metrics])),
        }

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Condition':<38} | {'Mean Reward':<16} | {'Footfall Sync':<18} | {'Stability':<10} | {'Energy':<8}")
    print("-" * 90)
    for name, m in all_results.items():
        print(
            f"{name:<38} | "
            f"{m['mean_reward']:>6.3f} +/- {m['std_reward']:<6.3f} | "
            f"{m['mean_footfall_sync']:>7.4f} +/- {m['std_footfall_sync']:<6.4f} | "
            f"{m['mean_stability']:>10.3f} | "
            f"{m['mean_energy']:>8.3f}"
        )
    print("=" * 90)

    # Save JSON summary
    summary_path = os.path.join(repo_root, "data", "ablation_a1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved numerical summary to: {summary_path}")

    # Plot Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    cond_names = [k.replace(" + PPO", "\n+ PPO") for k in all_results.keys()]
    rewards = [all_results[k]["mean_reward"] for k in all_results.keys()]
    reward_errs = [all_results[k]["std_reward"] for k in all_results.keys()]
    syncs = [all_results[k]["mean_footfall_sync"] for k in all_results.keys()]
    sync_errs = [all_results[k]["std_footfall_sync"] for k in all_results.keys()]

    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    # Reward subplot
    axes[0].bar(cond_names, rewards, yerr=reward_errs, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_ylabel("Cumulative Episode Reward", fontsize=11, fontweight="bold")
    axes[0].set_title("PPO Learned Episode Reward (Experiment A1)", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Footfall Sync subplot
    axes[1].bar(cond_names, syncs, yerr=sync_errs, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_ylabel("Physical Footfall Sync Score", fontsize=11, fontweight="bold")
    axes[1].set_title("Physical Footfall Beat Synchrony", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Experiment A1: Controlled PPO Ablation Study (No BPM Leakage)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(repo_root, "experiments", "ablation_a1_results.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved publication plot to: {fig_path}")


if __name__ == "__main__":
    run_experiment_a1()

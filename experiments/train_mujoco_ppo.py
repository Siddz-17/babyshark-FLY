"""
MuJoCo PPO Training Runner for 204-Actuator Drosophila Avatar.
Executes closed-loop PPO reinforcement learning directly against EPFL NeuroMechFly / MuJoCo 3.x.
Full telemetry tracking:
- Episode Reward
- Footfall Synchronization (actual contact touchdowns vs beat timestamps)
- Fall Rate (loss of upright equilibrium)
- Forward Velocity (mm/s)
- Energy Cost
- Policy Loss & Value Loss
- Policy Entropy
- Approximate KL Divergence
- PPO Clip Fraction
- Episode Length

Saves:
- Checkpoint: checkpoints/mujoco_ppo_milestone_{steps}.pt
- Telemetry JSON: data/mujoco_ppo_telemetry.json
- Training Curves Figure: experiments/mujoco_training_curves.png
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
from ppo.agent import PPOAgent
from ppo.buffer import RolloutBuffer


def train_mujoco_ppo(
    total_timesteps: int = 10000,
    num_envs: int = 2,
    num_steps: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    blind_to_tempo: bool = True,
    seed: int = 42,
    save_dir: str = "checkpoints",
):
    print("=" * 85)
    print("GENUINE MUJOCO 3.X PPO TRAINING RUNNER (204 ACTUATORS)")
    print(f"Total Steps: {total_timesteps} | Envs: {num_envs} | Steps/Rollout: {num_steps} | Blind to Tempo: {blind_to_tempo}")
    print("=" * 85)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(repo_root, "experiments"), exist_ok=True)

    print("Initializing MuJoCo environments (NeuroMechFly 204 actuators)...")
    envs = [
        DrosophilaDanceGymEnv(
            duration=3.0,
            dt=0.002,
            default_bpm=120.0,
            randomize_tempo=True,
            backend="mujoco",
            connectome_mode="biological",
            blind_to_tempo=blind_to_tempo,
        )
        for _ in range(num_envs)
    ]

    obs_dim = envs[0].observation_space.shape[0]  # 83
    action_dim = envs[0].action_space.shape[0]    # 8
    batch_size = num_envs * num_steps

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    buffer = RolloutBuffer(num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim, device=device)

    # Telemetry storage
    telemetry = {
        "global_steps": [],
        "iteration": [],
        "mean_reward": [],
        "footfall_sync": [],
        "fall_rate": [],
        "forward_velocity": [],
        "energy": [],
        "episode_length": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "throughput_fps": [],
    }

    # Reset environments
    next_obs_list = []
    for env in envs:
        o, _ = env.reset()
        next_obs_list.append(o)
    next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)

    global_step = 0
    start_time = time.time()
    num_iterations = total_timesteps // batch_size
    best_reward = -float("inf")

    # Rolling trackers
    ep_rewards = []
    ep_syncs = []
    ep_falls = []
    ep_vels = []
    ep_energies = []
    ep_lens = []

    cur_ep_lens = np.zeros(num_envs, dtype=int)
    cur_ep_rews = np.zeros(num_envs, dtype=float)
    cur_ep_syncs = [[] for _ in range(num_envs)]
    cur_ep_vels = [[] for _ in range(num_envs)]
    cur_ep_energies = [[] for _ in range(num_envs)]

    print("\nStarting MuJoCo PPO Rollouts...")
    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()

        for step in range(num_steps):
            global_step += num_envs

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            step_actions = action.cpu().numpy()
            step_rewards = []
            step_dones = []
            step_next_obs = []

            for env_idx, env in enumerate(envs):
                cur_ep_lens[env_idx] += 1
                o, r, term, trunc, info = env.step(step_actions[env_idx])
                d = term or trunc

                cur_ep_rews[env_idx] += r
                cur_ep_syncs[env_idx].append(info.get("beat_sync", 0.0))
                cur_ep_vels[env_idx].append(info.get("forward_velocity", 0.0))
                cur_ep_energies[env_idx].append(info.get("energy_penalty", 0.0))

                if d:
                    ep_rewards.append(cur_ep_rews[env_idx])
                    ep_lens.append(cur_ep_lens[env_idx])
                    ep_syncs.append(np.mean(cur_ep_syncs[env_idx]) if cur_ep_syncs[env_idx] else 0.0)
                    ep_vels.append(np.mean(cur_ep_vels[env_idx]) if cur_ep_vels[env_idx] else 0.0)
                    ep_energies.append(np.mean(cur_ep_energies[env_idx]) if cur_ep_energies[env_idx] else 0.0)
                    ep_falls.append(1.0 if info.get("is_fallen", False) else 0.0)

                    cur_ep_rews[env_idx] = 0.0
                    cur_ep_lens[env_idx] = 0
                    cur_ep_syncs[env_idx] = []
                    cur_ep_vels[env_idx] = []
                    cur_ep_energies[env_idx] = []

                    o, _ = env.reset()

                step_rewards.append(r)
                step_dones.append(d)
                step_next_obs.append(o)

            rewards_tensor = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(step_dones, dtype=torch.float32).to(device)

            buffer.insert(step, next_obs, action, logprob, rewards_tensor, dones_tensor, value)
            next_obs = torch.tensor(np.array(step_next_obs), dtype=torch.float32).to(device)
            next_done = dones_tensor

        # Compute GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            returns, advantages = buffer.compute_returns_and_advantages(next_value, next_done, gamma=gamma, gae_lambda=gae_lambda)

        b_obs = buffer.obs.reshape((-1, obs_dim))
        b_actions = buffer.actions.reshape((-1, action_dim))
        b_logprobs = buffer.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        pg_losses = []
        v_losses = []
        entropies = []
        approx_kls = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[minds], b_actions[minds])
                logratio = newlogprob - b_logprobs[minds]
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    approx_kls.append(approx_kl.item())
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_advantages = b_advantages[minds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[minds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

        fps = int(batch_size / (time.time() - iter_start + 1e-6))
        total_fps = int(global_step / (time.time() - start_time + 1e-6))

        m_rew = float(np.mean(ep_rewards[-20:])) if ep_rewards else float(rewards_tensor.mean().item())
        m_sync = float(np.mean(ep_syncs[-20:])) if ep_syncs else 0.0
        m_fall = float(np.mean(ep_falls[-20:])) if ep_falls else 0.0
        m_vel = float(np.mean(ep_vels[-20:])) if ep_vels else 0.0
        m_eng = float(np.mean(ep_energies[-20:])) if ep_energies else 0.0
        m_len = float(np.mean(ep_lens[-20:])) if ep_lens else float(step + 1)

        m_pg = float(np.mean(pg_losses))
        m_vl = float(np.mean(v_losses))
        m_ent = float(np.mean(entropies))
        m_kl = float(np.mean(approx_kls))
        m_clip = float(np.mean(clipfracs))

        # Record telemetry
        telemetry["global_steps"].append(global_step)
        telemetry["iteration"].append(iteration)
        telemetry["mean_reward"].append(m_rew)
        telemetry["footfall_sync"].append(m_sync)
        telemetry["fall_rate"].append(m_fall)
        telemetry["forward_velocity"].append(m_vel)
        telemetry["energy"].append(m_eng)
        telemetry["episode_length"].append(m_len)
        telemetry["policy_loss"].append(m_pg)
        telemetry["value_loss"].append(m_vl)
        telemetry["entropy"].append(m_ent)
        telemetry["approx_kl"].append(m_kl)
        telemetry["clip_fraction"].append(m_clip)
        telemetry["throughput_fps"].append(fps)

        if m_rew > best_reward and iteration >= 2:
            best_reward = m_rew
            best_ckpt_path = os.path.join(save_dir, "mujoco_best_dance_policy.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "global_step": global_step,
                "best_reward": best_reward,
                "telemetry": telemetry,
            }, best_ckpt_path)

        print(
            f"Iter {iteration:2d}/{num_iterations:2d} | "
            f"Step: {global_step:5d} | "
            f"Rew: {m_rew:>6.2f} | "
            f"Sync: {m_sync:>6.4f} | "
            f"Falls: {m_fall:>4.2f} | "
            f"KL: {m_kl:>6.4f} | "
            f"V-Loss: {m_vl:>6.3f} | "
            f"Ent: {m_ent:>5.2f} | "
            f"FPS: {fps} ({total_fps} avg)"
        )

    for env in envs:
        env.close()

    # Save final milestone checkpoint
    milestone_path = os.path.join(save_dir, f"mujoco_ppo_milestone_{total_timesteps//1000}k.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "global_step": global_step,
        "telemetry": telemetry,
    }, milestone_path)
    print(f"\nSaved milestone checkpoint: {milestone_path}")

    # Save telemetry JSON
    telem_path = os.path.join(repo_root, "data", "mujoco_ppo_telemetry.json")
    with open(telem_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved numerical telemetry: {telem_path}")

    # Generate publication-grade telemetry figure (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    steps_arr = np.array(telemetry["global_steps"])

    # 1. Episode Reward
    axes[0, 0].plot(steps_arr, telemetry["mean_reward"], color="#2ecc71", linewidth=2.0)
    axes[0, 0].set_title("Episode Reward (MuJoCo 204 DoF)", fontweight="bold")
    axes[0, 0].set_xlabel("Environment Steps")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Footfall Synchronization
    axes[0, 1].plot(steps_arr, telemetry["footfall_sync"], color="#3498db", linewidth=2.0)
    axes[0, 1].set_title("Physical Footfall Beat Synchrony", fontweight="bold")
    axes[0, 1].set_xlabel("Environment Steps")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Fall Rate & Stability
    axes[0, 2].plot(steps_arr, telemetry["fall_rate"], color="#e74c3c", linewidth=2.0)
    axes[0, 2].set_title("Fall Rate (Loss of Balance)", fontweight="bold")
    axes[0, 2].set_xlabel("Environment Steps")
    axes[0, 2].grid(True, linestyle="--", alpha=0.5)

    # 4. Policy & Value Losses
    axes[1, 0].plot(steps_arr, telemetry["policy_loss"], label="Policy Loss", color="#9b59b6")
    axes[1, 0].plot(steps_arr, telemetry["value_loss"], label="Value Loss", color="#e67e22")
    axes[1, 0].set_title("PPO Loss Curves", fontweight="bold")
    axes[1, 0].set_xlabel("Environment Steps")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 5. Approximate KL Divergence
    axes[1, 1].plot(steps_arr, telemetry["approx_kl"], color="#1abc9c", linewidth=2.0)
    axes[1, 1].axhline(0.01, color="red", linestyle=":", label="KL Target (0.01)")
    axes[1, 1].set_title("Approximate KL Divergence", fontweight="bold")
    axes[1, 1].set_xlabel("Environment Steps")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    # 6. Policy Entropy
    axes[1, 2].plot(steps_arr, telemetry["entropy"], color="#34495e", linewidth=2.0)
    axes[1, 2].set_title("Policy Exploration Entropy", fontweight="bold")
    axes[1, 2].set_xlabel("Environment Steps")
    axes[1, 2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("MuJoCo 3.x 204-Actuator PPO Training Dynamics", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig_path = os.path.join(repo_root, "experiments", "mujoco_training_curves.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved publication telemetry figure: {fig_path}")

    return telemetry


if __name__ == "__main__":
    train_mujoco_ppo(total_timesteps=10000, num_envs=2, num_steps=256)

"""
PPO Training Engine for Connectome-Driven Drosophila Dance Controller.
Optimized for local CPU/GPU execution (Ryzen 5 5600H + RTX 3050).
Implements Vectorized PPO with GAE, Checkpointing, and Telemetry Logging.
"""

import os
import time
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dance_gym_env import DrosophilaDanceGymEnv
from ppo.agent import PPOAgent


def train_ppo(
    total_timesteps: int = 40000,
    num_envs: int = 4,
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
    checkpoint_dir: str = "checkpoints",
    model_save_path: str = "best_dance_policy.pt",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_full_path = os.path.join(checkpoint_dir, model_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STARTING DROSOPHILA CONNECTOME PPO TRAINING")
    print(f"Device: {device} | Envs: {num_envs} | Steps/Env: {num_steps} | Total: {total_timesteps}")
    print("=" * 70)

    # Initialize parallel environments
    envs = [DrosophilaDanceGymEnv(duration=3.0, dt=0.002, randomize_tempo=True) for _ in range(num_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    # Initialize ActorCritic policy network
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Storage buffers
    batch_size = num_envs * num_steps
    obs_buffer = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32).to(device)
    actions_buffer = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32).to(device)
    logprobs_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
    rewards_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
    dones_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
    values_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)

    # Initial environment reset
    next_obs_list = []
    for env in envs:
        o, _ = env.reset()
        next_obs_list.append(o)
    next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)

    # Logging telemetry
    global_step = 0
    start_time = time.time()
    best_mean_reward = -float("inf")
    episode_rewards = [[] for _ in range(num_envs)]
    completed_rewards = []
    beat_sync_records = []
    training_steps_log = []
    mean_rewards_log = []

    num_iterations = total_timesteps // batch_size
    print(f"Total Iterations: {num_iterations} (Batch size: {batch_size})")

    for iteration in range(1, num_iterations + 1):
        # Annealing learning rate
        frac = 1.0 - (iteration - 1.0) / num_iterations
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        # 1. Rollout phase
        for step in range(num_steps):
            global_step += num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buffer[step] = value
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # Step all environments
            action_np = action.cpu().numpy()
            step_rewards = []
            for env_idx, env in enumerate(envs):
                next_o, r, term, trunc, info = env.step(action_np[env_idx])
                step_rewards.append(r)
                episode_rewards[env_idx].append(r)
                if "beat_sync" in info:
                    beat_sync_records.append(info["beat_sync"])

                if term or trunc:
                    completed_rewards.append(sum(episode_rewards[env_idx]))
                    episode_rewards[env_idx] = []
                    next_o, _ = env.reset()

                next_obs_list[env_idx] = next_o

            rewards_buffer[step] = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)

        # 2. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        # Flatten batch
        b_obs = obs_buffer.reshape((-1, obs_dim))
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        # 3. PPO Optimization
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # 4. Evaluation & Logging
        current_mean_rew = float(np.mean(completed_rewards[-30:])) if len(completed_rewards) > 0 else float(torch.mean(rewards_buffer).item()) * num_steps
        current_beat_sync = float(np.mean(beat_sync_records[-200:])) if len(beat_sync_records) > 0 else 0.0
        elapsed_time = time.time() - start_time
        fps = int(global_step / (elapsed_time + 1e-6))

        training_steps_log.append(global_step)
        mean_rewards_log.append(current_mean_rew)

        # Checkpoint saving
        if current_mean_rew > best_mean_reward and iteration > 3:
            best_mean_reward = current_mean_rew
            torch.save({
                "iteration": iteration,
                "global_step": global_step,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_reward": best_mean_reward,
            }, best_model_full_path)
            saved_indicator = " [BEST SAVED!]"
        else:
            saved_indicator = ""

        if iteration % 5 == 0 or iteration == num_iterations:
            print(f"Iter {iteration:3d}/{num_iterations} | Step: {global_step:5d} | "
                  f"Throughput: {fps} steps/s | Mean Rew: {current_mean_rew:.2f} | "
                  f"Beat Sync: {current_beat_sync:.3f}{saved_indicator}")

    print("-" * 70)
    print(f"PPO Training Complete! Best Mean Reward: {best_mean_reward:.2f}")
    print(f"Model saved to: {os.path.abspath(best_model_full_path)}")

    # Generate Learning Curve Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        plt.plot(training_steps_log, mean_rewards_log, color="dodgerblue", lw=2, label="Mean Episode Reward")
        plt.title("Connectome Drosophila Dance PPO Training Curve", fontsize=13)
        plt.xlabel("Environment Timesteps")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        curve_path = "ppo_training_curves.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Training curve plot saved to: {os.path.abspath(curve_path)}")
    except Exception as e:
        print(f"Could not save training plot: {e}")

    print("=" * 70)
    return best_model_full_path


if __name__ == "__main__":
    train_ppo(total_timesteps=30000, num_envs=4, num_steps=256)

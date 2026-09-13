"""
Modular PPO Training Runner using ppo.agent and ppo.buffer.
"""

import os
import time
import yaml
import numpy as np
import torch
import torch.optim as optim

from dance_gym_env import DrosophilaDanceGymEnv
from ppo.agent import PPOAgent
from ppo.buffer import RolloutBuffer


def train(config_path: str = "configs/ppo.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    env_cfg = cfg.get("env", {})
    ppo_cfg = cfg.get("ppo", {})
    ckpt_cfg = cfg.get("checkpoints", {})

    duration = env_cfg.get("duration", 3.0)
    dt = env_cfg.get("dt", 0.002)
    default_bpm = env_cfg.get("default_bpm", 120.0)
    backend = env_cfg.get("backend", "simple")
    connectome_mode = env_cfg.get("connectome_mode", "biological")

    total_timesteps = ppo_cfg.get("total_timesteps", 30000)
    num_envs = ppo_cfg.get("num_envs", 4)
    num_steps = ppo_cfg.get("num_steps", 256)
    learning_rate = ppo_cfg.get("learning_rate", 3e-4)
    gamma = ppo_cfg.get("gamma", 0.99)
    gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
    update_epochs = ppo_cfg.get("update_epochs", 4)
    minibatch_size = ppo_cfg.get("minibatch_size", 64)
    clip_coef = ppo_cfg.get("clip_coef", 0.2)
    ent_coef = ppo_cfg.get("ent_coef", 0.01)
    vf_coef = ppo_cfg.get("vf_coef", 0.5)

    save_dir = ckpt_cfg.get("save_dir", "checkpoints")
    model_name = ckpt_cfg.get("model_name", "best_dance_policy.pt")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"PPO MODULAR TRAINING RUNNER (Backend: {backend.upper()}, Mode: {connectome_mode.upper()})")
    print(f"Device: {device} | Envs: {num_envs} | Total Timesteps: {total_timesteps}")
    print("=" * 70)

    envs = [
        DrosophilaDanceGymEnv(
            duration=duration, dt=dt, default_bpm=default_bpm,
            backend=backend, connectome_mode=connectome_mode, randomize_tempo=True
        ) for _ in range(num_envs)
    ]

    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim, device)

    next_obs_list = [env.reset()[0] for env in envs]
    next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)

    batch_size = num_envs * num_steps
    num_iterations = total_timesteps // batch_size
    best_reward = -float("inf")
    completed_rewards = []
    global_step = 0
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        for step in range(num_steps):
            global_step += num_envs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            action_np = action.cpu().numpy()
            step_rewards = []
            for env_idx, env in enumerate(envs):
                next_o, r, term, trunc, _ = env.step(action_np[env_idx])
                step_rewards.append(r)
                if term or trunc:
                    completed_rewards.append(r)
                    next_o, _ = env.reset()
                next_obs_list[env_idx] = next_o

            r_tensor = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            buffer.insert(step, next_obs, action, logprob, r_tensor, next_done, value)
            next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            returns, advantages = buffer.compute_returns_and_advantages(next_value, next_done, gamma, gae_lambda)

        b_obs = buffer.obs.reshape((-1, obs_dim))
        b_logprobs = buffer.logprobs.reshape(-1)
        b_actions = buffer.actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                mb_advantages = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        cur_reward = float(np.mean(completed_rewards[-30:])) if completed_rewards else float(r_tensor.mean().item())
        if cur_reward > best_reward and iteration > 3:
            best_reward = cur_reward
            torch.save({"model_state_dict": agent.state_dict(), "best_reward": best_reward}, best_path)

        if iteration % 5 == 0 or iteration == num_iterations:
            fps = int(global_step / (time.time() - start_time + 1e-6))
            print(f"Iter {iteration:2d}/{num_iterations} | Step: {global_step:5d} | Throughput: {fps} steps/s | Mean Rew: {cur_reward:.2f}")

    print(f"Training Complete! Saved to {best_path}")


if __name__ == "__main__":
    train()

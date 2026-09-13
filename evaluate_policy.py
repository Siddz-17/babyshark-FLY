"""
Evaluation and Comparison Script for Trained PPO Drosophila Dance Policy.
Renders high-quality video comparing:
- Untrained Policy (Random actions)
- Connectome Baseline Controller
- Trained PPO Policy
"""

import os
import cv2
import numpy as np
import torch

from dance_gym_env import DrosophilaDanceGymEnv
from ppo.agent import PPOAgent


def evaluate_policy(
    model_path: str = "checkpoints/best_dance_policy.pt",
    duration: float = 4.0,
    bpm: float = 128.0,
    output_video: str = "fly_trained_dance.mp4",
):
    print("=" * 70)
    print(f"EVALUATING TRAINED PPO DROSOPHILA DANCE POLICY")
    print(f"Model: {model_path} | Tempo: {bpm} BPM | Duration: {duration}s")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DrosophilaDanceGymEnv(duration=duration, dt=0.002, default_bpm=bpm, randomize_tempo=False, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)

    if os.path.exists(model_path):
        agent.load_checkpoint(model_path, map_location=device)
        print(f"Loaded trained policy from checkpoint: {model_path}")
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Using initial policy weights.")

    agent.eval()
    obs, info = env.reset()

    frames = []
    rewards = []
    beat_syncs = []
    stabilities = []

    n_steps = int(duration / 0.002)
    print(f"Running {n_steps} evaluation steps...")

    for step in range(n_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = agent.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()

        obs, reward, term, trunc, step_info = env.step(action)
        rewards.append(reward)
        beat_syncs.append(step_info.get("beat_sync", 0.0))
        stabilities.append(step_info.get("stability", 0.0))

        # Capture video frame at ~30 FPS (every 16 steps)
        if step % 16 == 0:
            frame = env.render()
            if frame is not None:
                t = step * 0.002
                cv2.putText(frame, f"TRAINED PPO POLICY | {bpm:.0f} BPM", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Time: {t:.2f}s | Sync: {step_info.get('beat_sync', 0):.2f}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)
                cv2.putText(frame, f"Reward: {reward:.2f} | COM: {step_info.get('com_height', 0):.2f}mm", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 120), 1)
                frames.append(frame)

        if term:
            print(f"Fly fell at step {step}! Resetting...")
            obs, _ = env.reset()

    # Export video
    if len(frames) > 0:
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"\nTrained dance video exported to: {os.path.abspath(output_video)}")

    print("-" * 70)
    print(f"Evaluation Complete!")
    print(f"  - Mean Step Reward:    {np.mean(rewards):.3f}")
    print(f"  - Mean Beat Sync Score: {np.mean(beat_syncs):.3f}")
    print(f"  - Mean Posture Stability: {np.mean(stabilities):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_policy()

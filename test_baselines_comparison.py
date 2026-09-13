"""
Comparative Benchmark Experiment:
Evaluates all 5 controllers across identical 120 BPM audio tracks:
- Baseline A: Autonomous CPG (No music)
- Baseline B: Beat-Heuristic CPG (Rule-based tracking)
- Baseline C: Random MLP CPG (Un-optimized network)
- Baseline D: Connectome-Constrained CPG (FlyWire auditory pathway in biological mode)
- Baseline E: Trained PPO -> CPG (Learned reinforcement learning motor policy)
"""

import numpy as np

from audio_pipeline import AudioRhythmPipeline
from cpg_controller import DrosophilaCPG
from fly_env import DrosophilaFlyEnv
from dance_reward import DanceRewardEngine
from baselines import (
    BaselineA_AutonomousCPG,
    BaselineB_BeatHeuristicCPG,
    BaselineC_RandomMLP,
    BaselineD_ConnectomeCPG,
    BaselineE_TrainedPPO,
)


def evaluate_controller(name: str, controller, duration: float = 4.0, bpm: float = 120.0, dt: float = 0.002, backend: str = "simple"):
    audio = AudioRhythmPipeline(bpm=bpm, duration=duration, synthetic_type="dance_beat")
    cpg = DrosophilaCPG(dt=dt, default_freq=bpm / 60.0)
    env = DrosophilaFlyEnv(backend=backend, dt=dt)
    proprio = env.reset()
    reward_engine = DanceRewardEngine(nominal_com_height=1.25)

    if hasattr(controller, "reset"):
        controller.reset()

    n_steps = int(duration / dt)
    rewards = []
    beat_syncs = []
    stabilities = []
    energies = []

    for step in range(n_steps):
        t = step * dt
        audio_frame = audio.get_frame(t)

        if isinstance(controller, (BaselineD_ConnectomeCPG, BaselineE_TrainedPPO)):
            cpg_mod, _ = controller.step(audio_frame, proprio)
        else:
            cpg_mod = controller.step(audio_frame, proprio)

        phases, target_joints = cpg.step(cpg_mod)
        proprio = env.step(target_joints)

        breakdown = reward_engine.compute_reward(
            sim_time=t,
            beat_times=audio.beat_times,
            beat_phase=audio_frame.beat_phase,
            beat_pulse=audio_frame.beat_pulse,
            onset_strength=audio_frame.onset_strength,
            tempo_bpm=audio_frame.tempo_bpm,
            com_position=proprio.com_position,
            body_euler=proprio.body_euler,
            body_ang_vel=proprio.body_ang_vel,
            foot_contacts=proprio.foot_contacts,
            joint_angles=proprio.joint_angles,
            joint_velocities=proprio.joint_velocities,
            cpg_phases=phases,
        )

        rewards.append(breakdown.total_reward)
        beat_syncs.append(breakdown.beat_sync)
        stabilities.append(breakdown.stability)
        energies.append(breakdown.energy_penalty)

    return {
        "name": name,
        "mean_reward": float(np.mean(rewards)),
        "mean_beat_sync": float(np.mean(beat_syncs)),
        "mean_stability": float(np.mean(stabilities)),
        "mean_energy": float(np.mean(energies)),
    }


def run_benchmark(backend: str = "simple"):
    print("=" * 86)
    print(f"RUNNING DROSOPHILA DANCE BENCHMARK EXPERIMENT (Backend: {backend.upper()})")
    print("Comparing Baseline A, B, C, D (Connectome), and E (Trained PPO)")
    print("=" * 86)

    controllers = [
        ("Baseline A (Autonomous CPG)", BaselineA_AutonomousCPG(stepping_freq=2.5)),
        ("Baseline B (Beat-Heuristic CPG)", BaselineB_BeatHeuristicCPG()),
        ("Baseline C (Random MLP CPG)", BaselineC_RandomMLP()),
        ("Baseline D (Connectome-Constrained CPG)", BaselineD_ConnectomeCPG()),
        ("Baseline E (Trained PPO -> CPG)", BaselineE_TrainedPPO()),
    ]

    results = []
    for name, ctl in controllers:
        print(f"Evaluating {name}...")
        res = evaluate_controller(name, ctl, backend=backend)
        results.append(res)

    print("\n" + "=" * 86)
    print(f"{'Controller Name':<42} | {'Reward':>8} | {'Footfall Sync':>14} | {'Stability':>10} | {'Energy Cost':>11}")
    print("-" * 86)
    for r in results:
        print(f"{r['name']:<42} | {r['mean_reward']:>8.3f} | {r['mean_beat_sync']:>14.3f} | {r['mean_stability']:>10.3f} | {r['mean_energy']:>11.3f}")
    print("=" * 86)


if __name__ == "__main__":
    run_benchmark()

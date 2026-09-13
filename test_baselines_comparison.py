"""
Comparative Benchmark Experiment:
Evaluates Baseline A, Baseline B, Baseline C, and Baseline D across identical 120 BPM audio tracks:
- Baseline A: Null / No Music
- Baseline B: Heuristic Beat -> CPG
- Baseline C: Direct MLP Policy
- Baseline D: Bio-Inspired Connectome Pathway (JON -> AMMC -> DN -> CPG)
"""

import numpy as np

from audio_pipeline import AudioRhythmPipeline
from cpg_controller import DrosophilaCPG
from fly_env import DrosophilaFlyEnv
from dance_reward import DanceRewardEngine
from baselines import (
    BaselineA_NoMusic,
    BaselineB_BeatCPG,
    BaselineC_DirectMLP,
    BaselineD_ConnectomeAuditoryCPG,
)


def evaluate_controller(name: str, controller, duration: float = 4.0, bpm: float = 120.0, dt: float = 0.002):
    audio = AudioRhythmPipeline(bpm=bpm, duration=duration, synthetic_type="dance_beat")
    cpg = DrosophilaCPG(dt=dt, default_freq=bpm / 60.0)
    env = DrosophilaFlyEnv(dt=dt)
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

        if isinstance(controller, BaselineD_ConnectomeAuditoryCPG):
            cpg_mod, _ = controller.step(audio_frame, proprio)
        else:
            cpg_mod = controller.step(audio_frame, proprio)

        phases, target_joints = cpg.step(cpg_mod)
        proprio = env.step(target_joints)

        breakdown = reward_engine.compute_reward(
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


def run_benchmark():
    print("=" * 80)
    print("RUNNING DROSOPHILA DANCE BENCHMARK EXPERIMENT")
    print("Comparing Baseline A, Baseline B, Baseline C, and Baseline D (Connectome)")
    print("=" * 80)

    controllers = [
        ("Baseline A (No Music)", BaselineA_NoMusic(stepping_freq=2.5)),
        ("Baseline B (Heuristic Beat->CPG)", BaselineB_BeatCPG()),
        ("Baseline C (Direct MLP)", BaselineC_DirectMLP()),
        ("Baseline D (Connectome JON->AMMC->DN)", BaselineD_ConnectomeAuditoryCPG()),
    ]

    results = []
    for name, ctl in controllers:
        print(f"Evaluating {name}...")
        res = evaluate_controller(name, ctl)
        results.append(res)

    print("\n" + "=" * 80)
    print(f"{'Controller Name':<38} | {'Reward':>8} | {'Beat Sync':>10} | {'Stability':>10} | {'Energy Cost':>11}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<38} | {r['mean_reward']:>8.3f} | {r['mean_beat_sync']:>10.3f} | {r['mean_stability']:>10.3f} | {r['mean_energy']:>11.3f}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

"""
Rigorous Scientific Ablation Experiment:
Evaluates the causal necessity and specific contribution of biological connectome topology
under controlled rhythmic stimuli across multiple seeds and tempos.

Ablation Conditions:
1. Biological Connectome (FlyWire JON -> AMMC -> DN with mapped receptive fields)
2. Weight-Shuffled Connectome (Preserves weight distribution and sparsity, scrambles connections)
3. Degree-Preserving Shuffled Connectome (Preserves exact in-degree, out-degree, and weight distributions via Maslov-Sneppen swaps)
4. Direct MLP (No connectome intermediate)
5. Ablated Proprioception (Connectome active, body state zeroed out to test closed-loop feedback)

Outputs:
- Telemetry logging across seeds and tempos (110, 120, 130 BPM)
- Comparative statistics table (Reward, Footfall Sync, Stability, Energy)
- Publication-quality plot: ablation_results.png
- Data summary: data/ablation_summary.json
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from audio_pipeline import AudioRhythmPipeline
from connectome_auditory import DrosophilaAuditoryCircuit
from cpg_controller import DrosophilaCPG, CPGModulation
from fly_env import DrosophilaFlyEnv, ProprioceptionState
from dance_reward import DanceRewardEngine
from ppo_model import ActorCritic


def degree_preserving_shuffle(W: np.ndarray, n_swaps_factor: int = 15, seed: int = 42) -> np.ndarray:
    """
    Applies a Maslov-Sneppen style bipartite degree-preserving rewiring.
    Preserves exact in-degree, out-degree, and weight values of every neuron,
    while destroying specific biological wiring and tonotopy.
    """
    rng = np.random.RandomState(seed)
    W_shuffled = W.copy()
    rows, cols = np.where(W_shuffled > 0)
    n_edges = len(rows)
    if n_edges < 4:
        return W_shuffled

    total_swaps = n_edges * n_swaps_factor
    edges = list(zip(rows, cols))

    for _ in range(total_swaps):
        idx1, idx2 = rng.choice(len(edges), size=2, replace=False)
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]

        # Ensure edges do not share endpoints and cross-edges don't already exist
        if u1 != u2 and v1 != v2 and W_shuffled[u1, v2] == 0 and W_shuffled[u2, v1] == 0:
            # Swap edges
            w1 = W_shuffled[u1, v1]
            w2 = W_shuffled[u2, v2]

            W_shuffled[u1, v1] = 0
            W_shuffled[u2, v2] = 0
            W_shuffled[u1, v2] = w1
            W_shuffled[u2, v1] = w2

            edges[idx1] = (u1, v2)
            edges[idx2] = (u2, v1)

    return W_shuffled


def weight_shuffle(W: np.ndarray, seed: int = 42) -> np.ndarray:
    """Randomly permutes all values in the weight matrix, preserving distribution but not degree."""
    rng = np.random.RandomState(seed)
    flat = W.flatten()
    rng.shuffle(flat)
    return flat.reshape(W.shape)


def run_episode(
    condition: str,
    circuit: DrosophilaAuditoryCircuit,
    bpm: float,
    duration: float = 4.0,
    dt: float = 0.002,
    seed: int = 42,
    backend: str = "simple",
):
    audio = AudioRhythmPipeline(bpm=bpm, duration=duration, synthetic_type="dance_beat")
    cpg = DrosophilaCPG(dt=dt, default_freq=bpm / 60.0)
    env = DrosophilaFlyEnv(backend=backend, dt=dt)
    proprio = env.reset()
    reward_engine = DanceRewardEngine(nominal_com_height=1.25)
    circuit.reset()

    rng = np.random.RandomState(seed)
    # Direct MLP weights if testing Condition 4
    if condition == "Direct_MLP":
        W_mlp = rng.randn(8, 59) * 0.1

    n_steps = int(duration / dt)
    rewards = []
    beat_syncs = []
    stabilities = []
    energies = []

    for step in range(n_steps):
        t = step * dt
        audio_frame = audio.get_frame(t)

        # Handle Proprioception Ablation (Condition 5)
        if condition == "Ablated_Proprioception":
            # Mask proprioceptive feedback to zero
            effective_proprio = ProprioceptionState(
                joint_angles=np.zeros(18),
                joint_velocities=np.zeros(18),
                foot_contacts=np.zeros(6),
                body_euler=np.zeros(3),
                body_ang_vel=np.zeros(3),
                com_position=np.array([0.0, 0.0, 1.25]),
                com_velocity=np.zeros(3),
            )
        else:
            effective_proprio = proprio

        if condition == "Direct_MLP":
            # Bypasses connectome entirely
            audio_vec = np.array([
                audio_frame.beat_phase / (2.0 * np.pi),
                audio_frame.beat_pulse,
                audio_frame.onset_strength,
                audio_frame.low_band_energy,
                audio_frame.tempo_bpm / 120.0,
            ])
            inp = np.concatenate([audio_vec, effective_proprio.vector])
            out = np.tanh(W_mlp @ inp)
            target_freq = (bpm / 60.0) * (1.0 + 0.3 * out[0])
            target_amp = 1.0 + 0.3 * out[1]
            body_bob = max(0.0, 0.4 + 0.4 * out[2])
            lr_offset = 0.2 * out[3]
            cpg_mod = CPGModulation(
                frequency_hz=target_freq,
                amplitude=target_amp,
                phase_offset_lr=lr_offset,
                body_bob_amplitude=body_bob,
                swing_ratio=0.38,
                leg_amplitudes=np.ones(6),
            )
        else:
            # Connectome-driven conditions (1, 2, 3, 5)
            # Pure biological Mode B (NO beat pulse injected)
            neural = circuit.step(
                raw_amplitude=audio_frame.raw_amplitude,
                onset_strength=audio_frame.onset_strength,
                low_band=audio_frame.low_band_energy,
                mid_band=audio_frame.mid_band_energy,
                high_band=audio_frame.high_band_energy,
                bilateral_bias=0.0,
            )

            dn_transient = float(np.mean(neural.dn_activity[:8]))
            dn_sustained = float(np.mean(neural.dn_activity[8:16]))
            dn_asymmetry = float(np.mean(neural.dn_activity[16:20]) - np.mean(neural.dn_activity[20:24]))

            target_freq = (bpm / 60.0) * (0.85 + 0.35 * dn_transient)
            target_amp = 0.9 + 0.5 * dn_sustained
            body_bob = 0.3 + 0.6 * dn_transient
            phase_offset_lr = dn_asymmetry * 0.2

            # Closed-loop balance compensation from proprioception
            left_ground = np.sum(effective_proprio.foot_contacts[0::2])
            right_ground = np.sum(effective_proprio.foot_contacts[1::2])
            balance_bias = 0.1 * (left_ground - right_ground)

            cpg_mod = CPGModulation(
                frequency_hz=target_freq,
                amplitude=target_amp,
                phase_offset_lr=phase_offset_lr + balance_bias,
                body_bob_amplitude=body_bob,
                swing_ratio=0.38,
                leg_amplitudes=np.ones(6),
            )

        phases, target_joints = cpg.step(cpg_mod)
        proprio = env.step(target_joints)

        # Actual footfall event touchdown evaluation
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
        "mean_reward": float(np.mean(rewards)),
        "mean_beat_sync": float(np.mean(beat_syncs)),
        "mean_stability": float(np.mean(stabilities)),
        "mean_energy": float(np.mean(energies)),
    }


def run_full_ablation_study(
    tempos: list = [110.0, 120.0, 130.0],
    seeds: list = [42, 101, 777],
    output_dir: str = "data",
    plot_path: str = "ablation_results.png",
):
    print("=" * 86)
    print("STARTING SCIENTIFIC CONNECTOME ABLATION STUDY")
    print(f"Tempos: {tempos} BPM | Seeds: {seeds}")
    print("Conditions:")
    print("  1. Biological Connectome (FlyWire Mapped Topology)")
    print("  2. Weight-Shuffled Connectome (Permuted Synaptic Weights)")
    print("  3. Degree-Preserving Shuffled (Maslov-Sneppen Preserved In/Out Degrees)")
    print("  4. Direct MLP (No Connectome Intermediate)")
    print("  5. Ablated Proprioception (Connectome Active, Masked Body Feedback)")
    print("=" * 86)

    # Base circuits for conditions
    # 1. Biological
    circuit_bio = DrosophilaAuditoryCircuit(mode="biological")
    W_ja_orig = circuit_bio.W_jon_ammc_left.copy()
    W_dn_orig = circuit_bio.W_ammc_dn.copy()

    # 2. Weight Shuffled
    circuit_wt_shuf = DrosophilaAuditoryCircuit(mode="biological")
    circuit_wt_shuf.W_jon_ammc_left = weight_shuffle(W_ja_orig, seed=123)
    circuit_wt_shuf.W_jon_ammc_right = circuit_wt_shuf.W_jon_ammc_left.copy()
    circuit_wt_shuf.W_ammc_dn = weight_shuffle(W_dn_orig, seed=124)

    # 3. Degree-Preserving Shuffled
    circuit_deg_shuf = DrosophilaAuditoryCircuit(mode="biological")
    circuit_deg_shuf.W_jon_ammc_left = degree_preserving_shuffle(W_ja_orig, seed=456)
    circuit_deg_shuf.W_jon_ammc_right = circuit_deg_shuf.W_jon_ammc_left.copy()
    circuit_deg_shuf.W_ammc_dn = degree_preserving_shuffle(W_dn_orig, seed=457)

    conditions = {
        "1. Biological Connectome": circuit_bio,
        "2. Weight Shuffled": circuit_wt_shuf,
        "3. Degree-Preserving Shuffled": circuit_deg_shuf,
        "4. Direct MLP": circuit_bio,  # Uses Direct_MLP branch
        "5. Ablated Proprioception": circuit_bio,  # Uses masked proprio branch
    }

    results = {name: {"rewards": [], "sync": [], "stability": [], "energy": []} for name in conditions}

    for cond_name, ckt in conditions.items():
        cond_key = cond_name.split(". ")[1].replace(" ", "_")
        print(f"\nEvaluating: {cond_name}...")
        for bpm in tempos:
            for seed in seeds:
                res = run_episode(
                    condition=cond_key,
                    circuit=ckt,
                    bpm=bpm,
                    duration=3.5,
                    dt=0.002,
                    seed=seed,
                )
                results[cond_name]["rewards"].append(res["mean_reward"])
                results[cond_name]["sync"].append(res["mean_beat_sync"])
                results[cond_name]["stability"].append(res["mean_stability"])
                results[cond_name]["energy"].append(res["mean_energy"])

    # Aggregate statistics
    summary_table = []
    print("\n" + "=" * 90)
    print(f"{'Condition':<34} | {'Mean Reward':>12} | {'Footfall Sync':>14} | {'Stability':>10} | {'Energy':>10}")
    print("-" * 90)
    for name, data in results.items():
        mean_r = float(np.mean(data["rewards"]))
        std_r = float(np.std(data["rewards"]))
        mean_s = float(np.mean(data["sync"]))
        std_s = float(np.std(data["sync"]))
        mean_stab = float(np.mean(data["stability"]))
        mean_e = float(np.mean(data["energy"]))

        summary_table.append({
            "condition": name,
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
            "mean_footfall_sync": round(mean_s, 4),
            "std_footfall_sync": round(std_s, 4),
            "mean_stability": round(mean_stab, 4),
            "mean_energy": round(mean_e, 4),
        })

        print(f"{name:<34} | {mean_r:>6.3f} +/- {std_r:<4.2f} | {mean_s:>7.4f} +/- {std_s:<4.3f} | {mean_stab:>10.3f} | {mean_e:>10.3f}")
    print("=" * 90)

    # Save summary JSON
    os.makedirs(output_dir, exist_ok=True)
    summary_json_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nSaved numerical summary: {os.path.abspath(summary_json_path)}")

    # Generate Publication-Quality Figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = [s["condition"].split(". ")[1] for s in summary_table]
    rewards = [s["mean_reward"] for s in summary_table]
    r_errs = [s["std_reward"] for s in summary_table]
    syncs = [s["mean_footfall_sync"] for s in summary_table]
    s_errs = [s["std_footfall_sync"] for s in summary_table]

    colors = ["#2ecc71", "#e67e22", "#f39c12", "#95a5a6", "#e74c3c"]

    # Bar 1: Footfall Synchronization
    bars1 = ax1.bar(names, syncs, yerr=s_errs, capsize=5, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Footfall Event Sync Score", fontsize=11, fontweight="bold")
    ax1.set_title("Causal Effect of Connectome on Rhythm Locking", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)

    # Bar 2: Total Reward
    bars2 = ax2.bar(names, rewards, yerr=r_errs, capsize=5, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Total Mean Reward", fontsize=11, fontweight="bold")
    ax2.set_title("Overall Motor & Postural Performance", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9)

    plt.suptitle("Drosophila Neuromechanical Dance: Biological Connectome Ablation Study", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved publication plot: {os.path.abspath(plot_path)}")

    return summary_table


if __name__ == "__main__":
    run_full_ablation_study()

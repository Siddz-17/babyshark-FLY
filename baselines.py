"""
Comparative Baseline Controllers for Drosophila Rhythm & Dance Study:
- Baseline A: Autonomous CPG (No music sensitivity, constant tripod gait)
- Baseline B: Beat-Heuristic CPG (Rule-based tempo & onset tracking)
- Baseline C: Random MLP CPG (Un-optimized feedforward network benchmark)
- Baseline D: Connectome-Constrained CPG (Zero-shot biological auditory drive via FlyWire JON->AMMC->DN)
- Baseline E: Trained PPO -> CPG (Learned reinforcement learning motor policy)
"""

import os
import numpy as np
import torch
from typing import Dict, Optional, Tuple

from cpg_controller import CPGModulation
from audio_pipeline import AudioFrame
from connectome_auditory import DrosophilaAuditoryCircuit, NeuralState
from fly_env import ProprioceptionState
from ppo.agent import PPOAgent


class BaselineA_AutonomousCPG:
    """Baseline A: Autonomous CPG (Walks steadily, no music sensitivity)."""
    def __init__(self, stepping_freq: float = 2.5):
        self.freq = stepping_freq

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> CPGModulation:
        return CPGModulation(
            frequency_hz=self.freq,
            amplitude=1.0,
            phase_offset_lr=0.0,
            body_bob_amplitude=0.1,
            swing_ratio=0.35,
            leg_amplitudes=np.ones(6),
        )


class BaselineB_BeatHeuristicCPG:
    """Baseline B: Beat-Heuristic CPG (Engineered rule-based tracking)."""
    def __init__(self):
        pass

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> CPGModulation:
        target_freq = max(audio.tempo_bpm / 60.0, 1.0)
        amp = 0.85 + 0.45 * audio.onset_strength
        bob = 0.3 + 0.6 * audio.beat_pulse

        return CPGModulation(
            frequency_hz=target_freq,
            amplitude=amp,
            phase_offset_lr=0.0,
            body_bob_amplitude=bob,
            swing_ratio=0.38,
            leg_amplitudes=np.ones(6),
        )


class BaselineC_RandomMLP:
    """Baseline C: Random MLP CPG (Un-trained fixed feedforward network)."""
    def __init__(self, state_dim: int = 50, action_dim: int = 8, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(64, state_dim) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = rng.randn(action_dim, 64) * 0.1
        self.b2 = np.zeros(action_dim)

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> CPGModulation:
        audio_vec = np.array([
            audio.beat_phase / (2.0 * np.pi),
            audio.beat_pulse,
            audio.onset_strength,
            audio.low_band_energy,
            audio.tempo_bpm / 120.0,
        ])
        state = np.concatenate([audio_vec, proprio.vector])
        if len(state) < self.W1.shape[1]:
            state = np.pad(state, (0, self.W1.shape[1] - len(state)))
        else:
            state = state[:self.W1.shape[1]]

        h = np.tanh(self.W1 @ state + self.b1)
        out = np.tanh(self.W2 @ h + self.b2)

        freq = (audio.tempo_bpm / 60.0) * (1.0 + 0.3 * out[0])
        amp = 1.0 + 0.4 * out[1]
        bob = max(0.0, 0.4 + 0.4 * out[2])
        lr_offset = 0.2 * out[3]
        leg_amps = np.clip(1.0 + 0.2 * out[4:6], 0.5, 1.5)
        leg_amps_full = np.ones(6)
        leg_amps_full[:2] = leg_amps[0]
        leg_amps_full[2:] = leg_amps[1]

        return CPGModulation(
            frequency_hz=freq,
            amplitude=amp,
            phase_offset_lr=lr_offset,
            body_bob_amplitude=bob,
            swing_ratio=0.38,
            leg_amplitudes=leg_amps_full,
        )


class BaselineD_ConnectomeCPG:
    """Baseline D: Connectome-Constrained CPG (FlyWire auditory pathway in biological mode)."""
    def __init__(self, dt: float = 0.002):
        self.circuit = DrosophilaAuditoryCircuit(dt=dt, mode="biological")

    def reset(self):
        self.circuit.reset()

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> Tuple[CPGModulation, NeuralState]:
        neural = self.circuit.step(
            raw_amplitude=audio.raw_amplitude,
            onset_strength=audio.onset_strength,
            low_band=audio.low_band_energy,
            mid_band=audio.mid_band_energy,
            high_band=audio.high_band_energy,
            bilateral_bias=0.0,
        )

        dn_transient = float(np.mean(neural.dn_activity[:8]))
        dn_sustained = float(np.mean(neural.dn_activity[8:16]))
        dn_asymmetry = float(np.mean(neural.dn_activity[16:20]) - np.mean(neural.dn_activity[20:24]))

        target_freq = (audio.tempo_bpm / 60.0) * (0.85 + 0.35 * dn_transient)
        target_amp = 0.9 + 0.5 * dn_sustained
        body_bob = 0.3 + 0.6 * dn_transient
        phase_offset_lr = dn_asymmetry * 0.2

        # Closed-loop stabilization: if left vs right foot contacts differ, compensate
        left_ground = np.sum(proprio.foot_contacts[0::2])
        right_ground = np.sum(proprio.foot_contacts[1::2])
        balance_bias = 0.1 * (left_ground - right_ground)

        cpg_mod = CPGModulation(
            frequency_hz=target_freq,
            amplitude=target_amp,
            phase_offset_lr=phase_offset_lr + balance_bias,
            body_bob_amplitude=body_bob,
            swing_ratio=0.38,
            leg_amplitudes=np.ones(6),
        )
        return cpg_mod, neural


class BaselineE_TrainedPPO:
    """Baseline E: Trained PPO Policy (Closed-loop actor-critic network)."""
    def __init__(self, checkpoint_path: str = "checkpoints/best_dance_policy.pt", dt: float = 0.002):
        self.circuit = DrosophilaAuditoryCircuit(dt=dt, mode="biological")
        self.device = torch.device("cpu")
        self.agent = PPOAgent(obs_dim=83, action_dim=8).to(self.device)
        if os.path.exists(checkpoint_path):
            self.agent.load_checkpoint(checkpoint_path, map_location=self.device)
            self.agent.eval()

    def reset(self):
        self.circuit.reset()

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> Tuple[CPGModulation, NeuralState]:
        neural = self.circuit.step(
            raw_amplitude=audio.raw_amplitude,
            onset_strength=audio.onset_strength,
            low_band=audio.low_band_energy,
            mid_band=audio.mid_band_energy,
            high_band=audio.high_band_energy,
        )

        audio_features = np.array([
            audio.beat_phase / (2.0 * np.pi),
            audio.beat_pulse,
            audio.onset_strength,
            audio.low_band_energy,
            audio.tempo_bpm / 120.0,
        ], dtype=np.float32)

        obs = np.concatenate([proprio.vector.astype(np.float32), neural.dn_activity.astype(np.float32), audio_features])
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            act = self.agent.act_deterministic(obs_tensor).squeeze(0).numpy()

        dn_transient = float(np.mean(neural.dn_activity[:8]))
        dn_sustained = float(np.mean(neural.dn_activity[8:16]))
        dn_asymmetry = float(np.mean(neural.dn_activity[16:20]) - np.mean(neural.dn_activity[20:24]))

        base_freq = (audio.tempo_bpm / 60.0) * (0.85 + 0.35 * dn_transient)
        base_amp = 0.9 + 0.5 * dn_sustained
        base_bob = 0.3 + 0.6 * dn_transient
        base_lr = dn_asymmetry * 0.2

        leg_amps = np.ones(6)
        leg_amps[:2] += act[5] * 0.3
        leg_amps[2:4] += act[6] * 0.3
        leg_amps[4:] += act[7] * 0.3
        leg_amps = np.clip(leg_amps, 0.4, 1.6)

        cpg_mod = CPGModulation(
            frequency_hz=max(0.5, base_freq + act[0] * 0.8),
            amplitude=np.clip(base_amp + act[1] * 0.4, 0.4, 1.8),
            phase_offset_lr=base_lr + act[3] * 0.3,
            body_bob_amplitude=np.clip(base_bob + act[2] * 0.4, 0.0, 1.0),
            swing_ratio=np.clip(0.38 + act[4] * 0.08, 0.25, 0.55),
            leg_amplitudes=leg_amps,
        )
        return cpg_mod, neural

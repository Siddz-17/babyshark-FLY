"""
Comparative Baseline Controllers for Drosophila Rhythm & Dance Study:
- Baseline A: Null / Random Walker (No music sensitivity)
- Baseline B: Heuristic Beat -> CPG Phase Lock (Rule-based tracking)
- Baseline C: Direct MLP Policy (Audio features + Proprioception -> CPG, No connectome)
- Baseline D: Bio-Inspired Connectome Pathway (Audio -> JON -> AMMC -> DN -> CPG)
"""

import numpy as np
from typing import Dict, Tuple

from cpg_controller import CPGModulation
from audio_pipeline import AudioFrame
from connectome_auditory import DrosophilaAuditoryCircuit, NeuralState
from fly_env import ProprioceptionState


class BaselineA_NoMusic:
    """
    Baseline A: Null / Autonomous Walker.
    Ignores music audio entirely; walks with unmodulated constant tripod gait.
    """
    def __init__(self, stepping_freq: float = 3.0):
        self.freq = stepping_freq

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> CPGModulation:
        # Constant unmodulated tripod gait
        return CPGModulation(
            frequency_hz=self.freq,
            amplitude=1.0,
            phase_offset_lr=0.0,
            body_bob_amplitude=0.1,
            swing_ratio=0.35,
            leg_amplitudes=np.ones(6),
        )


class BaselineB_BeatCPG:
    """
    Baseline B: Heuristic Beat-Tracking CPG.
    Locks stepping frequency directly to estimated audio tempo (BPM)
    and drives body bobbing with onset pulses, without any intervening neural circuitry.
    """
    def __init__(self):
        pass

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> CPGModulation:
        target_freq = max(audio.tempo_bpm / 60.0, 1.0)
        # Heuristic bobbing and stepping amplitude driven by onset strength
        amp = 0.8 + 0.5 * audio.onset_strength
        bob = 0.3 + 0.7 * audio.beat_pulse

        return CPGModulation(
            frequency_hz=target_freq,
            amplitude=amp,
            phase_offset_lr=0.0,
            body_bob_amplitude=bob,
            swing_ratio=0.38,
            leg_amplitudes=np.ones(6),
        )


class BaselineC_DirectMLP:
    """
    Baseline C: Direct MLP Controller (Standard end-to-end RL without biological circuit).
    Maps raw audio features + proprioception directly to CPG parameters.
    """
    def __init__(self, state_dim: int = 50, action_dim: int = 10, seed: int = 42):
        rng = np.random.RandomState(seed)
        # 2-layer MLP weights [state_dim -> 64 -> action_dim]
        self.W1 = rng.randn(64, state_dim) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = rng.randn(action_dim, 64) * 0.1
        self.b2 = np.zeros(action_dim)

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> CPGModulation:
        # Construct raw observation vector
        audio_vec = np.array([
            audio.beat_phase,
            audio.beat_pulse,
            audio.onset_strength,
            audio.low_band_energy,
            audio.mid_band_energy,
            audio.high_band_energy,
            audio.tempo_bpm / 120.0,
        ])
        state = np.concatenate([audio_vec, proprio.vector])
        if len(state) < self.W1.shape[1]:
            # Pad if needed
            state = np.pad(state, (0, self.W1.shape[1] - len(state)))
        else:
            state = state[:self.W1.shape[1]]

        # Feedforward
        h = np.tanh(self.W1 @ state + self.b1)
        out = np.tanh(self.W2 @ h + self.b2)

        freq = (audio.tempo_bpm / 60.0) * (1.0 + 0.3 * out[0])
        amp = 1.0 + 0.4 * out[1]
        bob = max(0.0, 0.5 + 0.5 * out[2])
        lr_offset = 0.2 * out[3]
        leg_amps = np.clip(1.0 + 0.2 * out[4:10], 0.5, 1.5)

        return CPGModulation(
            frequency_hz=freq,
            amplitude=amp,
            phase_offset_lr=lr_offset,
            body_bob_amplitude=bob,
            swing_ratio=0.38,
            leg_amplitudes=leg_amps,
        )


class BaselineD_ConnectomeAuditoryCPG:
    """
    Baseline D: Connectome-Constrained Auditory Pathway.
    Audio is processed through JON-A/B -> AMMC -> Descending Neurons.
    DN motor signals modulate CPG oscillators in accordance with Drosophila neurobiology.
    """
    def __init__(self, dt: float = 0.002):
        self.circuit = DrosophilaAuditoryCircuit(dt=dt)

    def reset(self):
        self.circuit.reset()

    def step(self, audio: AudioFrame, proprio: ProprioceptionState) -> Tuple[CPGModulation, NeuralState]:
        neural = self.circuit.step(
            raw_amplitude=audio.raw_amplitude,
            onset_strength=audio.onset_strength,
            low_band=audio.low_band_energy,
            mid_band=audio.mid_band_energy,
            high_band=audio.high_band_energy,
            beat_pulse=audio.beat_pulse,
            bilateral_bias=0.0,
        )

        dn_transient = float(np.mean(neural.dn_activity[:8]))
        dn_sustained = float(np.mean(neural.dn_activity[8:16]))
        dn_asymmetry = float(np.mean(neural.dn_activity[16:20]) - np.mean(neural.dn_activity[20:24]))

        target_freq = (audio.tempo_bpm / 60.0) * (0.85 + 0.35 * dn_transient)
        target_amp = 0.9 + 0.6 * dn_sustained
        body_bob = 0.3 + 0.7 * dn_transient
        phase_offset_lr = dn_asymmetry * 0.25

        cpg_mod = CPGModulation(
            frequency_hz=target_freq,
            amplitude=target_amp,
            phase_offset_lr=phase_offset_lr,
            body_bob_amplitude=body_bob,
            swing_ratio=0.38,
            leg_amplitudes=np.ones(6),
        )

        return cpg_mod, neural

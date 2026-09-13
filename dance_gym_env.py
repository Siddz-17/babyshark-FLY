"""
Gymnasium Environment for Connectome-Driven Drosophila Dance Simulation.
Interfaces the closed-loop pipeline with standard Reinforcement Learning algorithms (PPO).

Supports:
- Dual simulation backends: 'simple' (fast numerical, 1,800+ steps/s) or 'mujoco' (genuine MuJoCo 3.x)
- Biological connectome mode (pure acoustic wave & power spectrum, no pre-computed beat injection)
- Observation Space (83 dims): 54 Proprioception + 24 Auditory DNs + 5 Audio features
- Action Space (8 dims): Continuous CPG parameter modulations
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from audio_pipeline import AudioRhythmPipeline, AudioFrame
from connectome_auditory import DrosophilaAuditoryCircuit, NeuralState
from cpg_controller import DrosophilaCPG, CPGModulation, LEG_NAMES
from fly_env import DrosophilaFlyEnv, ProprioceptionState
from dance_reward import DanceRewardEngine, RewardBreakdown


class DrosophilaDanceGymEnv(gym.Env):
    """
    Gymnasium environment where an agent learns to modulate CPG parameters
    using auditory connectome signals and proprioceptive feedback to dance in sync with music.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        duration: float = 4.0,           # Max episode duration in seconds
        dt: float = 0.002,                # Physics timestep (500 Hz)
        default_bpm: float = 120.0,
        randomize_tempo: bool = True,     # Randomize BPM on reset for robust policy learning
        backend: str = "simple",          # 'simple' or 'mujoco'
        connectome_mode: str = "biological",  # 'biological' or 'engineered'
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.max_duration = duration
        self.dt = dt
        self.default_bpm = default_bpm
        self.randomize_tempo = randomize_tempo
        self.backend = backend
        self.connectome_mode = connectome_mode
        self.render_mode = render_mode

        self.max_steps = int(self.max_duration / self.dt)

        # 1. Action Space: 8 continuous CPG parameter modulations
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 2. Observation Space: 54 (proprio) + 24 (DNs) + 5 (audio) = 83 dimensions
        obs_dim = 54 + 24 + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal subsystems
        self.cpg = DrosophilaCPG(dt=self.dt, default_freq=self.default_bpm / 60.0)
        self.fly_env = DrosophilaFlyEnv(backend=self.backend, dt=self.dt, enable_rendering=(render_mode == "rgb_array"))
        self.circuit = DrosophilaAuditoryCircuit(dt=self.dt, mode=self.connectome_mode)
        self.reward_engine = DanceRewardEngine(nominal_com_height=1.25)
        self.audio_pipeline: Optional[AudioRhythmPipeline] = None

        self.step_count = 0
        self.current_bpm = default_bpm

    def _get_obs(self, audio_frame: AudioFrame, proprio: ProprioceptionState, neural: NeuralState) -> np.ndarray:
        """Constructs the combined 83-dimensional observation vector."""
        audio_features = np.array([
            audio_frame.beat_phase / (2.0 * np.pi),
            audio_frame.beat_pulse,
            audio_frame.onset_strength,
            audio_frame.low_band_energy,
            audio_frame.tempo_bpm / 120.0,
        ], dtype=np.float32)

        proprio_vec = proprio.vector.astype(np.float32)
        dn_vec = neural.dn_activity.astype(np.float32)

        return np.concatenate([proprio_vec, dn_vec, audio_features])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        if self.randomize_tempo:
            self.current_bpm = float(self.np_random.uniform(100.0, 140.0))
        else:
            self.current_bpm = self.default_bpm

        self.audio_pipeline = AudioRhythmPipeline(
            bpm=self.current_bpm,
            duration=self.max_duration,
            synthetic_type="dance_beat",
        )

        self.cpg.reset()
        self.circuit.reset()
        self.reward_engine.reset()
        proprio = self.fly_env.reset()

        audio_frame = self.audio_pipeline.get_frame(0.0)
        neural_state = self.circuit.step(
            raw_amplitude=audio_frame.raw_amplitude,
            onset_strength=audio_frame.onset_strength,
            low_band=audio_frame.low_band_energy,
            mid_band=audio_frame.mid_band_energy,
            high_band=audio_frame.high_band_energy,
            beat_pulse=audio_frame.beat_pulse,
        )

        obs = self._get_obs(audio_frame, proprio, neural_state)
        info = {
            "tempo_bpm": self.current_bpm,
            "com_height": proprio.com_position[2],
            "backend": getattr(self.fly_env, "backend", self.backend),
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        t = self.step_count * self.dt

        audio_frame = self.audio_pipeline.get_frame(t)

        # Step connectome circuit (Biological Mode: no pre-computed beat injection!)
        neural_state = self.circuit.step(
            raw_amplitude=audio_frame.raw_amplitude,
            onset_strength=audio_frame.onset_strength,
            low_band=audio_frame.low_band_energy,
            mid_band=audio_frame.mid_band_energy,
            high_band=audio_frame.high_band_energy,
            beat_pulse=audio_frame.beat_pulse,
            bilateral_bias=0.0,
        )

        # Connectome DN baselines
        dn_transient = float(np.mean(neural_state.dn_activity[:8]))
        dn_sustained = float(np.mean(neural_state.dn_activity[8:16]))
        dn_asymmetry = float(np.mean(neural_state.dn_activity[16:20]) - np.mean(neural_state.dn_activity[20:24]))

        base_freq = (self.current_bpm / 60.0) * (0.85 + 0.35 * dn_transient)
        base_amp = 0.9 + 0.5 * dn_sustained
        base_bob = 0.3 + 0.6 * dn_transient
        base_lr = dn_asymmetry * 0.2

        act = np.clip(action, -1.0, 1.0)
        freq_mod = act[0] * 0.8
        amp_mod = act[1] * 0.4
        bob_mod = act[2] * 0.4
        lr_mod = act[3] * 0.3
        swing_mod = act[4] * 0.08

        leg_amps = np.ones(6)
        leg_amps[:2] += act[5] * 0.3
        leg_amps[2:4] += act[6] * 0.3
        leg_amps[4:] += act[7] * 0.3
        leg_amps = np.clip(leg_amps, 0.4, 1.6)

        cpg_mod = CPGModulation(
            frequency_hz=max(0.5, base_freq + freq_mod),
            amplitude=np.clip(base_amp + amp_mod, 0.4, 1.8),
            phase_offset_lr=base_lr + lr_mod,
            body_bob_amplitude=np.clip(base_bob + bob_mod, 0.0, 1.0),
            swing_ratio=np.clip(0.38 + swing_mod, 0.25, 0.55),
            leg_amplitudes=leg_amps,
        )

        phases, target_joints = self.cpg.step(cpg_mod)
        proprio = self.fly_env.step(target_joints)

        # Multi-objective reward with actual footfall event timing
        breakdown = self.reward_engine.compute_reward(
            sim_time=t,
            beat_times=self.audio_pipeline.beat_times,
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

        reward = float(breakdown.total_reward)
        terminated = bool(breakdown.is_fallen)
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._get_obs(audio_frame, proprio, neural_state)

        info = {
            "beat_sync": breakdown.beat_sync,
            "stability": breakdown.stability,
            "rhythmicity": breakdown.rhythmicity,
            "movement_quality": breakdown.movement_quality,
            "energy_penalty": breakdown.energy_penalty,
            "com_height": proprio.com_position[2],
            "is_fallen": breakdown.is_fallen,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.fly_env.render_frame()
        return None

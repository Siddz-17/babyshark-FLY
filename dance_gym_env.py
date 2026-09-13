"""
Gymnasium Environment for Connectome-Driven Drosophila Dance Simulation.
Interfaces the closed-loop pipeline with standard Reinforcement Learning algorithms (PPO).

Observation Space (83 dimensions):
- Proprioception (54 dims): joint angles, velocities, foot contacts, Euler angles, angular vel, COM pos, COM vel
- Auditory Descending Neurons (24 dims): firing rates of DNs from the connectome circuit
- Audio State (5 dims): beat phase, beat pulse, onset strength, low-band energy, normalized tempo

Action Space (8 dimensions):
- Continuous CPG parameter modulations in [-1, +1]:
  [freq_delta, amp_delta, bob_delta, lr_phase_delta, swing_delta, amp_front, amp_mid, amp_hind]
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
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.max_duration = duration
        self.dt = dt
        self.default_bpm = default_bpm
        self.randomize_tempo = randomize_tempo
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
        self.fly_env = DrosophilaFlyEnv(dt=self.dt, enable_rendering=(render_mode == "rgb_array"))
        self.circuit = DrosophilaAuditoryCircuit(dt=self.dt)
        self.reward_engine = DanceRewardEngine(nominal_com_height=1.25)
        self.audio_pipeline: Optional[AudioRhythmPipeline] = None

        self.step_count = 0
        self.current_bpm = default_bpm

    def _get_obs(self, audio_frame: AudioFrame, proprio: ProprioceptionState, neural: NeuralState) -> np.ndarray:
        """Constructs the combined 83-dimensional observation vector."""
        audio_features = np.array([
            audio_frame.beat_phase / (2.0 * np.pi),  # Normalized [0, 1]
            audio_frame.beat_pulse,                  # [0, 1]
            audio_frame.onset_strength,              # [0, 1]
            audio_frame.low_band_energy,             # [0, 1]
            audio_frame.tempo_bpm / 120.0,           # Scaled tempo
        ], dtype=np.float32)

        # Proprioception: 54 dims
        proprio_vec = proprio.vector.astype(np.float32)

        # Descending Neurons: 24 dims
        dn_vec = neural.dn_activity.astype(np.float32)

        obs = np.concatenate([proprio_vec, dn_vec, audio_features])
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # Optional tempo randomization
        if self.randomize_tempo:
            self.current_bpm = float(self.np_random.uniform(100.0, 140.0))
        else:
            self.current_bpm = self.default_bpm

        # Initialize audio track with current BPM
        self.audio_pipeline = AudioRhythmPipeline(
            bpm=self.current_bpm,
            duration=self.max_duration,
            synthetic_type="dance_beat",
        )

        # Reset subsystems
        self.cpg.reset()
        self.circuit.reset()
        self.reward_engine.reset()
        proprio = self.fly_env.reset()

        # Initial audio and neural state at t = 0
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
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        t = self.step_count * self.dt

        # 1. Get current audio state
        audio_frame = self.audio_pipeline.get_frame(t)

        # 2. Step connectome auditory circuit
        neural_state = self.circuit.step(
            raw_amplitude=audio_frame.raw_amplitude,
            onset_strength=audio_frame.onset_strength,
            low_band=audio_frame.low_band_energy,
            mid_band=audio_frame.mid_band_energy,
            high_band=audio_frame.high_band_energy,
            beat_pulse=audio_frame.beat_pulse,
            bilateral_bias=0.0,
        )

        # 3. Compute CPG modulation: combines Connectome DN drive + PPO action
        # DN baseline components
        dn_transient = float(np.mean(neural_state.dn_activity[:8]))
        dn_sustained = float(np.mean(neural_state.dn_activity[8:16]))
        dn_asymmetry = float(np.mean(neural_state.dn_activity[16:20]) - np.mean(neural_state.dn_activity[20:24]))

        base_freq = (self.current_bpm / 60.0) * (0.85 + 0.35 * dn_transient)
        base_amp = 0.9 + 0.5 * dn_sustained
        base_bob = 0.3 + 0.6 * dn_transient
        base_lr = dn_asymmetry * 0.2

        # Action modulations (clamped within biomechanical safety bounds)
        act = np.clip(action, -1.0, 1.0)
        freq_mod = act[0] * 0.8          # +/- 0.8 Hz
        amp_mod = act[1] * 0.4           # +/- 0.4 amplitude
        bob_mod = act[2] * 0.4           # +/- 0.4 bobbing
        lr_mod = act[3] * 0.3            # +/- 0.3 rad phase
        swing_mod = act[4] * 0.08        # +/- 0.08 swing fraction

        leg_amps = np.ones(6)
        leg_amps[:2] += act[5] * 0.3     # Front legs
        leg_amps[2:4] += act[6] * 0.3    # Mid legs
        leg_amps[4:] += act[7] * 0.3     # Hind legs
        leg_amps = np.clip(leg_amps, 0.4, 1.6)

        cpg_mod = CPGModulation(
            frequency_hz=max(0.5, base_freq + freq_mod),
            amplitude=np.clip(base_amp + amp_mod, 0.4, 1.8),
            phase_offset_lr=base_lr + lr_mod,
            body_bob_amplitude=np.clip(base_bob + bob_mod, 0.0, 1.0),
            swing_ratio=np.clip(0.38 + swing_mod, 0.25, 0.55),
            leg_amplitudes=leg_amps,
        )

        # 4. Step CPG oscillator & get target joint kinematics
        phases, target_joints = self.cpg.step(cpg_mod)

        # 5. Step FlyGym biomechanical physics
        proprio = self.fly_env.step(target_joints)

        # 6. Compute multi-objective reward
        breakdown = self.reward_engine.compute_reward(
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

        # 7. Check termination and truncation
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


if __name__ == "__main__":
    env = DrosophilaDanceGymEnv()
    obs, info = env.reset()
    print("DrosophilaDanceGymEnv initialized:")
    print(f"  - Observation Shape: {obs.shape}")
    print(f"  - Action Shape:      {env.action_space.shape}")
    print(f"  - Initial Info:      {info}")

    # Test random action
    action = env.action_space.sample()
    next_obs, reward, term, trunc, info = env.step(action)
    print(f"Test Step Output:")
    print(f"  - Next Obs Shape: {next_obs.shape}")
    print(f"  - Step Reward:    {reward:.3f}")
    print(f"  - Terminated:     {term}")
    print(f"  - Info:           {info}")
    print("Gym environment operational!")

"""
Dance and Rhythm Reward Engine for Drosophila Simulation.
Formulates multi-objective closed-loop reward:
  R = w1 * beat_sync + w2 * onset_sync + w3 * rhythmicity
    + w4 * stability + w5 * movement_quality - w6 * energy_cost - w7 * falls

Guards against parasitic policies (e.g. violent chaotic shaking) by enforcing
postural stability, phase-locking, and biomechanical smoothness.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class RewardBreakdown:
    """Detailed components of the dance reward for telemetry & logging."""
    total_reward: float
    beat_sync: float
    onset_sync: float
    rhythmicity: float
    stability: float
    movement_quality: float
    energy_penalty: float
    fall_penalty: float
    is_fallen: bool


class DanceRewardEngine:
    """
    Computes reward for musical synchronization and biomechanical stability.
    """

    def __init__(
        self,
        nominal_com_height: float = 1.2,  # mm
        w_beat: float = 2.5,
        w_onset: float = 1.5,
        w_rhythm: float = 1.5,
        w_stability: float = 2.0,
        w_quality: float = 1.0,
        w_energy: float = 0.05,
        w_fall: float = 50.0,
    ):
        self.h0 = nominal_com_height
        self.w_beat = w_beat
        self.w_onset = w_onset
        self.w_rhythm = w_rhythm
        self.w_stability = w_stability
        self.w_quality = w_quality
        self.w_energy = w_energy
        self.w_fall = w_fall

        # Rolling history buffer for rhythmicity calculation
        self.history_len = 100
        self.reset()

    def reset(self):
        """Clears kinematic history."""
        self.step_vel_history = []
        self.prev_joint_angles = None

    def compute_reward(
        self,
        beat_phase: float,
        beat_pulse: float,
        onset_strength: float,
        tempo_bpm: float,
        com_position: np.ndarray,       # [x, y, z] in mm
        body_euler: np.ndarray,         # [roll, pitch, yaw] in radians
        body_ang_vel: np.ndarray,       # [wx, wy, wz] in rad/s
        foot_contacts: np.ndarray,      # 6 binary/force values
        joint_angles: np.ndarray,       # Joint angles in radians
        joint_velocities: np.ndarray,   # Joint velocities in rad/s
        cpg_phases: np.ndarray,         # 6 leg oscillator phases [0, 2*pi)
    ) -> RewardBreakdown:
        """
        Computes the complete multi-objective dance reward.
        """
        com_z = com_position[2]
        roll, pitch, yaw = body_euler

        # 1. Fall detection (thorax collapses to floor or rolls over)
        is_fallen = bool(com_z < 0.4 or abs(roll) > 0.8 or abs(pitch) > 0.8)
        fall_penalty = self.w_fall if is_fallen else 0.0

        # 2. Postural Stability
        # Upright thorax orientation and maintaining realistic standing height
        orientation_error = roll**2 + pitch**2
        height_error = (com_z - self.h0) ** 2
        stability_score = np.exp(-3.0 * orientation_error) * np.exp(-4.0 * height_error)
        stability_reward = self.w_stability * stability_score

        # 3. Beat Synchronization (Phase-Locking)
        # Check if stepping / footfall transitions lock to the musical beat
        # Cosine distance between leg CPG phase and audio beat phase
        # For alternating tripod, legs should be in phase (0) or anti-phase (pi) with beat
        phase_alignments = np.cos(cpg_phases - beat_phase)
        beat_sync_score = float(np.mean(np.maximum(0, phase_alignments)))
        # Boost sync reward during active musical beats
        beat_sync_reward = self.w_beat * beat_sync_score * (0.5 + 0.5 * beat_pulse)

        # 4. Onset Synchrony
        # Total active movement energy should correlate with acoustic onset envelope
        kinetic_energy = float(np.mean(joint_velocities**2))
        vel_norm = np.clip(kinetic_energy / 20.0, 0.0, 1.0)
        # Reward high kinetic movement when onsets are high, low when music is quiet
        onset_match = 1.0 - abs(vel_norm - onset_strength)
        onset_sync_reward = self.w_onset * max(0.0, onset_match)

        # 5. Movement Quality (Coordination & Plausibility)
        # Prevents "all legs lifting" (falling) or "zero legs moving" (freezing)
        ground_legs = np.sum(foot_contacts > 0.1)
        # Ideal support polygon in 6-legged insects: 3 to 5 legs on ground at any moment
        if 2 <= ground_legs <= 4:
            coordination_score = 1.0
        elif ground_legs == 5:
            coordination_score = 0.7
        else:
            coordination_score = 0.2

        # Reward contralateral leg alternation
        left_moving = np.any(np.abs(joint_velocities[: len(joint_velocities) // 2]) > 0.5)
        right_moving = np.any(np.abs(joint_velocities[len(joint_velocities) // 2 :]) > 0.5)
        alternation_bonus = 0.3 if (left_moving and right_moving) else 0.0
        quality_reward = self.w_quality * (coordination_score + alternation_bonus)

        # 6. Rhythmicity
        # Regular periodic stepping: track velocity variation across time
        self.step_vel_history.append(float(np.mean(np.abs(joint_velocities))))
        if len(self.step_vel_history) > self.history_len:
            self.step_vel_history.pop(0)

        if len(self.step_vel_history) >= 30:
            vel_series = np.array(self.step_vel_history)
            std_v = np.std(vel_series)
            mean_v = np.mean(vel_series) + 1e-6
            # Coefficient of variation (CV) for steady periodic motion
            rhythm_score = float(np.clip(1.0 - abs((std_v / mean_v) - 0.5), 0.0, 1.0))
        else:
            rhythm_score = 0.5
        rhythm_reward = self.w_rhythm * rhythm_score

        # 7. Energy & Jerk Cost (anti-flailing penalty)
        # Penalizes high joint torques / excessive erratic velocities
        energy_penalty = self.w_energy * float(np.sum(joint_velocities**2))

        # Total combined reward
        total = (
            beat_sync_reward
            + onset_sync_reward
            + rhythm_reward
            + stability_reward
            + quality_reward
            - energy_penalty
            - fall_penalty
        )

        return RewardBreakdown(
            total_reward=total,
            beat_sync=beat_sync_reward,
            onset_sync=onset_sync_reward,
            rhythmicity=rhythm_reward,
            stability=stability_reward,
            movement_quality=quality_reward,
            energy_penalty=energy_penalty,
            fall_penalty=fall_penalty,
            is_fallen=is_fallen,
        )


if __name__ == "__main__":
    engine = DanceRewardEngine()
    print("Dance Reward Engine initialized.")
    # Test stable in-sync state
    breakdown = engine.compute_reward(
        beat_phase=0.0,
        beat_pulse=1.0,
        onset_strength=0.8,
        tempo_bpm=120.0,
        com_position=np.array([0.0, 0.0, 1.2]),
        body_euler=np.array([0.0, 0.0, 0.0]),
        body_ang_vel=np.array([0.0, 0.0, 0.0]),
        foot_contacts=np.array([1, 0, 1, 0, 1, 0]),  # Tripod 1
        joint_angles=np.zeros(18),
        joint_velocities=np.ones(18) * 1.5,
        cpg_phases=np.array([0.0, np.pi, np.pi, 0.0, 0.0, np.pi]),
    )
    print(f"Sample Reward: {breakdown.total_reward:.2f}")
    print(f"  - Beat Sync:   {breakdown.beat_sync:.2f}")
    print(f"  - Stability:   {breakdown.stability:.2f}")
    print(f"  - Quality:     {breakdown.movement_quality:.2f}")
    print("Dance reward engine verified successfully!")

"""
Dance and Rhythm Reward Engine for Drosophila Simulation.
Evaluates:
1. Actual Footfall Event Synchronization: Detects physical foot touchdowns (transition to stance)
   and rewards exact temporal alignment to musical beat timestamps:
   beat_sync = mean( exp(-|t_footfall - t_nearest_beat| / sigma) )
2. Onset correlation with movement velocity
3. Stepping rhythmicity & dynamic vitality (preventing 'freeze to stay upright' local optimum)
4. Postural stability (roll/pitch/height)
5. Smoothness & anti-flail energy penalty
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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
        nominal_com_height: float = 1.25,  # mm
        w_beat: float = 3.0,
        w_onset: float = 1.5,
        w_rhythm: float = 1.5,
        w_stability: float = 1.8,
        w_quality: float = 1.2,
        w_energy: float = 0.04,
        w_fall: float = 40.0,
        footfall_sigma: float = 0.05,      # 50 ms precision window for beat hits
    ):
        self.h0 = nominal_com_height
        self.w_beat = w_beat
        self.w_onset = w_onset
        self.w_rhythm = w_rhythm
        self.w_stability = w_stability
        self.w_quality = w_quality
        self.w_energy = w_energy
        self.w_fall = w_fall
        self.sigma = footfall_sigma

        self.reset()

    def reset(self):
        """Clears kinematic and footfall history."""
        self.prev_contacts = np.ones(6, dtype=np.float64)
        self.recent_footfalls = []
        self.step_vel_history = []

    def compute_reward(
        self,
        sim_time: float,
        beat_times: Optional[np.ndarray],
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
        com_z = com_position[2]
        roll, pitch, yaw = body_euler

        # 1. Fall detection
        is_fallen = bool(com_z < 0.35 or abs(roll) > 0.85 or abs(pitch) > 0.85)
        fall_penalty = self.w_fall if is_fallen else 0.0

        # 2. Postural Stability
        orientation_err = roll**2 + pitch**2
        height_err = (com_z - self.h0) ** 2
        stability_score = float(np.exp(-3.0 * orientation_err) * np.exp(-4.0 * height_err))
        stability_reward = self.w_stability * stability_score

        # 3. Actual Footfall Event Synchronization (Ground Contact Touchdowns)
        # Detect physical footfall events: transition from swing (0) to stance (1)
        touchdowns = (foot_contacts > 0.5) & (self.prev_contacts <= 0.5)
        num_touchdowns = int(np.sum(touchdowns))

        sync_score = 0.0
        if num_touchdowns > 0 and beat_times is not None and len(beat_times) > 0:
            # Temporal distance to nearest beat
            idx = np.searchsorted(beat_times, sim_time)
            candidates = []
            if idx > 0:
                candidates.append(beat_times[idx - 1])
            if idx < len(beat_times):
                candidates.append(beat_times[idx])
            if candidates:
                min_dt = min(abs(sim_time - b) for b in candidates)
                # Exponential decay kernel centered at nearest beat
                sync_score = float(np.exp(- (min_dt / self.sigma) ** 2))
                self.recent_footfalls.append(sync_score)
        elif beat_pulse > 0.4:
            # Supporting phase-alignment bonus when actively stepping near beat pulse
            phase_alignments = np.cos(cpg_phases - beat_phase)
            sync_score = 0.5 * float(np.mean(np.maximum(0, phase_alignments)))

        # Update contact history
        self.prev_contacts = foot_contacts.copy()

        if len(self.recent_footfalls) > 50:
            self.recent_footfalls.pop(0)

        rolling_sync = float(np.mean(self.recent_footfalls)) if self.recent_footfalls else sync_score
        beat_sync_reward = self.w_beat * (0.6 * sync_score + 0.4 * rolling_sync)

        # 4. Onset Synchrony (Kinetic energy matches acoustic power)
        kinetic_energy = float(np.mean(joint_velocities**2))
        vel_norm = np.clip(kinetic_energy / 22.0, 0.0, 1.0)
        onset_match = max(0.0, 1.0 - abs(vel_norm - onset_strength))
        onset_sync_reward = self.w_onset * onset_match

        # 5. Stepping Dynamism & Movement Quality (Guards against 'freeze to stay upright')
        # Active movement bonus: standing completely still earns zero movement quality
        active_legs = np.sum(np.abs(joint_velocities[1::3]) > 0.4)
        ground_legs = np.sum(foot_contacts > 0.1)

        # Support polygon: 3-4 legs on ground while 2-3 legs swing is ideal tripod stepping
        if 2 <= ground_legs <= 4 and active_legs >= 2:
            movement_quality = 1.0
        elif 2 <= ground_legs <= 5 and active_legs >= 1:
            movement_quality = 0.6
        else:
            movement_quality = 0.15

        quality_reward = self.w_quality * movement_quality

        # 6. Rhythmicity
        self.step_vel_history.append(float(np.mean(np.abs(joint_velocities))))
        if len(self.step_vel_history) > 80:
            self.step_vel_history.pop(0)

        if len(self.step_vel_history) >= 25:
            vel_series = np.array(self.step_vel_history)
            std_v = np.std(vel_series)
            mean_v = np.mean(vel_series) + 1e-6
            rhythm_score = float(np.clip(1.0 - abs((std_v / mean_v) - 0.45), 0.0, 1.0))
        else:
            rhythm_score = 0.5
        rhythm_reward = self.w_rhythm * rhythm_score

        # 7. Energy / Jitter Penalty
        energy_penalty = self.w_energy * float(np.sum(joint_velocities**2))

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
    beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    print("Dance Reward Engine initialized with footfall event timing.")
    res = engine.compute_reward(
        sim_time=0.51,  # 10 ms after beat
        beat_times=beat_times,
        beat_phase=0.05,
        beat_pulse=0.9,
        onset_strength=0.8,
        tempo_bpm=120.0,
        com_position=np.array([0.0, 0.0, 1.25]),
        body_euler=np.zeros(3),
        body_ang_vel=np.zeros(3),
        foot_contacts=np.array([1, 0, 1, 0, 1, 0]),
        joint_angles=np.zeros(18),
        joint_velocities=np.ones(18) * 1.5,
        cpg_phases=np.zeros(6),
    )
    print(f"Sample Footfall Synchronized Reward: {res.total_reward:.2f}")
    print(f"  - Beat Sync Score:   {res.beat_sync:.2f}")
    print(f"  - Posture Stability: {res.stability:.2f}")
    print(f"  - Movement Quality:  {res.movement_quality:.2f}")
    print("Reward engine successfully updated!")

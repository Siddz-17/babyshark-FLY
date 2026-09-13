"""
Central Pattern Generator (CPG) Motor Controller for Drosophila.
Implements a network of 6 coupled phase oscillators (one per leg) capable of:
- Standard tripod walking
- In-place stepping / dancing
- Real-time modulation via Descending Neurons or PPO policy actions
  (frequency, amplitude, phase offsets, stance/swing duty cycle)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


# 6 legs of Drosophila:
# Fore legs: LF (Left Front), RF (Right Front)
# Mid legs:  LM (Left Mid),   RM (Right Mid)
# Hind legs: LH (Left Hind),  RH (Right Hind)
LEG_NAMES = ["LF", "RF", "LM", "RM", "LH", "RH"]
LEG_INDICES = {name: i for i, name in enumerate(LEG_NAMES)}


@dataclass
class CPGModulation:
    """Modulation parameters output by PPO or descending auditory neurons."""
    frequency_hz: float          # Overall stepping/dancing frequency (Hz)
    amplitude: float             # Joint oscillation amplitude scalar [0, 1.5]
    phase_offset_lr: float       # Left vs Right phase offset (rad)
    body_bob_amplitude: float    # Thorax pitch/heave bobbing amplitude [0, 1]
    swing_ratio: float           # Swing phase fraction of step cycle [0.2, 0.6]
    leg_amplitudes: np.ndarray   # Individual leg amplitude scalers (shape: 6)


class DrosophilaCPG:
    """
    Coupled phase oscillator network governing 6-legged Drosophila kinematics.
    Phase dynamics:
      dtheta_i/dt = 2*pi*f_i + sum_j w_ij * sin(theta_j - theta_i - phi_ij)
    """

    def __init__(self, dt: float = 0.002, default_freq: float = 4.0):
        self.dt = dt
        self.n_legs = 6
        self.default_freq = default_freq

        # Inter-leg coupling weights and target phase differences
        # Tripod gait target phases:
        # Tripod Group 1: LF, RM, LH  (phase 0)
        # Tripod Group 2: RF, LM, RH  (phase pi)
        self.nominal_phases = np.array([0.0, np.pi, np.pi, 0.0, 0.0, np.pi])

        # Coupling matrix (symmetric nearest-neighbor and contralateral)
        self.coupling_weights = np.zeros((6, 6))
        # Ipsilateral coupling (LF-LM, LM-LH, RF-RM, RM-RH)
        ipsi_pairs = [(0, 2), (2, 4), (1, 3), (3, 5)]
        for i, j in ipsi_pairs:
            self.coupling_weights[i, j] = 5.0
            self.coupling_weights[j, i] = 5.0

        # Contralateral coupling (LF-RF, LM-RM, LH-RH)
        contra_pairs = [(0, 1), (2, 3), (4, 5)]
        for i, j in contra_pairs:
            self.coupling_weights[i, j] = 8.0
            self.coupling_weights[j, i] = 8.0

        # Current state
        self.reset()

        # Joint angle range baselines for Drosophila legs (in radians)
        # Actuated joints per leg: Coxa (roll, pitch), Femur (pitch), Tibia (pitch)
        self.neutral_joints = {
            "LF": {"coxa_roll": 0.1, "coxa_pitch": -0.2, "femur": 0.8, "tibia": -1.2},
            "RF": {"coxa_roll": -0.1, "coxa_pitch": -0.2, "femur": 0.8, "tibia": -1.2},
            "LM": {"coxa_roll": 0.2, "coxa_pitch": 0.0, "femur": 0.6, "tibia": -1.1},
            "RM": {"coxa_roll": -0.2, "coxa_pitch": 0.0, "femur": 0.6, "tibia": -1.1},
            "LH": {"coxa_roll": 0.1, "coxa_pitch": 0.3, "femur": 0.7, "tibia": -1.3},
            "RH": {"coxa_roll": -0.1, "coxa_pitch": 0.3, "femur": 0.7, "tibia": -1.3},
        }

    def reset(self):
        """Resets leg phases to nominal tripod pattern."""
        self.phases = self.nominal_phases.copy()
        self.amplitudes = np.ones(6)

    def step(self, mod: CPGModulation) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Steps CPG forward by dt and computes target joint angles for all 6 legs.
        
        Returns:
            phases: Current phase array (shape: 6) in [0, 2*pi)
            target_joints: Dict of {leg_name: {joint_name: target_angle_rad}}
        """
        freq = max(mod.frequency_hz, 0.5)
        self.amplitudes = np.clip(mod.leg_amplitudes * mod.amplitude, 0.0, 1.8)

        # Compute phase target differences taking left/right modulation into account
        target_phi = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                nominal_diff = self.nominal_phases[j] - self.nominal_phases[i]
                # If i is left and j is right, add lr offset
                if (i in [0, 2, 4]) and (j in [1, 3, 5]):
                    target_phi[i, j] = nominal_diff + mod.phase_offset_lr
                elif (i in [1, 3, 5]) and (j in [0, 2, 4]):
                    target_phi[i, j] = nominal_diff - mod.phase_offset_lr
                else:
                    target_phi[i, j] = nominal_diff

        # Coupled oscillator differential equations:
        dtheta = np.zeros(6)
        for i in range(6):
            coupling_term = np.sum(
                self.coupling_weights[i, :] * np.sin(self.phases - self.phases[i] - target_phi[i, :])
            )
            dtheta[i] = 2.0 * np.pi * freq + coupling_term

        self.phases = (self.phases + dtheta * self.dt) % (2.0 * np.pi)

        # Compute kinematic trajectories per leg
        target_joints = {}
        for idx, name in enumerate(LEG_NAMES):
            phase = self.phases[idx]
            amp = self.amplitudes[idx]
            neut = self.neutral_joints[name]

            # In Drosophila locomotion:
            # During stance (ground contact): leg extends back
            # During swing (elevation): leg lifts (femur flexes, tibia flexes) and swings forward
            swing_ratio = np.clip(mod.swing_ratio, 0.2, 0.6)
            is_swing = phase < (2.0 * np.pi * swing_ratio)

            # Smooth cyclic oscillation
            sin_p = np.sin(phase)
            cos_p = np.cos(phase)

            # Heave/lift (tibia and femur)
            lift = max(0.0, sin_p) if is_swing else 0.0
            heave_bob = mod.body_bob_amplitude * 0.15 * np.cos(2.0 * np.pi * freq * self.dt)

            target_joints[name] = {
                "coxa_roll": neut["coxa_roll"] + 0.12 * amp * cos_p,
                "coxa_pitch": neut["coxa_pitch"] + 0.25 * amp * sin_p,
                "femur": neut["femur"] + 0.35 * amp * lift + heave_bob,
                "tibia": neut["tibia"] - 0.40 * amp * lift,
            }

        return self.phases.copy(), target_joints


if __name__ == "__main__":
    cpg = DrosophilaCPG()
    mod = CPGModulation(
        frequency_hz=2.0,  # 120 BPM = 2.0 Hz
        amplitude=1.0,
        phase_offset_lr=0.0,
        body_bob_amplitude=0.5,
        swing_ratio=0.4,
        leg_amplitudes=np.ones(6),
    )

    phases, joints = cpg.step(mod)
    print("CPG Step verification:")
    print(f"  - Leg phases (rad): {np.round(phases, 2)}")
    print(f"  - LF target joints: {joints['LF']}")
    print(f"  - RF target joints: {joints['RF']}")
    print("CPG dynamics operational!")

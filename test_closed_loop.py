"""
Closed-Loop Integration Test:
Audio -> Connectome (JON -> AMMC -> DN) -> CPG Motor Controller -> FlyGym Physics Avatar -> Reward Engine

Demonstrates the real-time closed-loop signal flow and records diagnostic metrics.
"""

import os
import sys
import time
import numpy as np

from audio_pipeline import AudioRhythmPipeline, AudioFrame
from connectome_auditory import DrosophilaAuditoryCircuit, NeuralState
from cpg_controller import DrosophilaCPG, CPGModulation, LEG_NAMES
from fly_env import DrosophilaFlyEnv, ProprioceptionState
from dance_reward import DanceRewardEngine, RewardBreakdown


def run_closed_loop_simulation(
    duration: float = 4.0,
    bpm: float = 120.0,
    dt: float = 0.002,
    render_video: bool = True,
    output_video_path: str = "fly_dance_closed_loop.mp4",
):
    print("=" * 70)
    print("STARTING CONNECTOME-DRIVEN DROSOPHILA CLOSED-LOOP SIMULATION")
    print(f"Duration: {duration}s | BPM: {bpm} | Timestep: {dt*1000:.1f}ms")
    print("=" * 70)

    # 1. Initialize Audio / Rhythm Pipeline
    print("[1/5] Initializing Audio Pipeline (120 BPM synthetic dance beat)...")
    audio = AudioRhythmPipeline(bpm=bpm, duration=duration, synthetic_type="dance_beat")

    # 2. Initialize Connectome Auditory Pathway (JON -> AMMC -> DN)
    print("[2/5] Initializing Connectome Auditory Circuit (~160 neurons)...")
    circuit = DrosophilaAuditoryCircuit(n_jon_per_antenna=32, n_ammc_per_side=48, n_dns=24, dt=dt)

    # 3. Initialize CPG Motor Controller
    print("[3/5] Initializing Drosophila CPG Leg Oscillator...")
    cpg = DrosophilaCPG(dt=dt, default_freq=bpm / 60.0)

    # 4. Initialize Biomechanical Avatar & Physics
    print("[4/5] Initializing Drosophila Fly Biomechanical Environment...")
    env = DrosophilaFlyEnv(dt=dt, enable_rendering=render_video)
    proprio = env.reset()

    # 5. Initialize Dance Reward Engine
    print("[5/5] Initializing Dance & Rhythm Reward Engine...")
    reward_engine = DanceRewardEngine(nominal_com_height=1.25)

    # Telemetry storage
    times = []
    beat_pulses = []
    jon_mean_activity = []
    ammc_mean_activity = []
    dn_mean_activity = []
    cpg_freqs = []
    rewards = []
    com_heights = []
    video_frames = []

    n_steps = int(duration / dt)
    print(f"\nRunning {n_steps} closed-loop simulation steps...")
    start_wall_time = time.time()

    for step in range(n_steps):
        t = step * dt

        # A. Audio feature extraction at timestamp t
        audio_frame = audio.get_frame(t)

        # B. Sensory transduction through Connectome Auditory Pathway
        neural_state = circuit.step(
            raw_amplitude=audio_frame.raw_amplitude,
            onset_strength=audio_frame.onset_strength,
            low_band=audio_frame.low_band_energy,
            mid_band=audio_frame.mid_band_energy,
            high_band=audio_frame.high_band_energy,
            beat_pulse=audio_frame.beat_pulse,
            bilateral_bias=0.0,
        )

        # C. Connectome DN -> CPG Modulation (Baseline B: Direct Bio-Rhythm Mapping)
        # Descending Neurons modulate stepping frequency, amplitude, and body heave bobbing
        # DN 0-7 (Transient / beat sensitive) drive rhythmic bursts
        # DN 8-15 (Sustained energy) scale general movement vigor
        # DN 16-23 (Bilateral contrast) modulate left/right sway
        dn_transient = float(np.mean(neural_state.dn_activity[:8]))
        dn_sustained = float(np.mean(neural_state.dn_activity[8:16]))
        dn_asymmetry = float(np.mean(neural_state.dn_activity[16:20]) - np.mean(neural_state.dn_activity[20:24]))

        # Dynamic frequency locked to music tempo (120 BPM = 2.0 Hz stepping baseline)
        target_freq = (bpm / 60.0) * (0.8 + 0.4 * dn_transient)
        target_amplitude = 0.9 + 0.6 * dn_sustained
        body_bob = 0.4 + 0.6 * dn_transient
        phase_offset_lr = dn_asymmetry * 0.2

        cpg_mod = CPGModulation(
            frequency_hz=target_freq,
            amplitude=target_amplitude,
            phase_offset_lr=phase_offset_lr,
            body_bob_amplitude=body_bob,
            swing_ratio=0.38,
            leg_amplitudes=np.ones(6),
        )

        # D. Step CPG to produce coordinated joint targets
        phases, target_joints = cpg.step(cpg_mod)

        # E. Step FlyGym Biomechanical Physics
        proprio = env.step(target_joints)

        # F. Compute Multi-Objective Closed-Loop Reward
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

        # G. Log telemetry
        times.append(t)
        beat_pulses.append(audio_frame.beat_pulse)
        jon_mean_activity.append(float(np.mean(neural_state.jon_left)))
        ammc_mean_activity.append(float(np.mean(neural_state.ammc_left)))
        dn_mean_activity.append(float(np.mean(neural_state.dn_activity)))
        cpg_freqs.append(target_freq)
        rewards.append(breakdown.total_reward)
        com_heights.append(proprio.com_position[2])

        # Record video frame at ~30 FPS (every 16 steps at dt=2ms)
        if render_video and (step % 16 == 0):
            frame = env.render_frame()
            # Overlay telemetry HUD on frame
            import cv2
            hud_text_1 = f"Time: {t:.2f}s | Beat: {'*BEAT*' if audio_frame.beat_pulse > 0.6 else '     '}"
            hud_text_2 = f"JON: {np.mean(neural_state.jon_left):.2f} | DN: {np.mean(neural_state.dn_activity):.2f}"
            hud_text_3 = f"Freq: {target_freq:.2f} Hz | Reward: {breakdown.total_reward:.2f}"
            cv2.putText(frame, hud_text_1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, hud_text_2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)
            cv2.putText(frame, hud_text_3, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 120), 1)
            video_frames.append(frame)

        if (step + 1) % 500 == 0 or (step + 1) == n_steps:
            elapsed = time.time() - start_wall_time
            steps_per_sec = (step + 1) / elapsed
            print(f"  Step {step+1:4d}/{n_steps} ({t:.2f}s) | "
                  f"Throughput: {steps_per_sec:.0f} steps/s | "
                  f"Mean Reward: {np.mean(rewards[-500:]):.2f} | "
                  f"COM Height: {proprio.com_position[2]:.2f}mm")

    total_sim_time = time.time() - start_wall_time
    print("-" * 70)
    print(f"Simulation Complete in {total_sim_time:.2f} seconds ({n_steps / total_sim_time:.0f} steps/sec)")
    print(f"Average Step Reward:  {np.mean(rewards):.3f}")
    print(f"Average COM Height:   {np.mean(com_heights):.3f} mm (Stability Target: 1.25 mm)")
    print(f"Mean JON Activity:    {np.mean(jon_mean_activity):.3f}")
    print(f"Mean DN Activity:     {np.mean(dn_mean_activity):.3f}")

    # Render video if requested
    if render_video and len(video_frames) > 0:
        print(f"\nSaving {len(video_frames)} frames to video: {output_video_path}...")
        saved = False
        try:
            import cv2
            h, w, _ = video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
            for f in video_frames:
                writer.write(f)
            writer.release()
            print(f"Video saved successfully with OpenCV: {os.path.abspath(output_video_path)}")
            saved = True
        except Exception as e_cv:
            print(f"OpenCV writer note: {e_cv}")

        if not saved:
            try:
                import imageio
                rgb_frames = [f[:, :, ::-1] for f in video_frames]
                imageio.mimsave(output_video_path, rgb_frames, fps=30)
                print(f"Video saved successfully with ImageIO: {os.path.abspath(output_video_path)}")
            except Exception as e:
                print(f"Failed to save video: {e}")

    # Generate diagnostic plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(times, beat_pulses, label="Audio Beat Pulse", color="crimson", lw=1.5)
        axes[0].set_ylabel("Audio")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, jon_mean_activity, label="Mean JON Activity", color="orange", lw=1.2)
        axes[1].plot(times, ammc_mean_activity, label="Mean AMMC Activity", color="purple", lw=1.2)
        axes[1].plot(times, dn_mean_activity, label="Mean DN Activity", color="dodgerblue", lw=1.5)
        axes[1].set_ylabel("Neural Rate")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, cpg_freqs, label="CPG Stepping Freq (Hz)", color="teal", lw=1.5)
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(times, rewards, label="Total Dance Reward", color="forestgreen", lw=1.2)
        axes[3].set_ylabel("Reward")
        axes[3].set_xlabel("Time (seconds)")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, alpha=0.3)

        plt.suptitle(f"Connectome-Driven Drosophila Dance Closed-Loop (120 BPM)", fontsize=13)
        plt.tight_layout()
        plot_path = "closed_loop_telemetry.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Telemetry plot saved: {os.path.abspath(plot_path)}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print("=" * 70)
    return {
        "mean_reward": np.mean(rewards),
        "mean_com_height": np.mean(com_heights),
        "mean_dn_activity": np.mean(dn_mean_activity),
    }


if __name__ == "__main__":
    run_closed_loop_simulation()

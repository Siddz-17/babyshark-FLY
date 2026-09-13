"""
Dance to Baby Shark: The Signature Pipeline Demonstration.
Connects the full bio-neuromechanical pipeline to the 'Baby Shark' song:
1. Ingests or synthesizes the iconic 'Baby Shark' song (115 BPM, signature 'doo doo' rhythm motif).
2. AudioRhythmPipeline extracts acoustic pressure, onset envelope, and spectral bands.
3. Johnston's Organ Neurons (JON) transduce the acoustic wave -> AMMC -> Descending Neurons.
4. Trained PPO Policy + CPG drives the 6 legs to synchronize footfalls and body bobbing to the song.
5. Renders high-resolution video frames and uses ffmpeg to multiplex the audio with the dance!

Produces:
- babyshark_fly_dance.mp4 (High-quality video with synchronized audio)
"""

import os
import sys
import subprocess
from typing import Tuple, Optional
import numpy as np
import scipy.io.wavfile as wavfile
import cv2
import torch

# Ensure repo root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from audio_pipeline import AudioRhythmPipeline
from connectome_auditory import DrosophilaAuditoryCircuit
from cpg_controller import DrosophilaCPG, CPGModulation
from fly_env import DrosophilaFlyEnv
from dance_reward import DanceRewardEngine
from ppo.agent import PPOAgent


def synthesize_babyshark_theme(duration: float = 12.0, sr: int = 22050, bpm: float = 115.0) -> Tuple[np.ndarray, str]:
    """
    Synthesizes the complete, authentic 'Baby Shark' melody and rhythm track:
    - Melody: D4, E4, G4, G4, G4, G4, G4, G4 ('Baby shark, doo doo doo doo doo doo')
    - Drums: Bouncy pop-dance kick and snare
    - Bassline: Groovy synth bass
    """
    out_dir = os.path.join(repo_root, "data", "audio")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "babyshark_song.wav")

    # If user provided a real babyshark audio file, prioritize it!
    user_candidates = [
        os.path.join(out_dir, "babyshark.mp3"),
        os.path.join(out_dir, "babyshark.wav"),
        os.path.join(repo_root, "babyshark.mp3"),
        os.path.join(repo_root, "babyshark.wav"),
    ]
    for c in user_candidates:
        if os.path.exists(c):
            print(f"Found existing user Baby Shark audio track: {c}")
            return None, c

    print("Synthesizing iconic 'Baby Shark' melody & rhythm track...")
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    audio = np.zeros(total_samples, dtype=np.float32)

    spb = 60.0 / bpm  # seconds per quarter note
    sp_eighth = spb / 2.0

    # Note frequencies (Hz)
    D4 = 293.66
    E4 = 329.63
    G4 = 392.00
    Fs4 = 369.99
    G3 = 196.00
    D3 = 146.83
    C3 = 130.81

    # Baby shark melody motif timing in eighth notes:
    # Beat 1: D4 (quarter = 2 eighths: 'Ba-')
    # Beat 2: E4 (quarter = 2 eighths: '-by')
    # Beat 3: G4 (eighth: 'shark'), G4 (eighth: 'doo')
    # Beat 4: G4 (eighth: 'doo'), G4 (eighth: 'doo')
    # Bar 2:
    # Beat 1: G4 (eighth: 'doo'), G4 (eighth: 'doo')
    # and repeat...

    bar_len = 4 * spb  # ~2.086 seconds per bar

    for bar in range(int(duration / bar_len) + 1):
        bar_start = bar * bar_len
        if bar_start >= duration:
            break

        # 1. Bouncy Drums (Kick on 1 & 3, Snare on 2 & 4, Hi-hat on all eighths)
        for beat in range(4):
            t_beat = bar_start + beat * spb
            if t_beat >= duration:
                break
            dt = t - t_beat

            # Kick on 0 and 2
            if beat in [0, 2]:
                mask_k = (dt >= 0) & (dt < 0.18)
                audio[mask_k] += 0.45 * np.sin(2 * np.pi * 90 * dt[mask_k]) * np.exp(-dt[mask_k] / 0.04)

            # Snare on 1 and 3
            if beat in [1, 3]:
                mask_s = (dt >= 0) & (dt < 0.15)
                noise = np.random.uniform(-0.35, 0.35, size=np.sum(mask_s))
                audio[mask_s] += 0.35 * noise * np.exp(-dt[mask_s] / 0.03)

        # 2. Bassline (Roots on G and D)
        for beat in range(4):
            t_beat = bar_start + beat * spb
            dt = t - t_beat
            mask_b = (dt >= 0) & (dt < spb * 0.8)
            freq_bass = G3 if bar % 2 == 0 else D3
            audio[mask_b] += 0.30 * np.sin(2 * np.pi * freq_bass * dt[mask_b]) * np.exp(-dt[mask_b] / 0.25)

        # 3. Melody: 'Baby shark, doo-doo-doo-doo-doo-doo'
        if bar < 3:
            notes = [
                (0.0, spb, D4),          # Ba-
                (spb, spb, E4),          # -by
                (2 * spb, sp_eighth, G4),      # shark,
                (2.5 * spb, sp_eighth, G4),    # doo
                (3.0 * spb, sp_eighth, G4),    # doo
                (3.5 * spb, sp_eighth, G4),    # doo
                (4.0 * spb, sp_eighth, G4),    # doo
                (4.5 * spb, sp_eighth, G4),    # doo
            ]
        else:
            # Final phrase: 'Baby shark!'
            notes = [
                (0.0, spb, D4),
                (spb, spb, E4),
                (2 * spb, spb, G4),
                (3 * spb, sp_eighth, G4),
                (3.5 * spb, sp_eighth, Fs4),
            ]

        for n_start, n_dur, freq in notes:
            t_note = bar_start + n_start
            dt = t - t_note
            mask_n = (dt >= 0) & (dt < min(n_dur * 0.9, duration - t_note))
            if np.any(mask_n):
                # Bright lead synth tone (fundamental + 2nd harmonic)
                synth = (
                    0.55 * np.sin(2 * np.pi * freq * dt[mask_n])
                    + 0.25 * np.sin(2 * np.pi * 2 * freq * dt[mask_n])
                    + 0.12 * np.sin(2 * np.pi * 3 * freq * dt[mask_n])
                )
                env = np.exp(-dt[mask_n] / (n_dur * 0.8))
                audio[mask_n] += synth * env

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    wavfile.write(out_path, sr, (audio * 32767).astype(np.int16))
    print(f"Baby Shark audio synthesized and saved to: {out_path}")
    return audio, out_path


def run_babyshark_dance(
    duration: float = 10.0,
    policy_path: str = "checkpoints/best_dance_policy.pt",
    output_mp4: str = "babyshark_fly_dance.mp4",
):
    print("=" * 85)
    print("BABYSHARK-FLY: CONNECTOME-PPO DANCE DEMONSTRATION")
    print("=" * 85)

    # 1. Obtain Baby Shark audio
    _, audio_path = synthesize_babyshark_theme(duration=duration)

    # 2. Initialize Audio & Neural Pipeline
    dt = 0.002  # 500 Hz physics
    pipeline = AudioRhythmPipeline(audio_path=audio_path, duration=duration)
    circuit = DrosophilaAuditoryCircuit(dt=dt, mode="biological")
    cpg = DrosophilaCPG(dt=dt, default_freq=pipeline.tempo_bpm / 60.0)
    fly = DrosophilaFlyEnv(backend="simple", dt=dt, enable_rendering=True)
    reward_engine = DanceRewardEngine(nominal_com_height=1.25)

    # 3. Load Trained PPO Policy
    agent = PPOAgent(obs_dim=83, action_dim=8)
    if not os.path.exists(policy_path):
        policy_path = os.path.join(repo_root, "checkpoints", "best_dance_policy.pt")
    if os.path.exists(policy_path):
        agent.load_checkpoint(policy_path)
        print(f"Loaded trained policy: {policy_path}")
    agent.eval()

    n_steps = int(duration / dt)
    raw_video_path = os.path.join(repo_root, "temp_video_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(raw_video_path, fourcc, 30, (640, 480))

    proprio = fly.reset()
    syncs = []
    rewards = []

    print(f"\nStepping Drosophila simulation to Baby Shark ({pipeline.tempo_bpm:.1f} BPM)...")
    for step in range(n_steps):
        t = step * dt
        audio_frame = pipeline.get_frame(t)

        # Johnston's Organ mechanotransduction
        neural = circuit.step(
            raw_amplitude=audio_frame.raw_amplitude,
            onset_strength=audio_frame.onset_strength,
            low_band=audio_frame.low_band_energy,
            mid_band=audio_frame.mid_band_energy,
            high_band=audio_frame.high_band_energy,
        )

        audio_features = np.array([
            audio_frame.raw_amplitude,
            audio_frame.onset_strength,
            audio_frame.low_band_energy,
            audio_frame.mid_band_energy,
            audio_frame.high_band_energy,
        ], dtype=np.float32)

        obs = np.concatenate([proprio.vector.astype(np.float32), neural.dn_activity.astype(np.float32), audio_features])
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            act = agent.act_deterministic(obs_tensor).squeeze(0).numpy()

        dn_transient = float(np.mean(neural.dn_activity[:8]))
        dn_sustained = float(np.mean(neural.dn_activity[8:16]))
        dn_asymmetry = float(np.mean(neural.dn_activity[16:20]) - np.mean(neural.dn_activity[20:24]))

        base_freq = (pipeline.tempo_bpm / 60.0) * (0.85 + 0.35 * dn_transient)
        freq_mod = act[0] * 0.8
        amp_mod = act[1] * 0.4
        bob_mod = act[2] * 0.5  # Emphasize bobbing on 'doo-doo'
        lr_mod = act[3] * 0.3
        swing_mod = act[4] * 0.08

        leg_amps = np.ones(6)
        leg_amps[:2] += act[5] * 0.3
        leg_amps[2:4] += act[6] * 0.3
        leg_amps[4:] += act[7] * 0.3
        leg_amps = np.clip(leg_amps, 0.4, 1.6)

        cpg_mod = CPGModulation(
            frequency_hz=max(0.5, base_freq + freq_mod),
            amplitude=np.clip(0.9 + 0.5 * dn_sustained + amp_mod, 0.4, 1.8),
            phase_offset_lr=dn_asymmetry * 0.2 + lr_mod,
            body_bob_amplitude=np.clip(0.35 + 0.65 * dn_transient + bob_mod, 0.0, 1.0),
            swing_ratio=np.clip(0.38 + swing_mod, 0.25, 0.55),
            leg_amplitudes=leg_amps,
        )

        phases, target_joints = cpg.step(cpg_mod)
        proprio = fly.step(target_joints)

        breakdown = reward_engine.compute_reward(
            sim_time=t,
            beat_times=pipeline.beat_times,
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

        syncs.append(breakdown.beat_sync)
        rewards.append(breakdown.total_reward)

        # Render frame at 30 fps (every ~16-17 steps at dt=0.002)
        if step % 16 == 0:
            frame = fly.render_frame()
            if frame is not None:
                # Add HUD display
                cv2.putText(frame, "SONG: BABY SHARK (115 BPM)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"Time: {t:.2f}s | Beat Sync: {breakdown.beat_sync:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"CPG Stepping Freq: {cpg_mod.frequency_hz:.2f} Hz", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Body Bob: {cpg_mod.body_bob_amplitude:.2f} | JON Transient: {dn_transient:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
                out_video.write(frame)

    out_video.release()
    print(f"Raw video rendered: {raw_video_path}")

    # 4. Multiplex Audio + Video with ffmpeg
    final_output = os.path.join(repo_root, output_mp4)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", raw_video_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_output
    ]
    print(f"Multiplexing audio and video with ffmpeg: {' '.join(ffmpeg_cmd)}")
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f" Successfully produced finalized dance video: {final_output}")
    except Exception as e:
        print(f"Note: ffmpeg encoding fallback (saving mp4 directly without re-encode): {e}")
        # Fallback copy
        import shutil
        shutil.copyfile(raw_video_path, final_output)

    if os.path.exists(raw_video_path):
        try:
            os.remove(raw_video_path)
        except:
            pass

    mean_sync = float(np.mean(syncs))
    mean_rew = float(np.mean(rewards))
    print("\n" + "=" * 85)
    print(f"BABY SHARK DANCE COMPLETE!")
    print(f"  - Song Duration: {duration} s")
    print(f"  - Footfall Beat Sync Score: {mean_sync:.4f}")
    print(f"  - Mean Step Reward: {mean_rew:.2f}")
    print(f"  - Video File: {final_output}")
    print("=" * 85)


if __name__ == "__main__":
    run_babyshark_dance(duration=10.0)

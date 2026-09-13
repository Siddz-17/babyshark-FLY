"""
Audio and Rhythm Processing Pipeline for Drosophila Simulation.
Extracts continuous rhythm features (onset envelope, beat phase, tempo, multi-band spectral energy)
and provides time-synchronized frames matching the physics simulation timestep.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AudioFrame:
    """Audio feature state at a specific simulation timestamp."""
    time: float
    beat_phase: float       # Continuous phase in [0, 2*pi), 0 corresponds to beat onset
    beat_pulse: float       # Gaussian impulse centered at beat moments (0 to 1)
    onset_strength: float   # Transient onset strength normalized [0, 1]
    low_band_energy: float  # 100-300 Hz energy (pulse song / bass)
    mid_band_energy: float  # 300-800 Hz energy (sine song / midrange)
    high_band_energy: float # >800 Hz energy (percussion / treble)
    raw_amplitude: float    # Instantaneous normalized acoustic pressure [-1, 1]
    tempo_bpm: float        # Estimated tempo in BPM


class AudioRhythmPipeline:
    """
    Processes audio input (WAV/MP3 or synthetic pulses) into continuous
    features for the Johnston's Organ (JON) auditory neural model.
    """

    def __init__(
        self,
        audio_path: Optional[str] = None,
        sample_rate: int = 22050,
        bpm: float = 120.0,
        duration: float = 10.0,
        synthetic_type: str = "dance_beat",
    ):
        self.sample_rate = sample_rate
        self.duration = duration

        if audio_path is not None:
            self.load_audio_file(audio_path)
        else:
            self.generate_synthetic_track(bpm=bpm, duration=duration, track_type=synthetic_type)

        self._extract_features()

    def generate_synthetic_track(self, bpm: float = 120.0, duration: float = 10.0, track_type: str = "dance_beat"):
        """Generates a synthetic rhythmic test signal with known ground-truth beats."""
        self.duration = duration
        self.tempo_bpm = bpm
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        self.time_axis = t
        waveform = np.zeros_like(t)

        seconds_per_beat = 60.0 / bpm
        self.ground_truth_beat_times = np.arange(0.0, duration, seconds_per_beat)

        if track_type == "dance_beat":
            # 4/4 drum rhythm: kick on beats 1 & 3, snare/clap on 2 & 4
            for i, beat_t in enumerate(self.ground_truth_beat_times):
                dt = t - beat_t
                mask = (dt >= 0) & (dt < 0.25)
                if not np.any(mask):
                    continue

                if i % 2 == 0:
                    # Bass Kick: 150 Hz sweeping down to 50 Hz with exponential decay
                    kick_decay = np.exp(-dt[mask] / 0.08)
                    kick_freq = 150.0 * np.exp(-dt[mask] / 0.04) + 45.0
                    kick = np.sin(2 * np.pi * kick_freq * dt[mask]) * kick_decay
                    waveform[mask] += 0.8 * kick
                else:
                    # Snare: 250 Hz tone + white noise burst
                    snare_decay = np.exp(-dt[mask] / 0.12)
                    noise = (np.random.rand(np.sum(mask)) * 2.0 - 1.0)
                    tone = np.sin(2 * np.pi * 280.0 * dt[mask])
                    waveform[mask] += 0.7 * (0.5 * tone + 0.5 * noise) * snare_decay

        elif track_type == "courtship_song":
            # Drosophila courtship pulse song: train of ~200 Hz sinusoidal pulses every 35 ms (IPI)
            pulse_interval = 0.035  # 35 ms inter-pulse interval
            pulse_times = np.arange(0.2, duration, pulse_interval)
            for p_t in pulse_times:
                dt = t - p_t
                mask = (dt >= 0) & (dt < 0.015)
                if np.any(mask):
                    pulse_decay = np.sin(np.pi * dt[mask] / 0.015)  # Hanning envelope
                    pulse = np.sin(2 * np.pi * 220.0 * dt[mask]) * pulse_decay
                    waveform[mask] += 0.9 * pulse

        # Normalize waveform to [-1, 1]
        max_val = np.max(np.abs(waveform)) + 1e-8
        self.waveform = waveform / max_val

    def load_audio_file(self, audio_path: str):
        """Loads an audio file using librosa or soundfile."""
        try:
            import librosa
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            self.waveform = waveform
            self.sample_rate = sr
            self.duration = len(waveform) / sr
            self.time_axis = np.linspace(0, self.duration, len(waveform), endpoint=False)
            self.tempo_bpm = 120.0  # Will be refined in feature extraction
            self.ground_truth_beat_times = None
        except Exception as e:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            self.waveform = data
            self.sample_rate = sr
            self.duration = len(data) / sr
            self.time_axis = np.linspace(0, self.duration, len(data), endpoint=False)
            self.tempo_bpm = 120.0
            self.ground_truth_beat_times = None

    def _extract_features(self):
        """Extracts onset envelope, beat timestamps, and frequency band profiles."""
        hop_length = 512
        frame_rate = self.sample_rate / hop_length

        try:
            import librosa
            # 1. Onset envelope
            onset_env = librosa.onset.onset_strength(
                y=self.waveform,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            # Normalize onset strength
            onset_env = onset_env / (np.max(onset_env) + 1e-8)

            # 2. Beat tracking
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            if hasattr(tempo, "__len__"):
                tempo = float(tempo[0])
            self.tempo_bpm = float(tempo) if tempo > 30 else self.tempo_bpm
            beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=hop_length)
            self.beat_times = beat_times if len(beat_times) > 0 else self.ground_truth_beat_times

            # 3. Spectral energy bands
            stft = np.abs(librosa.stft(self.waveform, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=self.sample_rate)

            # Band 1: 100-300 Hz
            b1_idx = (freqs >= 100) & (freqs < 300)
            b1_energy = np.mean(stft[b1_idx, :], axis=0) if np.any(b1_idx) else np.zeros(stft.shape[1])

            # Band 2: 300-800 Hz
            b2_idx = (freqs >= 300) & (freqs < 800)
            b2_energy = np.mean(stft[b2_idx, :], axis=0) if np.any(b2_idx) else np.zeros(stft.shape[1])

            # Band 3: > 800 Hz
            b3_idx = freqs >= 800
            b3_energy = np.mean(stft[b3_idx, :], axis=0) if np.any(b3_idx) else np.zeros(stft.shape[1])

            # Normalize band energies
            self.b1_energy = b1_energy / (np.max(b1_energy) + 1e-8)
            self.b2_energy = b2_energy / (np.max(b2_energy) + 1e-8)
            self.b3_energy = b3_energy / (np.max(b3_energy) + 1e-8)
            self.onset_env = onset_env
            self.feature_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sample_rate, hop_length=hop_length)

        except Exception as e:
            # SciPy fallback if librosa is unavailable
            from scipy.signal import spectrogram
            freqs, times, Sxx = spectrogram(self.waveform, fs=self.sample_rate, nperseg=1024, noverlap=512)
            self.feature_times = times
            spectral_diff = np.diff(np.mean(Sxx, axis=0), prepend=0)
            onset_env = np.maximum(0, spectral_diff)
            self.onset_env = onset_env / (np.max(onset_env) + 1e-8)

            b1_idx = (freqs >= 100) & (freqs < 300)
            b2_idx = (freqs >= 300) & (freqs < 800)
            b3_idx = freqs >= 800

            self.b1_energy = np.mean(Sxx[b1_idx, :], axis=0) if np.any(b1_idx) else np.zeros_like(times)
            self.b2_energy = np.mean(Sxx[b2_idx, :], axis=0) if np.any(b2_idx) else np.zeros_like(times)
            self.b3_energy = np.mean(Sxx[b3_idx, :], axis=0) if np.any(b3_idx) else np.zeros_like(times)

            self.b1_energy /= (np.max(self.b1_energy) + 1e-8)
            self.b2_energy /= (np.max(self.b2_energy) + 1e-8)
            self.b3_energy /= (np.max(self.b3_energy) + 1e-8)

            if self.ground_truth_beat_times is not None:
                self.beat_times = self.ground_truth_beat_times
            else:
                spb = 60.0 / self.tempo_bpm
                self.beat_times = np.arange(0, self.duration, spb)

    def get_frame(self, t: float) -> AudioFrame:
        """
        Retrieves continuous synchronized audio feature state at simulation time t.
        """
        # Loop audio if t exceeds duration
        t_mod = t % self.duration

        # 1. Raw acoustic sample
        sample_idx = int(t_mod * self.sample_rate) % len(self.waveform)
        raw_amp = float(self.waveform[sample_idx])

        # 2. Feature interpolation
        onset = float(np.interp(t_mod, self.feature_times, self.onset_env))
        b1 = float(np.interp(t_mod, self.feature_times, self.b1_energy))
        b2 = float(np.interp(t_mod, self.feature_times, self.b2_energy))
        b3 = float(np.interp(t_mod, self.feature_times, self.b3_energy))

        # 3. Beat phase and pulse calculation
        # Find nearest previous and next beats
        if self.beat_times is not None and len(self.beat_times) >= 2:
            idx = np.searchsorted(self.beat_times, t_mod)
            if idx == 0:
                prev_beat = self.beat_times[0] - (60.0 / self.tempo_bpm)
                next_beat = self.beat_times[0]
            elif idx >= len(self.beat_times):
                prev_beat = self.beat_times[-1]
                next_beat = self.beat_times[-1] + (60.0 / self.tempo_bpm)
            else:
                prev_beat = self.beat_times[idx - 1]
                next_beat = self.beat_times[idx]

            interval = max(next_beat - prev_beat, 1e-4)
            fraction = (t_mod - prev_beat) / interval
            beat_phase = float((fraction * 2.0 * np.pi) % (2.0 * np.pi))

            # Narrow Gaussian pulse centered at the beat (phase ~ 0 or 2*pi)
            phase_dist = min(beat_phase, 2.0 * np.pi - beat_phase)
            beat_pulse = float(np.exp(-0.5 * (phase_dist / 0.15) ** 2))
        else:
            spb = 60.0 / self.tempo_bpm
            fraction = (t_mod % spb) / spb
            beat_phase = float(fraction * 2.0 * np.pi)
            phase_dist = min(beat_phase, 2.0 * np.pi - beat_phase)
            beat_pulse = float(np.exp(-0.5 * (phase_dist / 0.15) ** 2))

        return AudioFrame(
            time=t,
            beat_phase=beat_phase,
            beat_pulse=beat_pulse,
            onset_strength=onset,
            low_band_energy=b1,
            mid_band_energy=b2,
            high_band_energy=b3,
            raw_amplitude=raw_amp,
            tempo_bpm=self.tempo_bpm,
        )


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    print("Generating synthetic 120 BPM dance beat track...")
    pipeline = AudioRhythmPipeline(bpm=120.0, duration=4.0, synthetic_type="dance_beat")

    ts = np.linspace(0, 4.0, 800)
    frames = [pipeline.get_frame(t) for t in ts]

    phases = [f.beat_phase for f in frames]
    pulses = [f.beat_pulse for f in frames]
    onsets = [f.onset_strength for f in frames]
    b1s = [f.low_band_energy for f in frames]

    print(f"Pipeline initialized. Duration: {pipeline.duration}s, BPM: {pipeline.tempo_bpm}")
    print(f"Extracted {len(frames)} frames across 4.0 seconds.")
    print("Audio Pipeline verification complete!")

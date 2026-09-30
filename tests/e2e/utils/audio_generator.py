"""
Procedural Audio & Signal Generator for E2E Test Suite.

Provides synthetic signal generation (sine, chirp, noise, multitone, silence),
lossless and quantized WAV/PCM file serialization, and corrupted/adversarial
audio file generation for boundary and negative testing.
"""

import os
import struct
import wave
from pathlib import Path
from typing import Sequence, Tuple, Union, Optional
import numpy as np

try:
    from scipy.io import wavfile as scipy_wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioGenerator:
    """Procedural audio waveform generator and WAV/PCM serializer."""

    @staticmethod
    def generate_sine_wave(
        sample_rate: int = 24000,
        duration_sec: float = 1.0,
        freq: float = 440.0,
        amplitude: float = 0.9
    ) -> np.ndarray:
        """Generates a single-frequency sinusoidal waveform.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            duration_sec: Duration in seconds.
            freq: Frequency in Hz.
            amplitude: Peak amplitude in range (0, 1.0].
            
        Returns:
            np.ndarray of shape (num_samples,) with dtype float32.
        """
        num_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
        waveform = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        return waveform

    @staticmethod
    def generate_chirp(
        sample_rate: int = 24000,
        duration_sec: float = 1.0,
        f0: float = 20.0,
        f1: float = 8000.0,
        amplitude: float = 0.9
    ) -> np.ndarray:
        """Generates a linear frequency sweep (chirp) from f0 to f1.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            duration_sec: Duration in seconds.
            f0: Starting frequency in Hz.
            f1: Ending frequency in Hz.
            amplitude: Peak amplitude in range (0, 1.0].
            
        Returns:
            np.ndarray of shape (num_samples,) with dtype float32.
        """
        num_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
        # Instantaneous phase for linear chirp: 2*pi*(f0*t + ((f1 - f0)/(2*duration))*t^2)
        phase = 2.0 * np.pi * (f0 * t + ((f1 - f0) / (2.0 * duration_sec)) * (t ** 2))
        waveform = (amplitude * np.sin(phase)).astype(np.float32)
        return waveform

    @staticmethod
    def generate_white_noise(
        sample_rate: int = 24000,
        duration_sec: float = 1.0,
        amplitude: float = 0.9,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """Generates uniform white noise.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            duration_sec: Duration in seconds.
            amplitude: Peak amplitude limit in range (0, 1.0].
            seed: Random seed for deterministic reproducibility.
            
        Returns:
            np.ndarray of shape (num_samples,) with dtype float32.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        num_samples = int(sample_rate * duration_sec)
        return rng.uniform(-amplitude, amplitude, num_samples).astype(np.float32)

    @staticmethod
    def generate_silence(
        sample_rate: int = 24000,
        duration_sec: float = 1.0
    ) -> np.ndarray:
        """Generates zero-amplitude silence audio.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            duration_sec: Duration in seconds.
            
        Returns:
            np.ndarray of zeros with shape (num_samples,) and dtype float32.
        """
        num_samples = int(sample_rate * duration_sec)
        return np.zeros(num_samples, dtype=np.float32)

    @staticmethod
    def generate_multitone(
        sample_rate: int = 24000,
        duration_sec: float = 1.0,
        freqs: Sequence[float] = (440.0, 880.0, 1760.0),
        amplitude: float = 0.9
    ) -> np.ndarray:
        """Generates a composite waveform summing multiple harmonic frequencies.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            duration_sec: Duration in seconds.
            freqs: Sequence of frequencies in Hz.
            amplitude: Total peak amplitude normalization limit.
            
        Returns:
            np.ndarray of shape (num_samples,) with dtype float32.
        """
        num_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
        waveform = np.zeros(num_samples, dtype=np.float32)
        for f in freqs:
            waveform += np.sin(2.0 * np.pi * f * t).astype(np.float32)
        # Normalize to target peak amplitude
        max_val = np.max(np.abs(waveform))
        if max_val > 1e-6:
            waveform = (waveform / max_val) * amplitude
        return waveform.astype(np.float32)

    @staticmethod
    def generate_stereo(
        left_audio: np.ndarray,
        right_audio: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Combines two mono channels into a 2-channel interleaved stereo array.
        
        Args:
            left_audio: Left channel 1D float32 array.
            right_audio: Right channel 1D float32 array (if None, copies left).
            
        Returns:
            np.ndarray of shape (num_samples, 2) with dtype float32.
        """
        if right_audio is None:
            right_audio = left_audio.copy()
        min_len = min(len(left_audio), len(right_audio))
        stereo = np.column_stack((left_audio[:min_len], right_audio[:min_len]))
        return stereo.astype(np.float32)

    @staticmethod
    def save_wav(
        filepath: Union[str, Path],
        audio: np.ndarray,
        sample_rate: int = 24000,
        dtype: str = "float32"
    ) -> str:
        """Saves audio array to a standard RIFF/WAVE file on disk.
        
        Args:
            filepath: Destination file path.
            audio: 1D (mono) or 2D (stereo) float32 audio array normalized in [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz.
            dtype: Target audio encoding ("float32", "int16", "int32").
            
        Returns:
            Absolute string path to saved WAV file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if HAS_SCIPY:
            if dtype == "int16":
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                scipy_wavfile.write(str(filepath), sample_rate, pcm)
            elif dtype == "int32":
                pcm = (np.clip(audio, -1.0, 1.0) * 2147483647.0).astype(np.int32)
                scipy_wavfile.write(str(filepath), sample_rate, pcm)
            else:  # float32
                scipy_wavfile.write(str(filepath), sample_rate, audio.astype(np.float32))
        else:
            # Standard library wave module fallback
            channels = 1 if audio.ndim == 1 else audio.shape[1]
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(channels)
                wf.setframerate(sample_rate)
                if dtype == "int16":
                    wf.setsampwidth(2)
                    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                    wf.writeframes(pcm.tobytes())
                else:
                    # Float32 fallback through wave/struct
                    wf.setsampwidth(2)
                    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                    wf.writeframes(pcm.tobytes())

        return str(filepath.resolve())

    @staticmethod
    def save_raw_pcm(
        filepath: Union[str, Path],
        audio: np.ndarray,
        dtype: str = "float32"
    ) -> str:
        """Saves raw uncompressed PCM byte stream without headers.
        
        Args:
            filepath: Destination file path.
            audio: 1D or 2D float32 audio array.
            dtype: Target encoding ("float32" or "int16").
            
        Returns:
            Absolute string path to saved raw PCM file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            if dtype == "int16":
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                f.write(pcm.tobytes())
            else:
                f.write(audio.astype(np.float32).tobytes())

        return str(filepath.resolve())

    @staticmethod
    def save_corrupt_wav(
        filepath: Union[str, Path],
        corruption_type: str = "truncated_header"
    ) -> str:
        """Creates an intentionally corrupted or malformed WAV file for negative testing.
        
        Args:
            filepath: Destination file path.
            corruption_type: Type of corruption:
                - "truncated_header": Incomplete RIFF/WAVE header chunk.
                - "invalid_riff": Non-RIFF magic identifier.
                - "zero_bytes": 0-byte empty file.
                - "truncated_data": Valid header declaring 100,000 samples but truncated data payload.
                - "channel_overflow": 6-channel surround sound file (exceeds 2-channel engine limit).
                - "corrupt_sample_rate": 0 Hz sample rate in format chunk.
                
        Returns:
            Absolute string path to corrupted WAV file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            if corruption_type == "truncated_header":
                # Incomplete 16-byte header
                f.write(b"RIFF\x24\x00\x00\x00WAVEfmt ")
            elif corruption_type == "invalid_riff":
                # Corrupted magic bytes
                f.write(b"CORRUPT_MAGIC_HEADER_NOT_RIFF" + b"\x00" * 64)
            elif corruption_type == "zero_bytes":
                # 0-byte empty file
                pass
            elif corruption_type == "truncated_data":
                # Valid WAV header declaring 40,000 bytes of data, but only writing 16 bytes
                header = struct.pack(
                    "<4sI4s4sIHHIIHH4sI",
                    b"RIFF", 40036, b"WAVE",
                    b"fmt ", 16, 1, 1, 24000, 48000, 2, 16,
                    b"data", 40000
                )
                f.write(header)
                f.write(b"\x00" * 16)  # Far less than 40,000 bytes
            elif corruption_type == "channel_overflow":
                # 6-channel audio (5.1 surround sound)
                num_channels = 6
                sample_rate = 24000
                bits_per_sample = 16
                byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
                block_align = num_channels * (bits_per_sample // 8)
                data_size = 6000 * block_align
                header = struct.pack(
                    "<4sI4s4sIHHIIHH4sI",
                    b"RIFF", 36 + data_size, b"WAVE",
                    b"fmt ", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
                    b"data", data_size
                )
                f.write(header)
                f.write(np.zeros(6000 * num_channels, dtype=np.int16).tobytes())
            elif corruption_type == "corrupt_sample_rate":
                # Sample rate = 0
                header = struct.pack(
                    "<4sI4s4sIHHIIHH4sI",
                    b"RIFF", 36 + 1000, b"WAVE",
                    b"fmt ", 16, 1, 1, 0, 0, 2, 16,
                    b"data", 1000
                )
                f.write(header)
                f.write(b"\x00" * 1000)
            else:
                raise ValueError(f"Unknown corruption_type: {corruption_type}")

        return str(filepath.resolve())

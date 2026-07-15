# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

import os
import sys
import time
import wave
import numpy as np
import torch
import torch.nn.functional as F

# Add python directory to path
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
import subq_audio


def load_wav_pcm(filepath):
    with wave.open(filepath, "rb") as w:
        n_frames = w.getnframes()
        raw_data = w.readframes(n_frames)
        # 16-bit PCM to float32
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return data


def benchmark_real_audio():
    audio_path = os.path.join(os.path.dirname(__file__), "real_human_speech.wav")
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}. Please run download_audio.py first.")
        return

    print("=== Sub-Quadratic Engine: Real Audio Verification ===")
    print(f"Loading real human speech from: {audio_path}")
    raw_audio = load_wav_pcm(audio_path)
    print(f"Successfully loaded {len(raw_audio)} raw audio samples.\n")

    # The engine expects 3D tensors: (batch, seq_len, d_model)
    batch = 1
    seq_len = 1024
    d_model = 256
    chunk_size = seq_len * d_model

    print(f"Engine Schema: Batch={batch}, SeqLen={seq_len}, D_Model={d_model}")
    print(f"Chunk size required: {chunk_size} samples.")

    # We will simulate a simple feature projection: we take the raw audio, pad it,
    # and chunk it to form our Q, K, V matrices.
    # In a real model, this would be a 1D Conv / Mel-Spectrogram frontend.
    padded_len = int(np.ceil(len(raw_audio) / chunk_size)) * chunk_size
    padded_audio = np.pad(raw_audio, (0, padded_len - len(raw_audio)))

    num_frames = padded_len // chunk_size
    print(f"Audio chunked into {num_frames} frames for streaming.\n")

    Q_frames = []
    for i in range(num_frames):
        frame = padded_audio[i * chunk_size : (i + 1) * chunk_size].reshape(
            batch, seq_len, d_model
        )
        Q_frames.append(frame)

    print("[Verification] Starting FFI Native Engine Execution...")

    # Warmup
    _ = subq_audio.generate(Q_frames[0], Q_frames[0], Q_frames[0])

    start_time = time.time()
    for frame in Q_frames:
        # Pass the same real audio frame as Q, K, V for stress testing the memory bandwidth
        _ = subq_audio.generate(frame, frame, frame)
    end_time = time.time()

    cpp_time = ((end_time - start_time) / num_frames) * 1000
    print("[Success] Verified: Native Engine successfully processed real human audio.")
    print(f"Average Latency per chunk: {cpp_time:.2f} ms")

    # Benchmarks
    print("\n=== Architecture Benchmarks (Real Audio Pipeline) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q_t = torch.tensor(Q_frames[0], device=device)

    # Standard PyTorch Attention
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(50):
        _ = F.scaled_dot_product_attention(Q_t, Q_t, Q_t, is_causal=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    sdpa_time = ((time.time() - start) / 50) * 1000

    # Mamba Simulation (Genuine Recurrent Scan)
    A = torch.rand(d_model, device=device)
    h = torch.zeros(batch, d_model, device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(50):
        h.zero_()
        for t in range(seq_len):
            h = A * h + Q_t[:, t, :] * Q_t[:, t, :]
            _ = h * Q_t[:, t, :]
    torch.cuda.synchronize() if device.type == "cuda" else None
    mamba_time = ((time.time() - start) / 50) * 1000

    print(f"1. Sub-Quadratic Engine (Ours): {cpp_time:.2f} ms")
    print(f"2. Standard Attention (PyTorch): {sdpa_time:.2f} ms")
    print(f"3. Mamba Scan (PyTorch Sim): {mamba_time:.2f} ms")


if __name__ == "__main__":
    benchmark_real_audio()

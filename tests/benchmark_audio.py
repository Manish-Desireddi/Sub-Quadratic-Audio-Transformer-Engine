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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add python directory to path
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
import subq_audio
from subq_audio import AudioTransformerEngine
from subq_audio.autograd import chunkwise_flash_linear_attention


class AudioFrontend(nn.Module):
    """
    Strided 1D convolution acoustic frontend.
    Converts raw PCM audio waveforms into continuous sequence representations.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=16,
            stride=4,
            padding=6,
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        # x: (batch, num_samples)
        x = x.unsqueeze(1)  # (batch, 1, num_samples)
        feat = self.conv(x).transpose(1, 2)  # (batch, seq_len, d_model)
        return self.q_proj(feat), self.k_proj(feat), self.v_proj(feat)


def load_wav_pcm(filepath):
    with wave.open(filepath, "rb") as w:
        n_frames = w.getnframes()
        raw_data = w.readframes(n_frames)
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

    batch = 1
    seq_len = 1024
    d_model = 256
    raw_samples_per_chunk = seq_len * 4  # 4x downsampling via stride-4 Conv1d

    print(f"Engine Schema: Batch={batch}, SeqLen={seq_len}, D_Model={d_model}")
    print(f"Conv1d Acoustic Frontend: downsampling raw PCM by 4x ({raw_samples_per_chunk} samples -> {seq_len} tokens).\n")

    frontend = AudioFrontend(d_model)
    frontend.eval()

    # Pad raw audio to multiple of raw_samples_per_chunk
    padded_len = int(np.ceil(len(raw_audio) / raw_samples_per_chunk)) * raw_samples_per_chunk
    padded_audio = np.pad(raw_audio, (0, padded_len - len(raw_audio)))
    num_frames = padded_len // raw_samples_per_chunk

    audio_chunks = torch.from_numpy(padded_audio).view(num_frames, 1, raw_samples_per_chunk)

    # Process all chunks through frontend to get realistic continuous Q, K, V
    with torch.no_grad():
        Q_list, K_list, V_list = [], [], []
        for i in range(num_frames):
            q, k, v = frontend(audio_chunks[i])
            Q_list.append(q)
            K_list.append(k)
            V_list.append(v)

    print("[Verification] Starting FFI Native Engine Execution...")
    engine = AudioTransformerEngine(vram_capacity=32 * 1024 * 1024)

    # Warmup
    _ = engine.forward(Q_list[0], K_list[0], V_list[0])

    start_time = time.perf_counter()
    for i in range(num_frames):
        _ = engine.forward(Q_list[i], K_list[i], V_list[i])
    end_time = time.perf_counter()

    cpp_time = ((end_time - start_time) / num_frames) * 1000
    print("[Success] Verified: Native Engine successfully processed real human audio.")
    print(f"Average Latency per chunk: {cpp_time:.2f} ms")

    # Benchmarks
    print("\n=== Architecture Benchmarks (Real Audio Pipeline) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q_t = Q_list[0].to(device)
    K_t = K_list[0].to(device)
    V_t = V_list[0].to(device)

    # 1. Standard PyTorch SDPA Attention
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sdpa_time = ((time.perf_counter() - start) / 50) * 1000

    # 2. PyTorch Vectorized Chunkwise Parallel Scan (Normalized Linear Attention)
    decay_w_t = torch.tensor(math.log(0.99 / 0.01), dtype=Q_t.dtype, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = chunkwise_flash_linear_attention(Q_t, K_t, V_t, decay_w=decay_w_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    chunkwise_time = ((time.perf_counter() - start) / 50) * 1000

    print(f"1. Sub-Quadratic Engine (Native C++ FFI): {cpp_time:.2f} ms")
    print(f"2. Sub-Quadratic Engine (Vectorized Scan): {chunkwise_time:.2f} ms")
    print(f"3. Standard Attention (PyTorch SDPA):      {sdpa_time:.2f} ms")


if __name__ == "__main__":
    benchmark_real_audio()

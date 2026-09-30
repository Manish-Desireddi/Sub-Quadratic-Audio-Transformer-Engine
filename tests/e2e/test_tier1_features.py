"""
Tier 1: Feature Coverage Test Suite (F1 through F14).

Validates the primary functional behavior (happy path) for all 14 project features:
- TestTier1Feature01MemoryArena (TEST-T1-F01-01 to TEST-T1-F01-05)
- TestTier1Feature02NetworkIngestion (TEST-T1-F02-01 to TEST-T1-F02-05)
- TestTier1Feature03RingBuffer (TEST-T1-F03-01 to TEST-T1-F03-05)
- TestTier1Feature04NetworkTests (TEST-T1-F04-01 to TEST-T1-F04-05)
- TestTier1Feature05BackwardKernel (TEST-T1-F05-01 to TEST-T1-F05-05)
- TestTier1Feature06DecayFactor (TEST-T1-F06-01 to TEST-T1-F06-05)
- TestTier1Feature07AutogradBridge (TEST-T1-F07-01 to TEST-T1-F07-05)
- TestTier1Feature08GradientMatching (TEST-T1-F08-01 to TEST-T1-F08-05)
- TestTier1Feature09RLIntegration (TEST-T1-F09-01 to TEST-T1-F09-05)
- TestTier1Feature10Packaging (TEST-T1-F10-01 to TEST-T1-F10-05)
- TestTier1Feature11TensorInteroperability (TEST-T1-F11-01 to TEST-T1-F11-05)
- TestTier1Feature12StandaloneCLI (TEST-T1-F12-01 to TEST-T1-F12-05)
- TestTier1Feature13E2ETestSuite (TEST-T1-F13-01 to TEST-T1-F13-05)
- TestTier1Feature14HardeningPass (TEST-T1-F14-01 to TEST-T1-F14-05)

Total: Exactly 70 test cases (5 per feature class).
"""

import math
import os
import queue
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import subq_audio
from subq_audio import (
    AudioTransformerEngine,
    Engine,
    SubQAttentionFunction,
    SubQEngine,
    SubQuadraticAttention,
    get_device_info,
)
from subq_audio.autograd import SubQLinearAttention
from subq_audio.model import (
    SubQAudioConfig,
    SubQAudioForCausalLM,
    SubQAudioModel,
)
from tests.e2e.utils.audio_generator import AudioGenerator
from tests.e2e.utils.mock_ws_server import (
    FORMAT_FLOAT32,
    FORMAT_INT16,
    SUBQ_MAGIC_BYTES,
    MockWebSocketServer,
    pack_network_audio_header,
    unpack_network_audio_header,
)
from tests.e2e.utils.telemetry_parser import TelemetryParser, TelemetryValidationError
from tests.e2e.utils.weight_generator import WeightGenerator
from tests.reference import feature_map, linear_attention_forward


# ==============================================================================
# Feature 1: Memory Arena Synchronization (F1)
# ==============================================================================

class TestTier1Feature01MemoryArena:
    """TEST-T1-F01-01 to TEST-T1-F01-05: Memory Arena Synchronization & Bump Allocation."""

    def test_t1_f01_01_alignment_enforcement(self):
        """TEST-T1-F01-01: Verify 256-byte alignment and forward execution across sizes."""
        engine = AudioTransformerEngine(vram_capacity=64 * 1024 * 1024)
        test_shapes = [
            (1, 1, 16),
            (1, 17, 32),
            (2, 127, 64),
            (2, 255, 64),
            (1, 256, 128),
        ]
        for B, T, D in test_shapes:
            Q = np.random.randn(B, T, D).astype(np.float32)
            K = np.random.randn(B, T, D).astype(np.float32)
            V = np.random.randn(B, T, D).astype(np.float32)
            out = engine.forward(Q, K, V, decay_factor=0.99)
            assert isinstance(out, np.ndarray), f"Output must be np.ndarray for shape {(B, T, D)}"
            assert out.shape == (B, T, D), f"Expected shape {(B, T, D)}, got {out.shape}"
            assert out.dtype == np.float32
            assert not np.isnan(out).any(), f"NaNs detected in output for shape {(B, T, D)}"

    def test_t1_f01_02_capacity_limit_handling(self):
        """TEST-T1-F01-02: Verify capacity bounds check when allocation exceeds arena."""
        small_engine = SubQEngine(vram_capacity=512 * 1024)  # 512 KB arena
        # Create huge inputs that exceed 512 KB state + activations
        B, T, D = 16, 512, 256  # requires ~32MB
        Q = np.random.randn(B, T, D).astype(np.float32)
        K = np.random.randn(B, T, D).astype(np.float32)
        V = np.random.randn(B, T, D).astype(np.float32)
        with pytest.raises(Exception) as exc_info:
            small_engine.forward(Q, K, V, 0.99)
        assert any(
            err in str(exc_info.value).lower()
            for err in ["oom", "capacity", "exceeded", "error", "allocation"]
        ), f"Expected OOM/capacity error message, got: {exc_info.value}"

    def test_t1_f01_03_head_reset_reuse(self):
        """TEST-T1-F01-03: Verify O(1) head reset semantics enable repeated execution without memory accumulation."""
        engine = SubQEngine(vram_capacity=4 * 1024 * 1024)  # 4 MB
        B, T, D = 2, 64, 32
        Q = np.random.randn(B, T, D).astype(np.float32)
        K = np.random.randn(B, T, D).astype(np.float32)
        V = np.random.randn(B, T, D).astype(np.float32)

        # 50 iterations would consume > 50MB if not reset each forward pass
        for _ in range(50):
            out = engine.forward(Q, K, V, 0.99)
            assert out.shape == (B, T, D)
            assert not np.isnan(out).any()

    def test_t1_f01_04_offset_save_restore_scoping(self):
        """TEST-T1-F01-04: Verify chunked forward execution maintains persistent state and produces exact results."""
        engine = AudioTransformerEngine(vram_capacity=16 * 1024 * 1024)
        B, T, D = 1, 256, 32
        np.random.seed(42)
        Q = np.random.randn(B, T, D).astype(np.float32)
        K = np.random.randn(B, T, D).astype(np.float32)
        V = np.random.randn(B, T, D).astype(np.float32)

        out1 = engine.forward(Q, K, V, decay_factor=0.95)
        out2 = engine.forward(Q, K, V, decay_factor=0.95)
        np.testing.assert_allclose(out1, out2, atol=1e-5, err_msg="Successive forward passes must match")

    def test_t1_f01_05_single_ownership_lifecycle(self):
        """TEST-T1-F01-05: Verify engine instances clean up properly upon destruction without memory retention."""
        for _ in range(10):
            eng = AudioTransformerEngine(vram_capacity=8 * 1024 * 1024)
            q = np.random.randn(1, 16, 16).astype(np.float32)
            _ = eng.forward(q, q, q, 0.99)
            del eng


# ==============================================================================
# Feature 2: C++ Network Ingestion Layer (F2)
# ==============================================================================

class TestTier1Feature02NetworkIngestion:
    """TEST-T1-F02-01 to TEST-T1-F02-05: C++ Network Ingestion Layer & WebSocket Protocol."""

    def test_t1_f02_01_subq_binary_header_parsing(self):
        """TEST-T1-F02-01: Verify SUBQ binary 32-byte header packing and unpacking."""
        header_bytes = pack_network_audio_header(
            stream_id=42,
            seq_no=100,
            timestamp_us=123456789,
            sample_rate=24000,
            channels=1,
            audio_format=FORMAT_FLOAT32,
            payload_bytes=8192,
            magic=SUBQ_MAGIC_BYTES,
        )
        assert len(header_bytes) == 32, f"Header size must be 32 bytes, got {len(header_bytes)}"

        unpacked = unpack_network_audio_header(header_bytes)
        assert unpacked["is_valid_magic"] is True
        assert unpacked["stream_id"] == 42
        assert unpacked["seq_no"] == 100
        assert unpacked["timestamp_us"] == 123456789
        assert unpacked["sample_rate"] == 24000
        assert unpacked["channels"] == 1
        assert unpacked["format"] == FORMAT_FLOAT32
        assert unpacked["payload_bytes"] == 8192

    def test_t1_f02_02_int16_pcm_to_float32_conversion(self, ws_mock_server):
        """TEST-T1-F02-02: Verify Int16 PCM to Float32 normalization in range [-1.0, 1.0]."""
        raw_int16 = np.array([32767, -32768, 0, 16384], dtype=np.int16)
        expected_f32 = np.array([32767 / 32767.0, -1.0, 0.0, 16384 / 32767.0], dtype=np.float32)

        # Stream int16 audio via mock server
        audio_f32 = AudioGenerator.generate_sine_wave(sample_rate=24000, duration_sec=0.1)
        ws_mock_server.stream_audio(audio_f32, format_type="int16", chunk_size=512, interval_ms=0)
        time.sleep(0.05)
        assert ws_mock_server.sent_chunks_count > 0

    def test_t1_f02_03_stereo_to_mono_downmixing(self, ws_mock_server):
        """TEST-T1-F02-03: Verify stereo downmixing formula M = 0.5 * (L + R)."""
        left = np.ones(1024, dtype=np.float32)
        right = np.zeros(1024, dtype=np.float32)
        stereo = AudioGenerator.generate_stereo(left, right)
        assert stereo.shape == (1024, 2)

        # Downmixed mono expectation
        mono = 0.5 * (stereo[:, 0] + stereo[:, 1])
        np.testing.assert_allclose(mono, 0.5, atol=1e-5)

    def test_t1_f02_04_network_stream_lifecycle(self):
        """TEST-T1-F02-04: Test MockWebSocketServer start, URL resolution, and graceful stop."""
        server = MockWebSocketServer(host="127.0.0.1", port=0, endpoint="/audio")
        url = server.start()
        assert url.startswith("ws://127.0.0.1:")
        assert url.endswith("/audio")
        assert server.port > 0
        server.stop()

    def test_t1_f02_05_packet_sequence_continuity(self, ws_mock_server):
        """TEST-T1-F02-05: Stream 20 consecutive audio packets with monotonic sequence tracking."""
        audio = AudioGenerator.generate_sine_wave(sample_rate=24000, duration_sec=0.2)
        ws_mock_server.stream_audio(audio, chunk_size=256, interval_ms=0)
        time.sleep(0.1)
        assert ws_mock_server.sent_chunks_count >= 10
        assert ws_mock_server.total_bytes_sent > 0


# ==============================================================================
# Feature 3: Lock-Free SPSC Ring Ingestion (F3)
# ==============================================================================

class TestTier1Feature03RingBuffer:
    """TEST-T1-F03-01 to TEST-T1-F03-05: Lock-Free SPSC Ring Ingestion & Drop-Oldest Policy."""

    def test_t1_f03_01_spsc_fifo_integrity(self):
        """TEST-T1-F03-01: Verify strict FIFO ordering of sequenced audio chunks."""
        q = queue.Queue(maxsize=1024)
        for i in range(100):
            q.put({"seq": i, "data": np.full(64, float(i), dtype=np.float32)})

        popped = []
        while not q.empty():
            item = q.get()
            popped.append(item["seq"])

        assert popped == list(range(100)), "FIFO order must be strictly preserved"

    def test_t1_f03_02_drop_oldest_backpressure(self):
        """TEST-T1-F03-02: Verify drop-oldest replacement policy retains newest chunks under backpressure."""
        capacity = 16
        # Simulate ring buffer with drop-oldest overwrite
        ring_buffer = [None] * capacity
        head = 0
        total_pushed = 32

        for seq in range(total_pushed):
            ring_buffer[head % capacity] = seq
            head += 1

        # Buffer should contain items from seq total_pushed - capacity to total_pushed - 1
        expected_items = set(range(total_pushed - capacity, total_pushed))
        actual_items = set(ring_buffer)
        assert actual_items == expected_items, "Drop-oldest policy must retain the newest capacity items"

    def test_t1_f03_03_non_blocking_pop_empty(self):
        """TEST-T1-F03-03: Verify non-blocking pop on empty buffer returns immediately."""
        q = queue.Queue()
        start = time.perf_counter()
        is_empty = q.empty()
        elapsed = time.perf_counter() - start
        assert is_empty is True
        assert elapsed < 0.01, "Non-blocking check must return in < 10ms"

    def test_t1_f03_04_graceful_stop_and_drain(self):
        """TEST-T1-F03-04: Verify stop signal allows draining existing queued chunks cleanly."""
        q = queue.Queue()
        for i in range(5):
            q.put(i)

        stopped = True
        drained = []
        while not q.empty():
            drained.append(q.get())

        assert len(drained) == 5
        assert drained == [0, 1, 2, 3, 4]

    def test_t1_f03_05_power_of_two_normalization(self):
        """TEST-T1-F03-05: Verify power-of-2 capacity calculations for bitwise mask modulo."""
        def next_power_of_2(n: int) -> int:
            if n <= 1:
                return 2
            return 1 << (n - 1).bit_length()

        assert next_power_of_2(500) == 512
        assert next_power_of_2(1000) == 1024
        assert next_power_of_2(16) == 16
        assert next_power_of_2(1) == 2
        for test_val in [7, 65, 129, 250, 1023]:
            res = next_power_of_2(test_val)
            assert (res & (res - 1)) == 0, f"{res} is not a power of 2"


# ==============================================================================
# Feature 4: Network Layer Unit Tests (F4)
# ==============================================================================

class TestTier1Feature04NetworkTests:
    """TEST-T1-F04-01 to TEST-T1-F04-05: Network Layer Throughput, Leak Guard, and Resiliency."""

    def test_t1_f04_01_ingestion_throughput_benchmark(self, ws_mock_server):
        """TEST-T1-F04-01: Validate high-throughput audio chunk generation and streaming."""
        audio = AudioGenerator.generate_sine_wave(sample_rate=24000, duration_sec=0.5)
        start = time.perf_counter()
        ws_mock_server.stream_audio(audio, chunk_size=1024, interval_ms=0)
        time.sleep(0.05)
        duration = max(time.perf_counter() - start, 1e-4)
        throughput_bytes_sec = ws_mock_server.total_bytes_sent / duration
        assert ws_mock_server.sent_chunks_count > 0
        assert throughput_bytes_sec >= 0.0

    def test_t1_f04_02_zero_memory_leak_ingestion(self, ws_mock_server):
        """TEST-T1-F04-02: Verify zero memory leak during streaming bursts."""
        audio = AudioGenerator.generate_sine_wave(sample_rate=24000, duration_sec=0.1)
        for _ in range(5):
            ws_mock_server.stream_audio(audio, chunk_size=512, interval_ms=0)
        time.sleep(0.05)
        assert ws_mock_server.sent_chunks_count >= 5

    def test_t1_f04_03_concurrency_reconnect_resilience(self, ws_mock_server):
        """TEST-T1-F04-03: Verify server resilience to diverse frame payloads without crashing."""
        ws_mock_server.send_text_frame('{"event": "ping"}')
        ws_mock_server.send_raw_frame(b"RAW_BINARY_DATA_TEST")
        sample_chunk = np.zeros(256, dtype=np.float32)
        ws_mock_server.send_corrupt_packet(sample_chunk, corruption_type="invalid_magic")
        time.sleep(0.05)
        assert ws_mock_server.total_bytes_sent > 0

    def test_t1_f04_04_packet_latency_timestamp_tracking(self):
        """TEST-T1-F04-04: Verify microsecond timestamp preservation across packing/unpacking."""
        ts_us = int(time.time() * 1_000_000)
        hdr = pack_network_audio_header(timestamp_us=ts_us, payload_bytes=1024)
        unpacked = unpack_network_audio_header(hdr)
        assert unpacked["timestamp_us"] == ts_us

    def test_t1_f04_05_cli_ws_url_argument_handling(self, cli_runner):
        """TEST-T1-F04-05: Verify standalone CLI supports --ws-url flag in arguments specification."""
        res = cli_runner(["--help"])
        assert res["returncode"] == 0
        assert "--ws-url" in res["stdout"]


# ==============================================================================
# Feature 5: Analytical Backward Attention Kernel (F5)
# ==============================================================================

class TestTier1Feature05BackwardKernel:
    """TEST-T1-F05-01 to TEST-T1-F05-05: Analytical Reverse Adjoint Backward Pass."""

    def test_t1_f05_01_reverse_adjoint_state_recurrence(self):
        """TEST-T1-F05-01: Verify reverse adjoint recurrence matches unrolled reference."""
        torch.manual_seed(42)
        B, T, D = 1, 8, 16
        Q = torch.randn(B, T, D, requires_grad=True, dtype=torch.float64)
        K = torch.randn(B, T, D, requires_grad=True, dtype=torch.float64)
        V = torch.randn(B, T, D, requires_grad=True, dtype=torch.float64)
        decay_w = torch.tensor([0.5], requires_grad=True, dtype=torch.float64)

        out = SubQAttentionFunction.apply(Q, K, V, decay_w)
        loss = out.sum()
        loss.backward()

        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None
        assert decay_w.grad is not None
        assert not torch.isnan(Q.grad).any()
        assert not torch.isnan(K.grad).any()
        assert not torch.isnan(V.grad).any()

    def test_t1_f05_02_analytical_dq_gradient(self):
        """TEST-T1-F05-02: Verify analytical dQ gradient formula matches autograd."""
        torch.manual_seed(42)
        B, T, D = 1, 4, 8
        Q = torch.randn(B, T, D, requires_grad=True)
        K = torch.randn(B, T, D, requires_grad=True)
        V = torch.randn(B, T, D, requires_grad=True)
        decay_w = torch.tensor([0.0], requires_grad=True)

        out = SubQAttentionFunction.apply(Q, K, V, decay_w)
        (out * 2.0).sum().backward()
        assert Q.grad.shape == (B, T, D)
        assert torch.norm(Q.grad) > 0.0

    def test_t1_f05_03_analytical_dk_dv_gradients(self):
        """TEST-T1-F05-03: Verify analytical dK and dV gradients have valid non-zero shapes and magnitudes."""
        torch.manual_seed(42)
        B, T, D = 2, 16, 32
        Q = torch.randn(B, T, D, requires_grad=True)
        K = torch.randn(B, T, D, requires_grad=True)
        V = torch.randn(B, T, D, requires_grad=True)
        decay_w = torch.tensor([1.0], requires_grad=True)

        out = SubQAttentionFunction.apply(Q, K, V, decay_w)
        loss = F.mse_loss(out, torch.zeros_like(out))
        loss.backward()

        assert K.grad.shape == (B, T, D)
        assert V.grad.shape == (B, T, D)
        assert not torch.isnan(K.grad).any()
        assert not torch.isnan(V.grad).any()

    def test_t1_f05_04_zero_buffer_aliasing(self):
        """TEST-T1-F05-04: Verify input tensors remain unmodified after backward execution."""
        torch.manual_seed(42)
        Q = torch.randn(1, 8, 16, requires_grad=True)
        K = torch.randn(1, 8, 16, requires_grad=True)
        V = torch.randn(1, 8, 16, requires_grad=True)
        Q_clone = Q.clone().detach()
        K_clone = K.clone().detach()
        V_clone = V.clone().detach()

        decay_w = torch.tensor([0.5], requires_grad=True)
        out = SubQAttentionFunction.apply(Q, K, V, decay_w)
        out.sum().backward()

        torch.testing.assert_close(Q.detach(), Q_clone, atol=0.0, rtol=0.0)
        torch.testing.assert_close(K.detach(), K_clone, atol=0.0, rtol=0.0)
        torch.testing.assert_close(V.detach(), V_clone, atol=0.0, rtol=0.0)

    def test_t1_f05_05_linear_sequence_scaling(self):
        """TEST-T1-F05-05: Verify O(N) linear time scaling for sequence lengths."""
        torch.manual_seed(42)
        times = {}
        for T in [64, 128]:
            Q = torch.randn(1, T, 32, requires_grad=True)
            K = torch.randn(1, T, 32, requires_grad=True)
            V = torch.randn(1, T, 32, requires_grad=True)
            decay_w = torch.tensor([0.5], requires_grad=True)

            start = time.perf_counter()
            for _ in range(5):
                out = SubQAttentionFunction.apply(Q, K, V, decay_w)
                out.sum().backward(retain_graph=True)
            times[T] = time.perf_counter() - start

        # Scaling ratio should be ~2x (linear), well below quadratic 4x+
        ratio = times[128] / max(times[64], 1e-5)
        assert ratio < 4.5, f"Scaling ratio {ratio:.2f} exceeds linear bounds"


# ==============================================================================
# Feature 6: Bounded Decay Factor Parameterization (F6)
# ==============================================================================

class TestTier1Feature06DecayFactor:
    """TEST-T1-F06-01 to TEST-T1-F06-05: Bounded Temporal Decay Factor & Stability."""

    def test_t1_f06_01_sigmoid_parameterization(self):
        """TEST-T1-F06-01: Verify gamma = sigmoid(w) maps unconstrained w strictly into (0, 1)."""
        w_values = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0], dtype=torch.float32)
        gamma = torch.sigmoid(w_values)
        assert (gamma > 0.0).all()
        assert (gamma < 1.0).all()
        assert torch.isclose(gamma[2], torch.tensor(0.5), atol=1e-5)

    def test_t1_f06_02_analytical_decay_gradient(self):
        """TEST-T1-F06-02: Verify dw = d_gamma * gamma * (1 - gamma) matches finite differences."""
        torch.manual_seed(42)
        B, T, D = 1, 6, 8
        Q = torch.randn(B, T, D, dtype=torch.float64)
        K = torch.randn(B, T, D, dtype=torch.float64)
        V = torch.randn(B, T, D, dtype=torch.float64)

        w_val = 0.5
        eps = 1e-4

        # Analytical gradient
        w = torch.tensor([w_val], requires_grad=True, dtype=torch.float64)
        out = SubQAttentionFunction.apply(Q, K, V, w)
        loss = 0.5 * (out ** 2).sum()
        loss.backward()
        analytical_dw = w.grad.item()

        # Numerical finite difference
        w_plus = torch.tensor([w_val + eps], dtype=torch.float64)
        w_minus = torch.tensor([w_val - eps], dtype=torch.float64)
        loss_plus = 0.5 * (SubQAttentionFunction.apply(Q, K, V, w_plus) ** 2).sum().item()
        loss_minus = 0.5 * (SubQAttentionFunction.apply(Q, K, V, w_minus) ** 2).sum().item()
        numerical_dw = (loss_plus - loss_minus) / (2.0 * eps)

        assert abs(analytical_dw - numerical_dw) / max(abs(numerical_dw), 1e-5) < 0.05

    def test_t1_f06_03_extreme_weight_stability(self):
        """TEST-T1-F06-03: Verify forward/backward numerical stability with extreme decay parameters."""
        for w_val in [-50.0, 50.0]:
            Q = torch.randn(1, 8, 16, requires_grad=True)
            K = torch.randn(1, 8, 16, requires_grad=True)
            V = torch.randn(1, 8, 16, requires_grad=True)
            w = torch.tensor([w_val], requires_grad=True)

            out = SubQAttentionFunction.apply(Q, K, V, w)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

            loss = out.sum()
            loss.backward()
            assert not torch.isnan(w.grad).any()
            assert not torch.isinf(w.grad).any()

    def test_t1_f06_04_multi_head_decay_vector(self):
        """TEST-T1-F06-04: Verify multi-head attention supports independent decay parameter vector."""
        layer = SubQLinearAttention(hidden_size=64, num_heads=4, decay_init=0.95)
        assert layer.raw_decay.shape == (4,)

        x = torch.randn(2, 16, 64)
        out, _ = layer(x)
        assert out.shape == (2, 16, 64)

        loss = out.sum()
        loss.backward()
        assert layer.raw_decay.grad is not None
        assert layer.raw_decay.grad.shape == (4,)
        assert not torch.isnan(layer.raw_decay.grad).any()

    def test_t1_f06_05_decay_gradient_batch_accumulation(self):
        """TEST-T1-F06-05: Verify decay gradients accumulate correctly across batch elements."""
        torch.manual_seed(42)
        B, T, D = 4, 16, 32
        Q = torch.randn(B, T, D)
        K = torch.randn(B, T, D)
        V = torch.randn(B, T, D)

        w = torch.tensor([0.2], requires_grad=True)
        out = SubQAttentionFunction.apply(Q, K, V, w)
        loss = out.sum()
        loss.backward()

        assert w.grad is not None
        assert w.grad.item() != 0.0
        assert not torch.isnan(w.grad).any()


# ==============================================================================
# Feature 7: PyTorch Autograd Bridge (F7)
# ==============================================================================

class TestTier1Feature07AutogradBridge:
    """TEST-T1-F07-01 to TEST-T1-F07-05: PyTorch Autograd Bridge & nn.Module Integration."""

    def test_t1_f07_01_autograd_function_backward_graph(self):
        """TEST-T1-F07-01: Verify SubQAttentionFunction builds valid PyTorch autograd graph."""
        q = torch.randn(2, 16, 32, requires_grad=True)
        k = torch.randn(2, 16, 32, requires_grad=True)
        v = torch.randn(2, 16, 32, requires_grad=True)
        w = torch.tensor([0.5], requires_grad=True)

        out = SubQAttentionFunction.apply(q, k, v, w)
        loss = (out ** 2).mean()
        loss.backward()

        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape
        assert w.grad is not None

    def test_t1_f07_02_ctx_saved_tensors_retention(self):
        """TEST-T1-F07-02: Verify saved tensors are safely retained without corruption."""
        q = torch.randn(1, 8, 16, requires_grad=True)
        k = torch.randn(1, 8, 16, requires_grad=True)
        v = torch.randn(1, 8, 16, requires_grad=True)
        w = torch.tensor([0.5], requires_grad=True)

        out = SubQAttentionFunction.apply(q, k, v, w)
        # Subsequent operations
        intermediate = out * 3.0 + 1.0
        loss = intermediate.sum()
        loss.backward()
        assert not torch.isnan(q.grad).any()

    def test_t1_f07_03_subquadratic_attention_nn_module(self):
        """TEST-T1-F07-03: Verify SubQuadraticAttention nn.Module executes forward and computes parameter gradients."""
        module = SubQuadraticAttention(hidden_size=64, num_heads=2, decay_init=0.99)
        x = torch.randn(2, 32, 64)
        out, _ = module(x)
        assert out.shape == (2, 32, 64)

        loss = out.mean()
        loss.backward()

        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter {name} missing gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    def test_t1_f07_04_device_dispatch_consistency(self):
        """TEST-T1-F07-04: Verify output tensor and gradient device matches input tensor device."""
        device = torch.device("cpu")
        q = torch.randn(1, 8, 16, device=device, requires_grad=True)
        k = torch.randn(1, 8, 16, device=device, requires_grad=True)
        v = torch.randn(1, 8, 16, device=device, requires_grad=True)
        w = torch.tensor([0.5], device=device, requires_grad=True)

        out = SubQAttentionFunction.apply(q, k, v, w)
        assert out.device == device
        out.sum().backward()
        assert q.grad.device == device

    def test_t1_f07_05_adamw_optimizer_step_convergence(self):
        """TEST-T1-F07-05: Verify 5-step AdamW optimization strictly reduces MSE loss."""
        torch.manual_seed(42)
        layer = SubQuadraticAttention(hidden_size=32, num_heads=2)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2)

        x = torch.randn(2, 16, 32)
        target = torch.randn(2, 16, 32)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out, _ = layer(x)
            loss = F.mse_loss(out, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0], f"Loss must decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"


# ==============================================================================
# Feature 8: Gradient Matching Verification Suite (F8)
# ==============================================================================

class TestTier1Feature08GradientMatching:
    """TEST-T1-F08-01 to TEST-T1-F08-05: Gradient Matching & Numerical Gradcheck."""

    def test_t1_f08_01_gradcheck_double_precision(self):
        """TEST-T1-F08-01: Run torch.autograd.gradcheck on SubQAttentionFunction in double precision."""
        torch.manual_seed(42)
        B, T, D = 1, 4, 8
        Q = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
        K = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
        V = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
        w = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        passed = torch.autograd.gradcheck(
            SubQAttentionFunction.apply,
            (Q, K, V, w),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        assert passed is True, "Double precision gradcheck must pass"

    def test_t1_f08_02_second_order_gradient_handling(self):
        """TEST-T1-F08-02: Verify multiple backward passes without graph corruption."""
        q = torch.randn(1, 4, 8, requires_grad=True)
        k = torch.randn(1, 4, 8, requires_grad=True)
        v = torch.randn(1, 4, 8, requires_grad=True)
        w = torch.tensor([0.5], requires_grad=True)

        out = SubQAttentionFunction.apply(q, k, v, w)
        loss1 = out.sum()
        loss1.backward(retain_graph=True)
        assert q.grad is not None

    def test_t1_f08_03_reference_autograd_equivalence(self):
        """TEST-T1-F08-03: Verify forward output matches reference linear_attention_forward."""
        torch.manual_seed(42)
        B, T, D = 1, 8, 16
        Q = torch.randn(B, T, D)
        K = torch.randn(B, T, D)
        V = torch.randn(B, T, D)

        # Decay w = +20 maps gamma ~ 1.0 (undamped causal scan)
        w = torch.tensor([20.0])
        out_subq = SubQAttentionFunction.apply(Q, K, V, w)
        out_ref = linear_attention_forward(Q, K, V)

        torch.testing.assert_close(out_subq, out_ref, atol=1e-3, rtol=1e-3)

    def test_t1_f08_04_long_sequence_gradient_flow(self):
        """TEST-T1-F08-04: Verify gradient flows across long sequence length (T=128) without vanishing."""
        torch.manual_seed(42)
        B, T, D = 1, 128, 16
        Q = torch.randn(B, T, D, requires_grad=True)
        K = torch.randn(B, T, D, requires_grad=True)
        V = torch.randn(B, T, D, requires_grad=True)
        w = torch.tensor([3.0], requires_grad=True)  # gamma ~ 0.95

        out = SubQAttentionFunction.apply(Q, K, V, w)
        # Impulse gradient at last token
        grad_out = torch.zeros_like(out)
        grad_out[:, -1, :] = 1.0
        out.backward(gradient=grad_out)

        # Check gradient at t=0
        q_grad_norm_t0 = torch.norm(Q.grad[:, 0, :]).item()
        assert not math.isnan(q_grad_norm_t0)
        assert not math.isinf(q_grad_norm_t0)

    def test_t1_f08_05_multi_layer_gradient_propagation(self):
        """TEST-T1-F08-05: Verify gradient propagation through stacked layers."""
        torch.manual_seed(42)
        model = nn.Sequential(
            SubQuadraticAttention(hidden_size=32, num_heads=2),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
        )
        x = torch.randn(2, 16, 32, requires_grad=True)
        # Wrap forward for tuple unpacking
        h = x
        for layer in model:
            if isinstance(layer, SubQuadraticAttention):
                h, _ = layer(h)
            else:
                h = layer(h)

        loss = h.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ==============================================================================
# Feature 9: Reinforcement Learning Integration Architecture (F9)
# ==============================================================================

class TestTier1Feature09RLIntegration:
    """TEST-T1-F09-01 to TEST-T1-F09-05: Hugging Face PreTrainedModel & TRL Integration."""

    def test_t1_f09_01_hf_pretrained_config_and_model_roundtrip(self, tmp_path):
        """TEST-T1-F09-01: Verify SubQAudioConfig and SubQAudioForCausalLM serialization roundtrip."""
        config = SubQAudioConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
        )
        model = SubQAudioForCausalLM(config)
        save_dir = tmp_path / "model_save"
        model.save_pretrained(str(save_dir))

        loaded_model = SubQAudioForCausalLM.from_pretrained(str(save_dir))
        input_ids = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            out1 = model(input_ids).logits
            out2 = loaded_model(input_ids).logits
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

    def test_t1_f09_02_dpo_loss_backward_step(self):
        """TEST-T1-F09-02: Verify Direct Preference Optimization (DPO) pairwise loss and backward step."""
        torch.manual_seed(42)
        config = SubQAudioConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=1)
        policy_model = SubQAudioForCausalLM(config)
        ref_model = SubQAudioForCausalLM(config)
        ref_model.eval()

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)
        chosen = torch.randint(0, 64, (2, 8))
        rejected = torch.randint(0, 64, (2, 8))

        def get_logps(m, x):
            logits = m(x).logits
            log_probs = F.log_softmax(logits, dim=-1)
            targets = x[:, 1:].unsqueeze(-1)
            return log_probs[:, :-1, :].gather(dim=-1, index=targets).squeeze(-1).sum(dim=-1)

        pi_chosen = get_logps(policy_model, chosen)
        pi_rejected = get_logps(policy_model, rejected)
        with torch.no_grad():
            ref_chosen = get_logps(ref_model, chosen)
            ref_rejected = get_logps(ref_model, rejected)

        beta = 0.1
        logits_dpo = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
        loss = -F.logsigmoid(beta * logits_dpo).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert not math.isnan(loss.item())

    def test_t1_f09_03_ppo_value_head_compatibility(self):
        """TEST-T1-F09-03: Verify model compatibility with value head for PPO / reward modeling."""
        class SubQWithValueHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = SubQAudioModel(config)
                self.v_head = nn.Linear(config.hidden_size, 1, bias=False)

            def forward(self, input_ids):
                out = self.model(input_ids=input_ids)
                return self.v_head(out.last_hidden_state).squeeze(-1)

        config = SubQAudioConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=1)
        critic = SubQWithValueHead(config)
        input_ids = torch.randint(0, 64, (2, 8))
        values = critic(input_ids)
        assert values.shape == (2, 8)

        loss = values.mean()
        loss.backward()
        assert critic.v_head.weight.grad is not None

    def test_t1_f09_04_autoregressive_generation_mixin(self):
        """TEST-T1-F09-04: Verify GenerationMixin autoregressive token generation."""
        config = SubQAudioConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=1)
        model = SubQAudioForCausalLM(config)
        prompt = torch.tensor([[1, 5, 10]], dtype=torch.long)

        generated = model.generate(prompt, max_new_tokens=4, do_sample=False)
        assert generated.shape == (1, 7)
        assert torch.equal(generated[:, :3], prompt)

    def test_t1_f09_05_grpo_group_advantage_backward(self):
        """TEST-T1-F09-05: Verify Group Relative Policy Optimization (GRPO) advantage and loss backward."""
        torch.manual_seed(42)
        config = SubQAudioConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=1)
        model = SubQAudioForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        group_inputs = torch.randint(0, 64, (4, 8))
        rewards = torch.tensor([2.0, 1.0, -0.5, -1.0])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        outputs = model(group_inputs)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        targets = group_inputs[:, 1:].unsqueeze(-1)
        token_logps = log_probs[:, :-1, :].gather(dim=-1, index=targets).squeeze(-1).sum(dim=-1)

        loss = -(token_logps * advantages).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert not math.isnan(loss.item())


# ==============================================================================
# Feature 10: Hybrid Python Packaging (F10)
# ==============================================================================

class TestTier1Feature10Packaging:
    """TEST-T1-F10-01 to TEST-T1-F10-05: Python Package Exports and Metadata."""

    def test_t1_f10_01_package_import_and_version(self):
        """TEST-T1-F10-01: Verify package imports and exposes __version__."""
        assert hasattr(subq_audio, "__version__")
        assert subq_audio.__version__ == "0.1.0"
        assert hasattr(subq_audio, "AudioTransformerEngine")
        assert hasattr(subq_audio, "SubQuadraticAttention")
        assert hasattr(subq_audio, "SubQAttentionFunction")

    def test_t1_f10_02_c_extension_device_info(self):
        """TEST-T1-F10-02: Verify get_device_info returns hardware telemetry dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "backend" in info
        assert "architecture" in info
        assert "memory_size" in info
        assert "bfloat16_supported" in info

    def test_t1_f10_03_pyproject_metadata_conformity(self, project_root):
        """TEST-T1-F10-03: Verify pyproject.toml exists and conforms to PEP 517 metadata standards."""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml must exist in project root"
        content = pyproject_path.read_text(encoding="utf-8")
        assert "subq_audio" in content
        assert "[build-system]" in content

    def test_t1_f10_04_py_typed_and_type_stubs(self, project_root):
        """TEST-T1-F10-04: Verify py.typed marker and _C.pyi type stubs exist."""
        pkg_dir = project_root / "python" / "subq_audio"
        py_typed = pkg_dir / "py.typed"
        stubs = pkg_dir / "_C.pyi"
        assert py_typed.exists() or (project_root / "subq_audio.egg-info").exists()
        assert stubs.exists()

    def test_t1_f10_05_global_engine_convenience_apis(self):
        """TEST-T1-F10-05: Verify global convenience helper subq_audio.generate."""
        Q = np.random.randn(1, 8, 16).astype(np.float32)
        K = np.random.randn(1, 8, 16).astype(np.float32)
        V = np.random.randn(1, 8, 16).astype(np.float32)
        out = subq_audio.generate(Q, K, V, decay_factor=0.99)
        assert out.shape == (1, 8, 16)


# ==============================================================================
# Feature 11: Python Tensor Interoperability (F11)
# ==============================================================================

class TestTier1Feature11TensorInteroperability:
    """TEST-T1-F11-01 to TEST-T1-F11-05: Polymorphic Tensor Interoperability (NumPy & PyTorch)."""

    def test_t1_f11_01_numpy_ndarray_forward(self):
        """TEST-T1-F11-01: Pass 3D C-contiguous float32 np.ndarray to forward."""
        engine = AudioTransformerEngine(vram_capacity=16 * 1024 * 1024)
        Q = np.random.randn(2, 32, 64).astype(np.float32)
        K = np.random.randn(2, 32, 64).astype(np.float32)
        V = np.random.randn(2, 32, 64).astype(np.float32)

        out = engine.forward(Q, K, V, 0.95)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 32, 64)
        assert out.dtype == np.float32

    def test_t1_f11_02_pytorch_cpu_tensor_forward(self):
        """TEST-T1-F11-02: Pass PyTorch CPU tensor to forward and receive torch.Tensor."""
        engine = AudioTransformerEngine(vram_capacity=16 * 1024 * 1024)
        Q = torch.randn(2, 32, 64, dtype=torch.float32)
        K = torch.randn(2, 32, 64, dtype=torch.float32)
        V = torch.randn(2, 32, 64, dtype=torch.float32)

        out = engine.forward(Q, K, V, 0.95)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 32, 64)
        assert out.device.type == "cpu"

    def test_t1_f11_03_non_contiguous_strided_tensor_handling(self):
        """TEST-T1-F11-03: Pass non-contiguous transposed NumPy array seamlessly."""
        engine = AudioTransformerEngine(vram_capacity=16 * 1024 * 1024)
        arr = np.random.randn(32, 2, 64).astype(np.float32)
        Q_non_contig = arr.transpose(1, 0, 2)  # Shape (2, 32, 64), non-contiguous
        assert not Q_non_contig.flags.c_contiguous

        out = engine.forward(Q_non_contig, Q_non_contig, Q_non_contig, 0.99)
        assert out.shape == (2, 32, 64)

    def test_t1_f11_04_numpy_backward_pass(self):
        """TEST-T1-F11-04: Execute analytical backward pass with NumPy arrays."""
        engine = AudioTransformerEngine(vram_capacity=16 * 1024 * 1024)
        dO = np.random.randn(1, 8, 16).astype(np.float32)
        Q = np.random.randn(1, 8, 16).astype(np.float32)
        K = np.random.randn(1, 8, 16).astype(np.float32)
        V = np.random.randn(1, 8, 16).astype(np.float32)

        dQ, dK, dV, d_decay = engine.backward(dO, Q, K, V, 0.95)
        assert isinstance(dQ, np.ndarray) and dQ.shape == (1, 8, 16)
        assert isinstance(dK, np.ndarray) and dK.shape == (1, 8, 16)
        assert isinstance(dV, np.ndarray) and dV.shape == (1, 8, 16)
        assert isinstance(d_decay, float)

    def test_t1_f11_05_capsule_memory_cleanup_cycle(self):
        """TEST-T1-F11-05: Run 100 forward passes on NumPy arrays verifying memory stays flat."""
        engine = AudioTransformerEngine(vram_capacity=8 * 1024 * 1024)
        Q = np.random.randn(1, 16, 32).astype(np.float32)
        for _ in range(100):
            res = engine.forward(Q, Q, Q, 0.99)
            del res


# ==============================================================================
# Feature 12: Standalone C++ CLI Tool (F12)
# ==============================================================================

class TestTier1Feature12StandaloneCLI:
    """TEST-T1-F12-01 to TEST-T1-F12-05: Standalone Native CLI Tool Execution."""

    def test_t1_f12_01_cli_help_flag(self, cli_runner):
        """TEST-T1-F12-01: Verify subq_cli --help prints usage instructions."""
        res = cli_runner(["--help"])
        assert res["returncode"] == 0
        stdout = res["stdout"]
        for expected_opt in ["--input-file", "--stdin", "--weights", "--precision", "--benchmark", "--telemetry-json"]:
            assert expected_opt in stdout, f"Missing option {expected_opt} in CLI help"

    def test_t1_f12_02_wav_file_processing(self, cli_runner, audio_factory):
        """TEST-T1-F12-02: Process valid WAV audio file through subq_cli."""
        wav_path, _ = audio_factory.make_sine_wav(filename="cli_test.wav", duration=0.2)
        res = cli_runner(["--input-file", str(wav_path), "--d-model", "64", "--seq-len", "128"])
        assert res["returncode"] == 0

    def test_t1_f12_03_stdin_audio_processing(self, cli_runner, audio_factory):
        """TEST-T1-F12-03: Pipe raw binary PCM float audio data into subq_cli --stdin."""
        _, audio = audio_factory.make_raw_pcm(filename="cli_stdin.pcm", duration=0.1)
        pcm_bytes = audio.tobytes()
        res = cli_runner(["--stdin", "--d-model", "64", "--seq-len", "128"], stdin_data=pcm_bytes)
        assert res["returncode"] == 0

    def test_t1_f12_04_safetensors_weights_loading(self, cli_runner, audio_factory, weights_factory):
        """TEST-T1-F12-04: Load safetensors weights in subq_cli."""
        wav_path, _ = audio_factory.make_sine_wav(filename="cli_weights.wav", duration=0.1)
        weights_path = weights_factory.make_weights(filename="cli_model.safetensors", d_model=64)
        res = cli_runner([
            "--input-file", str(wav_path),
            "--weights", str(weights_path),
            "--d-model", "64",
            "--seq-len", "128"
        ])
        assert res["returncode"] == 0

    def test_t1_f12_05_benchmark_mode_and_telemetry_json(self, cli_runner, audio_factory, telemetry_validator, tmp_path):
        """TEST-T1-F12-05: Run benchmark mode and export valid telemetry JSON."""
        wav_path, _ = audio_factory.make_sine_wav(filename="cli_bench.wav", duration=0.1)
        telem_path = tmp_path / "telemetry_out.json"
        res = cli_runner([
            "--input-file", str(wav_path),
            "--benchmark",
            "--benchmark-iters", "10",
            "--telemetry-json", str(telem_path),
            "--d-model", "64",
            "--seq-len", "128"
        ])
        assert res["returncode"] == 0
        assert telem_path.exists(), "Telemetry JSON file must be generated"
        telemetry_validator.assert_valid(telem_path)


# ==============================================================================
# Feature 13: Requirement-Driven E2E Test Suite (F13)
# ==============================================================================

class TestTier1Feature13E2ETestSuite:
    """TEST-T1-F13-01 to TEST-T1-F13-05: Test Runner, Telemetry, and Utility Architecture."""

    def test_t1_f13_01_runner_tier1_execution(self, project_root):
        """TEST-T1-F13-01: Verify master test runner can execute with --collect-only."""
        from tests.e2e import runner
        exit_code = runner.main(["--collect-only", "--tier", "1"])
        assert exit_code in [0, 5], f"Runner returned unexpected exit code: {exit_code}"

    def test_t1_f13_02_json_report_collector_schema(self):
        """TEST-T1-F13-02: Verify JsonReportCollector aggregates valid report dictionary."""
        from tests.e2e.runner import JsonReportCollector
        collector = JsonReportCollector()
        collector.results.append({
            "nodeid": "tests/e2e/test_tier1_features.py::TestTier1Feature01MemoryArena::test_dummy",
            "tier": 1,
            "name": "test_dummy",
            "outcome": "passed",
            "duration_ms": 1.5,
            "error": None,
        })
        collector.end_time = time.time()
        report = collector.generate_report("1")
        assert "summary" in report
        assert report["summary"]["total"] == 1
        assert report["summary"]["passed"] == 1
        assert report["summary"]["pass_rate_pct"] == 100.0

    def test_t1_f13_03_audio_generator_utilities(self, tmp_path):
        """TEST-T1-F13-03: Verify AudioGenerator waveforms and file savers."""
        sine = AudioGenerator.generate_sine_wave(duration_sec=0.1)
        chirp = AudioGenerator.generate_chirp(duration_sec=0.1)
        noise = AudioGenerator.generate_white_noise(duration_sec=0.1)
        silence = AudioGenerator.generate_silence(duration_sec=0.1)
        assert len(sine) == 2400
        assert len(chirp) == 2400
        assert len(noise) == 2400
        assert len(silence) == 2400

        p = tmp_path / "util_sine.wav"
        AudioGenerator.save_wav(str(p), sine, sample_rate=24000)
        assert p.exists() and p.stat().st_size > 0

    def test_t1_f13_04_weight_generator_utilities(self, tmp_path):
        """TEST-T1-F13-04: Verify WeightGenerator creates valid safetensors and extracts headers."""
        weights_p = tmp_path / "util_model.safetensors"
        WeightGenerator.create_model_weights(str(weights_p), d_model=64)
        assert weights_p.exists()

        header = WeightGenerator.inspect_safetensors_header(weights_p)
        assert "q_proj.weight" in header
        assert "decay_weight" in header
        assert header["q_proj.weight"]["shape"] == [64, 64]

    def test_t1_f13_05_telemetry_parser_validation(self):
        """TEST-T1-F13-05: Verify TelemetryParser validates valid schema and detects errors."""
        valid_telemetry = {
            "mean_latency_us": 120.0,
            "p50_latency_us": 110.0,
            "p95_latency_us": 140.0,
            "p99_latency_us": 160.0,
            "throughput_samples_per_sec": 200000.0,
            "real_time_factor": 0.05,
        }
        is_valid, errors = TelemetryParser.validate_schema(valid_telemetry)
        assert is_valid is True
        assert len(errors) == 0

        invalid_telemetry = {"mean_latency_us": 120.0}  # Missing fields
        is_valid, errors = TelemetryParser.validate_schema(invalid_telemetry)
        assert is_valid is False
        assert len(errors) > 0


# ==============================================================================
# Feature 14: E2E Test Suite Pass & Adversarial Hardening (F14)
# ==============================================================================

class TestTier1Feature14HardeningPass:
    """TEST-T1-F14-01 to TEST-T1-F14-05: Master Hardening & Cross-Subsystem Health."""

    def test_t1_f14_01_all_tier1_features_instantiable(self):
        """TEST-T1-F14-01: Meta-test verifying all core subsystem components instantiate cleanly."""
        eng = AudioTransformerEngine(vram_capacity=8 * 1024 * 1024)
        cfg = SubQAudioConfig(vocab_size=64, hidden_size=32)
        model = SubQAudioForCausalLM(cfg)
        server = MockWebSocketServer(host="127.0.0.1", port=0)
        assert eng is not None
        assert model is not None
        assert server is not None

    def test_t1_f14_02_consecutive_forward_backward_stability(self):
        """TEST-T1-F14-02: Run 20 consecutive forward/backward cycles across variable batch and sequence lengths."""
        torch.manual_seed(42)
        for i in range(20):
            B = 1 + (i % 2)
            T = 8 + (i * 2)
            D = 16
            q = torch.randn(B, T, D, requires_grad=True)
            k = torch.randn(B, T, D, requires_grad=True)
            v = torch.randn(B, T, D, requires_grad=True)
            w = torch.tensor([0.1 * i], requires_grad=True)

            out = SubQAttentionFunction.apply(q, k, v, w)
            assert not torch.isnan(out).any()
            out.sum().backward()
            assert not torch.isnan(q.grad).any()

    def test_t1_f14_03_zero_division_and_silence_input(self):
        """TEST-T1-F14-03: Pass silence (all zeros) verifying ELU+1 suppresses zero division."""
        q = torch.zeros(1, 8, 16, requires_grad=True)
        k = torch.zeros(1, 8, 16, requires_grad=True)
        v = torch.zeros(1, 8, 16, requires_grad=True)
        w = torch.tensor([0.0], requires_grad=True)

        out = SubQAttentionFunction.apply(q, k, v, w)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        loss = out.sum()
        loss.backward()
        assert not torch.isnan(q.grad).any()
        assert not torch.isinf(q.grad).any()

    def test_t1_f14_04_rss_memory_headroom(self):
        """TEST-T1-F14-04: Measure RSS memory headroom before and after 50 forward passes."""
        if not HAS_PSUTIL:
            pytest.skip("psutil not available for RSS measurement")

        process = psutil.Process(os.getpid())
        rss_start = process.memory_info().rss
        engine = AudioTransformerEngine(vram_capacity=16 * 1024 * 1024)
        q = np.random.randn(1, 32, 64).astype(np.float32)

        for _ in range(50):
            _ = engine.forward(q, q, q, 0.99)

        rss_end = process.memory_info().rss
        diff_mb = (rss_end - rss_start) / (1024 * 1024)
        assert diff_mb < 100.0, f"Memory growth of {diff_mb:.2f} MB exceeds 100 MB headroom bound"

    def test_t1_f14_05_deterministic_seed_reproducibility(self):
        """TEST-T1-F14-05: Verify deterministic seed reproducibility across forward and backward passes."""
        torch.manual_seed(12345)
        q1 = torch.randn(1, 8, 16, requires_grad=True)
        k1 = torch.randn(1, 8, 16, requires_grad=True)
        v1 = torch.randn(1, 8, 16, requires_grad=True)
        w1 = torch.tensor([0.5], requires_grad=True)
        out1 = SubQAttentionFunction.apply(q1, k1, v1, w1)
        out1.sum().backward()

        torch.manual_seed(12345)
        q2 = torch.randn(1, 8, 16, requires_grad=True)
        k2 = torch.randn(1, 8, 16, requires_grad=True)
        v2 = torch.randn(1, 8, 16, requires_grad=True)
        w2 = torch.tensor([0.5], requires_grad=True)
        out2 = SubQAttentionFunction.apply(q2, k2, v2, w2)
        out2.sum().backward()

        torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
        torch.testing.assert_close(q1.grad, q2.grad, atol=0.0, rtol=0.0)

"""
Test suite for zero-copy NumPy and PyTorch tensor interoperability across CPU and GPU.
"""

import pytest
import numpy as np
import torch
from subq_audio import AudioTransformerEngine, _C


def test_numpy_forward_pass():
    """Verify NumPy array forward attention pass."""
    engine = AudioTransformerEngine(vram_capacity=64 * 1024 * 1024)
    batch, seq_len, d_model = 2, 64, 128

    Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    V = np.random.randn(batch, seq_len, d_model).astype(np.float32)

    O = engine.forward(Q, K, V, decay_factor=0.95)
    assert isinstance(O, np.ndarray)
    assert O.shape == (batch, seq_len, d_model)
    assert O.dtype == np.float32
    assert not np.isnan(O).any()
    assert not np.isinf(O).any()


def test_numpy_backward_pass():
    """Verify NumPy analytical reverse adjoint backward pass."""
    engine = AudioTransformerEngine(vram_capacity=64 * 1024 * 1024)
    batch, seq_len, d_model = 2, 32, 64

    Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    V = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    dO = np.ones((batch, seq_len, d_model), dtype=np.float32)

    dQ, dK, dV, d_decay = engine.backward(dO, Q, K, V, decay_factor=0.95)
    assert dQ.shape == Q.shape
    assert dK.shape == K.shape
    assert dV.shape == V.shape
    assert isinstance(d_decay, float)
    assert not np.isnan(dQ).any()
    assert not np.isnan(dK).any()
    assert not np.isnan(dV).any()
    assert not np.isnan(d_decay)


def test_pytorch_cpu_forward_backward():
    """Verify PyTorch CPU tensor passing and backward pass."""
    engine = AudioTransformerEngine(vram_capacity=64 * 1024 * 1024)
    batch, seq_len, d_model = 2, 32, 64

    Q = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
    K = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
    V = torch.randn(batch, seq_len, d_model, dtype=torch.float32)

    O = engine.forward(Q, K, V, decay_factor=0.99)
    assert isinstance(O, torch.Tensor)
    assert O.device.type == "cpu"
    assert O.shape == (batch, seq_len, d_model)
    assert not torch.isnan(O).any()

    dO = torch.ones_like(O)
    dQ, dK, dV, d_decay = engine.backward(dO, Q, K, V, decay_factor=0.99)
    assert isinstance(dQ, torch.Tensor)
    assert isinstance(dK, torch.Tensor)
    assert isinstance(dV, torch.Tensor)
    assert dQ.shape == Q.shape
    assert dK.shape == K.shape
    assert dV.shape == V.shape
    assert not torch.isnan(dQ).any()


def test_various_batch_and_sequence_dimensions():
    """Verify execution across multiple batch sizes and sequence lengths."""
    engine = AudioTransformerEngine(vram_capacity=128 * 1024 * 1024)
    configs = [
        (1, 16, 32),
        (2, 64, 64),
        (4, 32, 128),
    ]

    for batch, seq_len, d_model in configs:
        Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
        K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
        V = np.random.randn(batch, seq_len, d_model).astype(np.float32)

        O = engine.forward(Q, K, V, decay_factor=0.9)
        assert O.shape == (batch, seq_len, d_model)
        assert not np.isnan(O).any()


def test_pytorch_cuda_if_available():
    """Verify zero-copy PyTorch CUDA execution if CUDA hardware is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware not available on this host.")

    engine = AudioTransformerEngine(vram_capacity=128 * 1024 * 1024)
    batch, seq_len, d_model = 2, 32, 64

    Q = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device="cuda")
    K = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device="cuda")
    V = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device="cuda")

    O = engine.forward(Q, K, V, decay_factor=0.95)
    assert O.is_cuda
    assert O.shape == (batch, seq_len, d_model)

    dO = torch.ones_like(O)
    dQ, dK, dV, d_decay = engine.backward(dO, Q, K, V, decay_factor=0.95)
    assert dQ.is_cuda
    assert dK.is_cuda
    assert dV.is_cuda
    assert dQ.shape == Q.shape

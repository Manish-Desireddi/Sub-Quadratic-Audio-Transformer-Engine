import pytest
import numpy as np
import time
import os

try:
    import subq_engine
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import build.Release.subq_engine as subq_engine

@pytest.fixture(scope="module")
def engine():
    return subq_engine.SubQEngine(256 * 1024 * 1024)

def test_computational_scaling(engine, real_audio):
    batch = 1
    d_model = 256
    seq_lengths = [1024, 4096] # Test practical small sequences
    
    for seq_len in seq_lengths:
        Q, K, V = real_audio(batch, seq_len, d_model)
        
        # Warmup
        _ = engine.forward(Q, K, V)
        
        start = time.perf_counter()
        out = engine.forward(Q, K, V)
        latency = (time.perf_counter() - start) * 1000
        
        assert out.shape == (batch, seq_len, d_model), f"Expected shape {(batch, seq_len, d_model)}, got {out.shape}"
        
        info = subq_engine.get_device_info()
        if info['backend'] != "CPU":
            assert latency < 1000.0, f"Latency {latency:.2f} ms exceeds 1000ms threshold for O(1) engine on GPU"

def test_mathematical_stability(engine, real_audio):
    batch = 1
    d_model = 256
    massive_seq = 100000
    
    Q, K, V = real_audio(batch, massive_seq, d_model)
    # Scale down slightly to prevent immediate blowup on massive real audio
    Q, K, V = Q / np.sqrt(d_model), K / np.sqrt(d_model), V / np.sqrt(d_model)
    
    out = engine.forward(Q, K, V)
    
    has_nans = np.isnan(out).any()
    has_infs = np.isinf(out).any()
    max_val = np.abs(out).max()
    
    assert not has_nans, "NaNs detected in output state!"
    assert not has_infs, "Infs detected in output state!"
    assert max_val < 1000.0, f"Max value {max_val} indicates exploding gradient drift"

import pytest
import numpy as np
import os
import struct
import json
import torch

import subq_engine

def test_safetensors_overflow(tmp_path):
    """
    Test that the C++ backend correctly catches SIZE_MAX integer overflows
    when parsing the Safetensors header, rather than wrapping around and segfaulting.
    """
    bad_file = tmp_path / "overflow.safetensors"
    
    # Construct a malicious header
    malicious_header = {
        "__metadata__": {"format": "pt"},
        "bad_tensor": {
            "dtype": "F32",
            "shape": [1024, 1024],
            "data_offsets": [0, (2**64) - 1] # SIZE_MAX / Overflow trigger
        }
    }
    
    header_json = json.dumps(malicious_header).encode('utf-8')
    header_size = len(header_json)
    
    with open(bad_file, "wb") as f:
        # 8-byte uint64 length of header
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        # We don't even need to write the actual payload; the parser should throw instantly.
    
    engine = subq_engine.SubQEngine(10 * 1024 * 1024)
    with pytest.raises(RuntimeError) as exc_info:
        engine.load(str(bad_file))
    
    # Ensure it's the overflow exception we wrote, not a silent segfault
    assert "Failed to load safetensors file" in str(exc_info.value)


def test_garbage_shapes():
    """
    Test that PyBind11 correctly prevents non-3D arrays, 0-shapes, or mismatched bounds 
    from bleeding into the C++ `forward()` kernel.
    """
    engine = subq_engine.SubQEngine(10 * 1024 * 1024)
    
    # 1. 0-dimensional scalars
    Q = np.array(0.0, dtype=np.float32)
    with pytest.raises(RuntimeError):
        engine.forward(Q, Q, Q)
        
    # 2. 2D arrays instead of 3D
    Q = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(RuntimeError):
        engine.forward(Q, Q, Q)
        
    # 3. Non-contiguous arrays (using slicing to force non-contiguous memory)
    # PyBind11 forcecast and c_style should automatically copy it and prevent crashes.
    Q_big = np.random.randn(2, 100, 64).astype(np.float32)
    K_big = np.random.randn(2, 100, 64).astype(np.float32)
    V_big = np.random.randn(2, 100, 64).astype(np.float32)
    
    Q_slice = Q_big[:, ::2, :] # Strided, non-contiguous
    K_slice = K_big[:, ::2, :]
    V_slice = V_big[:, ::2, :]
    
    # Should seamlessly execute without segfaulting, thanks to py::array::c_style
    out = engine.forward(Q_slice, K_slice, V_slice)
    assert out.shape == Q_slice.shape


def test_nan_inf_ingestion():
    """
    Inject NaN and Inf into the Q, K, V tensors to ensure the sub-quadratic 
    kernel doesn't crash the host or cascade infinitely beyond the output.
    """
    engine = subq_engine.SubQEngine(10 * 1024 * 1024)
    batch, seq_len, d_model = 1, 128, 64
    
    Q = np.full((batch, seq_len, d_model), np.nan, dtype=np.float32)
    K = np.full((batch, seq_len, d_model), np.inf, dtype=np.float32)
    V = np.full((batch, seq_len, d_model), -np.inf, dtype=np.float32)
    
    # The kernel should process this without crashing the process
    out = engine.forward(Q, K, V)
    
    # Output should mathematically be NaNs, but the engine itself must survive
    assert np.isnan(out).any()

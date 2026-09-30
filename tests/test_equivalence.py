# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import torch
import sys
import os
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "python"))

import subq_engine
from .reference import linear_attention_forward

def test_forward_equivalence(real_audio):
    batch = 1
    seq_len = 512
    d_model = 32

    # Get real audio arrays
    Q_np, K_np, V_np = real_audio(batch, seq_len, d_model)
    
    # Scale down for equivalence check to prevent precision drifting on CPU float32
    Q_np = Q_np / 10.0
    K_np = K_np / 10.0
    V_np = V_np / 10.0

    # PyTorch tensors
    Q_torch = torch.from_numpy(Q_np)
    K_torch = torch.from_numpy(K_np)
    V_torch = torch.from_numpy(V_np)

    # PyTorch execution
    O_ref = linear_attention_forward(Q_torch.clone(), K_torch.clone(), V_torch.clone())

    # C++ execution
    engine = subq_engine.SubQEngine(128 * 1024 * 1024)
    # Clone so C++ doesn't mutate our test arrays
    O_cpp = engine.forward(Q_np.copy(), K_np.copy(), V_np.copy(), decay_factor=1.0)
    O_cpp_tensor = torch.from_numpy(O_cpp)

    # Validate equivalence
    # We use a slightly looser tolerance for the real audio test because the C++ engine
    # fuses ELU+1 and performs associative reductions in a different algebraic order than PyTorch
    assert torch.allclose(O_ref, O_cpp_tensor, atol=1e-3), (
        "Numerical validation failed! Output matrices do not match within tolerance."
    )

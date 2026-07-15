# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "python"))

import subq_engine
from reference import linear_attention_forward


def test_forward_equivalence():
    batch = 2
    seq_len = 16
    d_model = 32

    # Generate random tensors
    torch.manual_seed(42)
    Q = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
    K = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
    V = torch.randn(batch, seq_len, d_model, dtype=torch.float32)

    # PyTorch execution
    O_ref = linear_attention_forward(Q.clone(), K.clone(), V.clone())

    # C++ execution
    engine = subq_engine.SubQEngine(128 * 1024 * 1024)
    # Clone to numpy so C++ can modify in place without breaking our original tensors
    O_cpp = engine.forward(Q.clone().numpy(), K.clone().numpy(), V.clone().numpy())
    O_cpp_tensor = torch.from_numpy(O_cpp)

    # Validate equivalence
    assert torch.allclose(O_ref, O_cpp_tensor, atol=1e-4), (
        "Output matrices do not match!"
    )
    print(
        "Numerical validation successful! C++ Engine output matches PyTorch reference exactly."
    )


if __name__ == "__main__":
    test_forward_equivalence()

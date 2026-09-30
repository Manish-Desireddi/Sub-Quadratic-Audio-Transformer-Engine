# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from subq_audio.autograd import (
    SubQAttentionFunction,
    SubQLinearAttention,
    SubQuadraticAttention,
    feature_map,
    feature_map_prime,
)


def reference_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay_w: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for Sub-Quadratic Linear Attention.
    Uses autograd-tracked loop for ground-truth gradient comparison.
    """
    gamma = torch.sigmoid(decay_w)
    phi_Q = feature_map(q)
    phi_K = feature_map(k)

    if q.ndim == 4:
        B, H, T, D = q.shape
        out = torch.zeros_like(q)
        S = torch.zeros((B, H, D, D), dtype=q.dtype, device=q.device)
        for t in range(T):
            kt = phi_K[:, :, t, :].unsqueeze(3)  # [B, H, D, 1]
            vt = v[:, :, t, :].unsqueeze(2)      # [B, H, 1, D]
            if gamma.ndim == 0:
                g_t = gamma
            elif gamma.shape[0] == H:
                g_t = gamma.view(1, H, 1, 1)
            else:
                g_t = gamma.view(-1, H, 1, 1)
            S = g_t * S + torch.matmul(kt, vt)
            qt = phi_Q[:, :, t, :].unsqueeze(2)  # [B, H, 1, D]
            out[:, :, t, :] = torch.matmul(qt, S).squeeze(2)
        return out

    B, T, D = q.shape
    out = torch.zeros_like(q)
    S = torch.zeros((B, D, D), dtype=q.dtype, device=q.device)
    for t in range(T):
        kt = phi_K[:, t, :].unsqueeze(2)  # [B, D, 1]
        vt = v[:, t, :].unsqueeze(1)      # [B, 1, D]
        if gamma.ndim == 0:
            g_t = gamma
        else:
            g_t = gamma.view(-1, 1, 1)
        S = g_t * S + torch.bmm(kt, vt)
        qt = phi_Q[:, t, :].unsqueeze(1)  # [B, 1, D]
        out[:, t, :] = torch.bmm(qt, S).squeeze(1)
    return out


# ==============================================================================
# 1. Standard Gradcheck Suite (Double Precision)
# ==============================================================================

def test_gradcheck_nominal():
    torch.manual_seed(42)
    B, T, D = 2, 4, 8
    dtype = torch.float64

    q = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    w = torch.tensor(2.0, dtype=dtype, requires_grad=True)

    passed = gradcheck(
        SubQAttentionFunction.apply,
        (q, k, v, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )
    assert passed, "Nominal gradcheck failed"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_gradcheck_batch_variations(batch_size):
    torch.manual_seed(100 + batch_size)
    T, D = 4, 4
    dtype = torch.float64

    q = torch.randn(batch_size, T, D, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, T, D, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, T, D, dtype=dtype, requires_grad=True)
    w = torch.tensor(1.0, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q, k, v, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


@pytest.mark.parametrize("seq_len", [1, 2, 8, 16])
def test_gradcheck_seq_lens(seq_len):
    torch.manual_seed(200 + seq_len)
    B, D = 2, 4
    dtype = torch.float64

    q = torch.randn(B, seq_len, D, dtype=dtype, requires_grad=True)
    k = torch.randn(B, seq_len, D, dtype=dtype, requires_grad=True)
    v = torch.randn(B, seq_len, D, dtype=dtype, requires_grad=True)
    w = torch.tensor(0.5, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q, k, v, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


@pytest.mark.parametrize("head_dim", [2, 4, 8, 16])
def test_gradcheck_head_dims(head_dim):
    torch.manual_seed(300 + head_dim)
    B, T = 2, 4
    dtype = torch.float64

    q = torch.randn(B, T, head_dim, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, head_dim, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, head_dim, dtype=dtype, requires_grad=True)
    w = torch.tensor(1.5, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q, k, v, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


def test_gradcheck_4d_multi_head():
    torch.manual_seed(400)
    B, H, T, D = 2, 3, 4, 4
    dtype = torch.float64

    q = torch.randn(B, H, T, D, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=dtype, requires_grad=True)
    w = torch.randn(H, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q, k, v, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


def test_gradcheck_zero_inputs():
    torch.manual_seed(500)
    B, T, D = 2, 4, 4
    dtype = torch.float64

    q = torch.zeros(B, T, D, dtype=dtype, requires_grad=True)
    k = torch.zeros(B, T, D, dtype=dtype, requires_grad=True)
    v = torch.zeros(B, T, D, dtype=dtype, requires_grad=True)
    w = torch.tensor(0.0, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q, k, v, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


@pytest.mark.parametrize("w_val", [-15.0, -5.0, 0.0, 5.0, 15.0])
def test_gradcheck_decay_bounds(w_val):
    torch.manual_seed(600 + int(abs(w_val)))
    B, T, D = 1, 4, 4
    dtype = torch.float64

    q = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    w = torch.tensor(w_val, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q, k, v, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


def test_gradcheck_positive_negative_regimes():
    torch.manual_seed(700)
    B, T, D = 2, 4, 4
    dtype = torch.float64

    # Strictly positive regime (phi(x) = x + 1, phi'(x) = 1.0)
    q_pos = (torch.rand(B, T, D, dtype=dtype) + 1.0).requires_grad_(True)
    k_pos = (torch.rand(B, T, D, dtype=dtype) + 1.0).requires_grad_(True)
    v_pos = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    w = torch.tensor(1.2, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q_pos, k_pos, v_pos, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )

    # Strictly negative regime (phi(x) = exp(x), phi'(x) = exp(x))
    q_neg = (-torch.rand(B, T, D, dtype=dtype) - 1.0).requires_grad_(True)
    k_neg = (-torch.rand(B, T, D, dtype=dtype) - 1.0).requires_grad_(True)
    v_neg = torch.randn(B, T, D, dtype=dtype, requires_grad=True)

    assert gradcheck(
        SubQAttentionFunction.apply,
        (q_neg, k_neg, v_neg, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )


# ==============================================================================
# 2. Gradient Matching Against PyTorch Reference Implementation
# ==============================================================================

def test_grad_vs_reference_autograd():
    torch.manual_seed(800)
    B, T, D = 2, 16, 16
    dtype = torch.float64

    q_raw = torch.randn(B, T, D, dtype=dtype)
    k_raw = torch.randn(B, T, D, dtype=dtype)
    v_raw = torch.randn(B, T, D, dtype=dtype)
    w_raw = 2.5

    # 1. Custom Analytical Backward
    q1 = q_raw.clone().requires_grad_(True)
    k1 = k_raw.clone().requires_grad_(True)
    v1 = v_raw.clone().requires_grad_(True)
    w1 = torch.tensor(w_raw, dtype=dtype, requires_grad=True)

    out1 = SubQAttentionFunction.apply(q1, k1, v1, w1)
    target = torch.sin(out1.detach())
    loss1 = torch.sum((out1 - target) ** 2)
    loss1.backward()

    # 2. Reference Loop Backward
    q2 = q_raw.clone().requires_grad_(True)
    k2 = k_raw.clone().requires_grad_(True)
    v2 = v_raw.clone().requires_grad_(True)
    w2 = torch.tensor(w_raw, dtype=dtype, requires_grad=True)

    out2 = reference_linear_attention(q2, k2, v2, w2)
    loss2 = torch.sum((out2 - target) ** 2)
    loss2.backward()

    # Verify Forward Equivalence
    assert torch.allclose(out1, out2, atol=1e-7, rtol=1e-5), "Forward outputs differ from reference"

    # Verify Analytical Gradients match Reference
    assert torch.allclose(q1.grad, q2.grad, atol=1e-5, rtol=1e-4), f"dQ mismatch: max diff = {(q1.grad - q2.grad).abs().max()}"
    assert torch.allclose(k1.grad, k2.grad, atol=1e-5, rtol=1e-4), f"dK mismatch: max diff = {(k1.grad - k2.grad).abs().max()}"
    assert torch.allclose(v1.grad, v2.grad, atol=1e-5, rtol=1e-4), f"dV mismatch: max diff = {(v1.grad - v2.grad).abs().max()}"
    assert torch.allclose(w1.grad, w2.grad, atol=1e-5, rtol=1e-4), f"dw mismatch: max diff = {(w1.grad - w2.grad).abs().max()}"


# ==============================================================================
# 3. SubQLinearAttention Module Step & Stacked Backprop
# ==============================================================================

def test_subq_linear_attention_module_step():
    torch.manual_seed(900)
    B, T, hidden_size = 2, 8, 32
    attn = SubQLinearAttention(hidden_size=hidden_size, num_heads=4, decay_init=0.95)
    optimizer = torch.optim.AdamW(attn.parameters(), lr=1e-3)

    x = torch.randn(B, T, hidden_size, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.q_proj.weight.grad is not None and attn.q_proj.weight.grad.abs().sum() > 0
    assert attn.k_proj.weight.grad is not None and attn.k_proj.weight.grad.abs().sum() > 0
    assert attn.v_proj.weight.grad is not None and attn.v_proj.weight.grad.abs().sum() > 0
    assert attn.out_proj.weight.grad is not None and attn.out_proj.weight.grad.abs().sum() > 0
    assert attn.raw_decay.grad is not None and attn.raw_decay.grad.abs().sum() > 0

    init_decay = attn.decay_factor.clone().detach()
    optimizer.step()
    updated_decay = attn.decay_factor.detach()

    assert not torch.equal(init_decay, updated_decay)


def test_stacked_layers_backprop():
    d_model = 32
    seq_len = 16
    batch = 2

    model = nn.Sequential(
        SubQLinearAttention(hidden_size=d_model, num_heads=2, decay_init=0.95),
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        SubQLinearAttention(hidden_size=d_model, num_heads=2, decay_init=0.95),
    )

    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    out = x
    for layer in model:
        if isinstance(layer, SubQLinearAttention):
            out, _ = layer(out)
        else:
            out = layer(out)

    assert out.shape == (batch, seq_len, d_model)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

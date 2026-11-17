#
# Parity tests between the C++ engine and an autograd-tracked loop reference.
# Covers decayed forward and backward across the 64-token chunk boundary.
#

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from subq_audio import SubQEngine


def _loop_reference(q, k, v, gamma):
    """O_t = phi(q_t) S_t ; S_t = gamma * S_{t-1} + phi(k_t)^T v_t (float64, autograd-tracked)."""
    B, T, D = q.shape
    phi_q = F.elu(q) + 1.0
    phi_k = F.elu(k) + 1.0
    S = torch.zeros((B, D, D), dtype=q.dtype)
    outs = []
    for t in range(T):
        S = gamma * S + torch.bmm(phi_k[:, t].unsqueeze(2), v[:, t].unsqueeze(1))
        outs.append(torch.bmm(phi_q[:, t].unsqueeze(1), S).squeeze(1))
    return torch.stack(outs, dim=1)


def _inputs(B, T, D, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, D, generator=g, dtype=torch.float64) * 0.5 for _ in range(3)]


CASES = [
    # (batch, seq_len, d_model, gamma): T=130 spans two full chunks plus a tail
    (1, 7, 4, 0.5),
    (2, 64, 8, 0.9),
    (1, 65, 8, 0.99),
    (2, 130, 16, 0.95),
    (1, 130, 32, 0.8),
]


@pytest.mark.parametrize("B,T,D,gamma", CASES)
def test_decayed_forward_matches_reference(B, T, D, gamma):
    q, k, v = _inputs(B, T, D, seed=T + D)
    expected = _loop_reference(q, k, v, torch.tensor(gamma, dtype=torch.float64))

    engine = SubQEngine(128 * 1024 * 1024)
    out = engine.forward(
        q.float().numpy().copy(), k.float().numpy().copy(), v.float().numpy().copy(),
        decay_factor=gamma,
    )
    np.testing.assert_allclose(out, expected.numpy(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("B,T,D,gamma", CASES)
def test_decayed_backward_matches_reference_autograd(B, T, D, gamma):
    q, k, v = _inputs(B, T, D, seed=100 + T + D)
    dO = _inputs(B, T, D, seed=200 + T + D)[0]

    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    gr = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
    _loop_reference(qr, kr, vr, gr).backward(dO)

    engine = SubQEngine(128 * 1024 * 1024)
    dq, dk, dv, d_gamma = engine.backward(
        dO.float().numpy().copy(),
        q.float().numpy().copy(), k.float().numpy().copy(), v.float().numpy().copy(),
        decay_factor=gamma,
    )

    np.testing.assert_allclose(dq, qr.grad.numpy(), rtol=5e-3, atol=5e-3, err_msg="dQ")
    np.testing.assert_allclose(dk, kr.grad.numpy(), rtol=5e-3, atol=5e-3, err_msg="dK")
    np.testing.assert_allclose(dv, vr.grad.numpy(), rtol=5e-3, atol=5e-3, err_msg="dV")
    np.testing.assert_allclose(d_gamma, gr.grad.item(), rtol=5e-3, atol=5e-3, err_msg="d_gamma")


def test_small_values_are_not_flushed_to_zero():
    """Verify that small values above the float subnormal range are not zeroed."""
    B, T, D = 1, 4, 4
    q, k, v = _inputs(B, T, D, seed=7)
    v = v * 1e-18  # state entries ~1e-18: far above float subnormal (~1e-38), below the 1e-15 flush bound
    expected = _loop_reference(q, k, v, torch.tensor(0.9, dtype=torch.float64))
    assert 1e-30 < expected.abs().max().item() < 1e-15

    engine = SubQEngine(128 * 1024 * 1024)
    out = engine.forward(
        q.float().numpy().copy(), k.float().numpy().copy(), v.float().numpy().copy(),
        decay_factor=0.9,
    )
    np.testing.assert_allclose(out, expected.numpy(), rtol=1e-2, atol=0.0)

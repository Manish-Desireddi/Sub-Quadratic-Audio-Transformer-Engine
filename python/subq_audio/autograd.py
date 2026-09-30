# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    ELU+1 feature map: phi(x) = ELU(x) + 1.
    Guarantees non-negativity and positive-definite kernel evaluations.
    """
    return F.elu(x) + 1.0


def feature_map_prime(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of ELU+1 feature map:
    phi'(x) = 1.0 if x > 0 else exp(x).
    """
    return torch.where(x > 0.0, torch.ones_like(x), torch.exp(x))


class SubQAttentionFunction(torch.autograd.Function):
    """
    Exact analytical autograd bridge for Sub-Quadratic Associative Linear Attention.
    Implements reverse adjoint state recurrence:
      Gamma_t = gamma * Gamma_{t+1} + phi(Q_t)^T * dO_t
    with exact analytical gradients:
      dQ_t = (dO_t * S_t^T) * phi'(Q_t)
      dK_t = (V_t * Gamma_t^T) * phi'(K_t)
      dV_t = phi(K_t) * Gamma_t
      dgamma = sum_{t=1}^{T-1} <Gamma_{t+1}, S_t>_F
      dw = dgamma * gamma * (1 - gamma) where gamma = sigmoid(w)
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay_w: torch.Tensor) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"Shape mismatch: Q:{q.shape}, K:{k.shape}, V:{v.shape}")

        orig_shape = q.shape
        is_4d = (q.ndim == 4)  # [B, H, T, D]
        if is_4d:
            B, H, T, D = q.shape
            q_flat = q.reshape(B * H, T, D).contiguous()
            k_flat = k.reshape(B * H, T, D).contiguous()
            v_flat = v.reshape(B * H, T, D).contiguous()
        elif q.ndim == 3:
            B, T, D = q.shape
            H = 1
            q_flat = q.contiguous()
            k_flat = k.contiguous()
            v_flat = v.contiguous()
        else:
            raise ValueError(f"Expected 3D [B, T, D] or 4D [B, H, T, D] tensors, got {q.ndim}D")

        total_batches = q_flat.shape[0]  # B * H

        # Ensure decay_w is properly formatted and shaped
        if not isinstance(decay_w, torch.Tensor):
            decay_w_tensor = torch.tensor(decay_w, dtype=q.dtype, device=q.device)
        else:
            decay_w_tensor = decay_w.to(dtype=q.dtype, device=q.device).contiguous()

        # Compute gamma = sigmoid(w)
        gamma = torch.sigmoid(decay_w_tensor)
        if gamma.numel() == 1:
            gamma_expanded = gamma.repeat(total_batches)
        elif gamma.shape[0] == total_batches:
            gamma_expanded = gamma
        elif is_4d and gamma.shape[0] == H:
            gamma_expanded = gamma.repeat(B)
        else:
            gamma_expanded = gamma.expand(total_batches)

        # Feature maps
        phi_Q = feature_map(q_flat)
        phi_K = feature_map(k_flat)

        # Forward scan with state trajectory caching
        out = torch.zeros_like(q_flat)
        S_traj = torch.zeros((total_batches, T, D, D), dtype=q.dtype, device=q.device)
        S_curr = torch.zeros((total_batches, D, D), dtype=q.dtype, device=q.device)

        for t in range(T):
            gamma_t = gamma_expanded.view(total_batches, 1, 1)
            kt = phi_K[:, t, :].unsqueeze(2)  # [total_batches, D, 1]
            vt = v_flat[:, t, :].unsqueeze(1)  # [total_batches, 1, D]

            # S_t = gamma * S_{t-1} + phi(K_t)^T * V_t
            S_curr = gamma_t * S_curr + torch.bmm(kt, vt)
            S_traj[:, t, :, :] = S_curr

            # O_t = phi(Q_t) * S_t
            qt = phi_Q[:, t, :].unsqueeze(1)  # [total_batches, 1, D]
            out[:, t, :] = torch.bmm(qt, S_curr).squeeze(1)

        # Save for backward pass
        ctx.save_for_backward(q_flat, k_flat, v_flat, S_traj, decay_w_tensor)
        ctx.is_4d = is_4d
        ctx.orig_shape = orig_shape
        ctx.H = H
        ctx.B = B

        if is_4d:
            return out.reshape(orig_shape)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_flat, k_flat, v_flat, S_traj, decay_w_tensor = ctx.saved_tensors
        is_4d = ctx.is_4d
        orig_shape = ctx.orig_shape
        H = ctx.H
        B = ctx.B

        total_batches, T, D = q_flat.shape
        dO_flat = grad_output.reshape(total_batches, T, D).contiguous()

        # Compute gamma = sigmoid(w)
        gamma = torch.sigmoid(decay_w_tensor)
        if gamma.numel() == 1:
            gamma_expanded = gamma.repeat(total_batches)
        elif gamma.shape[0] == total_batches:
            gamma_expanded = gamma
        elif is_4d and gamma.shape[0] == H:
            gamma_expanded = gamma.repeat(B)
        else:
            gamma_expanded = gamma.expand(total_batches)

        phi_Q = feature_map(q_flat)
        phi_K = feature_map(k_flat)
        q_prime = feature_map_prime(q_flat)
        k_prime = feature_map_prime(k_flat)

        dQ = torch.zeros_like(q_flat)
        dK = torch.zeros_like(k_flat)
        dV = torch.zeros_like(v_flat)
        d_gamma_total = torch.zeros((total_batches,), dtype=q_flat.dtype, device=q_flat.device)

        # Reverse Adjoint Recurrence:
        # Gamma_t = gamma * Gamma_{t+1} + phi(Q_t)^T * dO_t
        Gamma_curr = torch.zeros((total_batches, D, D), dtype=q_flat.dtype, device=q_flat.device)

        for t in range(T - 1, -1, -1):
            gamma_t = gamma_expanded.view(total_batches, 1, 1)
            qt = phi_Q[:, t, :].unsqueeze(2)  # [total_batches, D, 1]
            dot = dO_flat[:, t, :].unsqueeze(1)  # [total_batches, 1, D]
            st = S_traj[:, t, :, :]  # [total_batches, D, D]

            # 1. Update Gamma_t = gamma * Gamma_{t+1} + phi(Q_t)^T * dO_t
            Gamma_curr = gamma_t * Gamma_curr + torch.bmm(qt, dot)

            # 2. dQ_t = (dO_t * S_t^T) * phi'(Q_t)
            dq_mapped = torch.bmm(dot, st.transpose(1, 2)).squeeze(1)
            dQ[:, t, :] = dq_mapped * q_prime[:, t, :]

            # 3. dV_t = phi(K_t) * Gamma_t
            kt_row = phi_K[:, t, :].unsqueeze(1)
            dV[:, t, :] = torch.bmm(kt_row, Gamma_curr).squeeze(1)

            # 4. dK_t = (V_t * Gamma_t^T) * phi'(K_t)
            vt_row = v_flat[:, t, :].unsqueeze(1)
            dk_mapped = torch.bmm(vt_row, Gamma_curr.transpose(1, 2)).squeeze(1)
            dK[:, t, :] = dk_mapped * k_prime[:, t, :]

            # 5. Accumulate d_gamma += <Gamma_t, S_{t-1}>_F (for t > 0)
            if t > 0:
                s_prev = S_traj[:, t - 1, :, :]
                d_gamma_total += torch.sum(Gamma_curr * s_prev, dim=(1, 2))

        # dw = d_gamma * gamma * (1 - gamma)
        dw_total = d_gamma_total * gamma_expanded * (1.0 - gamma_expanded)

        # Reduce dw to match original decay_w shape
        if decay_w_tensor.numel() == 1:
            dw = torch.sum(dw_total).reshape(decay_w_tensor.shape)
        elif is_4d and decay_w_tensor.shape[0] == H:
            dw = dw_total.view(B, H).sum(dim=0).reshape(decay_w_tensor.shape)
        elif dw_total.shape == decay_w_tensor.shape:
            dw = dw_total
        else:
            dw = dw_total.sum().reshape(decay_w_tensor.shape)

        if is_4d:
            dQ = dQ.reshape(orig_shape)
            dK = dK.reshape(orig_shape)
            dV = dV.reshape(orig_shape)

        return dQ, dK, dV, dw


class SubQLinearAttention(nn.Module):
    """
    Sub-Quadratic Associative Linear Attention layer.
    Features learnable decay parameterization w with gamma = sigmoid(w),
    multi-head projections, and Pre-LayerNorm compatibility.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        decay_init: float = 0.99,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if self.head_dim * num_heads != hidden_size:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        decay_init_clamped = min(max(decay_init, 1e-4), 1.0 - 1e-4)
        init_w = math.log(decay_init_clamped / (1.0 - decay_init_clamped))
        self.raw_decay = nn.Parameter(torch.full((num_heads,), init_w, dtype=torch.float32))

    @property
    def decay_factor(self) -> torch.Tensor:
        """Returns bounded decay factor gamma = sigmoid(raw_decay) in (0, 1)."""
        return torch.sigmoid(self.raw_decay)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = hidden_states.shape

        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        K = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        V = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]

        out = SubQAttentionFunction.apply(Q, K, V, self.raw_decay)  # [B, H, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        projected = self.out_proj(out)

        if use_cache:
            phi_K = feature_map(K)
            gamma = self.decay_factor.view(1, self.num_heads, 1, 1)
            S = torch.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            for t in range(L):
                kt = phi_K[:, :, t, :].unsqueeze(3)
                vt = V[:, :, t, :].unsqueeze(2)
                S = gamma * S + torch.matmul(kt, vt)
            return projected, S

        return projected, None


# Backward compatibility alias
SubQuadraticAttention = SubQLinearAttention

# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

import math
from typing import Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to the query and key tensors.
    Supports (B, L, D), (1, L, D), or (B, H, L, D) cos/sin representations.
    """
    if cos.ndim == 3 and q.ndim == 4:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    target_dtype: torch.dtype,
    target_len: int,
) -> Optional[torch.Tensor]:
    """
    Normalizes arbitrary 2D, 3D, or 4D attention masks into an active-token boolean/float
    indicator mask of shape [B, 1, T, 1].
    """
    if attention_mask is None:
        return None

    if attention_mask.dtype == torch.bool:
        valid_mask = attention_mask.to(dtype=target_dtype)
    elif torch.is_floating_point(attention_mask):
        if (attention_mask < -1.0).any():
            valid_mask = (attention_mask >= -1.0).to(dtype=target_dtype)
        else:
            valid_mask = (attention_mask > 0.0).to(dtype=target_dtype)
    else:
        valid_mask = (attention_mask != 0).to(dtype=target_dtype)

    # Normalize to [B, 1, T, 1]
    if valid_mask.ndim == 2:  # [B, T]
        valid_mask = valid_mask.unsqueeze(1).unsqueeze(-1)
    elif valid_mask.ndim == 3:  # [B, 1, T] or [B, T, 1]
        if valid_mask.shape[1] == 1 and valid_mask.shape[2] == target_len:
            valid_mask = valid_mask.unsqueeze(-1)
        elif valid_mask.shape[2] == 1 and valid_mask.shape[1] == target_len:
            valid_mask = valid_mask.unsqueeze(1)
        else:
            valid_mask = (valid_mask.sum(dim=-1, keepdim=True) > 0).to(dtype=target_dtype)
            if valid_mask.shape[1] == target_len:
                valid_mask = valid_mask.unsqueeze(1)
    elif valid_mask.ndim == 4:  # [B, 1, L, S] or [B, H, L, S]
        if valid_mask.shape[-2] != valid_mask.shape[-1]:
            raise ValueError(
                f"4D attention_mask must be square [B, 1|H, T, T]; got {tuple(valid_mask.shape)}"
            )
        # A token is active only if it is a live query (row has any visible key)
        # AND a live key (column is visible to some query). The column test is what
        # catches key-padding masks of shape [B, 1, T, T].
        active_query = valid_mask.sum(dim=-1) > 0  # [B, H', T]
        active_key = valid_mask.sum(dim=-2) > 0    # [B, H', T]
        valid_mask = (active_query & active_key).to(dtype=target_dtype).unsqueeze(-1)
        if valid_mask.shape[1] > 1:
            valid_mask = valid_mask[:, :1, :, :]

    return valid_mask


def _raw_chunkwise_scan(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: torch.Tensor,
    past_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Primitive kernelized associative scan over chunk blocks.
    Computes FLA recurrence for arbitrary value dimension Dv (e.g. D for numerator, 1 for denominator).
    """
    B, H, T, D = phi_q.shape
    Dv = v.shape[-1]

    S = (
        past_state.clone()
        if past_state is not None
        else torch.zeros((B, H, D, Dv), dtype=phi_q.dtype, device=phi_q.device)
    )

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        phi_q = F.pad(phi_q, (0, 0, 0, pad_len))
        phi_k = F.pad(phi_k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        log_gamma = F.pad(log_gamma, (0, pad_len), value=0.0)

    T_padded = phi_q.shape[2]
    num_chunks = T_padded // chunk_size

    phi_q_c = phi_q.view(B, H, num_chunks, chunk_size, D)
    phi_k_c = phi_k.view(B, H, num_chunks, chunk_size, D)
    v_c = v.view(B, H, num_chunks, chunk_size, Dv)
    log_gamma_c = log_gamma.view(B, H, num_chunks, chunk_size)

    z = torch.cumsum(log_gamma_c, dim=-1)
    z_diff = z.unsqueeze(-1) - z.unsqueeze(-2)
    causal_mask = torch.tril(torch.ones((chunk_size, chunk_size), device=phi_q.device, dtype=torch.bool))
    z_diff = z_diff.masked_fill(~causal_mask, float("-inf"))
    M = torch.exp(z_diff).to(phi_q.dtype)

    qk = torch.matmul(phi_q_c, phi_k_c.transpose(-1, -2)) * M
    out_intra = torch.matmul(qk, v_c)

    z_end = z[:, :, :, -1:]
    chunk_decay = torch.exp(z_end).unsqueeze(-1).to(phi_q.dtype)
    k_decay = torch.exp(z_end - z).unsqueeze(-1).to(phi_q.dtype)
    kv_chunks = torch.matmul((phi_k_c * k_decay).transpose(-1, -2), v_c)
    q_decay = torch.exp(z).unsqueeze(-1).to(phi_q.dtype)

    out_chunks = []
    for c in range(num_chunks):
        out_inter_c = torch.matmul(phi_q_c[:, :, c] * q_decay[:, :, c], S)
        out_chunks.append(out_intra[:, :, c] + out_inter_c)
        S = chunk_decay[:, :, c] * S + kv_chunks[:, :, c]

    out = torch.stack(out_chunks, dim=2).reshape(B, H, T_padded, Dv)
    if pad_len > 0:
        out = out[:, :, :T, :]

    return out.contiguous(), S


def _raw_chunkwise_backward(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: torch.Tensor,
    dO: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reverse adjoint scan for primitive chunkwise associative block.
    """
    B, H, T, D = phi_q.shape
    Dv = v.shape[-1]

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        phi_q = F.pad(phi_q, (0, 0, 0, pad_len))
        phi_k = F.pad(phi_k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        dO = F.pad(dO, (0, 0, 0, pad_len))
        log_gamma = F.pad(log_gamma, (0, pad_len), value=0.0)

    T_padded = phi_q.shape[2]
    num_chunks = T_padded // chunk_size

    phi_q_c = phi_q.view(B, H, num_chunks, chunk_size, D)
    phi_k_c = phi_k.view(B, H, num_chunks, chunk_size, D)
    v_c = v.view(B, H, num_chunks, chunk_size, Dv)
    dO_c = dO.view(B, H, num_chunks, chunk_size, Dv)
    log_gamma_c = log_gamma.view(B, H, num_chunks, chunk_size)

    z = torch.cumsum(log_gamma_c, dim=-1)
    z_diff = z.unsqueeze(-1) - z.unsqueeze(-2)
    causal_mask = torch.tril(torch.ones((chunk_size, chunk_size), device=phi_q.device, dtype=torch.bool))
    z_diff = z_diff.masked_fill(~causal_mask, float("-inf"))
    M = torch.exp(z_diff).to(phi_q.dtype)

    qk_unmasked = torch.matmul(phi_q_c, phi_k_c.transpose(-1, -2))

    z_end = z[:, :, :, -1:]
    chunk_decay = torch.exp(z_end).unsqueeze(-1).to(phi_q.dtype)
    k_decay = torch.exp(z_end - z).unsqueeze(-1).to(phi_q.dtype)
    kv_chunks = torch.matmul((phi_k_c * k_decay).transpose(-1, -2), v_c)
    q_decay = torch.exp(z).unsqueeze(-1).to(phi_q.dtype)

    # Pass 1: Forward checkpoints
    S_starts = []
    S_curr = torch.zeros((B, H, D, Dv), dtype=phi_q.dtype, device=phi_q.device)
    for c in range(num_chunks):
        S_starts.append(S_curr)
        S_curr = chunk_decay[:, :, c] * S_curr + kv_chunks[:, :, c]

    # Pass 2: Reverse adjoint scan
    dphi_q = torch.zeros_like(phi_q)
    dphi_k = torch.zeros_like(phi_k)
    dv = torch.zeros_like(v)
    dS_next = torch.zeros((B, H, D, Dv), dtype=phi_q.dtype, device=phi_q.device)
    dlog_gamma = torch.zeros_like(log_gamma_c)

    for c in range(num_chunks - 1, -1, -1):
        dO_chunk = dO_c[:, :, c]
        v_chunk = v_c[:, :, c]
        pq_chunk = phi_q_c[:, :, c]
        pk_chunk = phi_k_c[:, :, c]
        M_chunk = M[:, :, c]
        qk_u = qk_unmasked[:, :, c]

        A_chunk = qk_u * M_chunk
        dv_intra = torch.matmul(A_chunk.transpose(-1, -2), dO_chunk)
        dA_chunk = torch.matmul(dO_chunk, v_chunk.transpose(-1, -2))
        dScore = dA_chunk * M_chunk
        dpq_intra = torch.matmul(dScore, pk_chunk)
        dpk_intra = torch.matmul(dScore.transpose(-1, -2), pq_chunk)

        dM_score = dA_chunk * qk_u * M_chunk
        dM_score = torch.where(causal_mask, dM_score, torch.zeros_like(dM_score))
        dz_intra = dM_score.sum(dim=-1) - dM_score.sum(dim=-2)

        Sc = S_starts[c]
        qd = q_decay[:, :, c]
        kd = k_decay[:, :, c]
        cd = chunk_decay[:, :, c]

        dpq_inter = torch.matmul(dO_chunk, Sc.transpose(-1, -2)) * qd
        out_inter_c_val = torch.matmul(pq_chunk, Sc)
        dz_q = (dO_chunk * out_inter_c_val * qd).sum(dim=-1)

        dKV = dS_next
        dpk_inter = torch.matmul(v_chunk, dKV.transpose(-1, -2)) * kd
        dv_inter = torch.matmul(pk_chunk * kd, dKV)

        vk_term = (torch.matmul(v_chunk, dKV.transpose(-1, -2)) * pk_chunk).sum(dim=-1)
        dz_k = -vk_term * kd.squeeze(-1)
        dz_end_k = (vk_term * kd.squeeze(-1)).sum(dim=-1, keepdim=True)

        dcd = (dS_next * Sc).sum(dim=(-1, -2), keepdim=True)
        dz_end_cd = (dcd * cd).squeeze(-1)

        dz_c = dz_intra + dz_q + dz_k
        dz_c[:, :, -1:] += (dz_end_k + dz_end_cd)
        dlog_gamma[:, :, c] = torch.flip(torch.cumsum(torch.flip(dz_c, dims=[-1]), dim=-1), dims=[-1])

        dS_from_O = torch.matmul((pq_chunk * qd).transpose(-1, -2), dO_chunk)
        dS_next = dS_from_O + cd * dS_next

        dphi_q[:, :, c * chunk_size : (c + 1) * chunk_size] = dpq_intra + dpq_inter
        dphi_k[:, :, c * chunk_size : (c + 1) * chunk_size] = dpk_intra + dpk_inter
        dv[:, :, c * chunk_size : (c + 1) * chunk_size] = dv_intra + dv_inter

    if pad_len > 0:
        dphi_q = dphi_q[:, :, :T, :]
        dphi_k = dphi_k[:, :, :T, :]
        dv = dv[:, :, :T, :]
        dlog_gamma = dlog_gamma.view(B, H, T_padded)[:, :, :T]
    else:
        dlog_gamma = dlog_gamma.view(B, H, T)

    return dphi_q, dphi_k, dv, dlog_gamma


def _single_direction_causal_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_w: torch.Tensor,
    past_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    chunk_size: int = 64,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Core causal chunkwise scan engine with D^{-0.5} query scaling and running denominator normalizer Z_t.
    Ensures unit-scale output activations bounded as convex combinations of V.
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    phi_q = feature_map(q) * scale
    phi_k = feature_map(k)

    if not isinstance(decay_w, torch.Tensor):
        decay_w_t = torch.tensor(decay_w, dtype=q.dtype, device=q.device)
    else:
        decay_w_t = decay_w.to(dtype=q.dtype, device=q.device)

    if decay_w_t.ndim == 0:
        log_gamma = -F.softplus(-decay_w_t).view(1, 1, 1).expand(B, H, T)
    elif decay_w_t.ndim == 1:
        if decay_w_t.shape[0] == H:
            log_gamma = -F.softplus(-decay_w_t).view(1, H, 1).expand(B, H, T)
        else:
            log_gamma = -F.softplus(-decay_w_t).view(1, 1, -1).expand(B, H, T)
    elif decay_w_t.ndim == 2:
        if decay_w_t.shape[1] == T:
            log_gamma = -F.softplus(-decay_w_t).view(B, 1, T).expand(B, H, T)
        else:
            log_gamma = -F.softplus(-decay_w_t).view(B, H, 1).expand(B, H, T)
    elif decay_w_t.ndim == 3:
        if decay_w_t.shape == (B, H, T):
            log_gamma = -F.softplus(-decay_w_t)
        else:
            log_gamma = -F.softplus(-decay_w_t).expand(B, H, T)
    else:
        log_gamma = -F.softplus(-decay_w_t).expand(B, H, T)

    # Padding mask protection
    if valid_mask is not None:
        vm = valid_mask.expand(B, H, T, 1)
        phi_k = phi_k * vm
        v = v * vm
        vm_2d = valid_mask.squeeze(-1).expand(B, H, T)
        log_gamma = torch.where(vm_2d > 0, log_gamma, torch.zeros_like(log_gamma))

    # Parse past state
    S_past = None
    Z_past = None
    if isinstance(past_state, tuple):
        S_past = past_state[0]
        if len(past_state) > 1:
            Z_past = past_state[1]
    elif isinstance(past_state, torch.Tensor):
        S_past = past_state
        Z_past = getattr(past_state, "Z", None)

    # Compute numerator N
    N, S_next = _raw_chunkwise_scan(
        phi_q, phi_k, v, log_gamma, past_state=S_past, chunk_size=chunk_size
    )

    # Compute denominator Z with ones
    v_one = torch.ones((B, H, T, 1), dtype=q.dtype, device=q.device)
    if valid_mask is not None:
        v_one = v_one * valid_mask.expand(B, H, T, 1)
    Z_vec, Z_state = _raw_chunkwise_scan(
        phi_q, phi_k, v_one, log_gamma, past_state=Z_past, chunk_size=chunk_size
    )

    Z_clamped = torch.clamp(Z_vec, min=1e-5)
    out = N / Z_clamped

    if valid_mask is not None:
        out = out * valid_mask.expand_as(out)

    # Attach normalizer state Z to matrix state S for seamless O(1) caching
    setattr(S_next, "Z", Z_state)

    return out.contiguous(), S_next


def chunkwise_flash_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_w: torch.Tensor,
    past_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    chunk_size: int = 64,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized Chunkwise Parallel Flash Linear Attention.
    Supports causal autoregressive recurrence and bidirectional non-causal encoder scans,
    Grouped-Query Attention expansion, running denominator normalization, and padding mask state protection.
    """
    is_3d = (q.ndim == 3)
    if is_3d:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    B, H, T, D = q.shape
    valid_mask = _normalize_attention_mask(attention_mask, q.dtype, T)

    if is_causal:
        out, next_state = _single_direction_causal_scan(
            q, k, v, decay_w, past_state=past_state, chunk_size=chunk_size, valid_mask=valid_mask
        )
    else:
        # Bidirectional scan for non-causal encoder blocks (e.g. WhisperEncoder, BertModel)
        # 1. Forward scan (past-to-present)
        out_fwd, next_state = _single_direction_causal_scan(
            q, k, v, decay_w, past_state=past_state, chunk_size=chunk_size, valid_mask=valid_mask
        )

        # 2. Reverse scan (future-to-past)
        q_rev = torch.flip(q, dims=[2])
        k_rev = torch.flip(k, dims=[2])
        v_rev = torch.flip(v, dims=[2])
        decay_w_rev = torch.flip(decay_w, dims=[-1]) if isinstance(decay_w, torch.Tensor) and decay_w.ndim >= 2 else decay_w
        vm_rev = torch.flip(valid_mask, dims=[2]) if valid_mask is not None else None

        out_bwd_raw, _ = _single_direction_causal_scan(
            q_rev, k_rev, v_rev, decay_w_rev, past_state=None, chunk_size=chunk_size, valid_mask=vm_rev
        )
        out_bwd = torch.flip(out_bwd_raw, dims=[2])

        # 3. Deduplicate self-attention diagonal (in normalized linear attention, self-interaction = v)
        diag = v
        if valid_mask is not None:
            diag = diag * valid_mask.expand_as(diag)

        out = out_fwd + out_bwd - diag
        if valid_mask is not None:
            out = out * valid_mask.expand_as(out)

    if is_3d:
        out = out.squeeze(1)

    return out.contiguous(), next_state


def _single_direction_causal_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_w: torch.Tensor,
    grad_output: torch.Tensor,
    chunk_size: int = 64,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes exact analytical reverse adjoint gradients for single-direction causal Flash Linear Attention
    with quotient rule backpropagation through the running normalizer Z_t.
    """
    orig_w = decay_w
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    phi_q = feature_map(q) * scale
    phi_k = feature_map(k)

    if decay_w.ndim == 0:
        log_gamma = -F.softplus(-decay_w).view(1, 1, 1).expand(B, H, T)
    elif decay_w.ndim == 1:
        if decay_w.shape[0] == H:
            log_gamma = -F.softplus(-decay_w).view(1, H, 1).expand(B, H, T)
        else:
            log_gamma = -F.softplus(-decay_w).view(1, 1, -1).expand(B, H, T)
    elif decay_w.ndim == 2:
        if decay_w.shape[1] == T:
            log_gamma = -F.softplus(-decay_w).view(B, 1, T).expand(B, H, T)
        else:
            log_gamma = -F.softplus(-decay_w).view(B, H, 1).expand(B, H, T)
    elif decay_w.ndim == 3:
        if decay_w.shape == (B, H, T):
            log_gamma = -F.softplus(-decay_w)
        else:
            log_gamma = -F.softplus(-decay_w).expand(B, H, T)
    else:
        log_gamma = -F.softplus(-decay_w).expand(B, H, T)

    if valid_mask is not None:
        vm = valid_mask.expand(B, H, T, 1)
        phi_k = phi_k * vm
        v = v * vm
        vm_2d = valid_mask.squeeze(-1).expand(B, H, T)
        log_gamma = torch.where(vm_2d > 0, log_gamma, torch.zeros_like(log_gamma))
        grad_output = grad_output * valid_mask.expand(B, H, T, D)

    # Forward activations for quotient rule
    N, _ = _raw_chunkwise_scan(phi_q, phi_k, v, log_gamma, chunk_size=chunk_size)
    v_one = torch.ones((B, H, T, 1), dtype=q.dtype, device=q.device)
    if valid_mask is not None:
        v_one = v_one * valid_mask.expand(B, H, T, 1)
    Z_vec, _ = _raw_chunkwise_scan(phi_q, phi_k, v_one, log_gamma, chunk_size=chunk_size)
    Z_clamped = torch.clamp(Z_vec, min=1e-5)

    # Quotient rule adjoints
    dO_bar = grad_output / Z_clamped
    dZ_bar = -(grad_output * N).sum(dim=-1, keepdim=True) / (Z_clamped * Z_clamped)

    # Adjoint from N
    dpq_N, dpk_N, dv_N, dlg_N = _raw_chunkwise_backward(
        phi_q, phi_k, v, log_gamma, dO_bar, chunk_size=chunk_size
    )

    # Adjoint from Z
    dpq_Z, dpk_Z, _, dlg_Z = _raw_chunkwise_backward(
        phi_q, phi_k, v_one, log_gamma, dZ_bar, chunk_size=chunk_size
    )

    # Combine adjoints
    dphi_q = (dpq_N + dpq_Z) * scale
    dphi_k = dpk_N + dpk_Z
    dlog_gamma = dlg_N + dlg_Z

    dQ = dphi_q * feature_map_prime(q)
    dK = dphi_k * feature_map_prime(k)
    dV = dv_N

    if valid_mask is not None:
        vm_exp = valid_mask.expand(B, H, T, D)
        dQ = dQ * vm_exp
        dK = dK * vm_exp
        dV = dV * vm_exp

    dlog_gamma_flat = dlog_gamma.view(B, H, T)
    if orig_w.ndim == 0:
        dw = (dlog_gamma_flat * torch.sigmoid(-orig_w)).sum()
    elif orig_w.ndim == 1:
        if orig_w.shape[0] == H:
            dw = (dlog_gamma_flat * torch.sigmoid(-orig_w).view(1, H, 1)).sum(dim=(0, 2))
        else:
            dw = (dlog_gamma_flat * torch.sigmoid(-orig_w).view(1, 1, -1)).sum(dim=(0, 1))
    elif orig_w.ndim == 2:
        if orig_w.shape[1] == T:
            dw = (dlog_gamma_flat * torch.sigmoid(-orig_w).view(B, 1, T)).sum(dim=1)
        else:
            dw = (dlog_gamma_flat * torch.sigmoid(-orig_w).view(B, H, 1)).sum(dim=-1)
    elif orig_w.ndim == 3:
        dw = dlog_gamma_flat * torch.sigmoid(-orig_w)
    else:
        dw = dlog_gamma_flat.sum()

    return dQ, dK, dV, dw


class SubQAttentionFunction(torch.autograd.Function):
    """
    Exact analytical autograd bridge for Sub-Quadratic Associative Linear Attention.
    Implements Flash Linear Attention with vectorized chunkwise operations, running normalizer Z_t,
    padding mask protection, bidirectional support, and O(1) intermediate boundary state checkpoints.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_w: torch.Tensor,
        chunk_size: int = 64,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"Shape mismatch: Q:{q.shape}, K:{k.shape}, V:{v.shape}")

        out, _ = chunkwise_flash_linear_attention(
            q, k, v, decay_w, chunk_size=chunk_size, attention_mask=attention_mask, is_causal=is_causal
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            decay_w if isinstance(decay_w, torch.Tensor) else torch.tensor(decay_w, dtype=q.dtype, device=q.device),
        )
        ctx.chunk_size = chunk_size
        ctx.is_3d = (q.ndim == 3)
        ctx.is_causal = is_causal
        if attention_mask is not None:
            ctx.valid_mask = _normalize_attention_mask(attention_mask, q.dtype, q.shape[-2 if q.ndim >= 3 else 1])
        else:
            ctx.valid_mask = None
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v, decay_w = ctx.saved_tensors
        chunk_size = getattr(ctx, "chunk_size", 64)
        is_3d = getattr(ctx, "is_3d", False)
        is_causal = getattr(ctx, "is_causal", True)
        valid_mask = getattr(ctx, "valid_mask", None)

        if is_3d:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            grad_output = grad_output.unsqueeze(1)

        if is_causal:
            dQ, dK, dV, dw = _single_direction_causal_backward(
                q, k, v, decay_w, grad_output, chunk_size=chunk_size, valid_mask=valid_mask
            )
        else:
            # Full Multi-Directional Gradient Decomposition (is_causal=False)
            # 1. Forward causal adjoint scan (past-to-present)
            dQ_fwd, dK_fwd, dV_fwd, dw_fwd = _single_direction_causal_backward(
                q, k, v, decay_w, grad_output, chunk_size=chunk_size, valid_mask=valid_mask
            )

            # 2. Reverse causal adjoint scan (future-to-past)
            q_rev = torch.flip(q, dims=[2])
            k_rev = torch.flip(k, dims=[2])
            v_rev = torch.flip(v, dims=[2])
            decay_w_rev = (
                torch.flip(decay_w, dims=[-1])
                if isinstance(decay_w, torch.Tensor) and decay_w.ndim >= 2
                else decay_w
            )
            vm_rev = torch.flip(valid_mask, dims=[2]) if valid_mask is not None else None
            dO_rev = torch.flip(grad_output, dims=[2])

            dQ_bwd_raw, dK_bwd_raw, dV_bwd_raw, dw_bwd_raw = _single_direction_causal_backward(
                q_rev, k_rev, v_rev, decay_w_rev, dO_rev, chunk_size=chunk_size, valid_mask=vm_rev
            )
            dQ_bwd = torch.flip(dQ_bwd_raw, dims=[2])
            dK_bwd = torch.flip(dK_bwd_raw, dims=[2])
            dV_bwd = torch.flip(dV_bwd_raw, dims=[2])
            dw_bwd = (
                torch.flip(dw_bwd_raw, dims=[-1])
                if isinstance(decay_w, torch.Tensor) and decay_w.ndim >= 2
                else dw_bwd_raw
            )

            # 3. In normalized linear attention, diagonal self-term is simply v, so dV_diag = grad_output
            dO_masked = grad_output
            if valid_mask is not None:
                dO_masked = dO_masked * valid_mask.expand_as(dO_masked)

            dQ = dQ_fwd + dQ_bwd
            dK = dK_fwd + dK_bwd
            dV = dV_fwd + dV_bwd - dO_masked
            dw = dw_fwd + dw_bwd

        if is_3d:
            dQ = dQ.squeeze(1)
            dK = dK.squeeze(1)
            dV = dV.squeeze(1)

        grad_q = dQ.contiguous()
        grad_k = dK.contiguous()
        grad_v = dV.contiguous()
        grad_w = dw.contiguous()

        grads = (grad_q, grad_k, grad_v, grad_w)
        if len(ctx.needs_input_grad) > 4:
            return grads + (None,) * (len(ctx.needs_input_grad) - 4)
        return grads


class SubQLinearAttention(nn.Module):
    """
    Sub-Quadratic Associative Linear Attention layer.
    Features:
    - Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) support
    - Causal decoding and bidirectional encoder scanning modes
    - D^{-0.5} scaling & running denominator normalizer Z_t
    - Per-head RMSNorm stabilization
    - Rotary Position Embedding (RoPE) support
    - Padding mask state protection (attention_mask)
    - Learnable base temporal decay with optional dynamic input-dependent gating
    - Pre-LayerNorm and Hugging Face Transformer compatibility
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        num_key_value_heads: Optional[int] = None,
        decay_init: float = 0.99,
        bias: bool = False,
        use_data_dependent_gating: bool = True,
        is_causal: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.use_data_dependent_gating = use_data_dependent_gating
        self.is_causal = is_causal

        if self.head_dim * num_heads != hidden_size:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_heads {num_heads} must be divisible by num_key_value_heads {self.num_key_value_heads}")

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Per-head RMSNorm stabilization to maintain unit-scale activations
        if hasattr(nn, "RMSNorm"):
            self.head_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        else:
            self.head_norm = None

        decay_init_clamped = min(max(decay_init, 1e-4), 1.0 - 1e-4)
        init_w = math.log(decay_init_clamped / (1.0 - decay_init_clamped))
        self.raw_decay = nn.Parameter(torch.full((num_heads,), init_w, dtype=torch.float32))

        if self.use_data_dependent_gating:
            self.gate_proj = nn.Linear(hidden_size, num_heads, bias=bias)
        else:
            self.gate_proj = None

    @property
    def decay_factor(self) -> torch.Tensor:
        """Returns bounded base decay factor gamma = sigmoid(raw_decay) in (0, 1)."""
        return torch.sigmoid(self.raw_decay)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = hidden_states.shape

        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        K = self.k_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, KV_H, L, D]
        V = self.v_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, KV_H, L, D]

        # Apply Rotary Position Embeddings (RoPE) if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_key_value_groups > 1:
            K = K.repeat_interleave(self.num_key_value_groups, dim=1)  # [B, H, L, D]
            V = V.repeat_interleave(self.num_key_value_groups, dim=1)  # [B, H, L, D]

        if self.use_data_dependent_gating and self.gate_proj is not None:
            gate = self.gate_proj(hidden_states).transpose(1, 2)  # [B, H, L]
            decay_w = self.raw_decay.view(1, self.num_heads, 1) + gate  # [B, H, L]
        else:
            decay_w = self.raw_decay

        out, next_state = chunkwise_flash_linear_attention(
            Q, K, V, decay_w,
            past_state=past_state,
            attention_mask=attention_mask,
            is_causal=self.is_causal,
        )

        # Apply per-head RMSNorm
        if self.head_norm is not None:
            out = self.head_norm(out)
        else:
            out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-5)

        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        projected = self.out_proj(out)

        return projected, (next_state if use_cache else None)


# Backward compatibility alias
SubQuadraticAttention = SubQLinearAttention

# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

import math
import torch
import torch.nn.functional as F


def feature_map(x):
    # phi(x) = ELU(x) + 1
    return F.elu(x) + 1.0


def linear_attention_forward(Q, K, V, normalize: bool = False):
    """
    Reference PyTorch implementation for Sub-Quadratic Linear Attention.
    When normalize=False: O_t = phi(Q_t) * S_t (unnormalized, matching raw C++ engine)
    When normalize=True:  O_t = (phi(Q_t)/sqrt(D) * S_t) / (phi(Q_t)/sqrt(D) * Z_t) (normalized)
    S_t = S_{t-1} + phi(K_t)^T * V_t
    Z_t = Z_{t-1} + phi(K_t)

    Q, K, V: [batch, seq_len, d_model]
    Returns: O: [batch, seq_len, d_model]
    """
    batch, seq_len, d_model = Q.shape
    if normalize:
        scale = 1.0 / math.sqrt(d_model)
        phi_Q = feature_map(Q) * scale
    else:
        phi_Q = feature_map(Q)
    phi_K = feature_map(K)

    O = torch.zeros_like(Q)
    S = torch.zeros((batch, d_model, d_model), device=Q.device, dtype=Q.dtype)
    Z = torch.zeros((batch, d_model), device=Q.device, dtype=Q.dtype)

    for t in range(seq_len):
        q_t = phi_Q[:, t, :].unsqueeze(1)  # [batch, 1, d_model]
        k_t = phi_K[:, t, :].unsqueeze(2)  # [batch, d_model, 1]
        v_t = V[:, t, :].unsqueeze(1)      # [batch, 1, d_model]

        S = S + torch.bmm(k_t, v_t)
        if normalize:
            Z = Z + phi_K[:, t, :]
            num_t = torch.bmm(q_t, S).squeeze(1)
            denom_t = (phi_Q[:, t, :] * Z).sum(dim=-1, keepdim=True).clamp(min=1e-5)
            O[:, t, :] = num_t / denom_t
        else:
            o_t = torch.bmm(q_t, S)
            O[:, t, :] = o_t.squeeze(1)

    return O

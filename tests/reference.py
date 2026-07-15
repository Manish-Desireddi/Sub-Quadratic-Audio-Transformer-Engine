# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import torch
import torch.nn.functional as F


def feature_map(x):
    # phi(x) = ELU(x) + 1
    return F.elu(x) + 1.0


def linear_attention_forward(Q, K, V):
    """
    Reference PyTorch implementation for Sub-Quadratic Linear Attention.
    O_t = phi(Q_t) * S_t
    S_t = S_{t-1} + phi(K_t)^T * V_t

    Q, K, V: [batch, seq_len, d_model]
    Returns: O: [batch, seq_len, d_model]
    """
    phi_Q = feature_map(Q)
    phi_K = feature_map(K)

    batch, seq_len, d_model = Q.shape
    O = torch.zeros_like(Q)

    # Causal linear attention:
    # S_t = S_{t-1} + phi(K_t)^T * V_t
    # O_t = phi(Q_t) * S_t
    S = torch.zeros((batch, d_model, d_model), device=Q.device, dtype=Q.dtype)

    for t in range(seq_len):
        q_t = phi_Q[:, t, :].unsqueeze(1)  # [batch, 1, d_model]
        k_t = phi_K[:, t, :].unsqueeze(2)  # [batch, d_model, 1]
        v_t = V[:, t, :].unsqueeze(1)  # [batch, 1, d_model]

        # Outer product K_t^T * V_t -> [batch, d_model, d_model]
        S = S + torch.bmm(k_t, v_t)

        # O_t = Q_t * S -> [batch, 1, d_model]
        o_t = torch.bmm(q_t, S)
        O[:, t, :] = o_t.squeeze(1)

    return O

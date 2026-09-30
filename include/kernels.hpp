/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>

namespace subq {

template <typename T>
struct AttentionBackwardContext {
    const T* __restrict__ Q;             // [B, L, D] or [B, H, L, D]
    const T* __restrict__ K;             // [B, L, D] or [B, H, L, D]
    const T* __restrict__ V;             // [B, L, D] or [B, H, L, D]
    const T* __restrict__ dO;            // [B, L, D] or [B, H, L, D] (incoming grad)
    const T* __restrict__ S;             // Optional precomputed forward states [B, H, L, D, D] (nullptr if on-the-fly)
    T* __restrict__ dQ;                  // [B, L, D] or [B, H, L, D] (output grad wrt Q)
    T* __restrict__ dK;                  // [B, L, D] or [B, H, L, D] (output grad wrt K)
    T* __restrict__ dV;                  // [B, L, D] or [B, H, L, D] (output grad wrt V)
    T raw_decay_w;                       // Learnable unconstrained decay parameter w where gamma = sigmoid(w)
    T* __restrict__ d_raw_decay_w;       // Output grad wrt raw parameter w
    T* __restrict__ d_gamma;             // Output grad wrt gamma
    size_t batch_size;                   // B
    size_t num_heads;                    // H (default 1 if 3D)
    size_t seq_len;                      // L or T
    size_t head_dim;                     // D
};

template <typename T>
void subquadratic_attention_backward(AttentionBackwardContext<T>& ctx);

template <typename T>
void run_attention_backward_raw(AttentionBackwardContext<T>& ctx);

template <typename T>
void run_attention_forward_raw(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
    T* __restrict__ O, T* __restrict__ S,
    size_t batch, size_t num_heads, size_t seq_len, size_t head_dim, T gamma
);

} // namespace subq

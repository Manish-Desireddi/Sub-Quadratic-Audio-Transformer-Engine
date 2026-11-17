/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once

#include "tensor.hpp"
#include <cmath>
#include "gpu_macros.hpp"

struct AttentionContext {
    TensorDescriptor Q; // Query
    TensorDescriptor K; // Key
    TensorDescriptor V; // Value
    TensorDescriptor O; // Output
    TensorDescriptor S; // Recurrent State
    float decay_factor = 0.99f; // Temporal decay for the associative scan
};

void run_feature_map(TensorDescriptor& tensor);
void run_attention_forward(AttentionContext& ctx);
// Deprecated: reads dO from ctx.O and overwrites ctx.Q/K/V with dQ/dK/dV; assumes zero initial state.
void run_attention_backward(AttentionContext& ctx);

// Non-destructive backward (CPU). dO is read from ctx.O; gradients go to the separate dQ/dK/dV
// buffers ([batch, seq_len, d_model]), which must not alias the inputs (throws std::invalid_argument).
// S0 is the initial recurrence state [batch, d_model, d_model] used by the forward pass (nullptr = zeros).
// d_gamma, if non-null, receives dL/d(decay_factor). GPU builds throw std::runtime_error.
void run_attention_backward_ex(const AttentionContext& ctx, float* dQ, float* dK, float* dV,
                               float* d_gamma = nullptr, const float* S0 = nullptr);

/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include "tensor.hpp"

// Computes Layer Normalization on the last dimension of the tensor
// y = (x - mean) / sqrt(variance + eps) * gamma + beta
void run_layer_norm(TensorDescriptor& x, const TensorDescriptor& gamma, const TensorDescriptor& beta, float eps = 1e-5f);

// Computes a 2-layer MLP (Feed Forward Network) with SiLU activation
// hidden = SiLU(x * W1 + b1)
// out = hidden * W2 + b2
void run_ffn(const TensorDescriptor& x, TensorDescriptor& out, TensorDescriptor& hidden_buf, 
             const TensorDescriptor& W1, const TensorDescriptor& b1, 
             const TensorDescriptor& W2, const TensorDescriptor& b2);

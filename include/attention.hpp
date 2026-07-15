/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include "tensor.hpp"
#include "gpu_macros.hpp"

struct AttentionContext {
    TensorDescriptor Q; // Query
    TensorDescriptor K; // Key
    TensorDescriptor V; // Value
    TensorDescriptor O; // Output
    TensorDescriptor S; // Recurrent State
};

void run_feature_map(TensorDescriptor& tensor);
void run_attention_forward(AttentionContext& ctx);
void run_attention_backward(AttentionContext& ctx);

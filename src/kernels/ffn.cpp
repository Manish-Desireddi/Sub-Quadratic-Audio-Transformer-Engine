/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include "layers.hpp"
#include <cmath>

// SiLU activation function: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void run_ffn(const TensorDescriptor& x, TensorDescriptor& out, TensorDescriptor& hidden_buf, 
             const TensorDescriptor& W1, const TensorDescriptor& b1, 
             const TensorDescriptor& W2, const TensorDescriptor& b2) {
    // Basic CPU implementation of Linear -> SiLU -> Linear
    size_t batch = x.dimensions[0];
    size_t seq_len = x.dimensions[1];
    size_t d_model = x.dimensions[2];
    size_t hidden_dim = hidden_buf.dimensions[2];

    const float* x_ptr = static_cast<const float*>(x.data);
    float* h_ptr = static_cast<float*>(hidden_buf.data);
    float* out_ptr = static_cast<float*>(out.data);

    const float* w1_ptr = static_cast<const float*>(W1.data);
    const float* b1_ptr = static_cast<const float*>(b1.data);
    const float* w2_ptr = static_cast<const float*>(W2.data);
    const float* b2_ptr = static_cast<const float*>(b2.data);

    size_t tokens = batch * seq_len;

    // Linear 1 + SiLU
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < hidden_dim; ++h) {
            float val = b1_ptr[h];
            for (size_t d = 0; d < d_model; ++d) {
                val += x_ptr[t * d_model + d] * w1_ptr[d * hidden_dim + h];
            }
            h_ptr[t * hidden_dim + h] = silu(val);
        }
    }

    // Linear 2
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t d = 0; d < d_model; ++d) {
            float val = b2_ptr[d];
            for (size_t h = 0; h < hidden_dim; ++h) {
                val += h_ptr[t * hidden_dim + h] * w2_ptr[h * d_model + d];
            }
            out_ptr[t * d_model + d] = val;
        }
    }
}

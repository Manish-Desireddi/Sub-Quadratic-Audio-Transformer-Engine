/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include "layers.hpp"
#include <cmath>
#include <stdexcept>

void run_layer_norm(TensorDescriptor& x, const TensorDescriptor& gamma, const TensorDescriptor& beta, float eps) {
    if (x.dimensions.empty()) return;
    
    // CPU fallback implementation
    size_t batch = x.dimensions[0];
    size_t seq_len = x.dimensions[1];
    size_t d_model = x.dimensions[2];

    float* x_ptr = static_cast<float*>(x.data);
    const float* g_ptr = static_cast<const float*>(gamma.data);
    const float* b_ptr = static_cast<const float*>(beta.data);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t offset = b * seq_len * d_model + s * d_model;
            
            // Calculate Mean
            float sum = 0.0f;
            for (size_t d = 0; d < d_model; ++d) {
                sum += x_ptr[offset + d];
            }
            float mean = sum / d_model;

            // Calculate Variance
            float var_sum = 0.0f;
            for (size_t d = 0; d < d_model; ++d) {
                float diff = x_ptr[offset + d] - mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / d_model;

            // Normalize and apply scale/shift
            float inv_std = 1.0f / std::sqrt(variance + eps);
            for (size_t d = 0; d < d_model; ++d) {
                x_ptr[offset + d] = (x_ptr[offset + d] - mean) * inv_std * g_ptr[d] + b_ptr[d];
            }
        }
    }
}

/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include "attention.hpp"
#include "gpu_macros.hpp"

#if defined(USE_CUDA) || defined(USE_HIP)
__global__ void causal_linear_attention_kernel(
    const float* Q, const float* K, const float* V, float* O, float* S,
    int batch, int seq_len, int d_model
) {
    int b = blockIdx.x;
    int j = threadIdx.x; // Handles the j-th dimension of d_model

    if (j < d_model) {
        int s_offset = b * d_model * d_model;
        int qkv_offset_base = b * seq_len * d_model;

        for (int t = 0; t < seq_len; ++t) {
            int qkv_offset = qkv_offset_base + t * d_model;
            
            float v_val = V[qkv_offset + j];
            
            // Update state S[i, j] += K[i] * V[j]
            for (int i = 0; i < d_model; ++i) {
                float k_val = K[qkv_offset + i];
                S[s_offset + i * d_model + j] += k_val * v_val;
            }
            
            // Compute O[j] = sum_i Q[i] * S[i, j]
            float o_val = 0.0f;
            for (int i = 0; i < d_model; ++i) {
                float q_val = Q[qkv_offset + i];
                o_val += q_val * S[s_offset + i * d_model + j];
            }
            O[qkv_offset + j] = o_val;
        }
    }
}
#endif

void run_attention_forward(AttentionContext& ctx) {
    int batch = ctx.Q.dimensions[0];
    int seq_len = ctx.Q.dimensions[1];
    int d_model = ctx.Q.dimensions[2];

#if defined(USE_CUDA) || defined(USE_HIP)
    int threads = d_model; // Assuming d_model <= 1024, e.g., 256
    int blocks = batch;
    
    float* Q = static_cast<float*>(ctx.Q.data);
    float* K = static_cast<float*>(ctx.K.data);
    float* V = static_cast<float*>(ctx.V.data);
    float* O = static_cast<float*>(ctx.O.data);
    float* S = static_cast<float*>(ctx.S.data);
    
    causal_linear_attention_kernel<<<blocks, threads>>>(Q, K, V, O, S, batch, seq_len, d_model);
    CHECK_GPU_ERROR(gpuGetLastError());
#else
    float* Q = static_cast<float*>(ctx.Q.data);
    float* K = static_cast<float*>(ctx.K.data);
    float* V = static_cast<float*>(ctx.V.data);
    float* O = static_cast<float*>(ctx.O.data);
    float* S = static_cast<float*>(ctx.S.data);
    
    for (int b = 0; b < batch; ++b) {
        int s_offset = b * d_model * d_model;
        int qkv_offset_base = b * seq_len * d_model;
        
        for (int t = 0; t < seq_len; ++t) {
            int qkv_offset = qkv_offset_base + t * d_model;
            
            // Update S
            for (int i = 0; i < d_model; ++i) {
                float k_val = K[qkv_offset + i];
                for (int j = 0; j < d_model; ++j) {
                    float v_val = V[qkv_offset + j];
                    S[s_offset + i * d_model + j] += k_val * v_val;
                }
            }
            
            // Compute O
            for (int j = 0; j < d_model; ++j) {
                float o_val = 0.0f;
                for (int i = 0; i < d_model; ++i) {
                    float q_val = Q[qkv_offset + i];
                    float s_val = S[s_offset + i * d_model + j];
                    o_val += q_val * s_val;
                }
                O[qkv_offset + j] = o_val;
            }
        }
    }
#endif
}

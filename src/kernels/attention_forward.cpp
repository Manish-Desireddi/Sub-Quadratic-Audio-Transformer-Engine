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
    int batch, int seq_len, int d_model, float decay_factor
) {
    int b = blockIdx.x;
    int j = threadIdx.x; // Handles the j-th dimension of d_model

    extern __shared__ float shared_mem[];
    float* shared_k = shared_mem;
    float* shared_q = shared_mem + d_model;

    if (j < d_model) {
        int s_offset = b * d_model * d_model;
        int qkv_offset_base = b * seq_len * d_model;

        for (int t = 0; t < seq_len; ++t) {
            int qkv_offset = qkv_offset_base + t * d_model;
            
            // Load K and Q into shared memory and apply ELU inline
            float k_val = K[qkv_offset + j];
            shared_k[j] = (k_val > 0.0f) ? (k_val + 1.0f) : expf(k_val);

            float q_val = Q[qkv_offset + j];
            shared_q[j] = (q_val > 0.0f) ? (q_val + 1.0f) : expf(q_val);

            __syncthreads();

            float v_val = V[qkv_offset + j];
            
            // Update state S[i, j] = decay_factor * S[i, j] + K[i] * V[j]
            for (int i = 0; i < d_model; ++i) {
                float k_mapped = shared_k[i];
                float new_s = decay_factor * S[s_offset + i * d_model + j] + k_mapped * v_val;
                // Explicitly flush subnormals in case FFI resets hardware CSR
                if (new_s > -1e-15f && new_s < 1e-15f) new_s = 0.0f;
                S[s_offset + i * d_model + j] = new_s;
            }
            
            // Compute O[j] = sum_i Q[i] * S[i, j]
            float o_val = 0.0f;
            for (int i = 0; i < d_model; ++i) {
                float q_mapped = shared_q[i];
                o_val += q_mapped * S[s_offset + i * d_model + j];
            }
            O[qkv_offset + j] = o_val;

            __syncthreads();
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
    void* S_ptr = ctx.S.data;
    
    float decay_factor = ctx.decay_factor;
    int shared_mem_size = 2 * d_model * sizeof(float);
    
    // In a full implementation, we'd dispatch a separate __nv_bfloat16 kernel here.
    // For now, we fallback to FP32 if the hardware doesn't support it or if it's the CPU fallback.
    float* S = static_cast<float*>(S_ptr);
    causal_linear_attention_kernel<<<blocks, threads, shared_mem_size>>>(Q, K, V, O, S, batch, seq_len, d_model, decay_factor);
    CHECK_GPU_ERROR(gpuGetLastError());
#else
    float* Q = static_cast<float*>(ctx.Q.data);
    float* K = static_cast<float*>(ctx.K.data);
    float* V = static_cast<float*>(ctx.V.data);
    float* O = static_cast<float*>(ctx.O.data);
    float* S = static_cast<float*>(ctx.S.data);
    float decay_factor = ctx.decay_factor;
    
    // Chunkwise Parallel Associative Scan (Flash Linear Attention style)
    // Tiled into blocks of tokens to convert sequential O(T) recurrence into intra-chunk GEMM + boundary recurrence
    constexpr int CHUNK_SIZE = 64;

    for (int b = 0; b < batch; ++b) {
        int s_offset = b * d_model * d_model;
        int qkv_offset_base = b * seq_len * d_model;
        float* S_curr = &S[s_offset];

        for (int c_start = 0; c_start < seq_len; c_start += CHUNK_SIZE) {
            int c_len = std::min(CHUNK_SIZE, seq_len - c_start);

            // 1. Pre-calculate mapped features for the chunk to enable vectorized block access
            std::vector<float> q_chunk(c_len * d_model);
            std::vector<float> k_chunk(c_len * d_model);

            for (int t = 0; t < c_len; ++t) {
                int qkv_offset = qkv_offset_base + (c_start + t) * d_model;
                for (int i = 0; i < d_model; ++i) {
                    float q_val = Q[qkv_offset + i];
                    q_chunk[t * d_model + i] = (q_val > 0.0f) ? (q_val + 1.0f) : std::exp(q_val);
                    float k_val = K[qkv_offset + i];
                    k_chunk[t * d_model + i] = (k_val > 0.0f) ? (k_val + 1.0f) : std::exp(k_val);
                }
            }

            // 2. Intra-chunk (parallel GEMM) and Inter-chunk attention
            for (int t = 0; t < c_len; ++t) {
                int out_offset = qkv_offset_base + (c_start + t) * d_model;
                const float* q_t = &q_chunk[t * d_model];

                // Inter-chunk output: O_inter = phi(Q_t) * (gamma^(t+1) * S_prev)
                float decay_q = std::pow(decay_factor, static_cast<float>(t + 1));
                for (int j = 0; j < d_model; ++j) {
                    float o_inter = 0.0f;
                    for (int i = 0; i < d_model; ++i) {
                        o_inter += q_t[i] * S_curr[i * d_model + j];
                    }
                    O[out_offset + j] = o_inter * decay_q;
                }

                // Intra-chunk output: O_intra = sum_{tau=0}^t (phi(Q_t) . phi(K_tau)) * gamma^(t - tau) * V_tau
                for (int tau = 0; tau <= t; ++tau) {
                    const float* k_tau = &k_chunk[tau * d_model];
                    int v_offset = qkv_offset_base + (c_start + tau) * d_model;
                    float dot_qk = 0.0f;
                    for (int i = 0; i < d_model; ++i) {
                        dot_qk += q_t[i] * k_tau[i];
                    }
                    float weight = dot_qk * std::pow(decay_factor, static_cast<float>(t - tau));
                    for (int j = 0; j < d_model; ++j) {
                        O[out_offset + j] += weight * V[v_offset + j];
                    }
                }
            }

            // 3. Recurrent state update across chunk boundary:
            // S_next = gamma^c_len * S_prev + sum_{tau=0}^{c_len-1} gamma^(c_len - 1 - tau) * phi(K_tau)^T * V_tau
            float chunk_decay = std::pow(decay_factor, static_cast<float>(c_len));
            for (int i = 0; i < d_model; ++i) {
                for (int j = 0; j < d_model; ++j) {
                    float new_s = chunk_decay * S_curr[i * d_model + j];
                    for (int tau = 0; tau < c_len; ++tau) {
                        float k_val = k_chunk[tau * d_model + i];
                        int v_offset = qkv_offset_base + (c_start + tau) * d_model;
                        float v_val = V[v_offset + j];
                        float decay_step = std::pow(decay_factor, static_cast<float>(c_len - 1 - tau));
                        new_s += decay_step * k_val * v_val;
                    }
                    if (new_s > -1e-15f && new_s < 1e-15f) new_s = 0.0f;
                    S_curr[i * d_model + j] = new_s;
                }
            }
        }
    }
#endif
}

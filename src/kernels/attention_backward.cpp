/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include "attention.hpp"
#include "kernels.hpp"
#include "gpu_macros.hpp"
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace subq {

namespace {

// Inline ELU+1 feature map: phi(x) = ELU(x) + 1
template <typename T>
inline T phi_val(T x) {
    return (x > static_cast<T>(0)) ? (x + static_cast<T>(1)) : std::exp(x);
}

// Inline derivative of ELU+1 feature map: phi'(x) = 1 if x > 0 else exp(x)
template <typename T>
inline T phi_prime(T x) {
    return (x > static_cast<T>(0)) ? static_cast<T>(1) : std::exp(x);
}

// Logistic sigmoid
template <typename T>
inline T sigmoid_val(T w) {
    if (w > static_cast<T>(40)) return static_cast<T>(1);
    if (w < static_cast<T>(-40)) return static_cast<T>(0);
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-w));
}

} // anonymous namespace

template <typename T>
void subquadratic_attention_backward(AttentionBackwardContext<T>& ctx) {
    const size_t B = ctx.batch_size;
    const size_t H = (ctx.num_heads > 0) ? ctx.num_heads : 1;
    const size_t T_len = ctx.seq_len;
    const size_t D = ctx.head_dim;

    // Runtime safety checks against pointer aliasing
    if (ctx.dQ == ctx.Q || ctx.dQ == ctx.K || ctx.dQ == ctx.V ||
        ctx.dK == ctx.Q || ctx.dK == ctx.K || ctx.dK == ctx.V ||
        ctx.dV == ctx.Q || ctx.dV == ctx.K || ctx.dV == ctx.V) {
        throw std::invalid_argument("subquadratic_attention_backward: gradient outputs must not alias inputs");
    }

    const T w = ctx.raw_decay_w;
    const T gamma = sigmoid_val(w);

    // Zero-initialize output gradients
    const size_t total_qkv = B * H * T_len * D;
    std::memset(ctx.dQ, 0, total_qkv * sizeof(T));
    std::memset(ctx.dK, 0, total_qkv * sizeof(T));
    std::memset(ctx.dV, 0, total_qkv * sizeof(T));

    T total_d_gamma = static_cast<T>(0);

    // Chunkwise on-the-fly recomputation configuration
    constexpr size_t CHUNK_SIZE = 64;
    const size_t num_chunks = (T_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // State boundaries across chunks: only (num_chunks + 1) * D * D elements (O(T / C))
    std::vector<T> S_boundaries((num_chunks + 1) * D * D, static_cast<T>(0));
    // Fixed-size chunk state scratch: strictly O(1) memory bounded by C * D * D
    std::vector<T> S_chunk(CHUNK_SIZE * D * D, static_cast<T>(0));
    std::vector<T> S_curr(D * D, static_cast<T>(0));
    std::vector<T> Gamma(D * D, static_cast<T>(0));
    std::vector<T> q_mapped(D);
    std::vector<T> q_prime_vec(D);
    std::vector<T> k_mapped(D);
    std::vector<T> k_prime_vec(D);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t seq_offset = (b * H + h) * T_len * D;

            // =================================================================
            // PASS 1: Forward State Recurrence at chunk boundaries only
            // =================================================================
            if (ctx.S0 != nullptr) {
                std::memcpy(S_curr.data(), &ctx.S0[(b * H + h) * D * D], D * D * sizeof(T));
            } else {
                std::fill(S_curr.begin(), S_curr.end(), static_cast<T>(0));
            }
            std::fill(S_boundaries.begin(), S_boundaries.end(), static_cast<T>(0));

            for (size_t c = 0; c < num_chunks; ++c) {
                const size_t c_start = c * CHUNK_SIZE;
                const size_t c_end = std::min(c_start + CHUNK_SIZE, T_len);

                // Save chunk boundary state
                std::memcpy(&S_boundaries[c * D * D], S_curr.data(), D * D * sizeof(T));

                for (size_t t = c_start; t < c_end; ++t) {
                    const size_t qkv_offset = seq_offset + t * D;
                    for (size_t i = 0; i < D; ++i) {
                        k_mapped[i] = phi_val(ctx.K[qkv_offset + i]);
                    }

                    for (size_t i = 0; i < D; ++i) {
                        const T k_i = k_mapped[i];
                        for (size_t j = 0; j < D; ++j) {
                            const T prev_s = S_curr[i * D + j];
                            const T v_j = ctx.V[qkv_offset + j];
                            T new_s = gamma * prev_s + k_i * v_j;
                            if (subq::is_subnormal_or_zero(new_s)) {
                                new_s = static_cast<T>(0);
                            }
                            S_curr[i * D + j] = new_s;
                        }
                    }
                }
            }
            std::memcpy(&S_boundaries[num_chunks * D * D], S_curr.data(), D * D * sizeof(T));

            // =================================================================
            // PASS 2: Reverse Adjoint Scan chunk-by-chunk with on-the-fly recomputation
            // =================================================================
            std::fill(Gamma.begin(), Gamma.end(), static_cast<T>(0));

            for (int64_t c = static_cast<int64_t>(num_chunks) - 1; c >= 0; --c) {
                const size_t c_start = static_cast<size_t>(c) * CHUNK_SIZE;
                const size_t c_end = std::min(c_start + CHUNK_SIZE, T_len);
                const size_t c_len = c_end - c_start;

                // Recompute forward states for chunk c on-the-fly starting from S_boundaries[c]
                const T* S_start_c = &S_boundaries[static_cast<size_t>(c) * D * D];
                std::memcpy(S_curr.data(), S_start_c, D * D * sizeof(T));

                for (size_t step = 0; step < c_len; ++step) {
                    const size_t t = c_start + step;
                    const size_t qkv_offset = seq_offset + t * D;
                    const size_t chunk_s_offset = step * D * D;

                    for (size_t i = 0; i < D; ++i) {
                        k_mapped[i] = phi_val(ctx.K[qkv_offset + i]);
                    }

                    for (size_t i = 0; i < D; ++i) {
                        const T k_i = k_mapped[i];
                        for (size_t j = 0; j < D; ++j) {
                            const T prev_s = S_curr[i * D + j];
                            const T v_j = ctx.V[qkv_offset + j];
                            T new_s = gamma * prev_s + k_i * v_j;
                            if (subq::is_subnormal_or_zero(new_s)) {
                                new_s = static_cast<T>(0);
                            }
                            S_curr[i * D + j] = new_s;
                            S_chunk[chunk_s_offset + i * D + j] = new_s;
                        }
                    }
                }

                // Reverse Adjoint step inside chunk c
                for (int64_t step = static_cast<int64_t>(c_len) - 1; step >= 0; --step) {
                    const size_t t = c_start + static_cast<size_t>(step);
                    const size_t qkv_offset = seq_offset + t * D;
                    const size_t s_curr_offset = static_cast<size_t>(step) * D * D;
                    const T* S_prev = (step > 0) ? &S_chunk[(static_cast<size_t>(step) - 1) * D * D] : S_start_c;

                    // 1. Evaluate feature maps and derivatives
                    for (size_t i = 0; i < D; ++i) {
                        const T q_val = ctx.Q[qkv_offset + i];
                        q_mapped[i] = phi_val(q_val);
                        q_prime_vec[i] = phi_prime(q_val);

                        const T k_val = ctx.K[qkv_offset + i];
                        k_mapped[i] = phi_val(k_val);
                        k_prime_vec[i] = phi_prime(k_val);
                    }

                    // 2. Update Adjoint State: Gamma_t = gamma * Gamma_{t+1} + phi(Q_t)^T * dO_t
                    for (size_t i = 0; i < D; ++i) {
                        const T q_i = q_mapped[i];
                        for (size_t j = 0; j < D; ++j) {
                            const T do_j = ctx.dO[qkv_offset + j];
                            Gamma[i * D + j] = gamma * Gamma[i * D + j] + q_i * do_j;
                        }
                    }

                    // 3. Compute dQ_t = (dO_t * S_t^T) * phi'(Q_t)
                    for (size_t i = 0; i < D; ++i) {
                        T sum_do_s = static_cast<T>(0);
                        for (size_t j = 0; j < D; ++j) {
                            const T do_j = ctx.dO[qkv_offset + j];
                            const T s_val = S_chunk[s_curr_offset + i * D + j];
                            sum_do_s += do_j * s_val;
                        }
                        ctx.dQ[qkv_offset + i] = sum_do_s * q_prime_vec[i];
                    }

                    // 4. Compute dV_t = phi(K_t) * Gamma_t
                    for (size_t j = 0; j < D; ++j) {
                        T sum_k_gamma = static_cast<T>(0);
                        for (size_t i = 0; i < D; ++i) {
                            sum_k_gamma += k_mapped[i] * Gamma[i * D + j];
                        }
                        ctx.dV[qkv_offset + j] = sum_k_gamma;
                    }

                    // 5. Compute dK_t = (V_t * Gamma_t^T) * phi'(K_t)
                    for (size_t i = 0; i < D; ++i) {
                        T sum_v_gamma = static_cast<T>(0);
                        for (size_t j = 0; j < D; ++j) {
                            const T v_j = ctx.V[qkv_offset + j];
                            sum_v_gamma += Gamma[i * D + j] * v_j;
                        }
                        ctx.dK[qkv_offset + i] = sum_v_gamma * k_prime_vec[i];
                    }

                    // 6. Accumulate d_gamma += <Gamma_t, S_{t-1}>_F. At t = 0 S_{-1} is the initial
                    // state (zero when S0 is null, so this term vanishes in the default case).
                    for (size_t i = 0; i < D; ++i) {
                        for (size_t j = 0; j < D; ++j) {
                            total_d_gamma += Gamma[i * D + j] * S_prev[i * D + j];
                        }
                    }
                }
            }
        }
    }

    if (ctx.d_gamma) {
        *ctx.d_gamma = total_d_gamma;
    }
    if (ctx.d_raw_decay_w) {
        // dw = d_gamma * gamma * (1 - gamma)
        *ctx.d_raw_decay_w = total_d_gamma * gamma * (static_cast<T>(1) - gamma);
    }
}

template <typename T>
void run_attention_backward_raw(AttentionBackwardContext<T>& ctx) {
    subquadratic_attention_backward(ctx);
}

// Explicit template instantiations
template void subquadratic_attention_backward<float>(AttentionBackwardContext<float>& ctx);
template void subquadratic_attention_backward<double>(AttentionBackwardContext<double>& ctx);
template void run_attention_backward_raw<float>(AttentionBackwardContext<float>& ctx);
template void run_attention_backward_raw<double>(AttentionBackwardContext<double>& ctx);

#if defined(USE_CUDA) || defined(USE_HIP)
__global__ void causal_linear_attention_backward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int batch, int seq_len, int d_model, float decay_factor
) {
    int b = blockIdx.x;
    int j = threadIdx.x;
    if (j >= d_model) return;

    extern __shared__ float shared_mem[];
    float* shared_q = shared_mem;
    float* shared_k = shared_mem + d_model;
    float* shared_do = shared_mem + 2 * d_model;

    int qkv_base = b * seq_len * d_model;

    for (int t = 0; t < seq_len; ++t) {
        int qkv_offset = qkv_base + t * d_model;
        float q_val = Q[qkv_offset + j];
        float k_val = K[qkv_offset + j];
        float do_val = dO[qkv_offset + j];

        shared_q[j] = (q_val > 0.0f) ? (q_val + 1.0f) : expf(q_val);
        shared_k[j] = (k_val > 0.0f) ? (k_val + 1.0f) : expf(k_val);
        shared_do[j] = do_val;
        __syncthreads();

        float dphi_q_val = (q_val > 0.0f) ? 1.0f : expf(q_val);
        float dphi_k_val = (k_val > 0.0f) ? 1.0f : expf(k_val);

        dQ[qkv_offset + j] = shared_do[j] * dphi_q_val;
        dK[qkv_offset + j] = shared_do[j] * dphi_k_val;
        dV[qkv_offset + j] = shared_k[j] * shared_do[j];
        __syncthreads();
    }
}
#endif

} // namespace subq

void run_attention_backward(AttentionContext& ctx) {
    size_t batch = ctx.Q.dimensions[0];
    size_t seq_len = ctx.Q.dimensions[1];
    size_t d_model = ctx.Q.dimensions[2];

#if defined(USE_CUDA) || defined(USE_HIP)
    size_t total_elements = batch * seq_len * d_model;
    size_t total_bytes = total_elements * sizeof(float);

    float* dev_Q = static_cast<float*>(ctx.Q.data);
    float* dev_K = static_cast<float*>(ctx.K.data);
    float* dev_V = static_cast<float*>(ctx.V.data);
    float* dev_dO = static_cast<float*>(ctx.O.data);

    bool need_host_stage = ctx.Q.is_on_host;
    if (need_host_stage) {
        CHECK_GPU_ERROR(gpuMalloc(reinterpret_cast<void**>(&dev_Q), total_bytes));
        CHECK_GPU_ERROR(gpuMalloc(reinterpret_cast<void**>(&dev_K), total_bytes));
        CHECK_GPU_ERROR(gpuMalloc(reinterpret_cast<void**>(&dev_V), total_bytes));
        CHECK_GPU_ERROR(gpuMalloc(reinterpret_cast<void**>(&dev_dO), total_bytes));
        CHECK_GPU_ERROR(gpuMemcpy(dev_Q, ctx.Q.data, total_bytes, gpuMemcpyHostToDevice));
        CHECK_GPU_ERROR(gpuMemcpy(dev_K, ctx.K.data, total_bytes, gpuMemcpyHostToDevice));
        CHECK_GPU_ERROR(gpuMemcpy(dev_V, ctx.V.data, total_bytes, gpuMemcpyHostToDevice));
        CHECK_GPU_ERROR(gpuMemcpy(dev_dO, ctx.O.data, total_bytes, gpuMemcpyHostToDevice));
    }

    float* dQ = nullptr;
    float* dK = nullptr;
    float* dV = nullptr;

    CHECK_GPU_ERROR(gpuMalloc(reinterpret_cast<void**>(&dQ), total_bytes));
    CHECK_GPU_ERROR(gpuMalloc(reinterpret_cast<void**>(&dK), total_bytes));
    CHECK_GPU_ERROR(gpuMalloc(reinterpret_cast<void**>(&dV), total_bytes));

    int threads = static_cast<int>(d_model);
    int blocks = static_cast<int>(batch);
    int shared_mem_size = static_cast<int>(3 * d_model * sizeof(float));

    subq::causal_linear_attention_backward_kernel<<<blocks, threads, shared_mem_size>>>(
        dev_Q, dev_K, dev_V, dev_dO, dQ, dK, dV,
        static_cast<int>(batch), static_cast<int>(seq_len), static_cast<int>(d_model), ctx.decay_factor
    );
    CHECK_GPU_ERROR(gpuGetLastError());
    CHECK_GPU_ERROR(gpuDeviceSynchronize());

    if (need_host_stage) {
        CHECK_GPU_ERROR(gpuMemcpy(ctx.Q.data, dQ, total_bytes, gpuMemcpyDeviceToHost));
        CHECK_GPU_ERROR(gpuMemcpy(ctx.K.data, dK, total_bytes, gpuMemcpyDeviceToHost));
        CHECK_GPU_ERROR(gpuMemcpy(ctx.V.data, dV, total_bytes, gpuMemcpyDeviceToHost));
        CHECK_GPU_ERROR(gpuFree(dev_Q));
        CHECK_GPU_ERROR(gpuFree(dev_K));
        CHECK_GPU_ERROR(gpuFree(dev_V));
        CHECK_GPU_ERROR(gpuFree(dev_dO));
    } else {
        CHECK_GPU_ERROR(gpuMemcpy(ctx.Q.data, dQ, total_bytes, gpuMemcpyDeviceToDevice));
        CHECK_GPU_ERROR(gpuMemcpy(ctx.K.data, dK, total_bytes, gpuMemcpyDeviceToDevice));
        CHECK_GPU_ERROR(gpuMemcpy(ctx.V.data, dV, total_bytes, gpuMemcpyDeviceToDevice));
    }

    CHECK_GPU_ERROR(gpuFree(dQ));
    CHECK_GPU_ERROR(gpuFree(dK));
    CHECK_GPU_ERROR(gpuFree(dV));
#else
    // Deprecated in-place contract: dO is read from ctx.O and the gradients overwrite Q/K/V.
    // New code should call run_attention_backward_ex, which keeps inputs intact.
    // The initial recurrence state is assumed to be zero (matches the Python bindings).
    const size_t total = batch * seq_len * d_model;
    std::vector<float> dQ(total, 0.0f);
    std::vector<float> dK(total, 0.0f);
    std::vector<float> dV(total, 0.0f);

    run_attention_backward_ex(ctx, dQ.data(), dK.data(), dV.data(), nullptr, nullptr);

    std::memcpy(ctx.Q.data, dQ.data(), total * sizeof(float));
    std::memcpy(ctx.K.data, dK.data(), total * sizeof(float));
    std::memcpy(ctx.V.data, dV.data(), total * sizeof(float));
#endif
}

void run_attention_backward_ex(const AttentionContext& ctx, float* dQ, float* dK, float* dV,
                               float* d_gamma, const float* S0) {
#if defined(USE_CUDA) || defined(USE_HIP)
    (void)ctx; (void)dQ; (void)dK; (void)dV; (void)d_gamma; (void)S0;
    throw std::runtime_error("run_attention_backward_ex: GPU backward is not implemented");
#else
    const size_t batch = ctx.Q.dimensions[0];
    const size_t seq_len = ctx.Q.dimensions[1];
    const size_t d_model = ctx.Q.dimensions[2];

    // Aliasing is a hard error in all build types (previously assert-only, compiled out in Release).
    const void* inputs[] = {ctx.Q.data, ctx.K.data, ctx.V.data, ctx.O.data};
    for (const void* in : inputs) {
        if (in == dQ || in == dK || in == dV) {
            throw std::invalid_argument("run_attention_backward_ex: gradient outputs must not alias inputs");
        }
    }

    float d_gamma_local = 0.0f;
    float d_w = 0.0f;
    const float w_val = std::log(std::max(ctx.decay_factor, 1e-6f) / std::max(1.0f - ctx.decay_factor, 1e-6f));

    subq::AttentionBackwardContext<float> bwd_ctx;
    bwd_ctx.Q = static_cast<const float*>(ctx.Q.data);
    bwd_ctx.K = static_cast<const float*>(ctx.K.data);
    bwd_ctx.V = static_cast<const float*>(ctx.V.data);
    bwd_ctx.dO = static_cast<const float*>(ctx.O.data);  // incoming gradient travels in ctx.O
    bwd_ctx.S = nullptr;
    bwd_ctx.S0 = S0;
    bwd_ctx.dQ = dQ;
    bwd_ctx.dK = dK;
    bwd_ctx.dV = dV;
    bwd_ctx.raw_decay_w = w_val;
    bwd_ctx.d_raw_decay_w = &d_w;
    bwd_ctx.d_gamma = &d_gamma_local;
    bwd_ctx.batch_size = batch;
    bwd_ctx.num_heads = 1;
    bwd_ctx.seq_len = seq_len;
    bwd_ctx.head_dim = d_model;

    subq::subquadratic_attention_backward(bwd_ctx);

    if (d_gamma) {
        *d_gamma = d_gamma_local;
    }
#endif
}

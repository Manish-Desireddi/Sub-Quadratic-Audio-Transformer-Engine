/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include "attention.hpp"
#include "kernels.hpp"
#include "gpu_macros.hpp"
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cassert>

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

    // Safety checks against pointer aliasing
    assert(ctx.dQ != ctx.Q && ctx.dQ != ctx.K && ctx.dQ != ctx.V);
    assert(ctx.dK != ctx.Q && ctx.dK != ctx.K && ctx.dK != ctx.V);
    assert(ctx.dV != ctx.Q && ctx.dV != ctx.K && ctx.dV != ctx.V);

    const T w = ctx.raw_decay_w;
    const T gamma = sigmoid_val(w);

    // Zero-initialize output gradients
    const size_t total_qkv = B * H * T_len * D;
    std::memset(ctx.dQ, 0, total_qkv * sizeof(T));
    std::memset(ctx.dK, 0, total_qkv * sizeof(T));
    std::memset(ctx.dV, 0, total_qkv * sizeof(T));

    T total_d_gamma = static_cast<T>(0);

    // Scratch buffers
    std::vector<T> S_traj(T_len * D * D, static_cast<T>(0));
    std::vector<T> Gamma(D * D, static_cast<T>(0));
    std::vector<T> q_mapped(D);
    std::vector<T> q_prime_vec(D);
    std::vector<T> k_mapped(D);
    std::vector<T> k_prime_vec(D);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t seq_offset = (b * H + h) * T_len * D;

            // =================================================================
            // PASS 1: Forward State Recurrence (if not provided externally)
            // S_t = gamma * S_{t-1} + phi(K_t)^T * V_t
            // =================================================================
            const T* S_source = ctx.S ? (ctx.S + (b * H + h) * T_len * D * D) : S_traj.data();

            if (!ctx.S) {
                std::fill(S_traj.begin(), S_traj.end(), static_cast<T>(0));
                for (size_t t = 0; t < T_len; ++t) {
                    const size_t qkv_offset = seq_offset + t * D;
                    const size_t s_curr_offset = t * D * D;
                    const size_t s_prev_offset = (t > 0) ? ((t - 1) * D * D) : 0;

                    for (size_t i = 0; i < D; ++i) {
                        k_mapped[i] = phi_val(ctx.K[qkv_offset + i]);
                    }

                    for (size_t i = 0; i < D; ++i) {
                        const T k_i = k_mapped[i];
                        for (size_t j = 0; j < D; ++j) {
                            const T prev_s = (t > 0) ? S_traj[s_prev_offset + i * D + j] : static_cast<T>(0);
                            const T v_j = ctx.V[qkv_offset + j];
                            T new_s = gamma * prev_s + k_i * v_j;
                            if (new_s > -static_cast<T>(1e-15) && new_s < static_cast<T>(1e-15)) {
                                new_s = static_cast<T>(0);
                            }
                            S_traj[s_curr_offset + i * D + j] = new_s;
                        }
                    }
                }
            }

            // =================================================================
            // PASS 2: Reverse Adjoint Scan (t = T_len - 1 down to 0)
            // Gamma_t = gamma * Gamma_{t+1} + phi(Q_t)^T * dO_t
            // =================================================================
            std::fill(Gamma.begin(), Gamma.end(), static_cast<T>(0));

            for (int64_t t = static_cast<int64_t>(T_len) - 1; t >= 0; --t) {
                const size_t qkv_offset = seq_offset + t * D;
                const size_t s_curr_offset = t * D * D;

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
                        const T s_val = S_source[s_curr_offset + i * D + j];
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

                // 6. Accumulate d_gamma += <Gamma_t, S_{t-1}>_F (for t > 0)
                if (t > 0) {
                    const size_t s_prev_offset = (t - 1) * D * D;
                    for (size_t i = 0; i < D; ++i) {
                        for (size_t j = 0; j < D; ++j) {
                            total_d_gamma += Gamma[i * D + j] * S_source[s_prev_offset + i * D + j];
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

} // namespace subq

void run_attention_backward(AttentionContext& ctx) {
    size_t batch = ctx.Q.dimensions[0];
    size_t seq_len = ctx.Q.dimensions[1];
    size_t d_model = ctx.Q.dimensions[2];

    float* Q = static_cast<float*>(ctx.Q.data);
    float* K = static_cast<float*>(ctx.K.data);
    float* V = static_cast<float*>(ctx.V.data);
    float* dO = static_cast<float*>(ctx.O.data);
    
    // In legacy callers where ctx.Q holds gradients in-place, we allocate temp buffers if needed
    // or execute non-aliased backward
    std::vector<float> dQ(batch * seq_len * d_model, 0.0f);
    std::vector<float> dK(batch * seq_len * d_model, 0.0f);
    std::vector<float> dV(batch * seq_len * d_model, 0.0f);

    float d_gamma = 0.0f;
    float d_w = 0.0f;
    float w_val = std::log(std::max(ctx.decay_factor, 1e-6f) / std::max(1.0f - ctx.decay_factor, 1e-6f));

    subq::AttentionBackwardContext<float> bwd_ctx;
    bwd_ctx.Q = Q;
    bwd_ctx.K = K;
    bwd_ctx.V = V;
    bwd_ctx.dO = dO;
    bwd_ctx.S = nullptr;
    bwd_ctx.dQ = dQ.data();
    bwd_ctx.dK = dK.data();
    bwd_ctx.dV = dV.data();
    bwd_ctx.raw_decay_w = w_val;
    bwd_ctx.d_raw_decay_w = &d_w;
    bwd_ctx.d_gamma = &d_gamma;
    bwd_ctx.batch_size = batch;
    bwd_ctx.num_heads = 1;
    bwd_ctx.seq_len = seq_len;
    bwd_ctx.head_dim = d_model;

    subq::subquadratic_attention_backward(bwd_ctx);

    std::memcpy(ctx.Q.data, dQ.data(), batch * seq_len * d_model * sizeof(float));
    std::memcpy(ctx.K.data, dK.data(), batch * seq_len * d_model * sizeof(float));
    std::memcpy(ctx.V.data, dV.data(), batch * seq_len * d_model * sizeof(float));
}

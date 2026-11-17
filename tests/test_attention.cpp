/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "attention.hpp"
#include "kernels.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

TEST_CASE("Feature Map Activation (ELU + 1)", "[feature_map]") {
    std::vector<size_t> dims = {1, 1, 4};
    std::vector<float> h_data = {-1.0f, 2.0f, 0.0f, -2.0f}; 
    // Expected:
    // x = -1.0 => exp(-1) = 0.367879
    // x = 2.0  => 2.0 + 1 = 3.0
    // x = 0.0  => exp(0) = 1.0
    // x = -2.0 => exp(-2) = 0.135335
    
    float* d_data;
#if defined(USE_CUDA)
    cudaMalloc(&d_data, 4 * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
#elif defined(USE_HIP)
    hipMalloc(&d_data, 4 * sizeof(float));
    hipMemcpy(d_data, h_data.data(), 4 * sizeof(float), hipMemcpyHostToDevice);
#else
    d_data = h_data.data();
#endif

    TensorDescriptor desc{dims, {}, PrecisionMode::FP32, d_data, true};
    run_feature_map(desc);

#if defined(USE_CUDA)
    cudaMemcpy(h_data.data(), d_data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
#elif defined(USE_HIP)
    hipMemcpy(h_data.data(), d_data, 4 * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_data);
#endif

    REQUIRE_THAT(h_data[0], Catch::Matchers::WithinAbs(0.367879f, 1e-4f));
    REQUIRE_THAT(h_data[1], Catch::Matchers::WithinAbs(3.0f, 1e-4f));
    REQUIRE_THAT(h_data[2], Catch::Matchers::WithinAbs(1.0f, 1e-4f));
    REQUIRE_THAT(h_data[3], Catch::Matchers::WithinAbs(0.135335f, 1e-4f));
}

TEST_CASE("Analytical Backward Pass (Reverse Adjoint Scan)", "[attention_backward]") {
    const size_t B = 1;
    const size_t H = 1;
    const size_t T = 2;
    const size_t D = 2;

    std::vector<float> Q = {1.0f, -0.5f, 0.5f, 1.0f};
    std::vector<float> K = {0.0f, 1.0f, -1.0f, 0.0f};
    std::vector<float> V = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> dO = {1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<float> dQ(B * T * D, 0.0f);
    std::vector<float> dK(B * T * D, 0.0f);
    std::vector<float> dV(B * T * D, 0.0f);
    float d_gamma = 0.0f;
    float d_w = 0.0f;
    float raw_w = 0.0f; // sigmoid(0) = 0.5

    subq::AttentionBackwardContext<float> ctx;
    ctx.Q = Q.data();
    ctx.K = K.data();
    ctx.V = V.data();
    ctx.dO = dO.data();
    ctx.S = nullptr;
    ctx.dQ = dQ.data();
    ctx.dK = dK.data();
    ctx.dV = dV.data();
    ctx.raw_decay_w = raw_w;
    ctx.d_raw_decay_w = &d_w;
    ctx.d_gamma = &d_gamma;
    ctx.batch_size = B;
    ctx.num_heads = H;
    ctx.seq_len = T;
    ctx.head_dim = D;

    subq::subquadratic_attention_backward(ctx);

    // Verify non-zero gradients computed for all tensors
    for (size_t i = 0; i < B * T * D; ++i) {
        REQUIRE(!std::isnan(dQ[i]));
        REQUIRE(!std::isnan(dK[i]));
        REQUIRE(!std::isnan(dV[i]));
    }
    REQUIRE(!std::isnan(d_gamma));
    REQUIRE(!std::isnan(d_w));

    // For w = 0 -> gamma = 0.5 -> dw = d_gamma * 0.5 * 0.5 = d_gamma * 0.25
    REQUIRE_THAT(d_w, Catch::Matchers::WithinAbs(d_gamma * 0.25f, 1e-6f));
}

TEST_CASE("Double Precision Analytical Backward Pass", "[attention_backward_double]") {
    const size_t B = 1;
    const size_t H = 1;
    const size_t T = 2;
    const size_t D = 2;

    std::vector<double> Q = {0.5, -0.2, 0.8, 0.1};
    std::vector<double> K = {0.1, 0.9, -0.4, 0.3};
    std::vector<double> V = {1.5, 2.5, 0.5, 1.2};
    std::vector<double> dO = {0.7, 0.3, 1.1, 0.4};

    std::vector<double> dQ(B * T * D, 0.0);
    std::vector<double> dK(B * T * D, 0.0);
    std::vector<double> dV(B * T * D, 0.0);
    double d_gamma = 0.0;
    double d_w = 0.0;
    double raw_w = 1.0;

    subq::AttentionBackwardContext<double> ctx;
    ctx.Q = Q.data();
    ctx.K = K.data();
    ctx.V = V.data();
    ctx.dO = dO.data();
    ctx.S = nullptr;
    ctx.dQ = dQ.data();
    ctx.dK = dK.data();
    ctx.dV = dV.data();
    ctx.raw_decay_w = raw_w;
    ctx.d_raw_decay_w = &d_w;
    ctx.d_gamma = &d_gamma;
    ctx.batch_size = B;
    ctx.num_heads = H;
    ctx.seq_len = T;
    ctx.head_dim = D;

    subq::subquadratic_attention_backward(ctx);

    for (size_t i = 0; i < B * T * D; ++i) {
        REQUIRE(!std::isnan(dQ[i]));
        REQUIRE(!std::isnan(dK[i]));
        REQUIRE(!std::isnan(dV[i]));
    }
    REQUIRE(!std::isnan(d_gamma));
    REQUIRE(!std::isnan(d_w));

    double gamma = 1.0 / (1.0 + std::exp(-raw_w));
    REQUIRE_THAT(d_w, Catch::Matchers::WithinAbs(d_gamma * gamma * (1.0 - gamma), 1e-10));
}

namespace {

// Loss = sum(dO * O) for the recurrence S_t = gamma S_{t-1} + phi(K_t)^T V_t, O_t = phi(Q_t) S_t,
// starting from initial state S0 [D, D] (single batch, single head).
double loss_with_state(const std::vector<double>& Q, const std::vector<double>& K, const std::vector<double>& V,
                       const std::vector<double>& dO, const std::vector<double>& S0, double gamma,
                       size_t T, size_t D) {
    auto phi = [](double x) { return x > 0.0 ? x + 1.0 : std::exp(x); };
    std::vector<double> S = S0;
    double loss = 0.0;
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                S[i * D + j] = gamma * S[i * D + j] + phi(K[t * D + i]) * V[t * D + j];
            }
        }
        for (size_t j = 0; j < D; ++j) {
            double o = 0.0;
            for (size_t i = 0; i < D; ++i) o += phi(Q[t * D + i]) * S[i * D + j];
            loss += dO[t * D + j] * o;
        }
    }
    return loss;
}

} // namespace

TEST_CASE("Backward with nonzero initial state matches finite differences", "[attention_backward_state]") {
    // T = 70 crosses the 64-token chunk boundary; D = 3 keeps finite differences cheap.
    const size_t T = 70, D = 3;
    const double raw_w = 0.4;
    const double gamma = 1.0 / (1.0 + std::exp(-raw_w));

    std::vector<double> Q(T * D), K(T * D), V(T * D), dO(T * D), S0(D * D);
    for (size_t i = 0; i < T * D; ++i) {
        Q[i] = 0.6 * std::sin(0.7 * i + 0.1);
        K[i] = 0.5 * std::cos(0.9 * i + 0.3);
        V[i] = 0.8 * std::sin(1.3 * i + 0.5);
        dO[i] = 0.7 * std::cos(0.4 * i + 0.2);
    }
    for (size_t i = 0; i < D * D; ++i) S0[i] = 0.5 + 0.1 * i;

    std::vector<double> dQ(T * D), dK(T * D), dV(T * D);
    double d_gamma = 0.0, d_w = 0.0;
    subq::AttentionBackwardContext<double> ctx;
    ctx.Q = Q.data(); ctx.K = K.data(); ctx.V = V.data(); ctx.dO = dO.data();
    ctx.S = nullptr; ctx.S0 = S0.data();
    ctx.dQ = dQ.data(); ctx.dK = dK.data(); ctx.dV = dV.data();
    ctx.raw_decay_w = raw_w; ctx.d_raw_decay_w = &d_w; ctx.d_gamma = &d_gamma;
    ctx.batch_size = 1; ctx.num_heads = 1; ctx.seq_len = T; ctx.head_dim = D;
    subq::subquadratic_attention_backward(ctx);

    const double eps = 1e-6;
    auto check = [&](std::vector<double>& x, const std::vector<double>& analytic, const char* name) {
        for (size_t i = 0; i < x.size(); i += 5) {
            const double orig = x[i];
            x[i] = orig + eps;
            const double lp = loss_with_state(Q, K, V, dO, S0, gamma, T, D);
            x[i] = orig - eps;
            const double lm = loss_with_state(Q, K, V, dO, S0, gamma, T, D);
            x[i] = orig;
            INFO(name << "[" << i << "]");
            REQUIRE_THAT(analytic[i], Catch::Matchers::WithinAbs((lp - lm) / (2 * eps), 1e-5));
        }
    };
    check(Q, dQ, "dQ");
    check(K, dK, "dK");
    check(V, dV, "dV");

    const double lp = loss_with_state(Q, K, V, dO, S0, gamma + eps, T, D);
    const double lm = loss_with_state(Q, K, V, dO, S0, gamma - eps, T, D);
    REQUIRE_THAT(d_gamma, Catch::Matchers::WithinAbs((lp - lm) / (2 * eps), 1e-4));
}

TEST_CASE("run_attention_backward_ex keeps inputs intact and rejects aliasing", "[attention_backward_ex]") {
    const size_t B = 1, T = 5, D = 2;
    std::vector<float> Q = {1.f, -.5f, .5f, 1.f, .2f, .1f, -.3f, .4f, .9f, -.7f};
    std::vector<float> K = {0.f, 1.f, -1.f, 0.f, .3f, .2f, .6f, -.1f, .5f, .5f};
    std::vector<float> V = {1.f, 2.f, 3.f, 4.f, .5f, .6f, -.7f, .8f, .9f, 1.f};
    std::vector<float> dO(B * T * D, 1.0f);
    std::vector<float> S(B * D * D, 0.0f), O = dO;

    auto desc = [&](std::vector<float>& v) {
        return TensorDescriptor{{B, T, D}, {}, PrecisionMode::FP32, v.data(), true};
    };
    AttentionContext ctx;
    ctx.Q = desc(Q); ctx.K = desc(K); ctx.V = desc(V); ctx.O = desc(O);
    ctx.S = TensorDescriptor{{B, D, D}, {}, PrecisionMode::FP32, S.data(), true};
    ctx.decay_factor = 0.9f;

    const std::vector<float> Q0 = Q, K0 = K, V0 = V;
    std::vector<float> dQ(B * T * D), dK(B * T * D), dV(B * T * D);
    float d_gamma = 0.0f;
    run_attention_backward_ex(ctx, dQ.data(), dK.data(), dV.data(), &d_gamma);
    REQUIRE(Q == Q0);
    REQUIRE(K == K0);
    REQUIRE(V == V0);
    REQUIRE(!std::isnan(d_gamma));
    REQUIRE(d_gamma != 0.0f);

    // Legacy in-place entry must produce identical gradients.
    run_attention_backward(ctx);
    REQUIRE(Q == dQ);
    REQUIRE(K == dK);
    REQUIRE(V == dV);

    // Aliasing an output with an input is a hard error (not an assert).
    std::vector<float> a = Q0;
    ctx.Q = desc(a);
    REQUIRE_THROWS_AS(run_attention_backward_ex(ctx, a.data(), dK.data(), dV.data()), std::invalid_argument);
}

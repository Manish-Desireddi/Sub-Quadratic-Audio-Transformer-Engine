/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "attention.hpp"
#include "kernels.hpp"
#include <vector>
#include <cmath>

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

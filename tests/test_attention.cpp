/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "attention.hpp"
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

/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "attention.hpp"
#include <vector>
#include <cmath>

TEST_CASE("Feature Map Activation (ELU + 1)", "[feature_map]") {
    std::vector<size_t> dims = {4};
    std::vector<float> data = {-1.0f, 2.0f, 0.0f, -2.0f}; 
    // Expected:
    // x = -1.0 => exp(-1) = 0.367879
    // x = 2.0  => 2.0 + 1 = 3.0
    // x = 0.0  => exp(0) = 1.0
    // x = -2.0 => exp(-2) = 0.135335
    
    TensorDescriptor desc{dims, {}, PrecisionMode::FP32, data.data(), true};

    run_feature_map(desc);

    REQUIRE_THAT(data[0], Catch::Matchers::WithinAbs(0.367879f, 1e-4f));
    REQUIRE_THAT(data[1], Catch::Matchers::WithinAbs(3.0f, 1e-4f));
    REQUIRE_THAT(data[2], Catch::Matchers::WithinAbs(1.0f, 1e-4f));
    REQUIRE_THAT(data[3], Catch::Matchers::WithinAbs(0.135335f, 1e-4f));
}

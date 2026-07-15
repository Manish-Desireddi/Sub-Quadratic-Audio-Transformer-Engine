/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once

#include <vector>
#include <cstdint>

enum class PrecisionMode {
    FP32,
    FP16,
    BF16
};

struct TensorDescriptor {
    std::vector<size_t> dimensions;
    std::vector<size_t> strides;
    PrecisionMode precision;
    void* data; // Pointer to VRAM in Arena
    bool is_on_host; // false if in GPU VRAM
};

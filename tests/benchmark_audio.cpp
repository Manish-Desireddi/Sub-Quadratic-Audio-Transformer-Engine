/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "engine.hpp"
#include "arena.hpp"
#include "attention.hpp"

TEST_CASE("Native C++ Sub-Quadratic Engine Benchmark", "[benchmark]") {
    size_t vram_capacity = 256 * 1024 * 1024; // 256 MB
    MemoryArena arena(vram_capacity);
    Engine engine(arena);

    size_t batch = 1;
    size_t seq_len = 1024;
    size_t d_model = 256;

    size_t tensor_elements = batch * seq_len * d_model;
    size_t tensor_bytes = tensor_elements * sizeof(gpuHalf);
    size_t state_bytes = batch * d_model * d_model * sizeof(float);

    gpuHalf* d_Q = reinterpret_cast<gpuHalf*>(arena.allocate(tensor_bytes));
    gpuHalf* d_K = reinterpret_cast<gpuHalf*>(arena.allocate(tensor_bytes));
    gpuHalf* d_V = reinterpret_cast<gpuHalf*>(arena.allocate(tensor_bytes));
    gpuHalf* d_O = reinterpret_cast<gpuHalf*>(arena.allocate(tensor_bytes));
    float* d_S = reinterpret_cast<float*>(arena.allocate(state_bytes));

    AttentionContext ctx;
    ctx.Q = { {batch, seq_len, d_model}, {}, PrecisionMode::FP16, d_Q, true };
    ctx.K = { {batch, seq_len, d_model}, {}, PrecisionMode::FP16, d_K, true };
    ctx.V = { {batch, seq_len, d_model}, {}, PrecisionMode::FP16, d_V, true };
    ctx.S = { {batch, d_model, d_model}, {}, PrecisionMode::FP32, d_S, true }; 
    ctx.O = { {batch, seq_len, d_model}, {}, PrecisionMode::FP16, d_O, true };

    BENCHMARK("Forward Pass O(N) Scan (Seq=1024, d=256)") {
        engine.forward(ctx);
#if defined(USE_CUDA) || defined(USE_HIP)
        gpuDeviceSynchronize();
#endif
        return 0;
    };
}

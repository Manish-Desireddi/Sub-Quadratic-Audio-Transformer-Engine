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
    size_t tensor_bytes = tensor_elements * sizeof(uint16_t);
    size_t state_bytes = batch * d_model * d_model * sizeof(float);

    uint16_t* d_Q = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    uint16_t* d_K = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    uint16_t* d_V = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    uint16_t* d_O = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    float* d_S = reinterpret_cast<float*>(arena.allocate(state_bytes));

    AttentionContext ctx;
    ctx.Q.dimensions = {batch, seq_len, d_model}; ctx.Q.precision = PrecisionMode::FP16; ctx.Q.data = d_Q; ctx.Q.is_on_host = true;
    ctx.K.dimensions = {batch, seq_len, d_model}; ctx.K.precision = PrecisionMode::FP16; ctx.K.data = d_K; ctx.K.is_on_host = true;
    ctx.V.dimensions = {batch, seq_len, d_model}; ctx.V.precision = PrecisionMode::FP16; ctx.V.data = d_V; ctx.V.is_on_host = true;
    ctx.S.dimensions = {batch, d_model, d_model}; ctx.S.precision = PrecisionMode::FP32; ctx.S.data = d_S; ctx.S.is_on_host = true; 
    ctx.O.dimensions = {batch, seq_len, d_model}; ctx.O.precision = PrecisionMode::FP16; ctx.O.data = d_O; ctx.O.is_on_host = true;

    BENCHMARK("Forward Pass O(N) Scan (Seq=1024, d=256)") {
        engine.forward(ctx);
#if defined(USE_CUDA) || defined(USE_HIP)
        gpuDeviceSynchronize();
#endif
        return 0;
    };
}

/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "engine.hpp"
#include "arena.hpp"
#include "attention.hpp"
#include "gpu_macros.hpp"
#include "safetensors.hpp"

namespace py = pybind11;

class SubQEngineWrapper {
public:
    SubQEngineWrapper(size_t vram_capacity) 
        : arena_(vram_capacity), engine_(arena_) {}

    void load(const std::string& filepath) {
        try {
            SafetensorLoader loader(arena_);
            if (!loader.load(filepath)) {
                throw std::runtime_error("Failed to load safetensors file.");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Loader Error: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Unknown Loader Error occurred.");
        }
    }

    py::array_t<float> forward(py::array_t<float, py::array::c_style | py::array::forcecast> Q, 
                               py::array_t<float, py::array::c_style | py::array::forcecast> K, 
                               py::array_t<float, py::array::c_style | py::array::forcecast> V) {
        try {
            py::buffer_info buf_Q = Q.request();
            py::buffer_info buf_K = K.request();
            py::buffer_info buf_V = V.request();

            if (buf_Q.ndim != 3 || buf_K.ndim != 3 || buf_V.ndim != 3) {
                throw std::runtime_error("Number of dimensions must be 3 (batch, seq_len, d_model)");
            }

            size_t batch = buf_Q.shape[0];
            size_t seq_len = buf_Q.shape[1];
            size_t d_model = buf_Q.shape[2];

            // Allocate output array on host
            auto result = py::array_t<float>({batch, seq_len, d_model});
            py::buffer_info buf_out = result.request();

            // Reset arena for this forward pass
            arena_.reset_head();

            size_t bytes = batch * seq_len * d_model * sizeof(float);
            size_t state_bytes = batch * d_model * d_model * sizeof(float);
            
            // Allocate MemoryArena device pointers
            float* d_Q = reinterpret_cast<float*>(arena_.allocate(bytes));
            float* d_K = reinterpret_cast<float*>(arena_.allocate(bytes));
            float* d_V = reinterpret_cast<float*>(arena_.allocate(bytes));
            float* d_O = reinterpret_cast<float*>(arena_.allocate(bytes));
            float* d_S = reinterpret_cast<float*>(arena_.allocate(state_bytes));

            // Copy host numpy arrays to device arena
#if defined(USE_CUDA) || defined(USE_HIP)
            CHECK_GPU_ERROR(gpuMemcpy(d_Q, buf_Q.ptr, bytes, gpuMemcpyHostToDevice));
            CHECK_GPU_ERROR(gpuMemcpy(d_K, buf_K.ptr, bytes, gpuMemcpyHostToDevice));
            CHECK_GPU_ERROR(gpuMemcpy(d_V, buf_V.ptr, bytes, gpuMemcpyHostToDevice));
            CHECK_GPU_ERROR(gpuMemset(d_S, 0, state_bytes));
#else
            std::memcpy(d_Q, buf_Q.ptr, bytes);
            std::memcpy(d_K, buf_K.ptr, bytes);
            std::memcpy(d_V, buf_V.ptr, bytes);
            std::memset(d_S, 0, state_bytes);
#endif

            AttentionContext ctx;
            ctx.Q = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, d_Q, true };
            ctx.K = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, d_K, true };
            ctx.V = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, d_V, true };
            ctx.S = { {batch, d_model, d_model}, {}, PrecisionMode::FP32, d_S, true }; 
            ctx.O = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, d_O, true };

            engine_.forward(ctx);

            // Copy device results back to host numpy array
#if defined(USE_CUDA) || defined(USE_HIP)
            CHECK_GPU_ERROR(gpuMemcpy(buf_out.ptr, d_O, bytes, gpuMemcpyDeviceToHost));
#else
            std::memcpy(buf_out.ptr, d_O, bytes);
#endif

            return result;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Engine Error: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Unknown Engine Error occurred.");
        }
    }

private:
    MemoryArena arena_;
    Engine engine_;
};

PYBIND11_MODULE(subq_engine, m) {
    m.doc() = "Sub-Quadratic Audio Transformer Engine C++ backend";

    py::class_<SubQEngineWrapper>(m, "SubQEngine")
        .def(py::init<size_t>(), py::arg("vram_capacity") = 1024 * 1024 * 256)
        .def("load", &SubQEngineWrapper::load, py::arg("filepath"))
        .def("forward", &SubQEngineWrapper::forward, py::arg("Q"), py::arg("K"), py::arg("V"));
}

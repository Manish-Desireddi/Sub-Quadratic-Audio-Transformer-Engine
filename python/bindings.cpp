/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "engine.hpp"
#include "arena.hpp"
#include "attention.hpp"
#include "gpu_macros.hpp"
#include "safetensors.hpp"
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <mutex>

namespace py = pybind11;

py::dict get_device_info() {
    py::dict info;
#if defined(USE_CUDA)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    info["backend"] = "CUDA";
    info["architecture"] = std::string("SM ") + std::to_string(prop.major) + "." + std::to_string(prop.minor);
    info["memory_size"] = prop.totalGlobalMem;
    info["bfloat16_supported"] = (prop.major >= 8);
#elif defined(USE_HIP)
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    info["backend"] = "HIP";
    info["architecture"] = std::string(prop.gcnArchName);
    info["memory_size"] = prop.totalGlobalMem;
    std::string arch(prop.gcnArchName);
    info["bfloat16_supported"] = (arch.find("gfx11") != std::string::npos || arch.find("gfx90a") != std::string::npos);
#else
    info["backend"] = "CPU";
    info["architecture"] = "Host";
    info["memory_size"] = 0;
    info["bfloat16_supported"] = false;
#endif
    return info;
}

class SubQEngineWrapper {
public:
    SubQEngineWrapper(size_t vram_capacity) 
        : arena_(vram_capacity), engine_(arena_) {
        // Prevent subnormal floats from stalling the ALU
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }

    void load(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
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
                               py::array_t<float, py::array::c_style | py::array::forcecast> V,
                               float decay_factor = 0.99f) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
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

            // 1. Zero-Copy Handoff allocation
            size_t total_elements = batch * seq_len * d_model;
            float* host_out_data = new float[total_elements];
            py::capsule free_when_done(host_out_data, [](void *f) {
                float *foo = reinterpret_cast<float *>(f);
                delete[] foo;
            });

            auto result = py::array_t<float>(
                {batch, seq_len, d_model}, 
                {seq_len * d_model * sizeof(float), d_model * sizeof(float), sizeof(float)}, 
                host_out_data, 
                free_when_done
            );

            // 2. Chunking logic for O(1) VRAM footprint
            arena_.reset_head();
            size_t state_bytes = batch * d_model * d_model * sizeof(float);
            float* d_S = reinterpret_cast<float*>(arena_.allocate(state_bytes));

#if defined(USE_CUDA) || defined(USE_HIP)
            CHECK_GPU_ERROR(gpuMemset(d_S, 0, state_bytes));
#else
            std::memset(d_S, 0, state_bytes);
#endif

            // Save the offset so we can rewind back here for every chunk
            size_t saved_offset = arena_.get_offset();

            // Define a chunk size based on remaining arena capacity
            size_t available_bytes = arena_.get_total_bytes() - saved_offset;
            size_t max_chunk_size = available_bytes / (4 * batch * d_model * sizeof(float));
            size_t chunk_size = std::min(max_chunk_size, static_cast<size_t>(16384)); // Safe ceiling

            float* host_Q = static_cast<float*>(buf_Q.ptr);
            float* host_K = static_cast<float*>(buf_K.ptr);
            float* host_V = static_cast<float*>(buf_V.ptr);

            for (size_t t = 0; t < seq_len; t += chunk_size) {
                size_t current_chunk = std::min(chunk_size, seq_len - t);
                size_t chunk_bytes = batch * current_chunk * d_model * sizeof(float);

                arena_.set_offset(saved_offset); // Rewind pointer
                float* d_Q = reinterpret_cast<float*>(arena_.allocate(chunk_bytes));
                float* d_K = reinterpret_cast<float*>(arena_.allocate(chunk_bytes));
                float* d_V = reinterpret_cast<float*>(arena_.allocate(chunk_bytes));
                float* d_O = reinterpret_cast<float*>(arena_.allocate(chunk_bytes));

                for (size_t b = 0; b < batch; ++b) {
                    size_t src_offset = b * (seq_len * d_model) + t * d_model;
                    size_t dst_offset = b * (current_chunk * d_model);
                    size_t batch_chunk_bytes = current_chunk * d_model * sizeof(float);

#if defined(USE_CUDA) || defined(USE_HIP)
                    CHECK_GPU_ERROR(gpuMemcpy(d_Q + dst_offset, host_Q + src_offset, batch_chunk_bytes, gpuMemcpyHostToDevice));
                    CHECK_GPU_ERROR(gpuMemcpy(d_K + dst_offset, host_K + src_offset, batch_chunk_bytes, gpuMemcpyHostToDevice));
                    CHECK_GPU_ERROR(gpuMemcpy(d_V + dst_offset, host_V + src_offset, batch_chunk_bytes, gpuMemcpyHostToDevice));
#else
                    std::memcpy(d_Q + dst_offset, host_Q + src_offset, batch_chunk_bytes);
                    std::memcpy(d_K + dst_offset, host_K + src_offset, batch_chunk_bytes);
                    std::memcpy(d_V + dst_offset, host_V + src_offset, batch_chunk_bytes);
#endif
                }

                py::dict info = get_device_info();
                bool bf16_supported = info["bfloat16_supported"].cast<bool>();
                PrecisionMode precision = bf16_supported ? PrecisionMode::BF16 : PrecisionMode::FP32;

                AttentionContext ctx;
                ctx.Q = { {batch, current_chunk, d_model}, {}, PrecisionMode::FP32, d_Q, true };
                ctx.K = { {batch, current_chunk, d_model}, {}, PrecisionMode::FP32, d_K, true };
                ctx.V = { {batch, current_chunk, d_model}, {}, PrecisionMode::FP32, d_V, true };
                ctx.S = { {batch, d_model, d_model}, {}, precision, d_S, true }; 
                ctx.O = { {batch, current_chunk, d_model}, {}, PrecisionMode::FP32, d_O, true };
                ctx.decay_factor = decay_factor;

                engine_.forward(ctx);

                for (size_t b = 0; b < batch; ++b) {
                    size_t dst_offset = b * (seq_len * d_model) + t * d_model;
                    size_t src_offset = b * (current_chunk * d_model);
                    size_t batch_chunk_bytes = current_chunk * d_model * sizeof(float);

#if defined(USE_CUDA) || defined(USE_HIP)
                    CHECK_GPU_ERROR(gpuMemcpy(host_out_data + dst_offset, d_O + src_offset, batch_chunk_bytes, gpuMemcpyDeviceToHost));
#else
                    std::memcpy(host_out_data + dst_offset, d_O + src_offset, batch_chunk_bytes);
#endif
                }
            }

            return result;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Engine Error: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Unknown Engine Error occurred.");
        }
    }

private:
    std::mutex arena_mutex_;
    MemoryArena arena_;
    Engine engine_;
};

PYBIND11_MODULE(subq_engine, m) {
    m.doc() = "Sub-Quadratic Audio Transformer Engine C++ backend";

    m.def("get_device_info", &get_device_info, "Get hardware telemetry info");

    py::class_<SubQEngineWrapper>(m, "SubQEngine")
        .def(py::init<size_t>(), py::arg("vram_capacity") = 1024 * 1024 * 256)
        .def("load", &SubQEngineWrapper::load, py::arg("filepath"))
        .def("forward", &SubQEngineWrapper::forward, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("decay_factor") = 0.99f);
}

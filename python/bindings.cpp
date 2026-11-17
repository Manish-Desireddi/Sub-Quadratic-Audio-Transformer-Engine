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
#include "kernels.hpp"
#include "gpu_macros.hpp"
#include "safetensors.hpp"
#include "audio_stream.hpp"
#include <mutex>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#include <pmmintrin.h>
#define SUBQ_ENABLE_DENORMALS_FLUSH() do { \
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); \
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON); \
} while(0)
#else
#define SUBQ_ENABLE_DENORMALS_FLUSH() do {} while(0)
#endif

namespace py = pybind11;

py::dict get_device_info() {
    py::dict info;
#if defined(USE_CUDA)
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    info["backend"] = "CUDA";
    info["architecture"] = std::string("SM ") + std::to_string(prop.major) + "." + std::to_string(prop.minor);
    info["memory_size"] = prop.totalGlobalMem;
    info["bfloat16_supported"] = (prop.major >= 8);
#elif defined(USE_HIP)
    int device = 0;
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
    static int prepare_device(int device_id) {
#if defined(USE_CUDA)
        if (device_id >= 0) {
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaSetDevice failed for device ") + 
                                         std::to_string(device_id) + ": " + cudaGetErrorString(err));
            }
        }
#elif defined(USE_HIP)
        if (device_id >= 0) {
            hipError_t err = hipSetDevice(device_id);
            if (err != hipSuccess) {
                throw std::runtime_error(std::string("hipSetDevice failed for device ") + 
                                         std::to_string(device_id) + ": " + hipGetErrorString(err));
            }
        }
#else
        (void)device_id;
#endif
        return device_id;
    }

    explicit SubQEngineWrapper(size_t vram_capacity = 32 * 1024 * 1024, int device_id = 0) 
        : device_id_(prepare_device(device_id)), arena_(vram_capacity), engine_(arena_) {
        SUBQ_ENABLE_DENORMALS_FLUSH();
    }

    int device_id() const {
        return device_id_;
    }

    void load(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        try {
            CHECK_GPU_ERROR(gpuSetDevice(device_id_));
            SafetensorLoader loader(arena_);
            if (!loader.load(filepath)) {
                throw std::runtime_error("Failed to load safetensors file: " + filepath);
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
            CHECK_GPU_ERROR(gpuSetDevice(device_id_));
            py::buffer_info buf_Q = Q.request();
            py::buffer_info buf_K = K.request();
            py::buffer_info buf_V = V.request();

            if (buf_Q.ndim != 3 || buf_K.ndim != 3 || buf_V.ndim != 3) {
                throw std::runtime_error("Number of dimensions must be 3 (batch, seq_len, d_model)");
            }

            size_t batch = buf_Q.shape[0];
            size_t seq_len = buf_Q.shape[1];
            size_t d_model = buf_Q.shape[2];

            if (buf_K.shape[0] != batch || buf_K.shape[1] != seq_len || buf_K.shape[2] != d_model ||
                buf_V.shape[0] != batch || buf_V.shape[1] != seq_len || buf_V.shape[2] != d_model) {
                throw std::runtime_error("Shape mismatch between Q, K, V tensors");
            }

            size_t total_elements = batch * seq_len * d_model;
            float* host_out_data = new float[total_elements];
            py::capsule free_when_done(host_out_data, [](void *f) {
                delete[] reinterpret_cast<float*>(f);
            });

            auto result = py::array_t<float>(
                {batch, seq_len, d_model}, 
                {seq_len * d_model * sizeof(float), d_model * sizeof(float), sizeof(float)}, 
                host_out_data, 
                free_when_done
            );

            arena_.reset_head();
            size_t state_bytes = batch * d_model * d_model * sizeof(float);
            float* d_S = reinterpret_cast<float*>(arena_.allocate(state_bytes));

#if defined(USE_CUDA) || defined(USE_HIP)
            CHECK_GPU_ERROR(gpuMemset(d_S, 0, state_bytes));
#else
            std::memset(d_S, 0, state_bytes);
#endif

            size_t saved_offset = arena_.get_offset();
            size_t available_bytes = (arena_.get_total_bytes() > saved_offset) ? (arena_.get_total_bytes() - saved_offset) : 0;
            size_t denominator = 4 * batch * d_model * sizeof(float);
            size_t max_chunk_size = (denominator > 0) ? (available_bytes / denominator) : 1024;
            size_t chunk_size = std::max<size_t>(1, std::min(max_chunk_size, static_cast<size_t>(16384)));

            float* host_Q = static_cast<float*>(buf_Q.ptr);
            float* host_K = static_cast<float*>(buf_K.ptr);
            float* host_V = static_cast<float*>(buf_V.ptr);

            for (size_t t = 0; t < seq_len; t += chunk_size) {
                size_t current_chunk = std::min(chunk_size, seq_len - t);
                size_t chunk_bytes = batch * current_chunk * d_model * sizeof(float);

                arena_.set_offset(saved_offset);
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
            throw std::runtime_error(std::string("Engine Forward Error: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Unknown Engine Forward Error occurred.");
        }
    }

    void forward_raw(uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr, uintptr_t o_ptr,
                     size_t batch, size_t seq_len, size_t d_model, float decay_factor = 0.99f) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        try {
            CHECK_GPU_ERROR(gpuSetDevice(device_id_));
            arena_.reset_head();
            size_t state_bytes = batch * d_model * d_model * sizeof(float);
            float* d_S = reinterpret_cast<float*>(arena_.allocate(state_bytes));

#if defined(USE_CUDA) || defined(USE_HIP)
            CHECK_GPU_ERROR(gpuMemset(d_S, 0, state_bytes));
#else
            std::memset(d_S, 0, state_bytes);
#endif

            AttentionContext ctx;
            ctx.Q = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, reinterpret_cast<void*>(q_ptr), false };
            ctx.K = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, reinterpret_cast<void*>(k_ptr), false };
            ctx.V = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, reinterpret_cast<void*>(v_ptr), false };
            ctx.O = { {batch, seq_len, d_model}, {}, PrecisionMode::FP32, reinterpret_cast<void*>(o_ptr), false };
            ctx.S = { {batch, d_model, d_model}, {}, PrecisionMode::FP32, d_S, false };
            ctx.decay_factor = decay_factor;

            engine_.forward(ctx);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Engine Forward Raw Error: ") + e.what());
        }
    }

    py::tuple backward(py::array_t<float, py::array::c_style | py::array::forcecast> dO,
                       py::array_t<float, py::array::c_style | py::array::forcecast> Q,
                       py::array_t<float, py::array::c_style | py::array::forcecast> K,
                       py::array_t<float, py::array::c_style | py::array::forcecast> V,
                       float decay_factor = 0.99f) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        try {
            CHECK_GPU_ERROR(gpuSetDevice(device_id_));
            py::buffer_info buf_dO = dO.request();
            py::buffer_info buf_Q = Q.request();
            py::buffer_info buf_K = K.request();
            py::buffer_info buf_V = V.request();

            if (buf_dO.ndim != 3 || buf_Q.ndim != 3 || buf_K.ndim != 3 || buf_V.ndim != 3) {
                throw std::runtime_error("Number of dimensions must be 3 (batch, seq_len, d_model)");
            }

            size_t batch = buf_Q.shape[0];
            size_t seq_len = buf_Q.shape[1];
            size_t d_model = buf_Q.shape[2];
            size_t total_elements = batch * seq_len * d_model;

            float* dQ_data = new float[total_elements];
            float* dK_data = new float[total_elements];
            float* dV_data = new float[total_elements];

            float total_d_decay = 0.0f;
            float d_w = 0.0f;
            float clamped_decay = std::max(std::min(decay_factor, 1.0f - 1e-6f), 1e-6f);
            float w_val = std::log(clamped_decay / (1.0f - clamped_decay));

            subq::AttentionBackwardContext<float> bwd_ctx;
            bwd_ctx.Q = static_cast<const float*>(buf_Q.ptr);
            bwd_ctx.K = static_cast<const float*>(buf_K.ptr);
            bwd_ctx.V = static_cast<const float*>(buf_V.ptr);
            bwd_ctx.dO = static_cast<const float*>(buf_dO.ptr);
            bwd_ctx.S = nullptr;
            bwd_ctx.S0 = nullptr;
            bwd_ctx.dQ = dQ_data;
            bwd_ctx.dK = dK_data;
            bwd_ctx.dV = dV_data;
            bwd_ctx.raw_decay_w = w_val;
            bwd_ctx.d_raw_decay_w = &d_w;
            bwd_ctx.d_gamma = &total_d_decay;
            bwd_ctx.batch_size = batch;
            bwd_ctx.num_heads = 1;
            bwd_ctx.seq_len = seq_len;
            bwd_ctx.head_dim = d_model;

            engine_.backward(bwd_ctx);

            py::capsule free_dq(dQ_data, [](void *f) { delete[] reinterpret_cast<float*>(f); });
            py::capsule free_dk(dK_data, [](void *f) { delete[] reinterpret_cast<float*>(f); });
            py::capsule free_dv(dV_data, [](void *f) { delete[] reinterpret_cast<float*>(f); });

            auto dQ_arr = py::array_t<float>({batch, seq_len, d_model}, dQ_data, free_dq);
            auto dK_arr = py::array_t<float>({batch, seq_len, d_model}, dK_data, free_dk);
            auto dV_arr = py::array_t<float>({batch, seq_len, d_model}, dV_data, free_dv);

            return py::make_tuple(dQ_arr, dK_arr, dV_arr, total_d_decay);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Engine Backward Error: ") + e.what());
        }
    }

    float backward_raw(uintptr_t do_ptr, uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr,
                       uintptr_t dq_ptr, uintptr_t dk_ptr, uintptr_t dv_ptr,
                       size_t batch, size_t seq_len, size_t d_model, float decay_factor = 0.99f) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        try {
            CHECK_GPU_ERROR(gpuSetDevice(device_id_));
            size_t total_elements = batch * seq_len * d_model;

            float total_d_decay = 0.0f;
            float d_w = 0.0f;
            float clamped_decay = std::max(std::min(decay_factor, 1.0f - 1e-6f), 1e-6f);
            float w_val = std::log(clamped_decay / (1.0f - clamped_decay));

#if defined(USE_CUDA) || defined(USE_HIP)
            std::vector<float> host_dO(total_elements);
            std::vector<float> host_Q(total_elements);
            std::vector<float> host_K(total_elements);
            std::vector<float> host_V(total_elements);
            std::vector<float> host_dQ(total_elements, 0.0f);
            std::vector<float> host_dK(total_elements, 0.0f);
            std::vector<float> host_dV(total_elements, 0.0f);

            CHECK_GPU_ERROR(gpuMemcpy(host_dO.data(), reinterpret_cast<const void*>(do_ptr), total_elements * sizeof(float), gpuMemcpyDeviceToHost));
            CHECK_GPU_ERROR(gpuMemcpy(host_Q.data(), reinterpret_cast<const void*>(q_ptr), total_elements * sizeof(float), gpuMemcpyDeviceToHost));
            CHECK_GPU_ERROR(gpuMemcpy(host_K.data(), reinterpret_cast<const void*>(k_ptr), total_elements * sizeof(float), gpuMemcpyDeviceToHost));
            CHECK_GPU_ERROR(gpuMemcpy(host_V.data(), reinterpret_cast<const void*>(v_ptr), total_elements * sizeof(float), gpuMemcpyDeviceToHost));

            subq::AttentionBackwardContext<float> bwd_ctx;
            bwd_ctx.Q = host_Q.data();
            bwd_ctx.K = host_K.data();
            bwd_ctx.V = host_V.data();
            bwd_ctx.dO = host_dO.data();
            bwd_ctx.S = nullptr;
            bwd_ctx.S0 = nullptr;
            bwd_ctx.dQ = host_dQ.data();
            bwd_ctx.dK = host_dK.data();
            bwd_ctx.dV = host_dV.data();
            bwd_ctx.raw_decay_w = w_val;
            bwd_ctx.d_raw_decay_w = &d_w;
            bwd_ctx.d_gamma = &total_d_decay;
            bwd_ctx.batch_size = batch;
            bwd_ctx.num_heads = 1;
            bwd_ctx.seq_len = seq_len;
            bwd_ctx.head_dim = d_model;

            engine_.backward(bwd_ctx);

            CHECK_GPU_ERROR(gpuMemcpy(reinterpret_cast<void*>(dq_ptr), host_dQ.data(), total_elements * sizeof(float), gpuMemcpyHostToDevice));
            CHECK_GPU_ERROR(gpuMemcpy(reinterpret_cast<void*>(dk_ptr), host_dK.data(), total_elements * sizeof(float), gpuMemcpyHostToDevice));
            CHECK_GPU_ERROR(gpuMemcpy(reinterpret_cast<void*>(dv_ptr), host_dV.data(), total_elements * sizeof(float), gpuMemcpyHostToDevice));
#else
            subq::AttentionBackwardContext<float> bwd_ctx;
            bwd_ctx.Q = reinterpret_cast<const float*>(q_ptr);
            bwd_ctx.K = reinterpret_cast<const float*>(k_ptr);
            bwd_ctx.V = reinterpret_cast<const float*>(v_ptr);
            bwd_ctx.dO = reinterpret_cast<const float*>(do_ptr);
            bwd_ctx.S = nullptr;
            bwd_ctx.S0 = nullptr;
            bwd_ctx.dQ = reinterpret_cast<float*>(dq_ptr);
            bwd_ctx.dK = reinterpret_cast<float*>(dk_ptr);
            bwd_ctx.dV = reinterpret_cast<float*>(dv_ptr);
            bwd_ctx.raw_decay_w = w_val;
            bwd_ctx.d_raw_decay_w = &d_w;
            bwd_ctx.d_gamma = &total_d_decay;
            bwd_ctx.batch_size = batch;
            bwd_ctx.num_heads = 1;
            bwd_ctx.seq_len = seq_len;
            bwd_ctx.head_dim = d_model;

            engine_.backward(bwd_ctx);
#endif
            return total_d_decay;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Engine Backward Raw Error: ") + e.what());
        }
    }

private:
    int device_id_{0};
    std::mutex arena_mutex_;
    MemoryArena arena_;
    Engine engine_;
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "Sub-Quadratic Audio Transformer Engine C++ backend (_C)";

    m.def("get_device_info", &get_device_info, "Get hardware telemetry info (backend, memory, bfloat16)");

    py::class_<SubQEngineWrapper>(m, "SubQEngine")
        .def(py::init<size_t, int>(), 
             py::arg("vram_capacity") = 1024 * 1024 * 32,
             py::arg("device_id") = 0)
        .def_property_readonly("device_id", &SubQEngineWrapper::device_id, "Physical GPU device ID for isolated VRAM arena")
        .def("load", &SubQEngineWrapper::load, py::arg("filepath"), "Load weights from safetensors file")
        .def("forward", &SubQEngineWrapper::forward, 
             py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("decay_factor") = 0.99f,
             "Execute forward linear attention scan on NumPy arrays")
        .def("forward_raw", &SubQEngineWrapper::forward_raw,
             py::arg("q_ptr"), py::arg("k_ptr"), py::arg("v_ptr"), py::arg("o_ptr"),
             py::arg("batch"), py::arg("seq_len"), py::arg("d_model"), py::arg("decay_factor") = 0.99f,
             "Execute zero-copy forward pass on raw device pointers")
        .def("backward", &SubQEngineWrapper::backward,
             py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("decay_factor") = 0.99f,
             "Execute reverse adjoint backward pass returning (dQ, dK, dV, d_decay)")
        .def("backward_raw", &SubQEngineWrapper::backward_raw,
             py::arg("do_ptr"), py::arg("q_ptr"), py::arg("k_ptr"), py::arg("v_ptr"),
             py::arg("dq_ptr"), py::arg("dk_ptr"), py::arg("dv_ptr"),
             py::arg("batch"), py::arg("seq_len"), py::arg("d_model"), py::arg("decay_factor") = 0.99f,
             "Execute zero-copy backward pass on raw device pointers returning d_decay");

    py::class_<AudioStreamBuffer>(m, "AudioStreamBuffer")
        .def(py::init<size_t>(), py::arg("max_chunks") = 1024)
        .def("push_samples", [](AudioStreamBuffer& self, py::array_t<float, py::array::c_style | py::array::forcecast> samples, uint64_t timestamp) {
            AudioChunk chunk;
            chunk.timestamp = timestamp;
            py::buffer_info buf = samples.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            size_t n = static_cast<size_t>(buf.size);
            chunk.samples.assign(ptr, ptr + n);
            return self.push(chunk);
        }, py::arg("samples"), py::arg("timestamp") = 0, "Push raw audio samples into buffer")
        .def("pop_samples", [](AudioStreamBuffer& self, bool block) -> py::object {
            AudioChunk chunk;
            if (!self.pop(chunk, block)) {
                return py::none();
            }
            py::array_t<float> result(chunk.samples.size());
            py::buffer_info buf = result.request();
            std::memcpy(buf.ptr, chunk.samples.data(), chunk.samples.size() * sizeof(float));
            return py::make_tuple(chunk.timestamp, result);
        }, py::arg("block") = false, "Pop oldest audio chunk returning (timestamp, samples) or None")
        .def("stop", &AudioStreamBuffer::stop, "Signal buffer stop and shutdown ingestion")
        .def("capacity", &AudioStreamBuffer::capacity, "Get maximum chunk capacity")
        .def("is_stopped", &AudioStreamBuffer::is_stopped, "Check if buffer has been stopped")
        .def("size", &AudioStreamBuffer::size, "Get current number of queued chunks")
        .def("empty", &AudioStreamBuffer::empty, "Check if buffer is empty");
}

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
    explicit SubQEngineWrapper(size_t vram_capacity = 256 * 1024 * 1024) 
        : arena_(vram_capacity), engine_(arena_) {
        SUBQ_ENABLE_DENORMALS_FLUSH();
    }

    void load(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        try {
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

            std::memset(dQ_data, 0, total_elements * sizeof(float));
            std::memset(dK_data, 0, total_elements * sizeof(float));
            std::memset(dV_data, 0, total_elements * sizeof(float));

            const float* dO_ptr = static_cast<const float*>(buf_dO.ptr);
            const float* Q_ptr = static_cast<const float*>(buf_Q.ptr);
            const float* K_ptr = static_cast<const float*>(buf_K.ptr);
            const float* V_ptr = static_cast<const float*>(buf_V.ptr);

            float total_d_decay = 0.0f;

            // Compute exact analytical reverse adjoint linear attention gradients
            std::vector<float> S_history(seq_len * d_model * d_model, 0.0f);
            std::vector<float> Gamma(d_model * d_model, 0.0f);

            for (size_t b = 0; b < batch; ++b) {
                size_t b_offset = b * (seq_len * d_model);

                // 1. Forward pass to record state sequence S_t
                std::vector<float> current_S(d_model * d_model, 0.0f);
                for (size_t t = 0; t < seq_len; ++t) {
                    size_t t_offset = b_offset + t * d_model;
                    size_t hist_offset = t * (d_model * d_model);

                    for (size_t i = 0; i < d_model; ++i) {
                        float k_val = K_ptr[t_offset + i];
                        float phi_k = (k_val > 0.0f) ? (k_val + 1.0f) : std::exp(k_val);

                        for (size_t j = 0; j < d_model; ++j) {
                            float v_val = V_ptr[t_offset + j];
                            float new_s = decay_factor * current_S[i * d_model + j] + phi_k * v_val;
                            if (std::abs(new_s) < 1e-15f) new_s = 0.0f;
                            current_S[i * d_model + j] = new_s;
                            S_history[hist_offset + i * d_model + j] = new_s;
                        }
                    }
                }

                // 2. Backward reverse adjoint scan
                std::fill(Gamma.begin(), Gamma.end(), 0.0f);

                for (int t = static_cast<int>(seq_len) - 1; t >= 0; --t) {
                    size_t t_offset = b_offset + static_cast<size_t>(t) * d_model;
                    size_t hist_offset = static_cast<size_t>(t) * (d_model * d_model);

                    // Compute dQ_t: d\phi(Q_t) = dO_t * S_t^T
                    for (size_t i = 0; i < d_model; ++i) {
                        float q_val = Q_ptr[t_offset + i];
                        float dphi_q_val = (q_val > 0.0f) ? 1.0f : std::exp(q_val);

                        float dphi_Q_i = 0.0f;
                        for (size_t j = 0; j < d_model; ++j) {
                            float do_j = dO_ptr[t_offset + j];
                            float s_ij = S_history[hist_offset + i * d_model + j];
                            dphi_Q_i += do_j * s_ij;
                        }
                        dQ_data[t_offset + i] = dphi_Q_i * dphi_q_val;
                    }

                    // Update Gamma_t = decay_factor * Gamma_{t+1} + phi(Q_t)^T * dO_t
                    for (size_t i = 0; i < d_model; ++i) {
                        float q_val = Q_ptr[t_offset + i];
                        float phi_q = (q_val > 0.0f) ? (q_val + 1.0f) : std::exp(q_val);

                        for (size_t j = 0; j < d_model; ++j) {
                            float do_j = dO_ptr[t_offset + j];
                            Gamma[i * d_model + j] = decay_factor * Gamma[i * d_model + j] + phi_q * do_j;
                        }
                    }

                    // Compute dV_t = phi(K_t) * Gamma_t
                    for (size_t j = 0; j < d_model; ++j) {
                        float dv_j = 0.0f;
                        for (size_t i = 0; i < d_model; ++i) {
                            float k_val = K_ptr[t_offset + i];
                            float phi_k = (k_val > 0.0f) ? (k_val + 1.0f) : std::exp(k_val);
                            dv_j += phi_k * Gamma[i * d_model + j];
                        }
                        dV_data[t_offset + j] = dv_j;
                    }

                    // Compute dK_t = d\phi(K_t) * \phi'(K_t) where d\phi(K_t) = V_t * Gamma_t^T
                    for (size_t i = 0; i < d_model; ++i) {
                        float k_val = K_ptr[t_offset + i];
                        float dphi_k_val = (k_val > 0.0f) ? 1.0f : std::exp(k_val);

                        float dphi_K_i = 0.0f;
                        for (size_t j = 0; j < d_model; ++j) {
                            float v_j = V_ptr[t_offset + j];
                            dphi_K_i += v_j * Gamma[i * d_model + j];
                        }
                        dK_data[t_offset + i] = dphi_K_i * dphi_k_val;
                    }

                    // Accumulate decay gradient d_decay += Tr(S_{t-1}^T * Gamma_t)
                    if (t > 0) {
                        size_t prev_hist_offset = static_cast<size_t>(t - 1) * (d_model * d_model);
                        for (size_t i = 0; i < d_model; ++i) {
                            for (size_t j = 0; j < d_model; ++j) {
                                total_d_decay += S_history[prev_hist_offset + i * d_model + j] * Gamma[i * d_model + j];
                            }
                        }
                    }
                }
            }

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
            const float* dO = reinterpret_cast<const float*>(do_ptr);
            const float* Q = reinterpret_cast<const float*>(q_ptr);
            const float* K = reinterpret_cast<const float*>(k_ptr);
            const float* V = reinterpret_cast<const float*>(v_ptr);

            float* dQ = reinterpret_cast<float*>(dq_ptr);
            float* dK = reinterpret_cast<float*>(dk_ptr);
            float* dV = reinterpret_cast<float*>(dv_ptr);

            float total_d_decay = 0.0f;
            std::vector<float> S_history(seq_len * d_model * d_model, 0.0f);
            std::vector<float> Gamma(d_model * d_model, 0.0f);

            for (size_t b = 0; b < batch; ++b) {
                size_t b_offset = b * (seq_len * d_model);

                std::vector<float> current_S(d_model * d_model, 0.0f);
                for (size_t t = 0; t < seq_len; ++t) {
                    size_t t_offset = b_offset + t * d_model;
                    size_t hist_offset = t * (d_model * d_model);

                    for (size_t i = 0; i < d_model; ++i) {
                        float k_val = K[t_offset + i];
                        float phi_k = (k_val > 0.0f) ? (k_val + 1.0f) : std::exp(k_val);

                        for (size_t j = 0; j < d_model; ++j) {
                            float v_val = V[t_offset + j];
                            float new_s = decay_factor * current_S[i * d_model + j] + phi_k * v_val;
                            if (std::abs(new_s) < 1e-15f) new_s = 0.0f;
                            current_S[i * d_model + j] = new_s;
                            S_history[hist_offset + i * d_model + j] = new_s;
                        }
                    }
                }

                std::fill(Gamma.begin(), Gamma.end(), 0.0f);

                for (int t = static_cast<int>(seq_len) - 1; t >= 0; --t) {
                    size_t t_offset = b_offset + static_cast<size_t>(t) * d_model;
                    size_t hist_offset = static_cast<size_t>(t) * (d_model * d_model);

                    for (size_t i = 0; i < d_model; ++i) {
                        float q_val = Q[t_offset + i];
                        float dphi_q_val = (q_val > 0.0f) ? 1.0f : std::exp(q_val);

                        float dphi_Q_i = 0.0f;
                        for (size_t j = 0; j < d_model; ++j) {
                            float do_j = dO[t_offset + j];
                            float s_ij = S_history[hist_offset + i * d_model + j];
                            dphi_Q_i += do_j * s_ij;
                        }
                        dQ[t_offset + i] = dphi_Q_i * dphi_q_val;
                    }

                    for (size_t i = 0; i < d_model; ++i) {
                        float q_val = Q[t_offset + i];
                        float phi_q = (q_val > 0.0f) ? (q_val + 1.0f) : std::exp(q_val);

                        for (size_t j = 0; j < d_model; ++j) {
                            float do_j = dO[t_offset + j];
                            Gamma[i * d_model + j] = decay_factor * Gamma[i * d_model + j] + phi_q * do_j;
                        }
                    }

                    for (size_t j = 0; j < d_model; ++j) {
                        float dv_j = 0.0f;
                        for (size_t i = 0; i < d_model; ++i) {
                            float k_val = K[t_offset + i];
                            float phi_k = (k_val > 0.0f) ? (k_val + 1.0f) : std::exp(k_val);
                            dv_j += phi_k * Gamma[i * d_model + j];
                        }
                        dV[t_offset + j] = dv_j;
                    }

                    for (size_t i = 0; i < d_model; ++i) {
                        float k_val = K[t_offset + i];
                        float dphi_k_val = (k_val > 0.0f) ? 1.0f : std::exp(k_val);

                        float dphi_K_i = 0.0f;
                        for (size_t j = 0; j < d_model; ++j) {
                            float v_j = V[t_offset + j];
                            dphi_K_i += v_j * Gamma[i * d_model + j];
                        }
                        dK[t_offset + i] = dphi_K_i * dphi_k_val;
                    }

                    if (t > 0) {
                        size_t prev_hist_offset = static_cast<size_t>(t - 1) * (d_model * d_model);
                        for (size_t i = 0; i < d_model; ++i) {
                            for (size_t j = 0; j < d_model; ++j) {
                                total_d_decay += S_history[prev_hist_offset + i * d_model + j] * Gamma[i * d_model + j];
                            }
                        }
                    }
                }
            }

            return total_d_decay;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Engine Backward Raw Error: ") + e.what());
        }
    }

private:
    std::mutex arena_mutex_;
    MemoryArena arena_;
    Engine engine_;
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "Sub-Quadratic Audio Transformer Engine C++ backend (_C)";

    m.def("get_device_info", &get_device_info, "Get hardware telemetry info (backend, memory, bfloat16)");

    py::class_<SubQEngineWrapper>(m, "SubQEngine")
        .def(py::init<size_t>(), py::arg("vram_capacity") = 1024 * 1024 * 256)
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
}

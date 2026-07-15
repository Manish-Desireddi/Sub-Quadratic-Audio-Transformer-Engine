/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include "attention.hpp"
#include "gpu_macros.hpp"
#include <cmath>

#if defined(USE_CUDA) || defined(USE_HIP)
__global__ void elu_plus_one_kernel(float* x, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float val = x[idx];
        if (val > 0.0f) {
            x[idx] = val + 1.0f;
        } else {
            x[idx] = expf(val); // exp(x) - 1 + 1 = exp(x)
        }
    }
}
#endif

void run_feature_map(TensorDescriptor& x) {
    float* data = static_cast<float*>(x.data);
    
    int batch = x.dimensions[0];
    int seq_len = x.dimensions[1];
    int d_model = x.dimensions[2];
    int total_elements = batch * seq_len * d_model;

#if defined(USE_CUDA) || defined(USE_HIP)
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    elu_plus_one_kernel<<<blocks, threads>>>(data, total_elements);
    CHECK_GPU_ERROR(gpuGetLastError());
#else
    // CPU Fallback ELU + 1
    for (int i = 0; i < total_elements; ++i) {
        float val = data[i];
        if (val > 0.0f) {
            data[i] = val + 1.0f;
        } else {
            data[i] = std::exp(val);
        }
    }
#endif
}

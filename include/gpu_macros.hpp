/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GPU_SUCCESS hipSuccess
#define gpuError_t hipError_t
#define gpuMalloc hipMalloc
#define gpuMemset hipMemset
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuHostAlloc hipHostMalloc
#define gpuFreeHost hipHostFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuHostAllocDefault hipHostMallocDefault
#define gpuFree hipFree
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuHalf __fp16
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define GPU_SUCCESS cudaSuccess
#define gpuError_t cudaError_t
#define gpuMalloc cudaMalloc
#define gpuMemset cudaMemset
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuHostAlloc cudaHostAlloc
#define gpuFreeHost cudaFreeHost
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuHostAllocDefault cudaHostAllocDefault
#define gpuFree cudaFree
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuHalf half

#else
// CPU Fallback
#define USE_CPU 1
#define GPU_SUCCESS 0
typedef int gpuError_t;
#define gpuMemcpyHostToDevice 1
#define gpuMemcpyDeviceToHost 2
#define gpuHostAllocDefault 0

inline gpuError_t gpuHostAlloc(void **pHost, size_t size, unsigned int flags) {
    *pHost = std::malloc(size);
    return (*pHost != nullptr) ? GPU_SUCCESS : 1;
}

inline gpuError_t gpuFreeHost(void *pHost) {
    std::free(pHost);
    return GPU_SUCCESS;
}

inline gpuError_t gpuDeviceSynchronize() {
    return GPU_SUCCESS;
}

inline gpuError_t gpuMalloc(void **devPtr, size_t size) {
  *devPtr = std::malloc(size);
  return (*devPtr != nullptr) ? GPU_SUCCESS : 1;
}

inline gpuError_t gpuMemset(void *devPtr, int value, size_t count) {
  std::memset(devPtr, value, count);
  return GPU_SUCCESS;
}

inline gpuError_t gpuMemcpy(void *dst, const void *src, size_t count,
                            int kind) {
  std::memcpy(dst, src, count);
  return GPU_SUCCESS;
}

inline gpuError_t gpuFree(void *devPtr) {
  std::free(devPtr);
  return GPU_SUCCESS;
}

inline const char *gpuGetErrorString(gpuError_t error) {
  return (error == GPU_SUCCESS) ? "no error" : "CPU fallback error";
}

inline gpuError_t gpuGetLastError() {
  return GPU_SUCCESS;
}
#endif

#define CHECK_GPU_ERROR(val) check_gpu((val), #val, __FILE__, __LINE__)

inline void check_gpu(gpuError_t result, char const *const func,
                      const char *const file, int const line) {
  if (result != GPU_SUCCESS) {
    std::string msg =
        std::string("GPU error at ") + file + ":" + std::to_string(line) +
        " code=" + std::to_string(static_cast<unsigned int>(result)) + "(\"" +
        gpuGetErrorString(result) + "\") " + func;
    throw std::runtime_error(msg);
  }
}

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
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuHostAlloc hipHostMalloc
#define gpuFreeHost hipHostFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuHostAllocDefault hipHostMallocDefault
#define gpuFree hipFree
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
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
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuHostAlloc cudaHostAlloc
#define gpuFreeHost cudaFreeHost
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuHostAllocDefault cudaHostAllocDefault
#define gpuFree cudaFree
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuHalf half

#else
// CPU Fallback
#define USE_CPU 1
#define GPU_SUCCESS 0
typedef int gpuError_t;
#define gpuMemcpyHostToDevice 1
#define gpuMemcpyDeviceToHost 2
#define gpuMemcpyDeviceToDevice 3
#define gpuHostAllocDefault 0

#if defined(_MSC_VER)
#include <malloc.h>
#endif

inline gpuError_t gpuSetDevice(int device) {
    (void)device;
    return GPU_SUCCESS;
}

inline gpuError_t gpuGetDevice(int* device) {
    if (device) *device = 0;
    return GPU_SUCCESS;
}

inline gpuError_t gpuHostAlloc(void **pHost, size_t size, unsigned int flags) {
    (void)flags;
#if defined(_MSC_VER)
    *pHost = _aligned_malloc(size, 256);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(pHost, 256, size) != 0) {
        *pHost = nullptr;
    }
#else
    size_t rem = size % 256;
    size_t alloc_size = (rem == 0) ? size : (size + 256 - rem);
    *pHost = std::aligned_alloc(256, alloc_size);
#endif
    return (*pHost != nullptr) ? GPU_SUCCESS : 1;
}

inline gpuError_t gpuFreeHost(void *pHost) {
#if defined(_MSC_VER)
    _aligned_free(pHost);
#else
    std::free(pHost);
#endif
    return GPU_SUCCESS;
}

inline gpuError_t gpuDeviceSynchronize() {
    return GPU_SUCCESS;
}

inline gpuError_t gpuMalloc(void **devPtr, size_t size) {
#if defined(_MSC_VER)
    *devPtr = _aligned_malloc(size, 256);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(devPtr, 256, size) != 0) {
        *devPtr = nullptr;
    }
#else
    size_t rem = size % 256;
    size_t alloc_size = (rem == 0) ? size : (size + 256 - rem);
    *devPtr = std::aligned_alloc(256, alloc_size);
#endif
    return (*devPtr != nullptr) ? GPU_SUCCESS : 1;
}

inline gpuError_t gpuMemset(void *devPtr, int value, size_t count) {
    std::memset(devPtr, value, count);
    return GPU_SUCCESS;
}

inline gpuError_t gpuMemcpy(void *dst, const void *src, size_t count, int kind) {
    (void)kind;
    std::memcpy(dst, src, count);
    return GPU_SUCCESS;
}

inline gpuError_t gpuFree(void *devPtr) {
#if defined(_MSC_VER)
    _aligned_free(devPtr);
#else
    std::free(devPtr);
#endif
    return GPU_SUCCESS;
}

inline const char* gpuGetErrorString(gpuError_t error) {
    (void)error;
    return "CPU Success";
}

inline gpuError_t gpuGetLastError() {
    return GPU_SUCCESS;
}

typedef float gpuHalf;
#endif

// Robust error-checking macro for GPU operations
inline void check_gpu(gpuError_t result, char const *const func, const char *const file, int const line) {
    if (result != GPU_SUCCESS) {
        std::cerr << "GPU Error at " << file << ":" << line << " code=" << result << " \"" << gpuGetErrorString(result) << "\" in " << func << std::endl;
        throw std::runtime_error(std::string("GPU error: ") + gpuGetErrorString(result));
    }
}

#define CHECK_GPU_ERROR(val) check_gpu((val), #val, __FILE__, __LINE__)

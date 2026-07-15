/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include "arena.hpp"

MemoryArena::MemoryArena(size_t max_size_bytes)
    : base_ptr(nullptr), offset(0), capacity(max_size_bytes) {
    void* ptr;
    CHECK_GPU_ERROR(gpuMalloc(&ptr, capacity));
    base_ptr = static_cast<uint8_t*>(ptr);
    CHECK_GPU_ERROR(gpuMemset(base_ptr, 0, capacity));
}

MemoryArena::~MemoryArena() {
    if (base_ptr) {
        // We explicitly ignore errors in destructor to prevent std::terminate
        gpuFree(base_ptr);
        base_ptr = nullptr;
    }
}

uint8_t* MemoryArena::allocate(size_t bytes) {
    // 256-byte alignment is generally good for GPU architectures
    size_t alignment = 256;
    size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);

    if (offset + aligned_bytes > capacity) {
        throw std::runtime_error("OOM: MemoryArena capacity exceeded during allocation.");
    }

    uint8_t* ptr = base_ptr + offset;
    offset += aligned_bytes;
    return ptr;
}

void MemoryArena::reset_head() {
    offset = 0;
    CHECK_GPU_ERROR(gpuMemset(base_ptr, 0, capacity));
}

size_t MemoryArena::get_used_bytes() const {
    return offset;
}

size_t MemoryArena::get_total_bytes() const {
    return capacity;
}

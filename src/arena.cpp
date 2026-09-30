/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
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
    // Hard upper bound: refuse requests larger than arena capacity
    if (bytes == 0 || bytes > capacity) {
        throw std::runtime_error("OOM: Requested allocation exceeds arena capacity.");
    }

    // Safe alignment calculation — guard bytes + (alignment - 1) against wrap-around
    constexpr size_t alignment = 256;
    constexpr size_t align_mask = alignment - 1;
    size_t aligned_bytes;
    if (bytes > SIZE_MAX - align_mask) {
        throw std::runtime_error("OOM: Integer overflow computing aligned allocation size.");
    }
    aligned_bytes = (bytes + align_mask) & ~align_mask;

    // Guard offset accumulation against overflow
    size_t new_offset;
    if (capacity < aligned_bytes || offset > capacity - aligned_bytes) {
        throw std::runtime_error("OOM: MemoryArena capacity exceeded during allocation.");
    }
    new_offset = offset + aligned_bytes;

    uint8_t* ptr = base_ptr + offset;
    offset = new_offset;
    return ptr;
}

void MemoryArena::reset_head() {
    offset = 0;
    CHECK_GPU_ERROR(gpuMemset(base_ptr, 0, capacity));
}

size_t MemoryArena::get_offset() const {
    return offset;
}

void MemoryArena::set_offset(size_t saved_offset) {
    if (saved_offset <= capacity) {
        offset = saved_offset;
    } else {
        throw std::runtime_error("OOM: Invalid offset restoration.");
    }
}

size_t MemoryArena::get_used_bytes() const {
    return offset;
}

size_t MemoryArena::get_total_bytes() const {
    return capacity;
}

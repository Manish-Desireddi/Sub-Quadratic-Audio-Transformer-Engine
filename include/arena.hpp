/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include <cstdint>
#include <cstddef>
#include "gpu_macros.hpp"

class MemoryArena {
public:
    explicit MemoryArena(size_t max_size_bytes);
    ~MemoryArena();

    // Delete copy/move constructors to ensure single ownership
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) = delete;
    MemoryArena& operator=(MemoryArena&&) = delete;

    uint8_t* allocate(size_t bytes);
    void reset_head();
    size_t get_used_bytes() const;
    size_t get_total_bytes() const;

private:
    uint8_t* base_ptr;
    size_t offset;
    size_t capacity;
};

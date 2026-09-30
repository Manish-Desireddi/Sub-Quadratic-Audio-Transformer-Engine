/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "gpu_macros.hpp"

/**
 * @brief 256-byte aligned contiguous bump allocator for GPU VRAM and pinned host memory.
 * 
 * Provides O(1) allocation, offset checkpointing/rewinding for streaming chunk execution,
 * and deterministic memory management with zero fragmentation.
 */
class MemoryArena {
public:
    explicit MemoryArena(size_t max_size_bytes);
    ~MemoryArena();

    // Delete copy and move operations to enforce strict single ownership of raw GPU buffer
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) = delete;
    MemoryArena& operator=(MemoryArena&&) = delete;

    /**
     * @brief Allocates 256-byte aligned memory chunk from the arena.
     * @param bytes Number of bytes to allocate.
     * @return Pointer to the allocated memory block.
     * @throws std::runtime_error if allocation exceeds total capacity.
     */
    uint8_t* allocate(size_t bytes);

    /**
     * @brief Resets the allocation head to 0 and zero-initializes the entire arena buffer.
     */
    void reset_head();

    /**
     * @brief Gets current byte offset (high-water mark) within the arena.
     */
    size_t get_offset() const;

    /**
     * @brief Restores the allocation head to a previously saved offset checkpoint.
     * @param saved_offset Checkpoint offset previously obtained from get_offset().
     * @throws std::runtime_error if saved_offset exceeds capacity.
     */
    void set_offset(size_t saved_offset);

    /**
     * @brief Gets number of bytes currently allocated (equivalent to current offset).
     */
    size_t get_used_bytes() const;

    /**
     * @brief Gets total capacity of the arena in bytes.
     */
    size_t get_total_bytes() const;

private:
    uint8_t* base_ptr;
    size_t offset;
    size_t capacity;
};

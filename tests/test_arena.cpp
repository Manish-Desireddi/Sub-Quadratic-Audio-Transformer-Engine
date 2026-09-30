/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <catch2/catch_test_macros.hpp>
#include "arena.hpp"
#include <cstdint>

TEST_CASE("MemoryArena initialization and basic allocation", "[MemoryArena]") {
    size_t one_mb = 1024 * 1024;
    MemoryArena arena(one_mb);

    REQUIRE(arena.get_total_bytes() == one_mb);
    REQUIRE(arena.get_used_bytes() == 0);
    REQUIRE(arena.get_offset() == 0);

    uint8_t* ptr1 = arena.allocate(256);
    REQUIRE(ptr1 != nullptr);
    REQUIRE(arena.get_used_bytes() == 256);
    REQUIRE(arena.get_offset() == 256);

    size_t used_after_first = arena.get_used_bytes();
    uint8_t* ptr2 = arena.allocate(100);
    REQUIRE(ptr2 != nullptr);
    REQUIRE(ptr2 > ptr1);
    // 100 bytes aligned to 256 bytes = 256 bytes
    REQUIRE(arena.get_used_bytes() == used_after_first + 256);

    arena.reset_head();
    REQUIRE(arena.get_used_bytes() == 0);
    REQUIRE(arena.get_offset() == 0);
}

TEST_CASE("MemoryArena out-of-bounds allocation throws error", "[MemoryArena]") {
    size_t one_mb = 1024 * 1024;
    MemoryArena arena(one_mb);

    // Should succeed
    REQUIRE_NOTHROW(arena.allocate(one_mb - 256));

    // Should throw runtime_error
    REQUIRE_THROWS_AS(arena.allocate(512), std::runtime_error);
}

TEST_CASE("MemoryArena absolute boundary integrity", "[MemoryArena]") {
    size_t cap = 256 * 1024 * 1024; // 256MB boundary
    MemoryArena arena(cap);
    
    uint8_t* p1 = arena.allocate(cap - 256);
    REQUIRE(p1 != nullptr);
    
    // Attempting to exceed O(1) bound
    REQUIRE_THROWS_AS(arena.allocate(512), std::runtime_error);
    
    // Verify pointers fall strictly within allocated buffer
    uint8_t* base = arena.allocate(0); // Should point to current offset
    REQUIRE(reinterpret_cast<uintptr_t>(base) <= reinterpret_cast<uintptr_t>(p1) + cap);
}

TEST_CASE("MemoryArena 256-byte alignment verification", "[MemoryArena][alignment]") {
    size_t cap = 1024 * 1024;
    MemoryArena arena(cap);

    // Allocate odd sizes and verify 256-byte alignment on pointers and offsets
    uint8_t* p1 = arena.allocate(1);
    REQUIRE(p1 != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(p1) % 256 == 0);
    REQUIRE(arena.get_offset() == 256);

    uint8_t* p2 = arena.allocate(37);
    REQUIRE(p2 != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(p2) % 256 == 0);
    REQUIRE(arena.get_offset() == 512);

    uint8_t* p3 = arena.allocate(255);
    REQUIRE(p3 != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(p3) % 256 == 0);
    REQUIRE(arena.get_offset() == 768);

    uint8_t* p4 = arena.allocate(256);
    REQUIRE(p4 != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(p4) % 256 == 0);
    REQUIRE(arena.get_offset() == 1024);

    uint8_t* p5 = arena.allocate(257);
    REQUIRE(p5 != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(p5) % 256 == 0);
    REQUIRE(arena.get_offset() == 1024 + 512);
}

TEST_CASE("MemoryArena offset checkpointing and rewind", "[MemoryArena][checkpoint]") {
    size_t cap = 1024 * 1024;
    MemoryArena arena(cap);

    uint8_t* state_ptr = arena.allocate(1024);
    REQUIRE(state_ptr != nullptr);
    REQUIRE(arena.get_offset() == 1024);

    // Save checkpoint
    size_t checkpoint = arena.get_offset();

    // Allocate transient chunk 1
    uint8_t* chunk1 = arena.allocate(2048);
    REQUIRE(chunk1 != nullptr);
    REQUIRE(arena.get_offset() == 1024 + 2048);

    // Rewind back to checkpoint
    arena.set_offset(checkpoint);
    REQUIRE(arena.get_offset() == checkpoint);
    REQUIRE(arena.get_used_bytes() == 1024);

    // Allocate transient chunk 2 (should reuse the exact same address as chunk 1)
    uint8_t* chunk2 = arena.allocate(2048);
    REQUIRE(chunk2 == chunk1);
    REQUIRE(arena.get_offset() == 1024 + 2048);

    // Test invalid offset restoration
    REQUIRE_THROWS_AS(arena.set_offset(cap + 1024), std::runtime_error);
}

TEST_CASE("MemoryArena exhaustive alignment boundaries stress test", "[MemoryArena][stress][alignment]") {
    const size_t arena_size = 16 * 1024 * 1024; // 16 MB
    MemoryArena arena(arena_size);

    // Test a wide spectrum of sizes
    const size_t test_sizes[] = {
        1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257,
        511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
        8191, 8192, 8193, 16383, 16384, 16385, 32767, 32768, 65535, 65536,
        131071, 131072, 262143, 262144, 524287, 524288, 1048576
    };

    size_t prev_offset = 0;
    for (size_t sz : test_sizes) {
        uint8_t* ptr = arena.allocate(sz);
        REQUIRE(ptr != nullptr);
        
        // Exact 256-byte alignment invariant
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        REQUIRE(addr % 256 == 0);

        size_t cur_offset = arena.get_offset();
        REQUIRE(cur_offset % 256 == 0);

        size_t expected_aligned = (sz + 255) & ~255;
        REQUIRE(cur_offset == prev_offset + expected_aligned);
        prev_offset = cur_offset;

        // Verify write and read integrity at boundaries
        ptr[0] = static_cast<uint8_t>(sz & 0xFF);
        ptr[sz - 1] = static_cast<uint8_t>((sz * 3) & 0xFF);
        REQUIRE(ptr[0] == static_cast<uint8_t>(sz & 0xFF));
        REQUIRE(ptr[sz - 1] == static_cast<uint8_t>((sz * 3) & 0xFF));
    }
}

TEST_CASE("MemoryArena 2,000-iteration checkpoint and rewind stress test", "[MemoryArena][stress][checkpoint]") {
    const size_t arena_size = 4 * 1024 * 1024; // 4 MB
    MemoryArena arena(arena_size);

    // Initial persistent state allocation
    uint8_t* persistent_state = arena.allocate(4096);
    REQUIRE(persistent_state != nullptr);
    std::memset(persistent_state, 0xAB, 4096);

    const size_t checkpoint_offset = arena.get_offset();
    REQUIRE(checkpoint_offset == 4096);

    uint8_t* first_iteration_ptr = nullptr;

    // 2,000 iterations of allocate -> write -> read -> rewind
    const int total_iterations = 2000;
    for (int iter = 0; iter < total_iterations; ++iter) {
        // Allocate transient chunk with varying sizes
        size_t transient_size = 256 + ((iter * 37) % 8192);
        uint8_t* transient_ptr = arena.allocate(transient_size);
        REQUIRE(transient_ptr != nullptr);
        REQUIRE(reinterpret_cast<uintptr_t>(transient_ptr) % 256 == 0);

        if (iter == 0) {
            first_iteration_ptr = transient_ptr;
        } else {
            // Zero memory address drift: Every cycle must rewind to exact same base pointer
            REQUIRE(transient_ptr == first_iteration_ptr);
        }

        // Fill transient buffer with deterministic pattern based on iter
        uint8_t pattern = static_cast<uint8_t>((iter & 0x7F) + 1);
        std::memset(transient_ptr, pattern, transient_size);

        // Verify transient memory content integrity
        REQUIRE(transient_ptr[0] == pattern);
        REQUIRE(transient_ptr[transient_size / 2] == pattern);
        REQUIRE(transient_ptr[transient_size - 1] == pattern);

        // Verify persistent state was NOT corrupted
        REQUIRE(persistent_state[0] == static_cast<uint8_t>(0xAB));
        REQUIRE(persistent_state[4095] == static_cast<uint8_t>(0xAB));

        // Rewind back to checkpoint
        arena.set_offset(checkpoint_offset);
        REQUIRE(arena.get_offset() == checkpoint_offset);
        REQUIRE(arena.get_used_bytes() == checkpoint_offset);
    }

    // Final verification after 2,000 cycles
    REQUIRE(arena.get_offset() == checkpoint_offset);
    uint8_t* post_stress_ptr = arena.allocate(1024);
    REQUIRE(post_stress_ptr == first_iteration_ptr);
}

TEST_CASE("MemoryArena OOM capacity boundary stress and edge cases", "[MemoryArena][stress][oom]") {
    const size_t capacity = 1024 * 1024; // 1MB (1,048,576 bytes)
    MemoryArena arena(capacity);

    // 1. Allocate exact capacity
    REQUIRE_NOTHROW(arena.allocate(capacity));
    REQUIRE(arena.get_offset() == capacity);
    REQUIRE(arena.get_used_bytes() == capacity);

    // Any subsequent non-zero allocation must throw
    REQUIRE_THROWS_AS(arena.allocate(1), std::runtime_error);
    REQUIRE_THROWS_AS(arena.allocate(256), std::runtime_error);

    // 2. Reset and test off-by-one near boundary
    arena.reset_head();
    REQUIRE(arena.get_offset() == 0);

    // Allocate capacity - 256
    REQUIRE_NOTHROW(arena.allocate(capacity - 256));
    REQUIRE(arena.get_offset() == capacity - 256);

    // Allocate remaining 256 bytes (exact fill)
    REQUIRE_NOTHROW(arena.allocate(256));
    REQUIRE(arena.get_offset() == capacity);

    // Next 1 byte must fail
    REQUIRE_THROWS_AS(arena.allocate(1), std::runtime_error);

    // 3. Reset and test single over-capacity allocation
    arena.reset_head();
    REQUIRE_THROWS_AS(arena.allocate(capacity + 1), std::runtime_error);
    REQUIRE_THROWS_AS(arena.allocate(capacity + 256), std::runtime_error);
    REQUIRE_THROWS_AS(arena.allocate(capacity * 2), std::runtime_error);

    // Offset must remain 0 after rejected allocations
    REQUIRE(arena.get_offset() == 0);
}


/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <catch2/catch_test_macros.hpp>
#include "arena.hpp"

TEST_CASE("MemoryArena initialization and basic allocation", "[MemoryArena]") {
    size_t one_mb = 1024 * 1024;
    MemoryArena arena(one_mb);

    REQUIRE(arena.get_total_bytes() == one_mb);
    REQUIRE(arena.get_used_bytes() == 0);

    uint8_t* ptr1 = arena.allocate(256);
    REQUIRE(ptr1 != nullptr);
    REQUIRE(arena.get_used_bytes() > 0); // At least 256 bytes

    size_t used_after_first = arena.get_used_bytes();
    uint8_t* ptr2 = arena.allocate(100);
    REQUIRE(ptr2 != nullptr);
    REQUIRE(ptr2 > ptr1);
    REQUIRE(arena.get_used_bytes() > used_after_first); // Accounts for alignment

    arena.reset_head();
    REQUIRE(arena.get_used_bytes() == 0);
}

TEST_CASE("MemoryArena out-of-bounds allocation throws error", "[MemoryArena]") {
    size_t one_mb = 1024 * 1024;
    MemoryArena arena(one_mb);

    // Should succeed
    REQUIRE_NOTHROW(arena.allocate(one_mb - 256));

    // Should throw runtime_error
    REQUIRE_THROWS_AS(arena.allocate(512), std::runtime_error);
}

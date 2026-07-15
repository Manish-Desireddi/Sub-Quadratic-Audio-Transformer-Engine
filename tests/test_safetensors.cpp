/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include <catch2/catch_test_macros.hpp>
#include "safetensors.hpp"
#include "arena.hpp"
#include <fstream>
#include <cstdint>

TEST_CASE("Safetensors Loader parses correct header", "[safetensors]") {
    // 1. Create a dummy .safetensors file in memory/disk
    std::string dummy_file = "dummy.safetensors";
    
    // Valid JSON header describing 1 tensor and metadata
    std::string json_header = R"({"__metadata__": {"format": "pt"}, "weight": {"dtype": "F32", "shape": [10, 20], "data_offsets": [0, 800]}})";
    
    // Write 8-byte length prefix (little endian)
    uint64_t header_size = json_header.length();
    
    std::ofstream out(dummy_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(json_header.c_str(), header_size);
    
    // Write 800 bytes of dummy binary data
    std::vector<char> dummy_data(800, 0);
    out.write(dummy_data.data(), dummy_data.size());
    out.close();

    // 2. Load with SafetensorLoader
    MemoryArena arena(1024 * 1024 * 10); // 10MB
    SafetensorLoader loader(arena);
    
    bool result = loader.load(dummy_file);
    REQUIRE(result == true);
    
    // Cleanup
    std::remove(dummy_file.c_str());
}

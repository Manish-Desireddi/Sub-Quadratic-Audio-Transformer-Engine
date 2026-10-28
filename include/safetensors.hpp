/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include "tensor.hpp"
#include "arena.hpp"

// Define a simple structure for parsed tensor metadata from Safetensors
struct SafetensorMetadata {
    std::string dtype;
    std::vector<size_t> shape;
    size_t data_offset_start;
    size_t data_offset_end;
};

class SafetensorLoader {
public:
    explicit SafetensorLoader(MemoryArena& arena);

    // Loads a .safetensors file from disk and parses the JSON header.
    // Memory maps the file or reads it, then allocates tensors in the Arena.
    bool load(const std::string& filepath);

    // Checks whether a tensor exists in the loaded safetensors archive
    bool has_tensor(const std::string& name) const;

    // Retrieves a tensor descriptor by name
    TensorDescriptor get_tensor(const std::string& name) const;

    // Accessors for metadata and tensor descriptors
    const std::unordered_map<std::string, SafetensorMetadata>& metadata() const { return metadata_map_; }
    const std::unordered_map<std::string, TensorDescriptor>& tensors() const { return tensors_; }

private:
    MemoryArena& arena_;
    std::string filepath_;
    size_t data_start_offset_; // offset where raw binary data begins
    std::unordered_map<std::string, SafetensorMetadata> metadata_map_;
    std::unordered_map<std::string, TensorDescriptor> tensors_;
};

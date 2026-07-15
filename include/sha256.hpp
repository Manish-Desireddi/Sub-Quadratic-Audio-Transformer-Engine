/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include <string>
#include <cstdint>
#include <vector>

class SHA256 {
public:
    SHA256();
    void update(const uint8_t* data, size_t length);
    std::string digest();

private:
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];

    void transform(const uint8_t* data);
};

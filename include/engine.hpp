/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include "attention.hpp"
#include "arena.hpp"

class Engine {
public:
    Engine(MemoryArena& arena);
    void forward(AttentionContext& ctx);
    void backward(AttentionContext& ctx);
private:
    MemoryArena& arena_;
};

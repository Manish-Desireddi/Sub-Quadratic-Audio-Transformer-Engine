/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include "engine.hpp"

Engine::Engine(MemoryArena& arena) : arena_(arena) {}

void Engine::forward(AttentionContext& ctx) {
    run_feature_map(ctx.K);
    run_feature_map(ctx.Q);
    run_attention_forward(ctx);
}

void Engine::backward(AttentionContext& ctx) {
    run_attention_backward(ctx);
}

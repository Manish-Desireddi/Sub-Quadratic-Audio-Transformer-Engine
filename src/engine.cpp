/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include "engine.hpp"

Engine::Engine(MemoryArena& arena) : arena_(arena) {}

void Engine::forward(AttentionContext& ctx) {
    run_attention_forward(ctx);
}

void Engine::backward(AttentionContext& ctx) {
    run_attention_backward(ctx);
}

void Engine::backward(subq::AttentionBackwardContext<float>& ctx) {
    subq::run_attention_backward_raw(ctx);
}

void Engine::backward(subq::AttentionBackwardContext<double>& ctx) {
    subq::run_attention_backward_raw(ctx);
}

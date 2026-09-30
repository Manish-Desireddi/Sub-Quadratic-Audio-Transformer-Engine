/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include <vector>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <chrono>

// Represents a chunk of raw PCM float audio data
struct AudioChunk {
    std::vector<float> samples;
    uint64_t timestamp;
};

// Thread-safe lock-free ring-buffer for high-throughput asynchronous audio ingestion
class AudioStreamBuffer {
public:
    AudioStreamBuffer(size_t max_chunks = 1024)
        : max_chunks_(max_chunks), head_(0), tail_(0), stopped_(false) {
        if (max_chunks == 0 || (max_chunks & (max_chunks - 1)) != 0) {
            // Power of 2 required for fast modulo mask
            size_t power = 1;
            while(power < max_chunks) power *= 2;
            max_chunks_ = power;
        }
        buffer_.resize(max_chunks_);
        mask_ = max_chunks_ - 1;
    }

    // Push a new audio chunk into the buffer (Producer)
    bool push(const AudioChunk& chunk) {
        if (stopped_.load(std::memory_order_relaxed)) return false;
        
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) & mask_;
        
        size_t current_head = head_.load(std::memory_order_acquire);
        
        if (next_tail == current_head) {
            // Buffer full: Acoustic frame drop!
            // Try to advance the head pointer to overwrite the oldest unread chunk.
            // If the consumer pops concurrently, the CAS will fail, meaning space opened up anyway.
            size_t next_head = (current_head + 1) & mask_;
            head_.compare_exchange_strong(current_head, next_head, std::memory_order_release, std::memory_order_relaxed);
        }
        
        buffer_[current_tail] = chunk;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // Pop the oldest audio chunk from the buffer (Consumer)
    bool pop(AudioChunk& chunk, bool block = true) {
        while (true) {
            size_t current_head = head_.load(std::memory_order_acquire);

            if (current_head == tail_.load(std::memory_order_acquire)) {
                // Buffer empty
                if (!block || stopped_.load(std::memory_order_relaxed)) return false;
                std::this_thread::yield();
                continue;
            }

            AudioChunk staged = buffer_[current_head];
            size_t next_head = (current_head + 1) & mask_;

            // CAS prevents ABA: if producer advanced head_ (frame drop) between our load and
            // this store, the CAS fails and we retry from the updated position.
            if (head_.compare_exchange_weak(
                    current_head, next_head,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                chunk = std::move(staged);
                return true;
            }
            // CAS failed — producer advanced head_; retry with new head value
        }
    }


    void stop() {
        stopped_.store(true, std::memory_order_release);
    }

private:
    std::vector<AudioChunk> buffer_;
    size_t max_chunks_;
    size_t mask_;
    
    // Align atomics to cache lines to prevent false sharing between producer/consumer
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(64) std::atomic<bool> stopped_;
};

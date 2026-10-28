/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once
#include <vector>
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iterator>

constexpr size_t AUDIO_CHUNK_CAPACITY = 16384;

// Fixed-capacity static audio buffer eliminating all heap malloc/free during streaming
template <size_t MaxSamples = AUDIO_CHUNK_CAPACITY>
class StaticAudioBuffer {
public:
    StaticAudioBuffer() noexcept : size_(0) {}

    StaticAudioBuffer(std::initializer_list<float> init) noexcept : size_(0) {
        assign(init.begin(), init.end());
    }

    StaticAudioBuffer(const std::vector<float>& vec) noexcept : size_(0) {
        assign(vec.begin(), vec.end());
    }

    StaticAudioBuffer& operator=(const std::vector<float>& vec) {
        assign(vec.begin(), vec.end());
        return *this;
    }

    template <typename InputIt>
    void assign(InputIt first, InputIt last) {
        size_t n = static_cast<size_t>(std::distance(first, last));
        if (n > MaxSamples) {
            n = MaxSamples;
        }
        std::copy(first, first + n, data_.begin());
        size_ = n;
    }

    void assign(size_t n, float val) {
        resize(n, val);
    }

    void resize(size_t n, float val = 0.0f) {
        if (n > MaxSamples) {
            n = MaxSamples;
        }
        if (n > size_) {
            std::fill(data_.begin() + size_, data_.begin() + n, val);
        }
        size_ = n;
    }

    void push_back(float val) noexcept {
        if (size_ < MaxSamples) {
            data_[size_++] = val;
        }
    }

    void clear() noexcept {
        size_ = 0;
    }

    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr size_t capacity() const noexcept { return MaxSamples; }
    [[nodiscard]] constexpr size_t max_size() const noexcept { return MaxSamples; }

    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    float& operator[](size_t idx) noexcept { return data_[idx]; }
    const float& operator[](size_t idx) const noexcept { return data_[idx]; }

    float& at(size_t idx) {
        if (idx >= size_) throw std::out_of_range("StaticAudioBuffer out of range");
        return data_[idx];
    }
    const float& at(size_t idx) const {
        if (idx >= size_) throw std::out_of_range("StaticAudioBuffer out of range");
        return data_[idx];
    }

    float* begin() noexcept { return data_.data(); }
    float* end() noexcept { return data_.data() + size_; }
    const float* begin() const noexcept { return data_.data(); }
    const float* end() const noexcept { return data_.data() + size_; }

    bool operator==(const StaticAudioBuffer& other) const noexcept {
        if (size_ != other.size_) return false;
        return std::equal(begin(), end(), other.begin());
    }

private:
    std::array<float, MaxSamples> data_{};
    size_t size_{0};
};

// Represents a chunk of raw PCM float audio data with fixed static capacity
struct AudioChunk {
    StaticAudioBuffer<AUDIO_CHUNK_CAPACITY> samples;
    uint64_t timestamp{0};
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

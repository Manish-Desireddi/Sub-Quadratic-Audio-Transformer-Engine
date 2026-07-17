/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include "engine.hpp"
#include "arena.hpp"
#include "attention.hpp"
#include "audio_stream.hpp"
#include <iostream>
#include <thread>
#include <chrono>

// IEEE-754 compliant host-side float32 to float16 conversion
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exponent = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x007fffff;

    if (exponent <= 0) { // Denormal or zero
        if (exponent < -10) return sign; // Underflow to zero
        mantissa = (mantissa | 0x00800000) >> (1 - exponent);
        return sign | (mantissa >> 13);
    } else if (exponent == 0xff - 127 + 15) { // Inf or NaN
        if (mantissa == 0) return sign | 0x7c00; // Inf
        return sign | 0x7c00 | (mantissa >> 13); // NaN
    } else if (exponent > 30) { // Overflow to Inf
        return sign | 0x7c00;
    }
    return sign | (exponent << 10) | (mantissa >> 13);
}

// Forward declaration from src/audio_ingest.cpp
std::vector<float> load_wav(const std::string& filepath);

void producer_thread(AudioStreamBuffer& buffer, const std::string& audio_file) {
    try {
        std::vector<float> raw_audio = load_wav(audio_file);
        std::cout << "[Producer] Loaded " << raw_audio.size() << " samples from " << audio_file << "\n";
        
        // Simulate streaming by chunking the audio
        size_t chunk_size = 1024 * 256; // Fixed context size
        for (size_t i = 0; i < raw_audio.size(); i += chunk_size) {
            AudioChunk chunk;
            chunk.timestamp = i;
            size_t end = std::min(i + chunk_size, raw_audio.size());
            chunk.samples = std::vector<float>(raw_audio.begin() + i, raw_audio.begin() + end);
            
            // Pad if necessary
            if (chunk.samples.size() < chunk_size) {
                chunk.samples.resize(chunk_size, 0.0f);
            }
            
            buffer.push(chunk);
            // Simulate live real-time audio pacing (~50ms per chunk for benchmark)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    } catch (const std::exception& e) {
        std::cerr << "[Producer] Fatal Error: " << e.what() << "\n";
        throw;
    }
    
    buffer.stop();
    std::cout << "[Producer] Finished streaming audio.\n";
}

void consumer_thread(AudioStreamBuffer& buffer) {
    std::cout << "[Consumer] Initializing Native C++ Engine (Pinned Memory, Native Execution)...\n";
    size_t vram_capacity = 256 * 1024 * 1024; // 256 MB
    MemoryArena arena(vram_capacity);
    Engine engine(arena);
    
    size_t batch = 1;
    size_t seq_len = 1024;
    size_t d_model = 256;
    
    // In FP16 mode, each element is 2 bytes (uint16_t)
    size_t tensor_elements = batch * seq_len * d_model;
    size_t tensor_bytes = tensor_elements * sizeof(uint16_t);
    size_t state_bytes = batch * d_model * d_model * sizeof(float);
    
    // Pre-allocate persistent device memory
    uint16_t* d_Q = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    uint16_t* d_K = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    uint16_t* d_V = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    uint16_t* d_O = reinterpret_cast<uint16_t*>(arena.allocate(tensor_bytes));
    float* d_S = reinterpret_cast<float*>(arena.allocate(state_bytes));

    AttentionContext ctx;
    ctx.Q.dimensions = {batch, seq_len, d_model}; ctx.Q.precision = PrecisionMode::FP16; ctx.Q.data = d_Q; ctx.Q.is_on_host = true;
    ctx.K.dimensions = {batch, seq_len, d_model}; ctx.K.precision = PrecisionMode::FP16; ctx.K.data = d_K; ctx.K.is_on_host = true;
    ctx.V.dimensions = {batch, seq_len, d_model}; ctx.V.precision = PrecisionMode::FP16; ctx.V.data = d_V; ctx.V.is_on_host = true;
    ctx.S.dimensions = {batch, d_model, d_model}; ctx.S.precision = PrecisionMode::FP32; ctx.S.data = d_S; ctx.S.is_on_host = true; 
    ctx.O.dimensions = {batch, seq_len, d_model}; ctx.O.precision = PrecisionMode::FP16; ctx.O.data = d_O; ctx.O.is_on_host = true;

    AudioChunk chunk;
    int frames_processed = 0;
    
    // Allocate Pinned Host Memory for fast H2D transfers and FP16 conversion
    uint16_t* h_pinned_data = nullptr;
#if defined(USE_CUDA) || defined(USE_HIP)
    gpuHostAlloc((void**)&h_pinned_data, tensor_bytes, gpuHostAllocDefault);
#else
    h_pinned_data = new uint16_t[tensor_elements];
#endif
    
    auto process_start = std::chrono::high_resolution_clock::now();
    double total_execution_ms = 0.0;

    while (buffer.pop(chunk, true)) {
        // Cast raw float audio to FP16 in Pinned Memory via IEEE-754 bitwise math
        for (size_t i = 0; i < tensor_elements && i < chunk.samples.size(); ++i) {
            uint16_t fp16_val = float_to_half(chunk.samples[i]);
            std::memcpy(&h_pinned_data[i], &fp16_val, sizeof(uint16_t));
        }
        
        // H2D Transfer from Pinned Memory
#if defined(USE_CUDA) || defined(USE_HIP)
        gpuMemcpy(d_Q, h_pinned_data, tensor_bytes, gpuMemcpyHostToDevice);
        gpuMemcpy(d_K, h_pinned_data, tensor_bytes, gpuMemcpyHostToDevice);
        gpuMemcpy(d_V, h_pinned_data, tensor_bytes, gpuMemcpyHostToDevice);
        gpuMemset(d_S, 0, state_bytes);
        gpuDeviceSynchronize();
#endif

        // Strictly measure ON-DEVICE Execution Latency
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        engine.forward(ctx);
        
#if defined(USE_CUDA) || defined(USE_HIP)
        gpuDeviceSynchronize();
#endif
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_elapsed = iter_end - iter_start;
        total_execution_ms += iter_elapsed.count();
        
        frames_processed++;
    }
    
    auto process_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_elapsed = process_end - process_start;
    
    std::cout << "[Consumer] Processed " << frames_processed << " frames.\n";
    if (frames_processed > 0) {
        std::cout << "[Consumer] Average ON-DEVICE Execution Latency per Frame: " << (total_execution_ms / frames_processed) << " ms\n";
        std::cout << "[Consumer] Total Pipeline Time (including pinned H2D): " << total_elapsed.count() << " ms\n";
    }

#if defined(USE_CUDA) || defined(USE_HIP)
    gpuFreeHost(h_pinned_data);
#else
    delete[] h_pinned_data;
#endif
}

int main(int argc, char** argv) {
    std::cout << "=== Sub-Quadratic Audio Engine (Native C++) ===\n";
    
    std::string audio_file = "dummy_audio.wav";
    if (argc > 1) {
        audio_file = argv[1];
    }
    
    if (audio_file == "dummy_audio.wav") {
        std::cerr << "Usage: subq_cli <path_to_audio_file.wav>\n";
        return 1;
    }
    
    AudioStreamBuffer ring_buffer(16);
    
    std::thread prod(producer_thread, std::ref(ring_buffer), audio_file);
    std::thread cons(consumer_thread, std::ref(ring_buffer));
    
    prod.join();
    cons.join();
    
    std::cout << "Engine Shutdown Gracefully.\n";
    return 0;
}

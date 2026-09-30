/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "network_ingest.hpp"
#include "audio_stream.hpp"

#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXNetSystem.h>

#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>

namespace {

// Helper to poll for conditions with a timeout (avoids flaky sleep calls)
template <typename Predicate>
bool wait_for_condition(Predicate pred, std::chrono::milliseconds timeout = std::chrono::milliseconds(3000)) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

// Global network subsystem initializer for Windows Winsock
struct NetSystemGuard {
    NetSystemGuard() { ix::initNetSystem(); }
    ~NetSystemGuard() { ix::uninitNetSystem(); }
};
static NetSystemGuard s_net_guard;

// Helper to serialize NetworkAudioHeader + payload into a binary string
std::string build_packet(uint32_t stream_id, uint64_t seq_no, uint64_t timestamp_us,
                         uint32_t sample_rate, uint16_t channels, uint16_t format,
                         const void* payload_data, size_t payload_bytes,
                         uint32_t magic = subq::SUBQ_HEADER_MAGIC) {
    subq::NetworkAudioHeader header;
    header.magic = magic;
    header.stream_id = stream_id;
    header.seq_no = seq_no;
    header.timestamp_us = timestamp_us;
    header.sample_rate = sample_rate;
    header.channels = channels;
    header.format = format;
    header.payload_bytes = static_cast<uint32_t>(payload_bytes);

    std::string packet;
    packet.resize(sizeof(subq::NetworkAudioHeader) + payload_bytes);
    std::memcpy(&packet[0], &header, sizeof(subq::NetworkAudioHeader));
    if (payload_bytes > 0 && payload_data != nullptr) {
        std::memcpy(&packet[sizeof(subq::NetworkAudioHeader)], payload_data, payload_bytes);
    }
    return packet;
}

} // anonymous namespace

TEST_CASE("NetworkAudioHeader struct alignment and layout", "[network][header]") {
    REQUIRE(sizeof(subq::NetworkAudioHeader) == 36);
    REQUIRE(subq::SUBQ_HEADER_MAGIC == 0x53554251);
    REQUIRE(subq::SUBQ_HEADER_MAGIC_SWAPPED == 0x51425553);
}

TEST_CASE("Live stream packet ingestion over local WebSocket", "[network][ingest][websocket]") {
    const uint16_t test_port = 18081;
    subq::NetworkStreamConfig config;
    config.host = "127.0.0.1";
    config.port = test_port;
    config.endpoint = "/audio";
    config.sample_rate = 24000;
    config.channels = 1;
    config.chunk_size = 512;

    AudioStreamBuffer ring_buffer(64);
    auto receiver = subq::create_websocket_receiver();
    REQUIRE(receiver != nullptr);

    REQUIRE(receiver->start(config, ring_buffer));
    REQUIRE(receiver->is_active());

    // Setup client
    ix::WebSocket client;
    std::string url = "ws://127.0.0.1:" + std::to_string(test_port) + "/audio";
    client.setUrl(url);

    std::atomic<bool> connected{false};
    client.setOnMessageCallback([&connected](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) {
            connected.store(true);
        }
    });

    client.start();
    REQUIRE(wait_for_condition([&] { return connected.load(); }));

    // Send 10 packets of Float32 audio
    const size_t num_packets = 10;
    const size_t samples_per_packet = 512;
    std::vector<float> sample_data(samples_per_packet);
    for (size_t i = 0; i < samples_per_packet; ++i) {
        sample_data[i] = std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 24000.0f);
    }

    for (size_t i = 0; i < num_packets; ++i) {
        std::string pkt = build_packet(
            /*stream_id=*/1,
            /*seq_no=*/i,
            /*timestamp_us=*/1000000 + i * 21333,
            /*sample_rate=*/24000,
            /*channels=*/1,
            /*format=*/0, // Float32
            sample_data.data(),
            sample_data.size() * sizeof(float)
        );
        client.sendBinary(pkt);
    }

    // Wait for all 10 packets to be received
    REQUIRE(wait_for_condition([&] { return receiver->get_packets_received() >= num_packets; }));

    // Pop and verify chunks
    for (size_t i = 0; i < num_packets; ++i) {
        AudioChunk chunk;
        REQUIRE(ring_buffer.pop(chunk, /*block=*/false));
        REQUIRE(chunk.timestamp == 1000000 + i * 21333);
        REQUIRE(chunk.samples.size() == samples_per_packet);
        for (size_t s = 0; s < samples_per_packet; ++s) {
            REQUIRE_THAT(chunk.samples[s], Catch::Matchers::WithinAbs(sample_data[s], 1e-5f));
        }
    }

    REQUIRE(receiver->get_packets_received() == num_packets);
    REQUIRE(receiver->get_dropped_packets() == 0);
    REQUIRE(receiver->get_bytes_received() == num_packets * (sizeof(subq::NetworkAudioHeader) + samples_per_packet * sizeof(float)));

    client.stop();
    receiver->stop();
    REQUIRE_FALSE(receiver->is_active());
}

TEST_CASE("Timestamp continuity and sequence ordering", "[network][ordering][sequence]") {
    const uint16_t test_port = 18082;
    subq::NetworkStreamConfig config;
    config.host = "127.0.0.1";
    config.port = test_port;
    config.endpoint = "/audio";

    AudioStreamBuffer ring_buffer(256);
    auto receiver = subq::create_websocket_receiver();
    REQUIRE(receiver->start(config, ring_buffer));

    ix::WebSocket client;
    client.setUrl("ws://127.0.0.1:" + std::to_string(test_port) + "/audio");
    std::atomic<bool> connected{false};
    client.setOnMessageCallback([&connected](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) connected.store(true);
    });
    client.start();
    REQUIRE(wait_for_condition([&] { return connected.load(); }));

    const size_t packet_count = 50;
    const uint64_t base_ts = 5000000;
    const uint64_t interval_us = 10000;
    std::vector<float> dummy_samples(128, 0.123f);

    for (size_t i = 0; i < packet_count; ++i) {
        std::string pkt = build_packet(
            1, i, base_ts + i * interval_us, 24000, 1, 0,
            dummy_samples.data(), dummy_samples.size() * sizeof(float)
        );
        client.sendBinary(pkt);
    }

    REQUIRE(wait_for_condition([&] { return receiver->get_packets_received() >= packet_count; }));

    uint64_t last_ts = 0;
    for (size_t i = 0; i < packet_count; ++i) {
        AudioChunk chunk;
        REQUIRE(ring_buffer.pop(chunk, /*block=*/false));
        REQUIRE(chunk.timestamp == base_ts + i * interval_us);
        if (i > 0) {
            REQUIRE(chunk.timestamp > last_ts);
            REQUIRE(chunk.timestamp - last_ts == interval_us);
        }
        last_ts = chunk.timestamp;
    }

    client.stop();
    receiver->stop();
}

TEST_CASE("Format decoding: Float32 and Int16 Mono/Stereo", "[network][decoding][formats]") {
    const uint16_t test_port = 18083;
    subq::NetworkStreamConfig config;
    config.host = "127.0.0.1";
    config.port = test_port;

    AudioStreamBuffer ring_buffer(64);
    auto receiver = subq::create_websocket_receiver();
    REQUIRE(receiver->start(config, ring_buffer));

    ix::WebSocket client;
    client.setUrl("ws://127.0.0.1:" + std::to_string(test_port) + "/audio");
    std::atomic<bool> connected{false};
    client.setOnMessageCallback([&connected](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) connected.store(true);
    });
    client.start();
    REQUIRE(wait_for_condition([&] { return connected.load(); }));

    SECTION("Float32 Stereo Downmix to Mono") {
        // Interleaved Stereo: [L0, R0, L1, R1]
        std::vector<float> stereo_samples = {0.2f, 0.6f, -0.4f, 0.8f};
        std::string pkt = build_packet(1, 0, 100, 24000, /*channels=*/2, /*format=Float32*/0,
                                       stereo_samples.data(), stereo_samples.size() * sizeof(float));
        client.sendBinary(pkt);

        REQUIRE(wait_for_condition([&] { return receiver->get_packets_received() >= 1; }));
        AudioChunk chunk;
        REQUIRE(ring_buffer.pop(chunk, false));
        REQUIRE(chunk.samples.size() == 2);
        REQUIRE_THAT(chunk.samples[0], Catch::Matchers::WithinAbs(0.4f, 1e-5f)); // (0.2 + 0.6) / 2
        REQUIRE_THAT(chunk.samples[1], Catch::Matchers::WithinAbs(0.2f, 1e-5f)); // (-0.4 + 0.8) / 2
    }

    SECTION("Int16 Mono Normalized Conversion") {
        // Int16 samples: 0, 16384 (0.5), -32768 (-1.0), 32767 (~0.99997)
        std::vector<int16_t> pcm16 = {0, 16384, -32768, 32767};
        std::string pkt = build_packet(1, 1, 200, 24000, /*channels=*/1, /*format=Int16*/1,
                                       pcm16.data(), pcm16.size() * sizeof(int16_t));
        client.sendBinary(pkt);

        REQUIRE(wait_for_condition([&] { return receiver->get_packets_received() >= 1; }));
        AudioChunk chunk;
        REQUIRE(ring_buffer.pop(chunk, false));
        REQUIRE(chunk.samples.size() == 4);
        REQUIRE_THAT(chunk.samples[0], Catch::Matchers::WithinAbs(0.0f, 1e-4f));
        REQUIRE_THAT(chunk.samples[1], Catch::Matchers::WithinAbs(0.5f, 1e-4f));
        REQUIRE_THAT(chunk.samples[2], Catch::Matchers::WithinAbs(-1.0f, 1e-4f));
        REQUIRE_THAT(chunk.samples[3], Catch::Matchers::WithinAbs(0.999969f, 1e-4f));
    }

    SECTION("Int16 Stereo Normalized Downmix") {
        // Stereo pairs: (16384, 32767), (-32768, 0)
        std::vector<int16_t> pcm16_stereo = {16384, 32767, -32768, 0};
        std::string pkt = build_packet(1, 2, 300, 24000, /*channels=*/2, /*format=Int16*/1,
                                       pcm16_stereo.data(), pcm16_stereo.size() * sizeof(int16_t));
        client.sendBinary(pkt);

        REQUIRE(wait_for_condition([&] { return receiver->get_packets_received() >= 1; }));
        AudioChunk chunk;
        REQUIRE(ring_buffer.pop(chunk, false));
        REQUIRE(chunk.samples.size() == 2);
        REQUIRE_THAT(chunk.samples[0], Catch::Matchers::WithinAbs(0.74998f, 1e-4f)); // (0.5 + 1.0)/2
        REQUIRE_THAT(chunk.samples[1], Catch::Matchers::WithinAbs(-0.5f, 1e-4f));    // (-1.0 + 0.0)/2
    }

    SECTION("Raw PCM Fallback Ingestion") {
        std::vector<float> raw_floats = {0.1f, 0.2f, 0.3f, 0.4f};
        std::string raw_pkt(reinterpret_cast<const char*>(raw_floats.data()), raw_floats.size() * sizeof(float));
        client.sendBinary(raw_pkt);

        REQUIRE(wait_for_condition([&] { return receiver->get_packets_received() >= 1; }));
        AudioChunk chunk;
        REQUIRE(ring_buffer.pop(chunk, false));
        REQUIRE(chunk.samples.size() == 4);
        REQUIRE_THAT(chunk.samples[0], Catch::Matchers::WithinAbs(0.1f, 1e-5f));
        REQUIRE_THAT(chunk.samples[3], Catch::Matchers::WithinAbs(0.4f, 1e-5f));
    }

    SECTION("Corrupt / Malformed Packet Rejection") {
        // Bad magic header
        std::vector<float> data = {1.0f, 2.0f};
        std::string bad_pkt = build_packet(1, 99, 999, 24000, 1, 0,
                                           data.data(), data.size() * sizeof(float),
                                           /*magic=*/0xDEADBEEF);
        client.sendBinary(bad_pkt);

        // Truncated packet
        std::string trunc_pkt = bad_pkt.substr(0, 16);
        client.sendBinary(trunc_pkt);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        AudioChunk chunk;
        REQUIRE_FALSE(ring_buffer.pop(chunk, /*block=*/false));
        REQUIRE(receiver->get_dropped_packets() >= 1);
    }

    client.stop();
    receiver->stop();
}

TEST_CASE("Backpressure: Atomic CAS drop-oldest frame policy", "[network][backpressure][lockfree]") {
    const uint16_t test_port = 18084;
    subq::NetworkStreamConfig config;
    config.host = "127.0.0.1";
    config.port = test_port;

    // Small ring buffer of size 8
    AudioStreamBuffer small_buffer(8);
    auto receiver = subq::create_websocket_receiver();
    REQUIRE(receiver->start(config, small_buffer));

    ix::WebSocket client;
    client.setUrl("ws://127.0.0.1:" + std::to_string(test_port) + "/audio");
    std::atomic<bool> connected{false};
    client.setOnMessageCallback([&connected](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) connected.store(true);
    });
    client.start();
    REQUIRE(wait_for_condition([&] { return connected.load(); }));

    // Burst 50 packets without popping to force buffer overwrite
    const size_t burst_count = 50;
    std::vector<float> sample = {0.42f};
    for (size_t i = 0; i < burst_count; ++i) {
        std::string pkt = build_packet(1, i, 1000 + i, 24000, 1, 0, sample.data(), sizeof(float));
        client.sendBinary(pkt);
    }

    REQUIRE(wait_for_condition([&] { return receiver->get_packets_received() >= burst_count; }));

    // Pop all remaining chunks
    std::vector<AudioChunk> popped;
    AudioChunk chunk;
    while (small_buffer.pop(chunk, /*block=*/false)) {
        popped.push_back(chunk);
    }

    // Buffer capacity is 8, so we should have at most 8 newest chunks
    REQUIRE(popped.size() <= 8);
    REQUIRE(popped.size() > 0);

    // Oldest popped chunk must have a timestamp from the latter half of the burst
    REQUIRE(popped.front().timestamp >= 1000 + (burst_count - 8));
    REQUIRE(popped.back().timestamp == 1000 + burst_count - 1);

    client.stop();
    receiver->stop();
}

TEST_CASE("Clean start/stop lifecycle and zero resource leaks", "[network][lifecycle][raii]") {
    const uint16_t test_port = 18085;
    subq::NetworkStreamConfig config;
    config.host = "127.0.0.1";
    config.port = test_port;

    AudioStreamBuffer ring_buffer(32);

    SECTION("Rapid Start/Stop Cycling on Same Port") {
        auto receiver = subq::create_websocket_receiver();
        for (int iter = 0; iter < 5; ++iter) {
            REQUIRE(receiver->start(config, ring_buffer));
            REQUIRE(receiver->is_active());
            receiver->stop();
            REQUIRE_FALSE(receiver->is_active());
        }
    }

    SECTION("Consumer Unblocking on Buffer Stop") {
        std::atomic<bool> consumer_exited{false};
        std::thread consumer([&] {
            AudioChunk chunk;
            bool res = ring_buffer.pop(chunk, /*block=*/true);
            consumer_exited.store(!res);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ring_buffer.stop();

        REQUIRE(wait_for_condition([&] { return consumer_exited.load(); }));
        consumer.join();
    }

    SECTION("RAII Destructor Safe Teardown") {
        {
            auto receiver = subq::create_websocket_receiver();
            REQUIRE(receiver->start(config, ring_buffer));
            REQUIRE(receiver->is_active());
            // Exiting block destroys receiver without explicit stop()
        }
        // Port must be immediately re-bindable
        auto receiver2 = subq::create_websocket_receiver();
        REQUIRE(receiver2->start(config, ring_buffer));
        REQUIRE(receiver2->is_active());
        receiver2->stop();
    }
}

TEST_CASE("AudioStreamBuffer high-concurrency producer-consumer stress test", "[audio_stream][stress][concurrency]") {
    const size_t buffer_size = 128;
    AudioStreamBuffer ring_buffer(buffer_size);

    const size_t total_chunks = 50000;
    const size_t samples_per_chunk = 64;

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> chunks_popped{0};
    std::atomic<uint64_t> last_popped_timestamp{0};
    std::atomic<bool> monotonic_timestamps{true};
    std::atomic<bool> samples_valid{true};

    // Consumer thread
    std::thread consumer([&] {
        while (true) {
            AudioChunk chunk;
            bool popped = ring_buffer.pop(chunk, /*block=*/false);
            if (!popped) {
                if (producer_done.load(std::memory_order_relaxed)) {
                    // Try one last time to avoid race condition
                    if (!ring_buffer.pop(chunk, /*block=*/false)) {
                        break;
                    }
                } else {
                    std::this_thread::yield();
                    continue;
                }
            }
            {
                size_t count = chunks_popped.fetch_add(1, std::memory_order_relaxed);
                if (count > 0) {
                    uint64_t prev = last_popped_timestamp.load(std::memory_order_relaxed);
                    if (chunk.timestamp <= prev) {
                        monotonic_timestamps.store(false, std::memory_order_relaxed);
                    }
                }
                last_popped_timestamp.store(chunk.timestamp, std::memory_order_relaxed);

                // Verify sample validity (no torn writes or NaN/Inf)
                if (chunk.samples.size() != samples_per_chunk) {
                    samples_valid.store(false, std::memory_order_relaxed);
                } else {
                    for (float s : chunk.samples) {
                        if (std::isnan(s) || std::isinf(s)) {
                            samples_valid.store(false, std::memory_order_relaxed);
                        }
                    }
                }
            }
        }
    });

    // Producer thread
    std::thread producer([&] {
        std::vector<float> base_samples(samples_per_chunk, 0.5f);
        for (size_t i = 0; i < total_chunks; ++i) {
            AudioChunk chunk;
            chunk.timestamp = 1000 + i;
            chunk.samples = base_samples;
            chunk.samples[0] = static_cast<float>(i); // mark with sequence
            
            while (!ring_buffer.push(chunk)) {
                std::this_thread::yield();
            }
            if (i % 1000 == 0) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();

    // Drain remaining chunks
    AudioChunk remaining;
    while (ring_buffer.pop(remaining, false)) {
        chunks_popped.fetch_add(1, std::memory_order_relaxed);
        if (remaining.timestamp <= last_popped_timestamp.load(std::memory_order_relaxed)) {
            monotonic_timestamps.store(false, std::memory_order_relaxed);
        }
        last_popped_timestamp.store(remaining.timestamp, std::memory_order_relaxed);
    }

    ring_buffer.stop();
    consumer.join();

    REQUIRE(chunks_popped.load() > 0);
    REQUIRE(chunks_popped.load() <= total_chunks);
    REQUIRE(monotonic_timestamps.load());
    REQUIRE(samples_valid.load());
}

TEST_CASE("AudioStreamBuffer atomic CAS drop-oldest overrun stress test", "[audio_stream][stress][overrun]") {
    const size_t small_capacity = 32;
    AudioStreamBuffer ring_buffer(small_capacity);

    const size_t flood_count = 10000;
    std::vector<float> dummy_data(32, 0.1234f);

    // Push 10,000 items in a tight loop with NO consumer reading to stress CAS drop-oldest loop
    for (size_t i = 0; i < flood_count; ++i) {
        AudioChunk chunk;
        chunk.timestamp = 50000 + i;
        chunk.samples = dummy_data;
        REQUIRE(ring_buffer.push(chunk));
    }

    // Now pop all remaining elements in the buffer
    std::vector<AudioChunk> retained;
    AudioChunk chunk;
    while (ring_buffer.pop(chunk, /*block=*/false)) {
        retained.push_back(chunk);
    }

    // Must have retained at most small_capacity elements
    REQUIRE(retained.size() <= small_capacity);
    REQUIRE(retained.size() > 0);

    // Verify retained chunks are strictly the most recent ones
    uint64_t expected_min_ts = 50000 + (flood_count - small_capacity);
    REQUIRE(retained.front().timestamp >= expected_min_ts);
    REQUIRE(retained.back().timestamp == 50000 + flood_count - 1);

    // Verify strict monotonic ordering of retained elements
    for (size_t i = 1; i < retained.size(); ++i) {
        REQUIRE(retained[i].timestamp > retained[i - 1].timestamp);
        REQUIRE(retained[i].samples.size() == 32);
        REQUIRE_THAT(retained[i].samples[0], Catch::Matchers::WithinAbs(0.1234f, 1e-5f));
    }
}


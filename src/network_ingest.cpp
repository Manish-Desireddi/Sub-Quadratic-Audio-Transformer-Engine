/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include "network_ingest.hpp"
#include <ixwebsocket/IXWebSocketServer.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXNetSystem.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <memory>

namespace subq {

class WebSocketAudioReceiver : public INetworkAudioReceiver {
public:
    WebSocketAudioReceiver()
        : active_(false),
          target_buffer_(nullptr),
          packets_received_(0),
          bytes_received_(0),
          dropped_packets_(0),
          out_of_order_packets_(0),
          malformed_packets_(0),
          client_connections_(0),
          last_seq_no_(0),
          has_last_seq_(false),
          net_initialized_(false) {}

    ~WebSocketAudioReceiver() override {
        stop();
    }

    bool start(const NetworkStreamConfig& config, AudioStreamBuffer& target_buffer) override {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (active_.load(std::memory_order_relaxed)) {
            return true; // Already running
        }

        config_ = config;
        target_buffer_ = &target_buffer;

        // Reset telemetry counters
        packets_received_.store(0, std::memory_order_relaxed);
        bytes_received_.store(0, std::memory_order_relaxed);
        dropped_packets_.store(0, std::memory_order_relaxed);
        out_of_order_packets_.store(0, std::memory_order_relaxed);
        malformed_packets_.store(0, std::memory_order_relaxed);
        client_connections_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> seq_lock(seq_mutex_);
            last_seq_no_ = 0;
            has_last_seq_ = false;
        }

        // 1. Initialize platform networking (Winsock on Windows)
        if (!net_initialized_) {
            ix::initNetSystem();
            net_initialized_ = true;
        }

        // 2. Instantiate IXWebSocket server
        server_ = std::make_unique<ix::WebSocketServer>(config_.port, config_.host);

        // 3. Configure TLS if enabled
        if (config_.enable_ssl && !config_.ssl_cert_path.empty()) {
            ix::SocketTLSOptions tls_options;
            tls_options.certFile = config_.ssl_cert_path;
            tls_options.keyFile = config_.ssl_key_path;
            server_->setTLSOptions(tls_options);
        }

        // 4. Configure connection callback
        server_->setOnConnectionCallback([this](std::weak_ptr<ix::WebSocket> webSocketWeak,
                                                std::shared_ptr<ix::ConnectionState> connectionState) {
            (void)connectionState;
            client_connections_.fetch_add(1, std::memory_order_relaxed);

            if (auto webSocket = webSocketWeak.lock()) {
                webSocket->setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
                    handle_message(msg);
                });
            }
        });

        // 5. Bind and start listening
        auto res = server_->listen();
        if (!res.first) {
            std::cerr << "[NetworkIngest] Failed to bind to " << config_.host 
                      << ":" << config_.port << " - " << res.second << "\n";
            return false;
        }

        server_->start();
        active_.store(true, std::memory_order_release);
        return true;
    }

    void stop() override {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (!active_.load(std::memory_order_relaxed)) {
            return;
        }

        active_.store(false, std::memory_order_release);

        if (server_) {
            server_->stop();
            server_.reset();
        }

        if (net_initialized_) {
            ix::uninitNetSystem();
            net_initialized_ = false;
        }

        target_buffer_ = nullptr;
    }

    bool is_active() const override {
        return active_.load(std::memory_order_acquire);
    }

    size_t get_packets_received() const override {
        return packets_received_.load(std::memory_order_relaxed);
    }

    size_t get_bytes_received() const override {
        return bytes_received_.load(std::memory_order_relaxed);
    }

    size_t get_dropped_packets() const override {
        return dropped_packets_.load(std::memory_order_relaxed);
    }

    NetworkIngestStats get_stats() const override {
        NetworkIngestStats stats;
        stats.packets_received = packets_received_.load(std::memory_order_relaxed);
        stats.bytes_received = bytes_received_.load(std::memory_order_relaxed);
        stats.dropped_packets = dropped_packets_.load(std::memory_order_relaxed);
        stats.out_of_order_packets = out_of_order_packets_.load(std::memory_order_relaxed);
        stats.malformed_packets = malformed_packets_.load(std::memory_order_relaxed);
        stats.client_connections = client_connections_.load(std::memory_order_relaxed);
        return stats;
    }

private:
    void handle_message(const ix::WebSocketMessagePtr& msg) {
        if (!active_.load(std::memory_order_acquire) || target_buffer_ == nullptr) {
            return;
        }

        if (msg->type != ix::WebSocketMessageType::Message) {
            return;
        }

        const uint8_t* data = reinterpret_cast<const uint8_t*>(msg->str.data());
        size_t size = msg->str.size();

        if (size == 0) return;

        // Check for packaged SUBQ binary header
        if (size >= sizeof(NetworkAudioHeader)) {
            const NetworkAudioHeader* hdr = reinterpret_cast<const NetworkAudioHeader*>(data);
            if (hdr->magic == SUBQ_HEADER_MAGIC || hdr->magic == SUBQ_HEADER_MAGIC_SWAPPED) {
                parse_headered_packet(hdr, data + sizeof(NetworkAudioHeader), size - sizeof(NetworkAudioHeader), size);
                return;
            }
        }

        // Fallback for unheadered raw PCM streams
        if (config_.raw_pcm_fallback) {
            parse_raw_pcm_packet(data, size);
        } else {
            malformed_packets_.fetch_add(1, std::memory_order_relaxed);
            dropped_packets_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void parse_headered_packet(const NetworkAudioHeader* hdr, const uint8_t* payload, size_t actual_payload_bytes, size_t total_wire_bytes) {
        // Enforce maximum payload size to prevent malicious header-declared oversized allocations.
        // 8 MiB covers ~42 seconds of 48 kHz stereo Float32, well beyond any single streaming chunk.
        constexpr size_t kMaxPayloadBytes = 8u * 1024u * 1024u;
        if (hdr->payload_bytes > kMaxPayloadBytes ||
            hdr->payload_bytes != actual_payload_bytes ||
            hdr->sample_rate == 0 ||
            (hdr->channels != 1 && hdr->channels != 2)) {
            malformed_packets_.fetch_add(1, std::memory_order_relaxed);
            dropped_packets_.fetch_add(1, std::memory_order_relaxed);
            return;
        }


        // Track packet continuity & drops
        {
            std::lock_guard<std::mutex> seq_lock(seq_mutex_);
            uint64_t current_seq = hdr->seq_no;
            if (has_last_seq_) {
                if (current_seq > last_seq_no_ + 1) {
                    uint64_t gap = current_seq - (last_seq_no_ + 1);
                    dropped_packets_.fetch_add(gap, std::memory_order_relaxed);
                } else if (current_seq <= last_seq_no_) {
                    out_of_order_packets_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            last_seq_no_ = current_seq;
            has_last_seq_ = true;
        }

        AudioChunk chunk;
        chunk.timestamp = hdr->timestamp_us;

        if (hdr->format == static_cast<uint16_t>(AudioPayloadFormat::Float32)) {
            if (actual_payload_bytes % (hdr->channels * sizeof(float)) != 0) {
                malformed_packets_.fetch_add(1, std::memory_order_relaxed);
                dropped_packets_.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            size_t num_samples = actual_payload_bytes / sizeof(float);
            const float* float_src = reinterpret_cast<const float*>(payload);

            if (hdr->channels == 1) {
                chunk.samples.assign(float_src, float_src + num_samples);
            } else if (hdr->channels == 2) {
                size_t mono_frames = num_samples / 2;
                chunk.samples.resize(mono_frames);
                for (size_t i = 0; i < mono_frames; ++i) {
                    chunk.samples[i] = 0.5f * (float_src[2 * i] + float_src[2 * i + 1]);
                }
            }
        } else if (hdr->format == static_cast<uint16_t>(AudioPayloadFormat::Int16)) {
            if (actual_payload_bytes % (hdr->channels * sizeof(int16_t)) != 0) {
                malformed_packets_.fetch_add(1, std::memory_order_relaxed);
                dropped_packets_.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            size_t num_samples = actual_payload_bytes / sizeof(int16_t);
            const int16_t* int16_src = reinterpret_cast<const int16_t*>(payload);
            constexpr float kScale = 1.0f / 32768.0f;

            if (hdr->channels == 1) {
                chunk.samples.resize(num_samples);
                for (size_t i = 0; i < num_samples; ++i) {
                    chunk.samples[i] = static_cast<float>(int16_src[i]) * kScale;
                }
            } else if (hdr->channels == 2) {
                size_t mono_frames = num_samples / 2;
                chunk.samples.resize(mono_frames);
                constexpr float kStereoScale = 0.5f / 32768.0f;
                for (size_t i = 0; i < mono_frames; ++i) {
                    chunk.samples[i] = (static_cast<float>(int16_src[2 * i]) + 
                                        static_cast<float>(int16_src[2 * i + 1])) * kStereoScale;
                }
            }
        } else {
            malformed_packets_.fetch_add(1, std::memory_order_relaxed);
            dropped_packets_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Push to lock-free ring buffer
        if (!chunk.samples.empty() && target_buffer_) {
            target_buffer_->push(chunk);
            packets_received_.fetch_add(1, std::memory_order_relaxed);
            bytes_received_.fetch_add(total_wire_bytes, std::memory_order_relaxed);
        }
    }

    void parse_raw_pcm_packet(const uint8_t* data, size_t size) {
        AudioChunk chunk;
        chunk.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

        size_t ch = (config_.channels == 0) ? 1 : config_.channels;

        if (size % (ch * sizeof(float)) == 0) {
            size_t num_samples = size / sizeof(float);
            const float* float_src = reinterpret_cast<const float*>(data);
            if (ch == 1) {
                chunk.samples.assign(float_src, float_src + num_samples);
            } else {
                size_t mono_frames = num_samples / ch;
                chunk.samples.resize(mono_frames);
                float ch_scale = 1.0f / static_cast<float>(ch);
                for (size_t i = 0; i < mono_frames; ++i) {
                    float sum = 0.0f;
                    for (size_t c = 0; c < ch; ++c) {
                        sum += float_src[i * ch + c];
                    }
                    chunk.samples[i] = sum * ch_scale;
                }
            }
        } else if (size % (ch * sizeof(int16_t)) == 0) {
            size_t num_samples = size / sizeof(int16_t);
            const int16_t* int16_src = reinterpret_cast<const int16_t*>(data);
            float scale = (1.0f / 32768.0f) / static_cast<float>(ch);
            size_t mono_frames = num_samples / ch;
            chunk.samples.resize(mono_frames);
            for (size_t i = 0; i < mono_frames; ++i) {
                float sum = 0.0f;
                for (size_t c = 0; c < ch; ++c) {
                    sum += static_cast<float>(int16_src[i * ch + c]);
                }
                chunk.samples[i] = sum * scale;
            }
        } else {
            malformed_packets_.fetch_add(1, std::memory_order_relaxed);
            dropped_packets_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (!chunk.samples.empty() && target_buffer_) {
            target_buffer_->push(chunk);
            packets_received_.fetch_add(1, std::memory_order_relaxed);
            bytes_received_.fetch_add(size, std::memory_order_relaxed);
        }
    }

    std::mutex lifecycle_mutex_;
    std::mutex seq_mutex_;
    std::atomic<bool> active_;
    AudioStreamBuffer* target_buffer_;
    NetworkStreamConfig config_;
    std::unique_ptr<ix::WebSocketServer> server_;

    // Telemetry atomics
    std::atomic<uint64_t> packets_received_;
    std::atomic<uint64_t> bytes_received_;
    std::atomic<uint64_t> dropped_packets_;
    std::atomic<uint64_t> out_of_order_packets_;
    std::atomic<uint64_t> malformed_packets_;
    std::atomic<uint64_t> client_connections_;

    // Sequence tracking
    uint64_t last_seq_no_;
    bool has_last_seq_;
    bool net_initialized_;
};

std::unique_ptr<INetworkAudioReceiver> create_websocket_receiver() {
    return std::make_unique<WebSocketAudioReceiver>();
}

} // namespace subq

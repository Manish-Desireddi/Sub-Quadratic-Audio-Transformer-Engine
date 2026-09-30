/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once

#include "audio_stream.hpp"
#include <string>
#include <memory>
#include <cstdint>
#include <atomic>
#include <vector>
#include <chrono>

namespace subq {

#pragma pack(push, 1)
/**
 * @brief Binary header prepended to audio streaming frames over WebSocket.
 * 
 * Total size: Exactly 36 bytes.
 */
struct NetworkAudioHeader {
    uint32_t magic;         ///< Protocol magic identifier: 0x53554251 ("SUBQ")
    uint32_t stream_id;     ///< Logical stream ID (e.g. client ID or channel ID)
    uint64_t seq_no;        ///< Monotonically increasing packet sequence number
    uint64_t timestamp_us;  ///< Microsecond timestamp (source capture time or epoch)
    uint32_t sample_rate;   ///< Sample rate in Hz (e.g. 16000, 24000, 48000)
    uint16_t channels;      ///< Channel count: 1 = Mono, 2 = Stereo
    uint16_t format;        ///< Encoding: 0 = Float32 PCM, 1 = Int16 PCM
    uint32_t payload_bytes; ///< Byte size of audio payload following this header
};
#pragma pack(pop)

static_assert(sizeof(NetworkAudioHeader) == 36, "NetworkAudioHeader must be exactly 36 bytes with 1-byte packing.");

/// Expected magic word in host little-endian format ('S','U','B','Q')
constexpr uint32_t SUBQ_HEADER_MAGIC = 0x53554251;

/// Byte-swapped magic word representation for cross-endian network clients
constexpr uint32_t SUBQ_HEADER_MAGIC_SWAPPED = 0x51425553;

/**
 * @brief Supported audio payload encoding formats.
 */
enum class AudioPayloadFormat : uint16_t {
    Float32 = 0,
    Int16 = 1
};

/**
 * @brief Configuration parameters for network streaming ingestion.
 */
struct NetworkStreamConfig {
    std::string host = "0.0.0.0";      ///< Listen address (server) or target host (client)
    uint16_t port = 8080;               ///< WebSocket TCP port
    std::string endpoint = "/audio";    ///< WebSocket URL endpoint / path
    size_t sample_rate = 24000;         ///< Default audio sample rate in Hz
    size_t channels = 1;                ///< Default audio channel count (1 = mono, 2 = stereo)
    size_t chunk_size = 2048;           ///< Nominal audio samples per chunk
    bool enable_ssl = false;            ///< Enable TLS / WSS encryption
    std::string ssl_cert_path = "";     ///< Path to TLS certificate (server)
    std::string ssl_key_path = "";      ///< Path to TLS private key (server)
    bool raw_pcm_fallback = true;       ///< Allow unheadered raw PCM frames
};

/**
 * @brief Diagnostic telemetry and packet counters.
 */
struct NetworkIngestStats {
    uint64_t packets_received = 0;      ///< Total valid audio packets received
    uint64_t bytes_received = 0;        ///< Total bytes ingested
    uint64_t dropped_packets = 0;      ///< Packets dropped due to sequence gaps or buffer overflow
    uint64_t out_of_order_packets = 0;  ///< Out-of-order sequence arrivals
    uint64_t malformed_packets = 0;     ///< Malformed or corrupted frames rejected
    uint64_t client_connections = 0;    ///< Total active/historical client connections
};

/**
 * @brief Abstract interface for network audio ingestion receivers.
 */
class INetworkAudioReceiver {
public:
    virtual ~INetworkAudioReceiver() = default;

    /**
     * @brief Start the network audio receiver asynchronously.
     * @param config Network stream configuration parameters.
     * @param target_buffer Reference to lock-free ring buffer for decoded AudioChunks.
     * @return true if socket bound and listening/connected, false on error.
     */
    virtual bool start(const NetworkStreamConfig& config, AudioStreamBuffer& target_buffer) = 0;

    /**
     * @brief Stop the receiver, terminate background network loops, and close connections.
     */
    virtual void stop() = 0;

    /**
     * @brief Query active running state.
     * @return true if receiver is active and accepting frames.
     */
    virtual bool is_active() const = 0;

    /**
     * @brief Get count of valid audio packets received.
     */
    virtual size_t get_packets_received() const = 0;

    /**
     * @brief Get total count of audio payload bytes received.
     */
    virtual size_t get_bytes_received() const = 0;

    /**
     * @brief Get count of dropped audio packets.
     */
    virtual size_t get_dropped_packets() const = 0;

    /**
     * @brief Get comprehensive diagnostic telemetry.
     */
    virtual NetworkIngestStats get_stats() const = 0;
};

/**
 * @brief Factory function creating a WebSocket audio receiver instance.
 * @return std::unique_ptr to INetworkAudioReceiver implementation.
 */
std::unique_ptr<INetworkAudioReceiver> create_websocket_receiver();

} // namespace subq

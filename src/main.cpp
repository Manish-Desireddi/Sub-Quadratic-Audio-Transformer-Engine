/*
 * Copyright (c) 2026 Manish. All rights reserved.
 *
 * This work is licensed under the terms of the GNU GPLv3 license.
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <fcntl.h>
#include <io.h>
#endif

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "arena.hpp"
#include "attention.hpp"
#include "audio_stream.hpp"
#include "engine.hpp"
#include "gpu_macros.hpp"
#include "network_ingest.hpp"
#include "safetensors.hpp"
#include "telemetry.hpp"

// Forward declaration from src/audio_ingest.cpp
std::vector<float> load_wav(const std::string &filepath);

namespace {
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    g_shutdown_requested.store(true, std::memory_order_relaxed);
  }
}

inline uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(float));
  uint32_t sign = (x >> 16) & 0x8000;
  uint32_t exponent = ((x >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = x & 0x007fffff;

  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x00800000) >> (1 - exponent);
    return static_cast<uint16_t>(sign | (mantissa >> 13));
  } else if (exponent == 0xff - 127 + 15) {
    if (mantissa == 0)
      return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | 0x7c00 | (mantissa >> 13));
  } else if (exponent > 30) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

std::string get_backend_name() {
#if defined(USE_CUDA)
  return "CUDA";
#elif defined(USE_HIP)
  return "HIP";
#else
  return "CPU";
#endif
}
} // namespace

void wav_producer(AudioStreamBuffer &buffer, const std::string &audio_file,
                  size_t chunk_size, uint32_t sample_rate, bool benchmark_mode,
                  bool verbose) {
  try {
    std::vector<float> raw_audio = load_wav(audio_file);
    if (verbose) {
      std::cout << "[Producer] Ingested WAV: " << raw_audio.size()
                << " samples from '" << audio_file << "'\n";
    }

    double chunk_duration_sec = (sample_rate > 0)
                                    ? (static_cast<double>(chunk_size) /
                                       static_cast<double>(sample_rate))
                                    : 0.05;
    auto pacing_delay = std::chrono::microseconds(
        static_cast<int64_t>(chunk_duration_sec * 1.0e6));

    for (size_t i = 0; i < raw_audio.size() &&
                       !g_shutdown_requested.load(std::memory_order_relaxed);
         i += chunk_size) {
      AudioChunk chunk;
      chunk.timestamp = static_cast<uint64_t>(i);
      size_t end = std::min(i + chunk_size, raw_audio.size());
      chunk.samples.assign(raw_audio.begin() + i, raw_audio.begin() + end);

      if (chunk.samples.size() < chunk_size) {
        chunk.samples.resize(chunk_size, 0.0f);
      }

      buffer.push(chunk);

      if (!benchmark_mode) {
        std::this_thread::sleep_for(pacing_delay);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[Producer Error] " << e.what() << "\n";
  }
  buffer.stop();
}

void stdin_producer(AudioStreamBuffer &buffer, size_t chunk_size,
                    const std::string &pcm_format, uint32_t sample_rate,
                    bool benchmark_mode, bool verbose) {
#if defined(_WIN32) || defined(_WIN64)
  _setmode(_fileno(stdin), _O_BINARY);
#endif

  if (verbose) {
    std::cout << "[Producer] Reading raw PCM stream from standard input ("
              << pcm_format << ")...\n";
  }

  double chunk_duration_sec =
      (sample_rate > 0)
          ? (static_cast<double>(chunk_size) / static_cast<double>(sample_rate))
          : 0.05;
  auto pacing_delay = std::chrono::microseconds(
      static_cast<int64_t>(chunk_duration_sec * 1.0e6));
  uint64_t total_samples_read = 0;

  bool is_s16le = (pcm_format == "s16le");
  std::vector<int16_t> s16_buf(chunk_size);
  std::vector<float> f32_buf(chunk_size);

  while (!g_shutdown_requested.load(std::memory_order_relaxed) &&
         !std::cin.eof()) {
    AudioChunk chunk;
    chunk.timestamp = total_samples_read;

    if (is_s16le) {
      std::cin.read(reinterpret_cast<char *>(s16_buf.data()),
                    chunk_size * sizeof(int16_t));
      size_t bytes_read = static_cast<size_t>(std::cin.gcount());
      size_t samples_read = bytes_read / sizeof(int16_t);
      if (samples_read == 0)
        break;

      chunk.samples.resize(chunk_size, 0.0f);
      for (size_t i = 0; i < samples_read; ++i) {
        chunk.samples[i] = static_cast<float>(s16_buf[i]) / 32768.0f;
      }
      total_samples_read += samples_read;
    } else {
      std::cin.read(reinterpret_cast<char *>(f32_buf.data()),
                    chunk_size * sizeof(float));
      size_t bytes_read = static_cast<size_t>(std::cin.gcount());
      size_t samples_read = bytes_read / sizeof(float);
      if (samples_read == 0)
        break;

      chunk.samples.resize(chunk_size, 0.0f);
      std::memcpy(chunk.samples.data(), f32_buf.data(),
                  samples_read * sizeof(float));
      total_samples_read += samples_read;
    }

    buffer.push(chunk);

    if (!benchmark_mode) {
      std::this_thread::sleep_for(pacing_delay);
    }
  }
  buffer.stop();
}

void synthetic_producer(AudioStreamBuffer &buffer, size_t chunk_size,
                        size_t total_chunks, uint32_t sample_rate,
                        bool benchmark_mode, bool verbose) {
  if (verbose) {
    std::cout << "[Producer] Generating " << total_chunks
              << " synthetic audio chunks (" << chunk_size
              << " samples/chunk)...\n";
  }

  double chunk_duration_sec =
      (sample_rate > 0)
          ? (static_cast<double>(chunk_size) / static_cast<double>(sample_rate))
          : 0.05;
  auto pacing_delay = std::chrono::microseconds(
      static_cast<int64_t>(chunk_duration_sec * 1.0e6));

  const double freq1 = 440.0;
  const double freq2 = 880.0;
  const double two_pi = 6.283185307179586;

  for (size_t c = 0; c < total_chunks &&
                     !g_shutdown_requested.load(std::memory_order_relaxed);
       ++c) {
    AudioChunk chunk;
    chunk.timestamp = c * chunk_size;
    chunk.samples.resize(chunk_size);

    for (size_t i = 0; i < chunk_size; ++i) {
      double t = static_cast<double>(chunk.timestamp + i) /
                 static_cast<double>(sample_rate);
      float s = static_cast<float>(0.5 * std::sin(two_pi * freq1 * t) +
                                   0.25 * std::sin(two_pi * freq2 * t));
      chunk.samples[i] = s;
    }

    buffer.push(chunk);

    if (!benchmark_mode) {
      std::this_thread::sleep_for(pacing_delay);
    }
  }
  buffer.stop();
}

void engine_consumer(AudioStreamBuffer &buffer,
                     subq::TelemetryCollector &collector, size_t batch_size,
                     size_t seq_len, size_t d_model, float decay_factor,
                     const std::string &precision_str, size_t vram_mb,
                     const std::string &weights_path, bool verbose,
                     bool debug) {
  size_t vram_bytes = vram_mb * 1024 * 1024;
  MemoryArena arena(vram_bytes);
  Engine engine(arena);

  if (!weights_path.empty()) {
    if (verbose) {
      std::cout << "[Consumer] Loading model weights from: " << weights_path
                << "\n";
    }
    SafetensorLoader loader(arena);
    if (!loader.load(weights_path)) {
      std::cerr << "[Consumer Warning] Failed to parse weights file: "
                << weights_path << "\n";
    }
  }

  size_t tensor_elements = batch_size * seq_len * d_model;
  size_t tensor_bytes = tensor_elements * sizeof(float);
  size_t state_bytes = batch_size * d_model * d_model * sizeof(float);

  PrecisionMode precision = PrecisionMode::FP32;
  if (precision_str == "fp16")
    precision = PrecisionMode::FP16;
  else if (precision_str == "bf16")
    precision = PrecisionMode::BF16;

  // Pre-allocate persistent engine buffers
  float *d_Q = reinterpret_cast<float *>(arena.allocate(tensor_bytes));
  float *d_K = reinterpret_cast<float *>(arena.allocate(tensor_bytes));
  float *d_V = reinterpret_cast<float *>(arena.allocate(tensor_bytes));
  float *d_O = reinterpret_cast<float *>(arena.allocate(tensor_bytes));
  float *d_S = reinterpret_cast<float *>(arena.allocate(state_bytes));

  AttentionContext ctx;
  ctx.Q = {{batch_size, seq_len, d_model}, {}, precision, d_Q, true};
  ctx.K = {{batch_size, seq_len, d_model}, {}, precision, d_K, true};
  ctx.V = {{batch_size, seq_len, d_model}, {}, precision, d_V, true};
  ctx.S = {{batch_size, d_model, d_model}, {}, PrecisionMode::FP32, d_S, true};
  ctx.O = {{batch_size, seq_len, d_model}, {}, precision, d_O, true};
  ctx.decay_factor = decay_factor;

  AudioChunk chunk;
  size_t frame_index = 0;

  if (verbose) {
    std::cout << "[Consumer] Engine initialized on " << get_backend_name()
              << " (Arena: " << vram_mb << " MB). Ready for inference."
              << std::endl;
  }

  std::vector<float> h_Q(tensor_elements);
  std::vector<float> h_K(tensor_elements);
  std::vector<float> h_V(tensor_elements);

  while (buffer.pop(chunk, true) &&
         !g_shutdown_requested.load(std::memory_order_relaxed)) {
    // Map audio chunk samples across feature dimensions on CPU
    for (size_t b = 0; b < batch_size; ++b) {
      size_t b_offset = b * (seq_len * d_model);
      for (size_t t = 0; t < seq_len; ++t) {
        float sample_val = (t < chunk.samples.size()) ? chunk.samples[t] : 0.0f;
        for (size_t d = 0; d < d_model; ++d) {
          float val = sample_val * (1.0f + static_cast<float>(d) * 0.001f);
          h_Q[b_offset + t * d_model + d] = val;
          h_K[b_offset + t * d_model + d] = val;
          h_V[b_offset + t * d_model + d] = val;
        }
      }
    }

    // Copy to GPU memory
    gpuMemcpy(d_Q, h_Q.data(), tensor_bytes, gpuMemcpyHostToDevice);
    gpuMemcpy(d_K, h_K.data(), tensor_bytes, gpuMemcpyHostToDevice);
    gpuMemcpy(d_V, h_V.data(), tensor_bytes, gpuMemcpyHostToDevice);

    // Measure ON-DEVICE Execution Latency in microseconds
    auto start_time = std::chrono::high_resolution_clock::now();

    engine.forward(ctx);

#if defined(USE_CUDA) || defined(USE_HIP)
    gpuDeviceSynchronize();
#endif

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_us =
        std::chrono::duration<double, std::micro>(end_time - start_time)
            .count();

    size_t samples_in_chunk = chunk.samples.size();
    size_t tokens_in_chunk = batch_size * seq_len;
    collector.record_iteration(elapsed_us, samples_in_chunk, tokens_in_chunk);

    if (verbose && (frame_index % 25 == 0 || frame_index < 5)) {
      std::cout << "[Consumer] Frame #" << frame_index
                << " | Latency: " << std::fixed << std::setprecision(1)
                << elapsed_us << " us"
                << " (" << std::setprecision(3) << (elapsed_us / 1000.0)
                << " ms)\n";
    }

    if (debug) {
      std::cout << "[Debug] Frame " << frame_index
                << " Arena Offset: " << arena.get_offset() << " / "
                << arena.get_total_bytes() << " bytes\n";
    }

    frame_index++;
  }

  if (verbose) {
    std::cout << "[Consumer] Engine processed " << frame_index
              << " frames total.\n";
  }
}

int main(int argc, char **argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  CLI::App app{"Sub-Quadratic Audio Transformer Engine Native CLI", "subq_cli"};

  std::string input_file = "";
  bool use_stdin = false;
  std::string ws_url = "";

  uint32_t sample_rate = 16000;
  uint16_t channels = 1;
  std::string pcm_format = "f32le";

  std::string weights_path = "";
  size_t d_model = 256;
  size_t seq_len = 1024;
  size_t batch_size = 1;
  float decay_factor = 0.99f;
  std::string precision_str = "fp32";
  size_t vram_mb = 256;

  bool benchmark_mode = false;
  size_t benchmark_iters = 100;
  size_t warmup_iters = 10;

  std::string telemetry_json_path = "";
  bool verbose = false;
  bool debug = false;
  bool show_version = false;

  // Option Groups
  auto *input_group = app.add_option_group("Input Source Options");
  input_group
      ->add_option("-i,--input-file", input_file,
                   "Path to input WAV audio file")
      ->check(CLI::ExistingFile);
  input_group->add_flag("--stdin", use_stdin,
                        "Read raw PCM audio stream from standard input");
  input_group->add_option(
      "--ws-url", ws_url,
      "WebSocket stream endpoint (e.g. ws://localhost:8080/audio)");

  auto *audio_group = app.add_option_group("Audio Format Options");
  audio_group->add_option("-r,--sample-rate", sample_rate, "Sample rate in Hz")
      ->default_val(16000);
  audio_group
      ->add_option("-c,--channels", channels,
                   "Audio channels (1=Mono, 2=Stereo)")
      ->default_val(1);
  audio_group
      ->add_option("--pcm-format", pcm_format,
                   "Stdin PCM format: f32le or s16le")
      ->check(CLI::IsMember({"f32le", "s16le"}))
      ->default_val("f32le");

  auto *model_group = app.add_option_group("Model & Engine Configuration");
  model_group
      ->add_option("-w,--weights", weights_path,
                   "Path to .safetensors model file")
      ->check(CLI::ExistingFile);
  model_group
      ->add_option("-d,--d-model", d_model, "Model hidden dimension (d_model)")
      ->default_val(256);
  model_group
      ->add_option("-s,--seq-len", seq_len, "Chunk sequence length (seq_len)")
      ->default_val(1024);
  model_group->add_option("-b,--batch-size", batch_size, "Inference batch size")
      ->default_val(1);
  model_group
      ->add_option("--decay-factor", decay_factor,
                   "Decay coefficient gamma (0, 1)")
      ->default_val(0.99f);
  model_group
      ->add_option("-p,--precision", precision_str,
                   "Precision: fp32, fp16, bf16")
      ->check(CLI::IsMember({"fp32", "fp16", "bf16"}))
      ->default_val("fp32");
  model_group
      ->add_option("-m,--vram-mb", vram_mb, "Memory Arena capacity in MB")
      ->default_val(256);

  auto *bench_group = app.add_option_group("Benchmark & Execution Mode");
  bench_group->add_flag("--benchmark", benchmark_mode,
                        "Run in headless benchmark mode (unpaced)");
  bench_group
      ->add_option("-n,--benchmark-iters", benchmark_iters,
                   "Benchmark iterations count")
      ->default_val(100);
  bench_group
      ->add_option("--warmup-iters", warmup_iters, "Warmup iterations count")
      ->default_val(10);

  auto *logging_group = app.add_option_group("Telemetry & Logging");
  logging_group->add_option(
      "--telemetry-json", telemetry_json_path,
      "Path to export structured microsecond JSON report");
  logging_group->add_flag("-v,--verbose", verbose,
                          "Enable verbose per-frame progress and latency logs");
  logging_group->add_flag(
      "--debug", debug,
      "Enable low-level arena offset and hardware debug logs");
  logging_group->add_flag("-V,--version", show_version,
                          "Display engine version and hardware backend info");

  CLI11_PARSE(app, argc, argv);

  if (show_version) {
    std::cout << "SubQ Audio Transformer Engine v1.0.1 (" << get_backend_name()
              << " Backend)\n";
    return 0;
  }

  std::cout << "=== Sub-Quadratic Audio Transformer Engine (Native C++) ==="
            << std::endl;
  std::cout << " Backend: " << get_backend_name()
            << " | Precision: " << precision_str
            << " | Context: [B=" << batch_size << ", L=" << seq_len
            << ", D=" << d_model << "]" << std::endl;

  AudioStreamBuffer ring_buffer(32);
  subq::TelemetryCollector collector(warmup_iters);

  std::unique_ptr<subq::INetworkAudioReceiver> net_receiver = nullptr;
  std::thread producer;

  if (!ws_url.empty()) {
    std::cout << "[Network Ingestion] Connecting to WebSocket: " << ws_url
              << "\n";
    net_receiver = subq::create_websocket_receiver();
    subq::NetworkStreamConfig net_config;
    net_config.host = ws_url;
    net_config.sample_rate = sample_rate;
    net_config.channels = channels;
    net_config.chunk_size = seq_len;
    if (!net_receiver->start(net_config, ring_buffer)) {
      std::cerr
          << "[Network Ingestion Error] Failed to start WebSocket receiver.\n";
      return 1;
    }
  } else if (!input_file.empty()) {
    producer = std::thread(wav_producer, std::ref(ring_buffer), input_file,
                           seq_len, sample_rate, benchmark_mode, verbose);
  } else if (use_stdin) {
    producer = std::thread(stdin_producer, std::ref(ring_buffer), seq_len,
                           pcm_format, sample_rate, benchmark_mode, verbose);
  } else {
    // Synthetic benchmark stream
    size_t total_chunks = benchmark_iters + warmup_iters;
    benchmark_mode = true;
    producer = std::thread(synthetic_producer, std::ref(ring_buffer), seq_len,
                           total_chunks, sample_rate, benchmark_mode, verbose);
  }

  // Run Engine Consumer Thread
  std::thread consumer(engine_consumer, std::ref(ring_buffer),
                       std::ref(collector), batch_size, seq_len, d_model,
                       decay_factor, precision_str, vram_mb, weights_path,
                       verbose, debug);

  if (producer.joinable()) {
    producer.join();
  }

  if (net_receiver && net_receiver->is_active()) {
    net_receiver->stop();
    ring_buffer.stop();
  }

  if (consumer.joinable()) {
    consumer.join();
  }

  // Compute and Output Telemetry Metrics
  size_t peak_arena_bytes =
      (batch_size * seq_len * d_model * 4 * sizeof(float)) +
      (batch_size * d_model * d_model * sizeof(float));
  size_t total_arena_capacity = vram_mb * 1024 * 1024;
  subq::TelemetryMetrics metrics = collector.compute_metrics(
      get_backend_name(), precision_str, batch_size, seq_len, d_model,
      sample_rate, peak_arena_bytes, total_arena_capacity);

  collector.print_summary(metrics, std::cout);

  if (!telemetry_json_path.empty()) {
    if (metrics.export_json(telemetry_json_path)) {
      std::cout << "[Telemetry] Metrics report successfully saved to: "
                << telemetry_json_path << "\n";
    }
  }

  std::cout << "Engine Shutdown Gracefully.\n";
  return 0;
}

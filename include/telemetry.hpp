/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
 */

#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <nlohmann/json.hpp>

namespace subq {

inline std::string get_iso8601_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32) || defined(_WIN64)
    gmtime_s(&tm_buf, &now_c);
#else
    gmtime_r(&now_c, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return std::string(buf);
}

struct TelemetryMetrics {
    // Execution metadata
    std::string timestamp;
    std::string hardware_backend;
    std::string precision;
    size_t batch_size = 1;
    size_t seq_len = 1024;
    size_t d_model = 256;
    uint32_t sample_rate = 16000;
    
    // Iteration & Sample counts
    size_t warmup_iterations = 0;
    size_t measured_iterations = 0;
    size_t total_samples_processed = 0;
    size_t total_tokens_processed = 0;
    
    // Duration metrics (seconds & microseconds)
    double total_audio_duration_sec = 0.0;
    double total_compute_duration_sec = 0.0;
    
    // Latency statistics (microseconds)
    double latency_min_us = 0.0;
    double latency_max_us = 0.0;
    double latency_mean_us = 0.0;
    double latency_stddev_us = 0.0;
    double latency_p50_us = 0.0;
    double latency_p90_us = 0.0;
    double latency_p95_us = 0.0;
    double latency_p99_us = 0.0;
    
    // Performance & Efficiency
    double real_time_factor = 0.0; // RTF = compute_time / audio_time
    double rtf_speedup = 0.0;      // 1.0 / RTF
    double throughput_samples_per_sec = 0.0;
    double throughput_tokens_per_sec = 0.0;
    
    // Hardware Memory
    size_t peak_arena_bytes = 0;
    size_t total_arena_capacity_bytes = 0;
    double arena_utilization_pct = 0.0;
    
    nlohmann::json to_json() const {
        nlohmann::json j;
        j["metadata"] = {
            {"timestamp", timestamp},
            {"backend", hardware_backend},
            {"precision", precision},
            {"batch_size", batch_size},
            {"seq_len", seq_len},
            {"d_model", d_model},
            {"sample_rate", sample_rate}
        };
        j["counts"] = {
            {"warmup_iterations", warmup_iterations},
            {"measured_iterations", measured_iterations},
            {"total_samples", total_samples_processed},
            {"total_tokens", total_tokens_processed}
        };
        j["durations"] = {
            {"audio_duration_sec", total_audio_duration_sec},
            {"compute_duration_sec", total_compute_duration_sec}
        };
        j["latency_microseconds"] = {
            {"min", latency_min_us},
            {"max", latency_max_us},
            {"mean", latency_mean_us},
            {"stddev", latency_stddev_us},
            {"p50", latency_p50_us},
            {"p90", latency_p90_us},
            {"p95", latency_p95_us},
            {"p99", latency_p99_us}
        };
        j["throughput"] = {
            {"real_time_factor", real_time_factor},
            {"speedup_x_realtime", rtf_speedup},
            {"samples_per_sec", throughput_samples_per_sec},
            {"tokens_per_sec", throughput_tokens_per_sec}
        };
        j["memory"] = {
            {"peak_arena_bytes", peak_arena_bytes},
            {"peak_arena_mb", static_cast<double>(peak_arena_bytes) / (1024.0 * 1024.0)},
            {"total_arena_capacity_bytes", total_arena_capacity_bytes},
            {"total_arena_capacity_mb", static_cast<double>(total_arena_capacity_bytes) / (1024.0 * 1024.0)},
            {"arena_utilization_pct", arena_utilization_pct}
        };

        // Flat aliases for E2E testing Python TelemetryParser validation
        j["mean_latency_us"] = latency_mean_us;
        j["p50_latency_us"] = latency_p50_us;
        j["p95_latency_us"] = latency_p95_us;
        j["p99_latency_us"] = latency_p99_us;
        j["throughput_samples_per_sec"] = throughput_samples_per_sec;
        j["real_time_factor"] = real_time_factor;

        return j;
    }

    bool export_json(const std::string& filepath) const {
        if (filepath.empty()) return false;
        std::ofstream out(filepath);
        if (!out.is_open()) {
            std::cerr << "[Telemetry] Failed to open file for JSON export: " << filepath << "\n";
            return false;
        }
        out << to_json().dump(4) << "\n";
        return true;
    }
};

class TelemetryCollector {
public:
    explicit TelemetryCollector(size_t warmup_iters = 10)
        : warmup_iters_(warmup_iters), current_iter_(0), samples_recorded_(0), tokens_recorded_(0) {}
    
    void record_iteration(double latency_us, size_t samples, size_t tokens) {
        current_iter_++;
        if (current_iter_ <= warmup_iters_) {
            return; // Discard warmup passes
        }
        latencies_us_.push_back(latency_us);
        samples_recorded_ += samples;
        tokens_recorded_ += tokens;
    }
    
    TelemetryMetrics compute_metrics(
        const std::string& backend,
        const std::string& precision,
        size_t batch_size,
        size_t seq_len,
        size_t d_model,
        uint32_t sample_rate,
        size_t peak_arena_bytes,
        size_t total_arena_capacity
    ) const {
        TelemetryMetrics m;
        m.timestamp = get_iso8601_timestamp();
        m.hardware_backend = backend;
        m.precision = precision;
        m.batch_size = batch_size;
        m.seq_len = seq_len;
        m.d_model = d_model;
        m.sample_rate = sample_rate;
        m.warmup_iterations = warmup_iters_;
        m.measured_iterations = latencies_us_.size();
        m.total_samples_processed = samples_recorded_;
        m.total_tokens_processed = tokens_recorded_;
        m.peak_arena_bytes = peak_arena_bytes;
        m.total_arena_capacity_bytes = total_arena_capacity;
        m.arena_utilization_pct = (total_arena_capacity > 0) 
            ? (100.0 * static_cast<double>(peak_arena_bytes) / static_cast<double>(total_arena_capacity)) 
            : 0.0;
        
        if (latencies_us_.empty()) {
            return m;
        }
        
        std::vector<double> sorted = latencies_us_;
        std::sort(sorted.begin(), sorted.end());
        size_t N = sorted.size();
        
        m.latency_min_us = sorted.front();
        m.latency_max_us = sorted.back();
        
        double sum_us = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        m.latency_mean_us = sum_us / static_cast<double>(N);
        m.total_compute_duration_sec = sum_us / 1.0e6;
        
        double sq_diff = 0.0;
        for (double v : sorted) {
            sq_diff += (v - m.latency_mean_us) * (v - m.latency_mean_us);
        }
        m.latency_stddev_us = (N > 1) ? std::sqrt(sq_diff / static_cast<double>(N - 1)) : 0.0;
        
        auto get_percentile = [&](double p) -> double {
            if (N == 1) return sorted[0];
            double idx = p * static_cast<double>(N - 1);
            size_t lower = static_cast<size_t>(std::floor(idx));
            size_t upper = static_cast<size_t>(std::ceil(idx));
            double weight = idx - static_cast<double>(lower);
            return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
        };
        
        m.latency_p50_us = get_percentile(0.50);
        m.latency_p90_us = get_percentile(0.90);
        m.latency_p95_us = get_percentile(0.95);
        m.latency_p99_us = get_percentile(0.99);
        
        m.total_audio_duration_sec = (sample_rate > 0) 
            ? (static_cast<double>(samples_recorded_) / static_cast<double>(sample_rate)) 
            : 0.0;
        
        m.real_time_factor = (m.total_audio_duration_sec > 0.0) 
            ? (m.total_compute_duration_sec / m.total_audio_duration_sec) 
            : 0.0;
        
        m.rtf_speedup = (m.real_time_factor > 0.0) ? (1.0 / m.real_time_factor) : 0.0;
        
        m.throughput_samples_per_sec = (m.total_compute_duration_sec > 0.0) 
            ? (static_cast<double>(samples_recorded_) / m.total_compute_duration_sec) 
            : 0.0;
        
        m.throughput_tokens_per_sec = (m.total_compute_duration_sec > 0.0) 
            ? (static_cast<double>(tokens_recorded_) / m.total_compute_duration_sec) 
            : 0.0;
        
        return m;
    }

    void print_summary(const TelemetryMetrics& m, std::ostream& os = std::cout) const {
        os << "\n=======================================================\n";
        os << "     Sub-Quadratic Audio Transformer Telemetry Report  \n";
        os << "=======================================================\n";
        os << " Hardware Backend:          " << m.hardware_backend << "\n";
        os << " Precision Mode:            " << m.precision << "\n";
        os << " Dimensions (B x L x D):    " << m.batch_size << " x " << m.seq_len << " x " << m.d_model << "\n";
        os << " Sample Rate:               " << m.sample_rate << " Hz\n";
        os << " Total Chunks Measured:     " << m.measured_iterations << " (warmup: " << m.warmup_iterations << ")\n";
        os << " Total Audio Duration:      " << std::fixed << std::setprecision(3) << m.total_audio_duration_sec << " s\n";
        os << " Total Compute Duration:    " << std::fixed << std::setprecision(3) << m.total_compute_duration_sec << " s\n";
        os << "-------------------------------------------------------\n";
        os << " Latency Min:               " << std::fixed << std::setprecision(2) << m.latency_min_us << " us (" << (m.latency_min_us / 1000.0) << " ms)\n";
        os << " Latency Mean:              " << std::fixed << std::setprecision(2) << m.latency_mean_us << " us (" << (m.latency_mean_us / 1000.0) << " ms)\n";
        os << " Latency Max:               " << std::fixed << std::setprecision(2) << m.latency_max_us << " us (" << (m.latency_max_us / 1000.0) << " ms)\n";
        os << " Latency StdDev:            " << std::fixed << std::setprecision(2) << m.latency_stddev_us << " us\n";
        os << " Latency p50 (Median):      " << std::fixed << std::setprecision(2) << m.latency_p50_us << " us\n";
        os << " Latency p90:               " << std::fixed << std::setprecision(2) << m.latency_p90_us << " us\n";
        os << " Latency p95:               " << std::fixed << std::setprecision(2) << m.latency_p95_us << " us\n";
        os << " Latency p99:               " << std::fixed << std::setprecision(2) << m.latency_p99_us << " us\n";
        os << "-------------------------------------------------------\n";
        os << " Real-Time Factor (RTF):    " << std::fixed << std::setprecision(4) << m.real_time_factor << "\n";
        os << " Real-Time Speedup:         " << std::fixed << std::setprecision(1) << m.rtf_speedup << "x faster than real-time\n";
        os << " Sample Throughput:         " << std::fixed << std::setprecision(0) << m.throughput_samples_per_sec << " samples/sec\n";
        os << " Token Throughput:          " << std::fixed << std::setprecision(0) << m.throughput_tokens_per_sec << " tokens/sec\n";
        os << "-------------------------------------------------------\n";
        os << " Peak Arena Memory Used:    " << std::fixed << std::setprecision(2) << (static_cast<double>(m.peak_arena_bytes) / (1024.0 * 1024.0)) << " MB\n";
        os << " Total Arena Capacity:      " << std::fixed << std::setprecision(2) << (static_cast<double>(m.total_arena_capacity_bytes) / (1024.0 * 1024.0)) << " MB\n";
        os << " Arena Utilization:         " << std::fixed << std::setprecision(2) << m.arena_utilization_pct << " %\n";
        os << "=======================================================\n\n";
    }

    size_t get_measured_count() const { return latencies_us_.size(); }

private:
    size_t warmup_iters_;
    size_t current_iter_;
    std::vector<double> latencies_us_;
    size_t samples_recorded_;
    size_t tokens_recorded_;
};

} // namespace subq

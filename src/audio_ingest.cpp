/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the GNU GPLv3 license.  
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include <sndfile.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

std::vector<float> load_wav(const std::string& filepath) {
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open(filepath.c_str(), SFM_READ, &sfinfo);
    if (!infile) {
        throw std::runtime_error("Failed to open audio file: " + filepath);
    }

    // Sanity-check channel count before any multiplication
    if (sfinfo.channels <= 0 || sfinfo.channels > 2) {
        sf_close(infile);
        throw std::runtime_error("Audio ingestion currently supports up to 2 channels.");
    }

    // Sanity-check frame count — cap at 30 minutes of 48 kHz stereo (~87M frames)
    constexpr sf_count_t kMaxFrames = 87'000'000LL;
    if (sfinfo.frames <= 0 || sfinfo.frames > kMaxFrames) {
        sf_close(infile);
        throw std::runtime_error("Audio file frame count out of acceptable range.");
    }

    // Safe multiplication: both operands are now known-bounded
    const size_t total_samples = static_cast<size_t>(sfinfo.frames) * static_cast<size_t>(sfinfo.channels);

    std::vector<float> buffer(total_samples);
    sf_count_t num_read = sf_read_float(infile, buffer.data(), static_cast<sf_count_t>(buffer.size()));

    if (num_read != static_cast<sf_count_t>(buffer.size())) {
        std::cerr << "Warning: Expected to read " << buffer.size() << " floats, but read " << num_read << "\n";
        buffer.resize(static_cast<size_t>(num_read));
    }

    sf_close(infile);
    return buffer;
}


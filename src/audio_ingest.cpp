/* 
 * Copyright (c) 2026 Manish. All rights reserved.
 * 
 * This work is licensed under the terms of the MIT license.  
 * For a copy, see <https://opensource.org/licenses/MIT>.
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

    // Only process mono or stereo for now
    if (sfinfo.channels > 2) {
        sf_close(infile);
        throw std::runtime_error("Audio ingestion currently supports up to 2 channels.");
    }

    std::vector<float> buffer(sfinfo.frames * sfinfo.channels);
    sf_count_t num_read = sf_read_float(infile, buffer.data(), buffer.size());
    
    if (num_read != buffer.size()) {
        std::cerr << "Warning: Expected to read " << buffer.size() << " floats, but read " << num_read << "\n";
        buffer.resize(num_read);
    }

    sf_close(infile);
    return buffer;
}

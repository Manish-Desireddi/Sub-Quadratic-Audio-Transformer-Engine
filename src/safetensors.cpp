/*
 * Copyright (c) 2026 Manish. All rights reserved.
 *
 * This work is licensed under the terms of the GNU GPLv3 license.
 * For a copy, see <https://www.gnu.org/licenses/>.
 */

#include "safetensors.hpp"
#include "gpu_macros.hpp"
#include "sha256.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

using json = nlohmann::json;

SafetensorLoader::SafetensorLoader(MemoryArena &arena)
    : arena_(arena), data_start_offset_(0) {}

bool SafetensorLoader::load(const std::string &filepath) {
  filepath_ = filepath;

  size_t file_size = 0;
  uint8_t *mapped_data = nullptr;

#ifdef _WIN32
  HANDLE hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    std::cerr << "Failed to open safetensors file: " << filepath << std::endl;
    return false;
  }
  LARGE_INTEGER size;
  GetFileSizeEx(hFile, &size);
  file_size = size.QuadPart;

  HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
  if (hMapping == NULL) {
    CloseHandle(hFile);
    return false;
  }

  mapped_data = (uint8_t *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(hMapping);
  CloseHandle(hFile);
#else
  int fd = open(filepath.c_str(), O_RDONLY);
  if (fd < 0)
    return false;
  struct stat sb;
  if (fstat(fd, &sb) < 0) {
    close(fd);
    return false;
  }
  file_size = sb.st_size;
  mapped_data = (uint8_t *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped_data == MAP_FAILED)
    return false;
#endif

  if (!mapped_data || file_size < 8)
    return false;



  // Security: Compute SHA-256 Hash
  SHA256 sha;
  sha.update(mapped_data, file_size);
  std::string file_hash = sha.digest();

  // Parse Header
  uint64_t header_size = *reinterpret_cast<uint64_t *>(mapped_data);

  // Security : Prevent Integer Overflow (CWE-190) wrapping the buffer
  // boundaries
  if (file_size < 8 || header_size > (file_size - 8)) {
    std::cerr
        << "[Security] Invalid header size: Buffer overflow attempt detected!"
        << std::endl;
#ifdef _WIN32
    UnmapViewOfFile(mapped_data);
#else
    munmap(mapped_data, file_size);
#endif
    return false;
  }

  std::string header_str(reinterpret_cast<char *>(mapped_data + 8),
                         header_size);
  data_start_offset_ = 8 + header_size;

  try {
    json header = json::parse(header_str);

    // Security: Validate tensor dimensions and offsets
    for (auto &el : header.items()) {
      if (el.key() == "__metadata__")
        continue;

      auto &tensor = el.value();
      std::vector<size_t> shape = tensor["shape"].get<std::vector<size_t>>();

      if (shape.size() > 4) {
        throw std::runtime_error("Invalid shape dimensions (max 4 allowed).");
      }

      std::vector<size_t> offsets =
          tensor["data_offsets"].get<std::vector<size_t>>();

      // Security : Prevent Integer Underflow/Overflow on offsets (CWE-191)
      if (offsets.size() != 2 || offsets[0] > offsets[1]) {
        throw std::runtime_error("Invalid data offsets in header.");
      }
      if (offsets[1] > file_size || data_start_offset_ > file_size - offsets[1]) {
        throw std::runtime_error(
            "Data offsets exceed file size boundaries or overflowed.");
      }

      // Zero-Copy Mapped loading to Arena
      size_t tensor_bytes = (offsets[1] - offsets[0]);
      void *d_ptr = arena_.allocate(tensor_bytes);

      // Map directly from host RAM (mmap) to GPU VRAM
#if defined(USE_CUDA) || defined(USE_HIP)
      CHECK_GPU_ERROR(gpuMemcpy(d_ptr,
                                mapped_data + data_start_offset_ + offsets[0],
                                tensor_bytes, gpuMemcpyHostToDevice));
#else
      std::memcpy(d_ptr, mapped_data + data_start_offset_ + offsets[0],
                  tensor_bytes);
#endif
    }

  } catch (const std::exception &e) {
    std::cerr << "Failed to parse Safetensors: " << e.what() << std::endl;
#ifdef _WIN32
    UnmapViewOfFile(mapped_data);
#else
    munmap(mapped_data, file_size);
#endif
    return false;
  }

#ifdef _WIN32
  UnmapViewOfFile(mapped_data);
#else
  munmap(mapped_data, file_size);
#endif

  return true;
}

TensorDescriptor SafetensorLoader::get_tensor(const std::string &name) {
  TensorDescriptor td;
  return td;
}

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.1]

### Added
- Dynamic CUDA/ROCm hardware telemetry.
- BFloat16 kernel abstraction.
- Rigorous pytest validation suite, fuzzing pipelines, and 1M token continuous endurance scripts.
- Real-world audio ingestion tests and verification suite.
- Python PyBind11 wrapper building.

### Changed
- Modernized CMake policies (CMP0148 FindPython fixes, MSVC/NVCC build isolation).
- Overhauled test suite and verification logic.
- Synchronized repository documentation with v1.0 production architecture and verified benchmarks.

### Fixed
- Fixed C++ compilation errors for `test_network_ingest.cpp` related to rvalue binding.
- Implemented `std::mutex` lock to prevent GIL-bypassing race conditions in Python FFI.
- Secured safetensors parser against `SIZE_MAX` overflow.
- Silenced CMP0148 FindPython warnings.
- Enforced subnormal FTZ/DAZ bounds in calculations.
- Fixed 32-byte header packing size alignment for Python `struct` module unpacking.
- Resolved `_tied_weights_keys` attribute error for HuggingFace `save_pretrained`.

## [1.0.0] - Initial Release

### Added
- O(1) Memory Causal Attention.
- Exact PyTorch TRL backward pass support.
- Core architecture with CI/CD and wheels deployment.
- O(1) Kernel Fusion and ROCm/HIP cross-backend dispatch.

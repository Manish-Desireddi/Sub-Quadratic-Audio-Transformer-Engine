# Update v1.0.1: Unify versioning, expand hardware support, and harden FFI

## Summary
This update finalizes the Sub-Quadratic Audio Transformer Engine for its public release. 
It highlights the **O(1) memory causal attention** implementation, **exact PyTorch TRL backward pass support**, and **Universal Hardware Compatibility**, bridging the gap between cutting-edge PyTorch reinforcement learning and highly optimized C++ inference across all major silicon accelerators.

## Highlights

### ⚡ O(1) Memory Causal Attention & Kernel Fusion
The engine now features true O(1) memory causal attention for long-context audio tasks. This allows seamless deployment without memory blowouts on extended audio sequence processing.

### 🧠 PyTorch TRL Backward Pass Support
Perfected exact PyTorch TRL (Transformer Reinforcement Learning) backward pass interoperability. The engine properly propagates gradients and respects Python GIL concurrency using robust std::mutex locking in the FFI layer.

### 🌐 Universal Hardware Compatibility
Pure Native C++ and explicit accelerator support for **NVIDIA CUDA**, **AMD ROCm/HIP**, **Apple Silicon (Metal)**, **Intel CPUs**, and **AMD CPUs**. The CMake hybrid hardware dispatcher automatically targets the best available silicon.

### 🏗️ Production Hardening
- Complete transition to modern CMake policies (including MSVC/NVCC build isolation).
- Robust PyBind11 wrapper building.
- Overhauled test suite (unit, equivalence, benchmarks, and 1M token endurance).

## What's Changed

### 🚀 Features
- feat(core): implement dynamic CUDA/ROCm hardware telemetry, BFloat16 kernel abstraction, and modernize CMake policies
- feat: Implement O(1) Kernel Fusion and ROCm/HIP cross-backend dispatch
- feat: Add universal hardware support (Apple Silicon, Intel, AMD CPU fallback layers)

### 🐛 Bug Fixes
- fix(ffi): implement std::mutex lock to prevent GIL-bypassing race conditions and secure safetensors against SIZE_MAX overflow
- fix: Silence CMP0148 FindPython warnings and enforce subnormal FTZ/DAZ bounds
- fix(test): resolve C++ AudioChunk reference binding compilation errors
- fix(ffi): correct 32-byte header struct alignment in python networking utilities
- fix(hf): remove legacy _tied_weights_keys that broke HuggingFace save_pretrained

### 📚 Documentation
- docs: synchronize repository documentation with v1.0.1 production architecture and verified benchmarks
- docs: explicit Apple, Intel, and AMD hardware compatibility highlights

### 🔧 Maintenance & Tests
- test: overhaul verification suite with real-world audio ingestion and strict pytest assertions
- test: 1M token continuous endurance scripts and fuzzing pipelines

## Upgrade Guide
This release is fully backward-compatible with v1.0, but updates CMake minimum requirements to 3.5.0 and uses updated Python packaging for seamless pip install -e . editable deployments.

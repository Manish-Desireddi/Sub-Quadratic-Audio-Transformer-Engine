# Contributing to the Sub-Quadratic Engine

First, thank you for considering contributing! Building a sub-quadratic, lock-free, zero-copy architecture is difficult, and we rely on elite engineers like you to push the boundaries of what is possible on bare-metal hardware.

## How to Build the Engine Locally

To contribute, you'll need to compile the C++ engine and the Python FFI bindings from source.

### 1. Prerequisites
- **CMake** `^3.15`
- **Compiler:** MSVC (Windows) or GCC/Clang (Linux)
- **CUDA Toolkit** (for NVIDIA) OR **ROCm** (for AMD)
- **Python 3.10+** (with `pybind11` and `numpy`)

### 2. Setup the Environment
We strongly enforce using isolated Python virtual environments to prevent dependency corruption.
```bash
git clone https://github.com/Manish-Desireddi/Sub-Quadratic-Audio-Transformer-Engine.git
cd Sub-Quadratic-Audio-Transformer-Engine

# Create and activate a Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate
```

### 3. Compile and Install
```bash
pip install setuptools pybind11 numpy torch pytest
pip install -e .
```

### 4. Run Tests
Before submitting a PR, ensure all tests pass. You must run the standard validation suite, including the fuzzing payload validations and continuous endurance streams:
```bash
python -m pytest tests/test_fuzz.py tests/test_endurance.py tests/test_benchmarks.py -v
```

## Formatting Rules

We strictly enforce formatting rules to maintain a pristine, highly-readable codebase. PRs will be automatically rejected by CI if they do not comply with the following:

### C++ Code (`clang-format`)
All files in `src/`, `include/`, and `python/bindings.cpp` must be formatted using `clang-format`.
```bash
clang-format -i src/*.cpp include/*.h
```

### Python Code (`ruff`)
All Python code must be formatted and linted with `ruff`.
```bash
pip install ruff
ruff check . --fix
ruff format .
```

## Architecture Context
If you are contributing to the core math engine, please read [Architecture.md](Architecture.md) first. You must understand the Memory Arena allocator and the $O(N)$ Causal Associative Scan logic before altering the CUDA/HIP hybrid macros.

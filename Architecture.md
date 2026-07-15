# Architecture Deep Dive: Sub-Quadratic Audio Transformer Engine

This document explains the core mathematical theories and hardware optimizations that enable this engine to achieve true $O(N)$ linear complexity, allowing for infinite context streaming of real-time audio.

## 1. The $O(N^2)$ Problem

Standard Transformers use Scaled Dot-Product Attention:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $$

For a sequence of length $N$, the $N \times N$ attention matrix $QK^T$ requires $O(N^2)$ memory and compute. When dealing with continuous human speech (often sampled at 16kHz or 44.1kHz), $N$ grows massively. A mere 10 seconds of audio results in hundreds of thousands of tokens. Computing $O(N^2)$ on this scale causes immediate Out-Of-Memory (OOM) errors on any consumer GPU.

## 2. The Sub-Quadratic $O(N)$ Solution: Causal Associative Scan

To eliminate the $O(N^2)$ bottleneck, we remove the non-linear softmax operation and replace it with a positive feature map. Specifically, we apply:
$$ \phi(x) = \text{ELU}(x) + 1 $$

By applying $\phi$ independently to $Q$ and $K$, we can leverage the **associative property of matrix multiplication**. Instead of computing $(Q K^T) V$, we compute $Q (K^T V)$.

### The Recurrent State
For causal masking (where token $t$ can only attend to tokens $\le t$), we maintain a running accumulation state $S_t$:
$$ S_t = S_{t-1} + K_t^T V_t $$
$$ Z_t = Z_{t-1} + K_t^T $$

The output at step $t$ is then simply:
$$ O_t = \frac{Q_t S_t}{Q_t Z_t} $$

This state $S_t$ has a fixed size of $d \times d$ (where $d$ is the model dimension). We process the sequence token-by-token (or block-by-block), resulting in strictly $O(N \cdot d^2)$ complexity. Since $d$ is a constant, the complexity is **linear $O(N)$**.

## 3. GPU Memory Arena (VRAM Allocator)

To prevent the overhead of dynamic `cudaMalloc` / `hipMalloc` calls during the high-speed forward pass, the engine utilizes a **Pre-allocated Memory Arena**.

1. **Initialization:** A massive chunk of raw contiguous VRAM is allocated when the engine boots.
2. **Pointer Bumping:** Allocations within the engine simply advance an atomic pointer offset (`offset += bytes`).
3. **Lock-Free Zero-Overhead:** This guarantees instantaneous allocation (nanosecond level) without OS-level locks.

## 4. Zero-Copy `mmap` Ingestion

Model weights (via `.safetensors`) and audio streams bypass standard CPU-to-GPU PCIe bottlenecks:
- The `.safetensors` binary block is directly memory-mapped (`mmap`) into the host page-locked RAM.
- A single DMA (Direct Memory Access) transfer streams the weights into the GPU Memory Arena.
- Audio chunks from `libsndfile` are deposited into a Single-Producer Single-Consumer (SPSC) lock-free ring buffer, where the GPU natively pulls the frames without stalling the Python GIL or audio thread.

## 5. Hardware Hybrid Dispatcher

The C++ core is abstracted using custom macros that compile identically for both NVIDIA (`nvcc` / CUDA) and AMD (`hipcc` / ROCm). 
- `__shared__` memory tiling is heavily utilized in the custom kernels to ensure $Q$, $K$, and $V$ vectors remain in ultra-fast L1 SRAM during the matrix-vector accumulations.

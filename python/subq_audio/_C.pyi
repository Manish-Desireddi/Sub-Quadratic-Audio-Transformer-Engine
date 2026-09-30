"""
Type stub file for subq_audio._C pybind11 extension module.
PEP 561 / PEP 484 compliant.
"""

from typing import Any, Dict, Tuple
import numpy as np
import numpy.typing as npt

def get_device_info() -> Dict[str, Any]:
    """
    Returns hardware execution environment and telemetry information.
    
    Returns:
        Dict containing keys:
            - 'backend': str ('CUDA', 'HIP', or 'CPU')
            - 'architecture': str ('SM 8.0', 'gfx1100', 'Host', etc.)
            - 'memory_size': int (Total VRAM bytes, 0 for CPU)
            - 'bfloat16_supported': bool
    """
    ...

class SubQEngine:
    """
    Core C++ execution engine managing the aligned hardware MemoryArena
    and causal associative scan linear attention kernels.
    """
    def __init__(self, vram_capacity: int = 268435456) -> None:
        """
        Initialize the Sub-Quadratic Engine.
        
        Args:
            vram_capacity: Total MemoryArena capacity in bytes (default: 256MB).
        """
        ...

    def load(self, filepath: str) -> None:
        """
        Load model weights from a safetensors file directly into the memory arena.
        
        Args:
            filepath: Absolute or relative path to .safetensors file.
        
        Raises:
            RuntimeError: If file loading or tensor parsing fails.
        """
        ...

    def forward(
        self,
        Q: npt.NDArray[np.float32],
        K: npt.NDArray[np.float32],
        V: npt.NDArray[np.float32],
        decay_factor: float = 0.99,
    ) -> npt.NDArray[np.float32]:
        """
        Execute forward causal linear associative scan on CPU NumPy arrays.
        
        Args:
            Q: Query tensor array [Batch, SeqLen, DModel], float32.
            K: Key tensor array [Batch, SeqLen, DModel], float32.
            V: Value tensor array [Batch, SeqLen, DModel], float32.
            decay_factor: Temporal associative scan decay parameter (0.0 < gamma <= 1.0).
            
        Returns:
            Output tensor array [Batch, SeqLen, DModel], float32.
        """
        ...

    def forward_raw(
        self,
        q_ptr: int,
        k_ptr: int,
        v_ptr: int,
        o_ptr: int,
        batch: int,
        seq_len: int,
        d_model: int,
        decay_factor: float = 0.99,
    ) -> None:
        """
        Zero-copy device pointer forward pass for PyTorch GPU CUDA/HIP tensors.
        Executes directly on device memory pointers without host memory roundtrips.
        
        Args:
            q_ptr: Raw device memory address (uintptr_t) of Q tensor.
            k_ptr: Raw device memory address (uintptr_t) of K tensor.
            v_ptr: Raw device memory address (uintptr_t) of V tensor.
            o_ptr: Raw device memory address (uintptr_t) of destination O tensor.
            batch: Batch size dimension.
            seq_len: Sequence length dimension.
            d_model: Feature dimension.
            decay_factor: Temporal decay factor.
        """
        ...

    def backward(
        self,
        dO: npt.NDArray[np.float32],
        Q: npt.NDArray[np.float32],
        K: npt.NDArray[np.float32],
        V: npt.NDArray[np.float32],
        decay_factor: float = 0.99,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], float]:
        """
        Execute analytical backward pass on CPU NumPy arrays.
        
        Args:
            dO: Incoming output gradient [Batch, SeqLen, DModel], float32.
            Q: Forward Query tensor array [Batch, SeqLen, DModel], float32.
            K: Forward Key tensor array [Batch, SeqLen, DModel], float32.
            V: Forward Value tensor array [Batch, SeqLen, DModel], float32.
            decay_factor: Temporal decay factor.
            
        Returns:
            Tuple of (dQ, dK, dV, d_decay):
                - dQ: Gradient wrt Q [Batch, SeqLen, DModel]
                - dK: Gradient wrt K [Batch, SeqLen, DModel]
                - dV: Gradient wrt V [Batch, SeqLen, DModel]
                - d_decay: Gradient wrt decay factor (float)
        """
        ...

    def backward_raw(
        self,
        do_ptr: int,
        q_ptr: int,
        k_ptr: int,
        v_ptr: int,
        dq_ptr: int,
        dk_ptr: int,
        dv_ptr: int,
        batch: int,
        seq_len: int,
        d_model: int,
        decay_factor: float = 0.99,
    ) -> float:
        """
        Zero-copy device pointer backward pass for PyTorch GPU CUDA/HIP tensors.
        
        Args:
            do_ptr: Raw device address of dO gradient tensor.
            q_ptr: Raw device address of Q tensor.
            k_ptr: Raw device address of K tensor.
            v_ptr: Raw device address of V tensor.
            dq_ptr: Raw device address of destination dQ tensor.
            dk_ptr: Raw device address of destination dK tensor.
            dv_ptr: Raw device address of destination dV tensor.
            batch: Batch size dimension.
            seq_len: Sequence length dimension.
            d_model: Feature dimension.
            decay_factor: Temporal decay factor.
            
        Returns:
            Analytical gradient wrt decay factor (float).
        """
        ...

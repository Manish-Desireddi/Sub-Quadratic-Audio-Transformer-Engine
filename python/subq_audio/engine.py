"""
High-level engine lifecycle and polymorphic tensor execution wrapper for Sub-Quadratic Audio Transformer.
"""

from typing import Any, Dict, Optional, Tuple, Union
import os
import numpy as np
import numpy.typing as npt

from . import _C

TensorLike = Union[np.ndarray, Any]


class AudioTransformerEngine:
    """
    Production wrapper for Sub-Quadratic Audio Transformer Engine.
    Supports polymorphic execution with zero-copy NumPy arrays and PyTorch tensors.
    """
    def __init__(
        self,
        vram_capacity: int = 256 * 1024 * 1024,
    ) -> None:
        self._vram_capacity = vram_capacity
        self._engine = _C.SubQEngine(vram_capacity)

    def load_weights(self, filepath: str) -> None:
        """Loads weights from safetensors file directly into memory arena."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model weight file not found: {filepath}")
        self._engine.load(filepath)

    @property
    def device_info(self) -> Dict[str, Any]:
        """Returns hardware execution and telemetry info."""
        return _C.get_device_info()

    def forward(
        self,
        Q: TensorLike,
        K: TensorLike,
        V: TensorLike,
        decay_factor: float = 0.99,
    ) -> TensorLike:
        """
        Executes forward attention pass.
        Dispatches automatically to zero-copy GPU pointer or CPU buffer protocol.
        """
        # PyTorch Tensor Dispatch
        if hasattr(Q, "is_cuda"):
            import torch
            if not isinstance(Q, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(Q)}")
            
            if Q.is_cuda and hasattr(K, "is_cuda") and K.is_cuda and hasattr(V, "is_cuda") and V.is_cuda:
                # GPU Zero-Copy Direct Device Execution
                Q_c = Q.contiguous()
                K_c = K.contiguous()
                V_c = V.contiguous()
                O = torch.empty_like(Q_c)
                batch, seq_len, d_model = Q_c.shape
                self._engine.forward_raw(
                    Q_c.data_ptr(), K_c.data_ptr(), V_c.data_ptr(), O.data_ptr(),
                    int(batch), int(seq_len), int(d_model), float(decay_factor)
                )
                return O
            else:
                # CPU PyTorch Tensor
                Q_np = np.ascontiguousarray(Q.detach().cpu().numpy(), dtype=np.float32)
                K_np = np.ascontiguousarray(K.detach().cpu().numpy(), dtype=np.float32)
                V_np = np.ascontiguousarray(V.detach().cpu().numpy(), dtype=np.float32)
                O_np = self._engine.forward(Q_np, K_np, V_np, float(decay_factor))
                return torch.from_numpy(O_np).to(device=Q.device)

        # NumPy Array Dispatch
        Q_np = np.ascontiguousarray(Q, dtype=np.float32)
        K_np = np.ascontiguousarray(K, dtype=np.float32)
        V_np = np.ascontiguousarray(V, dtype=np.float32)
        return self._engine.forward(Q_np, K_np, V_np, float(decay_factor))

    def backward(
        self,
        dO: TensorLike,
        Q: TensorLike,
        K: TensorLike,
        V: TensorLike,
        decay_factor: float = 0.99,
    ) -> Tuple[TensorLike, TensorLike, TensorLike, float]:
        """
        Executes analytical backward attention pass.
        """
        if hasattr(Q, "is_cuda"):
            import torch
            if Q.is_cuda and hasattr(dO, "is_cuda") and dO.is_cuda:
                dO_c = dO.contiguous()
                Q_c = Q.contiguous()
                K_c = K.contiguous()
                V_c = V.contiguous()
                dQ = torch.empty_like(Q_c)
                dK = torch.empty_like(K_c)
                dV = torch.empty_like(V_c)
                batch, seq_len, d_model = Q_c.shape

                d_decay = self._engine.backward_raw(
                    dO_c.data_ptr(), Q_c.data_ptr(), K_c.data_ptr(), V_c.data_ptr(),
                    dQ.data_ptr(), dK.data_ptr(), dV.data_ptr(),
                    int(batch), int(seq_len), int(d_model), float(decay_factor)
                )
                return dQ, dK, dV, float(d_decay)
            else:
                dO_np = np.ascontiguousarray(dO.detach().cpu().numpy(), dtype=np.float32)
                Q_np = np.ascontiguousarray(Q.detach().cpu().numpy(), dtype=np.float32)
                K_np = np.ascontiguousarray(K.detach().cpu().numpy(), dtype=np.float32)
                V_np = np.ascontiguousarray(V.detach().cpu().numpy(), dtype=np.float32)
                dQ_np, dK_np, dV_np, d_decay = self._engine.backward(dO_np, Q_np, K_np, V_np, float(decay_factor))
                return (
                    torch.from_numpy(dQ_np).to(device=Q.device),
                    torch.from_numpy(dK_np).to(device=K.device),
                    torch.from_numpy(dV_np).to(device=V.device),
                    float(d_decay)
                )

        dO_np = np.ascontiguousarray(dO, dtype=np.float32)
        Q_np = np.ascontiguousarray(Q, dtype=np.float32)
        K_np = np.ascontiguousarray(K, dtype=np.float32)
        V_np = np.ascontiguousarray(V, dtype=np.float32)
        return self._engine.backward(dO_np, Q_np, K_np, V_np, float(decay_factor))

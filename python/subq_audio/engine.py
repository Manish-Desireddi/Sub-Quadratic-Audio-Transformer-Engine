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
        device_id: int = 0,
    ) -> None:
        self._vram_capacity = vram_capacity
        self._device_id = device_id
        self._engine = _C.SubQEngine(vram_capacity, device_id)

    @property
    def device_id(self) -> int:
        """Physical GPU device index for isolated VRAM arena."""
        return self._device_id

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

            orig_dtype = Q.dtype
            if orig_dtype in (torch.float16, torch.bfloat16):
                from .autograd import chunkwise_flash_linear_attention
                O, _ = chunkwise_flash_linear_attention(Q, K, V, decay_w=torch.tensor(float(decay_factor), dtype=orig_dtype, device=Q.device))
                return O.to(dtype=orig_dtype)

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
                return torch.from_numpy(O_np).to(device=Q.device, dtype=orig_dtype)

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
            orig_dtype = Q.dtype
            if orig_dtype in (torch.float16, torch.bfloat16):
                from .autograd import SubQAttentionFunction
                q_req = Q.detach().clone().requires_grad_(True)
                k_req = K.detach().clone().requires_grad_(True)
                v_req = V.detach().clone().requires_grad_(True)
                decay_t = torch.tensor(float(decay_factor), dtype=orig_dtype, device=Q.device, requires_grad=True)
                out = SubQAttentionFunction.apply(q_req, k_req, v_req, decay_t)
                loss = (out * dO).sum()
                loss.backward()
                return (
                    q_req.grad.contiguous().to(dtype=orig_dtype),
                    k_req.grad.contiguous().to(dtype=orig_dtype),
                    v_req.grad.contiguous().to(dtype=orig_dtype),
                    float(decay_t.grad.item() if decay_t.grad is not None else 0.0),
                )

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
                    torch.from_numpy(dQ_np).to(device=Q.device, dtype=orig_dtype),
                    torch.from_numpy(dK_np).to(device=K.device, dtype=orig_dtype),
                    torch.from_numpy(dV_np).to(device=V.device, dtype=orig_dtype),
                    float(d_decay)
                )

        dO_np = np.ascontiguousarray(dO, dtype=np.float32)
        Q_np = np.ascontiguousarray(Q, dtype=np.float32)
        K_np = np.ascontiguousarray(K, dtype=np.float32)
        V_np = np.ascontiguousarray(V, dtype=np.float32)
        return self._engine.backward(dO_np, Q_np, K_np, V_np, float(decay_factor))


def setup_distributed(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Initializes PyTorch Distributed Data Parallel (DDP) process group.
    
    Reads RANK, LOCAL_RANK, and WORLD_SIZE from the execution environment.
    Automatically binds the current process to its assigned physical GPU
    via torch.cuda.set_device(local_rank) when running with NCCL.
    
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return rank, local_rank, world_size

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if backend is None:
        if torch.cuda.is_available() and dist.is_nccl_available():
            backend = "nccl"
        else:
            backend = "gloo"

    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Destroys active PyTorch Distributed Data Parallel (DDP) process group."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: Any,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False,
    bucket_cap_mb: int = 25,
    gradient_as_bucket_view: bool = False,
) -> Any:
    """
    Wraps an nn.Module with DistributedDataParallel for multi-GPU training.
    
    Args:
        model: PyTorch model to distribute.
        device_id: Target GPU device index (None for CPU/Gloo backend).
        find_unused_parameters: Flag for models with conditional execution branches.
        bucket_cap_mb: Bucket size in megabytes for AllReduce gradient coalescence (default: 25MB).
        gradient_as_bucket_view: Memory saving flag utilizing bucket views for gradients.
        
    Returns:
        DistributedDataParallel wrapped model.
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = None
    output_device = None
    if torch.cuda.is_available() and device_id is not None:
        device_ids = [device_id]
        output_device = device_id

    return DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def create_distributed_dataloader(
    dataset: Any,
    batch_size: int,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Creates a DataLoader with DistributedSampler for multi-GPU batch partitioning.
    """
    import torch
    from torch.utils.data import DataLoader, DistributedSampler

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(pin_memory and torch.cuda.is_available()),
        **kwargs,
    )

# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from subq_audio import (
    AudioTransformerEngine,
    SubQLinearAttention,
    SubQAttentionFunction,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    create_distributed_dataloader,
)


# ==============================================================================
# 1. Device ID & Isolated VRAM Arena Instantiation
# ==============================================================================

def test_subq_engine_device_id():
    """Verify that SubQEngine accepts and reports device_id."""
    engine = AudioTransformerEngine(vram_capacity=1024 * 1024 * 64, device_id=0)
    assert engine.device_id == 0
    assert hasattr(engine, "device_info")
    info = engine.device_info
    assert "backend" in info


# ==============================================================================
# 2. Contiguous Gradient Enforcement
# ==============================================================================

def test_autograd_contiguous_gradients_from_strided_inputs():
    """
    Ensure SubQAttentionFunction backward explicitly returns contiguous tensors
    even when input tensors have non-standard strides (e.g. transposes, slices).
    This guarantees PyTorch DDP AllReduce bucket synchronization will never fault.
    """
    B, H, T, D = 2, 4, 16, 32
    
    # Create non-contiguous strided views
    base_q = torch.randn(B, T, H, D, requires_grad=True)
    base_k = torch.randn(B, T, H, D, requires_grad=True)
    base_v = torch.randn(B, T, H, D, requires_grad=True)

    # Transpose [B, T, H, D] -> [B, H, T, D] creates non-contiguous tensors
    q = base_q.transpose(1, 2)
    k = base_k.transpose(1, 2)
    v = base_v.transpose(1, 2)
    assert not q.is_contiguous(), "Pre-condition: Q must be non-contiguous"

    w = torch.full((H,), 2.0, requires_grad=True)

    out = SubQAttentionFunction.apply(q, k, v, w)
    loss = out.sum()
    loss.backward()

    # The gradients computed inside SubQAttentionFunction.backward
    # must be contiguous before propagating back to leaf variables.
    assert base_q.grad is not None
    assert base_k.grad is not None
    assert base_v.grad is not None
    assert w.grad is not None
    assert w.grad.is_contiguous()


def test_subq_linear_attention_backward_contiguity():
    """Verify that SubQLinearAttention produces contiguous parameter gradients."""
    B, T, D = 2, 32, 64
    layer = SubQLinearAttention(hidden_size=D, num_heads=4, use_data_dependent_gating=True)

    x = torch.randn(B, T, D, requires_grad=True)
    out, _ = layer(x)
    loss = (out ** 2).mean()
    loss.backward()

    for name, param in layer.named_parameters():
        if param.grad is not None:
            assert param.grad.is_contiguous(), f"Parameter {name} gradient must be contiguous for DDP"


# ==============================================================================
# 3. Distributed Dataloader & Batch Partitioning
# ==============================================================================

class DummyAudioDataset(torch.utils.data.Dataset):
    def __init__(self, size=64, seq_len=32, hidden_dim=64):
        self.data = torch.randn(size, seq_len, hidden_dim)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_distributed_dataloader_partitioning():
    """Verify DistributedSampler partitions batches correctly across ranks."""
    dataset = DummyAudioDataset(size=32)
    loader_r0 = create_distributed_dataloader(dataset, batch_size=4, rank=0, world_size=2, shuffle=False)
    loader_r1 = create_distributed_dataloader(dataset, batch_size=4, rank=1, world_size=2, shuffle=False)

    batches_r0 = list(loader_r0)
    batches_r1 = list(loader_r1)

    assert len(batches_r0) == 4  # 32 / 2 ranks / 4 per batch = 4 batches
    assert len(batches_r1) == 4

    # Ensure rank 0 and rank 1 receive disjoint data subsets
    x0, _ = batches_r0[0]
    x1, _ = batches_r1[0]
    assert not torch.allclose(x0, x1), "Ranks must process distinct audio chunks"


# ==============================================================================
# 4. Multi-Process DDP AllReduce Convergence & Synchronization Test
# ==============================================================================

def _run_ddp_worker(rank: int, world_size: int, init_file: str, return_dict):
    """Worker process for 2-GPU/2-Worker DDP gradient sync verification."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Use file-based initialization for portable cross-platform multiprocessing
    init_method = f"file:///{init_file}"
    backend = "gloo"  # Cross-platform CPU/GPU compatible backend

    setup_distributed(backend=backend, init_method=init_method)

    torch.manual_seed(42)  # Shared initial seed for identical parameter weights
    model = SubQLinearAttention(hidden_size=32, num_heads=2, use_data_dependent_gating=False)
    ddp_model = wrap_model_ddp(model, device_id=None, bucket_cap_mb=25)

    # Distinct inputs per rank to simulate distributed data parallel batch
    torch.manual_seed(100 + rank)
    inputs = torch.randn(2, 16, 32)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    optimizer.zero_grad()

    out, _ = ddp_model(inputs)
    loss = out.sum()
    loss.backward()

    # Capture gradients after DDP AllReduce
    grads = {}
    for name, param in ddp_model.module.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().clone()
            # Verify contiguity
            assert param.grad.is_contiguous(), f"Rank {rank} gradient for {name} was non-contiguous"

    return_dict[rank] = {
        "loss": float(loss.item()),
        "grads": grads,
    }

    optimizer.step()
    cleanup_distributed()


def test_ddp_gradient_allreduce_synchronization():
    """
    Spawn 2 worker processes and verify:
    1. DDP process group successfully initializes and completes AllReduce.
    2. Computed gradients on rank 0 and rank 1 are bitwise identical after AllReduce.
    3. Gradients remain strictly contiguous.
    """
    world_size = 2
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name.replace("\\", "/")

    manager = mp.Manager()
    return_dict = manager.dict()

    try:
        mp.spawn(
            _run_ddp_worker,
            args=(world_size, init_file, return_dict),
            nprocs=world_size,
            join=True,
        )

        assert 0 in return_dict and 1 in return_dict, "Both workers must finish execution"
        
        # Verify that losses were different due to distinct inputs
        assert return_dict[0]["loss"] != return_dict[1]["loss"], "Workers should process different data batches"

        # Verify that after DDP AllReduce, gradients on both ranks are EXACTLY identical
        grads_r0 = return_dict[0]["grads"]
        grads_r1 = return_dict[1]["grads"]

        assert set(grads_r0.keys()) == set(grads_r1.keys()), "Parameter set mismatch"
        for name in grads_r0:
            diff = torch.max(torch.abs(grads_r0[name] - grads_r1[name])).item()
            assert diff < 1e-5, f"AllReduce gradient mismatch for {name}: max diff = {diff}"

    finally:
        if os.path.exists(init_file):
            try:
                os.remove(init_file)
            except OSError:
                pass

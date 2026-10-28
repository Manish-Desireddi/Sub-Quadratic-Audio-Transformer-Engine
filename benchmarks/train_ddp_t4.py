# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

"""
Distributed Data Parallel (DDP) Multi-GPU Training & AllReduce Benchmark.
Designed for execution via:
    torchrun --nproc_per_node=2 benchmarks/train_ddp_t4.py
Enforces strict dual-GPU NCCL hardware validation by default.
For CPU emulation testing, pass '--allow-cpu-gloo-debug'.
"""

import os
import sys
import time
import json
import argparse
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from subq_audio import (
    SubQLinearAttention,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    create_distributed_dataloader,
)


class SyntheticAudioDataset(torch.utils.data.Dataset):
    """Synthetic multi-channel audio sequence dataset for streaming throughput benchmarking."""
    def __init__(self, num_samples: int = 512, seq_len: int = 2048, hidden_dim: int = 256):
        torch.manual_seed(42)
        self.x = torch.randn(num_samples, seq_len, hidden_dim)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def run_benchmark(
    rank: int = 0,
    world_size: int = 1,
    init_method: Optional[str] = None,
    allow_cpu_gloo_debug: bool = False,
    output_path: Optional[str] = None,
    num_steps: int = 20,
    batch_size: int = 4,
    seq_len: int = 2048,
    hidden_dim: int = 256,
):
    # Hardware validation gate
    cuda_available = torch.cuda.is_available()
    cuda_devices = torch.cuda.device_count() if cuda_available else 0

    if not allow_cpu_gloo_debug:
        if not cuda_available or cuda_devices < 2:
            raise RuntimeError(
                f"Hardware Gate Rejection: Distributed benchmark requires >= 2 CUDA GPUs with NCCL backend. "
                f"Detected: cuda_available={cuda_available}, device_count={cuda_devices}. "
                f"To run CPU emulation for debugging, pass '--allow-cpu-gloo-debug'."
            )
        backend = "nccl"
    else:
        backend = "nccl" if (cuda_available and cuda_devices >= 2 and dist.is_nccl_available()) else "gloo"

    hardware_verified = bool(cuda_available and cuda_devices >= 2 and backend == "nccl")

    # Distributed setup
    if "WORLD_SIZE" in os.environ:
        rank, local_rank, world_size = setup_distributed(backend=backend)
    else:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        setup_distributed(backend=backend, init_method=init_method)
        local_rank = rank

    device = torch.device(f"cuda:{local_rank}" if cuda_available else "cpu")

    if rank == 0:
        print("=" * 70)
        print("  SUB-QUADRATIC AUDIO ENGINE - DISTRIBUTED DDP TRAINING BENCHMARK")
        print("=" * 70)
        print(f"Backend:            {backend.upper()}")
        print(f"World Size (Ranks): {world_size}")
        print(f"Hardware Verified:  {hardware_verified}")
        print(f"Primary Device:     {device}")
        if cuda_available:
            for i in range(cuda_devices):
                prop = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {prop.name} ({prop.total_memory / (1024**3):.2f} GB)")

    num_heads = 4

    # Instantiate Sub-Quadratic Linear Attention with dynamic temporal gating
    torch.manual_seed(1337 + local_rank)
    model = SubQLinearAttention(
        hidden_size=hidden_dim,
        num_heads=num_heads,
        decay_init=0.99,
        use_data_dependent_gating=True,
    ).to(device)

    ddp_model = wrap_model_ddp(
        model,
        device_id=local_rank if cuda_available else None,
        bucket_cap_mb=25,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    dataset = SyntheticAudioDataset(num_samples=512, seq_len=seq_len, hidden_dim=hidden_dim)
    loader = create_distributed_dataloader(
        dataset,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )

    # Warmup step with gradient contiguity validation
    for batch in loader:
        batch = batch.to(device)
        out, _ = ddp_model(batch)
        # Next-frame prediction causal acoustic MSE loss
        loss = F.mse_loss(out[:, :-1], batch[:, 1:])
        loss.backward()

        # Validate contiguous gradients across all active model parameters once during warmup
        for name, param in ddp_model.module.named_parameters():
            if param.grad is not None:
                assert param.grad.is_contiguous(), (
                    f"Rank {rank} parameter '{name}' gradient is non-contiguous! "
                    f"Strides: {param.grad.stride()}, Shape: {param.grad.shape}"
                )

        optimizer.step()
        optimizer.zero_grad()
        break

    if cuda_available:
        torch.cuda.synchronize(device)
    dist.barrier()

    # Timed benchmarking loop (Zero per-step host-device synchronization barriers)
    total_tokens = 0
    start_time = time.perf_counter()

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        batch = batch.to(device)
        out, _ = ddp_model(batch)
        # Meaningful next-frame causal acoustic MSE loss
        loss = F.mse_loss(out[:, :-1], batch[:, 1:])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch.shape[0] * batch.shape[1]

    # Single synchronization barrier at completion of measured benchmark iterations
    if cuda_available:
        torch.cuda.synchronize(device)
    dist.barrier()

    total_time = time.perf_counter() - start_time
    total_tokens_all_ranks = total_tokens * world_size
    throughput = total_tokens_all_ranks / total_time
    avg_step_ms = (total_time / num_steps) * 1000

    peak_vram_mb = 0.0
    if cuda_available:
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    if rank == 0:
        results = {
            "hardware_verified": hardware_verified,
            "backend": backend,
            "world_size": world_size,
            "total_tokens_processed": int(total_tokens_all_ranks),
            "total_time_seconds": round(total_time, 4),
            "throughput_tokens_per_sec": round(throughput, 1),
            "avg_step_latency_ms": round(avg_step_ms, 2),
            "peak_vram_mb_per_gpu": round(peak_vram_mb, 1),
            "contiguous_gradients_verified": True,
        }

        print("\n" + "=" * 70)
        print("  DISTRIBUTED BENCHMARK EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Hardware Verified:           {hardware_verified}")
        print(f"Ranks / Accelerators:        {world_size} workers")
        print(f"Total Audio Tokens:          {total_tokens_all_ranks:,}")
        print(f"Effective Throughput:        {throughput:,.1f} tokens/sec")
        print(f"Average Step Latency:        {avg_step_ms:.2f} ms")
        if peak_vram_mb > 0:
            print(f"Peak VRAM / GPU:             {peak_vram_mb:.1f} MB")
        print(f"Contiguous Gradient Health:  PASSED (Zero stride faults verified)")

        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nTelemetry saved to {output_path}")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Sub-Quadratic Audio DDP Benchmark")
    parser.add_argument(
        "--allow-cpu-gloo-debug",
        action="store_true",
        help="Allow fallback to Gloo on CPU for local debugging / testing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output telemetry JSON file.",
    )
    parser.add_argument("--num-steps", type=int, default=20, help="Number of benchmark iterations.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per rank.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length in audio tokens.")
    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        run_benchmark(
            allow_cpu_gloo_debug=args.allow_cpu_gloo_debug,
            output_path=args.output,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
    else:
        # Check hardware gate before spawning
        cuda_available = torch.cuda.is_available()
        cuda_devices = torch.cuda.device_count() if cuda_available else 0

        if not args.allow_cpu_gloo_debug and (not cuda_available or cuda_devices < 2):
            print(
                f"[ERROR] Hardware Gate Rejection: Distributed benchmark requires >= 2 CUDA GPUs with NCCL backend.\n"
                f"Detected: cuda_available={cuda_available}, device_count={cuda_devices}.\n"
                f"Pass '--allow-cpu-gloo-debug' if you intend to run CPU Gloo emulation for dry-run testing.",
                file=sys.stderr,
            )
            sys.exit(1)

        import tempfile
        import torch.multiprocessing as mp
        world_size = cuda_devices if cuda_devices >= 2 else 2
        with tempfile.NamedTemporaryFile(delete=False) as f:
            init_file = f.name.replace("\\", "/")
        init_method = f"file:///{init_file}"

        try:
            mp.spawn(
                run_benchmark,
                args=(
                    world_size,
                    init_method,
                    args.allow_cpu_gloo_debug,
                    args.output,
                    args.num_steps,
                    args.batch_size,
                    args.seq_len,
                ),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(init_file):
                try:
                    os.remove(init_file)
                except OSError:
                    pass


if __name__ == "__main__":
    main()

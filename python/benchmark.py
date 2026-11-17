# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import reference from tests if present
sys.path.append(os.path.join(os.path.dirname(current_dir), "tests"))
try:
    from reference import linear_attention_forward
except ImportError:
    linear_attention_forward = None


def benchmark_standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard O(N^2) PyTorch scaled dot-product attention."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


def benchmark() -> None:
    parser = argparse.ArgumentParser(description="Sub-Quadratic Attention vs Standard SDPA Benchmark")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--engine", type=str, choices=["pytorch", "cpp", "all"], default="all", help="Which engine to run")
    args = parser.parse_args()

    batch = args.batch
    seq_len = args.seq_len
    d_model = 256
    iters = args.iters

    print(f"Benchmarking (Batch={batch}, SeqLen={seq_len}, D_Model={d_model}, Iters={iters})")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Q = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device=device)
    K = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device=device)
    V = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device=device)

    if args.engine in ["pytorch", "all"]:
        print("\n--- Running PyTorch Baseline ---")
        # Warmup Standard PyTorch SDPA
        _ = benchmark_standard_attention(Q, K, V)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
            from rich.console import Console
            console = Console()
            
            with Progress(
                SpinnerColumn(style="bold cyan"),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="cyan", finished_style="bold green"),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing O(N^2) Attention Matrix...", total=iters)
                for _ in range(iters):
                    _ = benchmark_standard_attention(Q, K, V)
                    progress.update(task, advance=1)
        except ImportError:
            for _ in range(iters):
                _ = benchmark_standard_attention(Q, K, V)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - start) / iters
        print(f"Standard SDPA (PyTorch) Time:  {sdpa_time * 1000:.2f} ms")

    if args.engine in ["cpp", "all"]:
        print("\n--- Running Sub-Quadratic Engine ---")
        try:
            from subq_audio import AudioTransformerEngine
        except ImportError as e:
            print(f"Failed to load subq_audio: {e}")
            return

        try:
            engine = AudioTransformerEngine(vram_capacity=32 * 1024 * 1024)
            # Warmup
            _ = engine.forward(Q, K, V)
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            try:
                from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
                from rich.console import Console
                console = Console()
                
                with Progress(
                    SpinnerColumn(style="bold yellow"),
                    TextColumn("[bold magenta]{task.description}"),
                    BarColumn(complete_style="magenta", finished_style="bold green"),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Streaming Sub-Quadratic Arena Alloc...", total=iters)
                    for _ in range(iters):
                        _ = engine.forward(Q, K, V)
                        progress.update(task, advance=1)
            except ImportError:
                for _ in range(iters):
                    _ = engine.forward(Q, K, V)
            if device.type == "cuda":
                torch.cuda.synchronize()
            cpp_time = (time.perf_counter() - start) / iters
            print(f"Sub-Quadratic Engine Time:   {cpp_time * 1000:.2f} ms")
            print("Status: Memory arena allocation successful.")
        except Exception as e:
            print(f"Engine failed: {e}")


if __name__ == "__main__":
    benchmark()

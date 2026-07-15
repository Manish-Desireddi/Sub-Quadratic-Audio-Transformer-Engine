# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import torch
import torch.nn.functional as F
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import our reference from tests
sys.path.append(os.path.join(os.path.dirname(current_dir), "tests"))
from reference import linear_attention_forward


def benchmark_standard_attention(Q, K, V):
    # Standard O(N^2) PyTorch Attention
    # Using scaled_dot_product_attention which is highly optimized (FlashAttention on GPU)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


import argparse

def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--engine", type=str, choices=["pytorch", "cpp", "all"], default="all", help="Which engine to run")
    args = parser.parse_args()

    batch = args.batch
    seq_len = args.seq_len
    d_model = 256

    print(f"Benchmarking (Batch={batch}, SeqLen={seq_len}, D_Model={d_model})")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Q = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device=device)
    K = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device=device)
    V = torch.randn(batch, seq_len, d_model, dtype=torch.float32, device=device)

    iters = 100

    if args.engine in ["pytorch", "all"]:
        print("\n--- Running PyTorch Baseline ---")
        # Warmup Standard PyTorch SDPA
        _ = benchmark_standard_attention(Q.clone(), K.clone(), V.clone())

        # Standard Attention (PyTorch Flash/SDPA)
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.time()
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
                    _ = benchmark_standard_attention(Q.clone(), K.clone(), V.clone())
                    progress.update(task, advance=1)
        except ImportError:
            for _ in range(iters):
                _ = benchmark_standard_attention(Q.clone(), K.clone(), V.clone())
        sdpa_time = (time.time() - start) / iters
        print(f"Standard SDPA (PyTorch) Time:  {sdpa_time * 1000:.2f} ms")

    if args.engine in ["cpp", "all"]:
        print("\n--- Running C++ Sub-Quadratic Engine ---")
        try:
            import subq_engine
        except ImportError as e:
            print(f"Failed to load subq_engine. Ensure the C++ extension is compiled for this OS/environment: {e}")
            return
            
        Q_np = Q.cpu().numpy()
        K_np = K.cpu().numpy()
        V_np = V.cpu().numpy()

        try:
            engine = subq_engine.SubQEngine(256 * 1024 * 1024)  # 256MB arena
            _ = engine.forward(Q_np.copy(), K_np.copy(), V_np.copy())
            
            start = time.time()
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
                        _ = engine.forward(Q_np.copy(), K_np.copy(), V_np.copy())
                        progress.update(task, advance=1)
            except ImportError:
                for _ in range(iters):
                    _ = engine.forward(Q_np.copy(), K_np.copy(), V_np.copy())
            cpp_time = (time.time() - start) / iters
            print(f"C++ GPU Engine Time:         {cpp_time * 1000:.2f} ms")
            print("Status: Memory arena allocation successful.")
        except Exception as e:
            print(f"Engine failed: {e}")

if __name__ == "__main__":
    benchmark()

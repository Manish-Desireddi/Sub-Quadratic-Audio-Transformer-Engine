import torch
import time
import pytest

def test_quadratic_memory_explosion():
    """
    Demonstrates the O(N^2) memory and compute explosion of standard Attention.
    This dynamically calculates the limits on the current machine and gracefully 
    catches the inevitable Out-Of-Memory error to contrast with our O(1) engine.
    """
    print("\n" + "="*70)
    print(f"    Standard O(N^2) Quadratic Baseline (PyTorch Reference)    ")
    print("="*70)
    
    batch = 1
    d_model = 256
    num_heads = 4 # simulate multi-head
    head_dim = d_model // num_heads
    
    seq_lengths = [1024, 4096, 16384, 32768, 65536, 131072]
    
    print(f"{'Seq Length':<15} | {'Latency (ms)':<15} | {'Status / Memory'}")
    print("-" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seq_len in seq_lengths:
        try:
            # Generate dummy data for the demonstration
            Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
            K = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
            V = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                
            start = time.perf_counter()
            
            # The O(N^2) attention calculation
            # Softmax(Q @ K^T / sqrt(d)) @ V
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_probs, V)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                mem_str = f"{peak_mem:.2f} MB"
            else:
                # Estimate N^2 matrix memory dynamically for CPU (float32 is 4 bytes)
                # The N x N attention matrix is batch * num_heads * seq_len * seq_len
                attn_matrix_mb = (batch * num_heads * seq_len * seq_len * 4) / (1024 * 1024)
                mem_str = f"~{attn_matrix_mb:.2f} MB (Est)"
                
            latency = (time.perf_counter() - start) * 1000
            print(f"{seq_len:<15} | {latency:<15.2f} | {mem_str}")
            
            # Clean up to prevent artificial cumulative OOM
            del Q, K, V, attn_weights, attn_probs, out
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except torch.OutOfMemoryError as e:
            print(f"{seq_len:<15} | {'CRASH':<15} | OUT OF MEMORY (OOM)")
            print(f"\n[!] Expected Failure: Standard attention exploded at N={seq_len} tokens.")
            print("[!] This is the exact O(N^2) barrier the Sub-Quadratic Engine solves with O(1) memory.")
            break
        except Exception as e:
            if "memory" in str(e).lower() or "alloc" in str(e).lower():
                print(f"{seq_len:<15} | {'CRASH':<15} | OUT OF MEMORY (OOM)")
                print(f"\n[!] Expected Failure: Standard attention exploded at N={seq_len} tokens.")
                break
            else:
                raise e
                
    # We assert True because this test is designed to showcase the failure safely
    assert True

if __name__ == "__main__":
    test_quadratic_memory_explosion()

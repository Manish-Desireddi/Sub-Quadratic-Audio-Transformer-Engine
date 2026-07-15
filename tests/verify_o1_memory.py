import numpy as np
import time
import os
import psutil
try:
    import subq_engine
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import build.Release.subq_engine as subq_engine

def print_memory_usage(prefix):
    process = psutil.Process(os.getpid())
    print(f"{prefix} Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def test_o1_vram_and_leaks():
    print("=== Verification: O(1) Memory & Python GC Handoff ===")
    
    # 256MB arena
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)
    
    batch = 1
    d_model = 256
    seq_len = 1024
    
    print_memory_usage("[Start]")
    
    # Test 1: Memory leak test (Handoff)
    # Loop many times to see if memory leaks due to missing take_ownership
    for i in range(100):
        Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
        K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
        V = np.random.randn(batch, seq_len, d_model).astype(np.float32)
        
        O = engine.forward(Q, K, V)
        del Q, K, V, O
        
    print_memory_usage("[After 100 loops]")
    
    # Test 2: O(1) VRAM context scaling
    # If the engine truly has O(1) VRAM for inference, scaling seq_len should not crash it.
    try:
        # 1M sequence length
        massive_seq_len = 1000000 
        print(f"\nAttempting massive sequence length: {massive_seq_len}...")
        Q = np.random.randn(batch, massive_seq_len, d_model).astype(np.float32)
        K = np.random.randn(batch, massive_seq_len, d_model).astype(np.float32)
        V = np.random.randn(batch, massive_seq_len, d_model).astype(np.float32)
        O = engine.forward(Q, K, V)
        print("Success! Processed massive sequence without OOM.")
    except Exception as e:
        print(f"FAILED: Engine threw exception on massive sequence: {e}")

if __name__ == '__main__':
    test_o1_vram_and_leaks()

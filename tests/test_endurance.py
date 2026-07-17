import pytest
import numpy as np
import os
import psutil
import threading
import time

import subq_engine

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_million_token_stream():
    """
    Simulate a continuous 24-hour style firehose by chunking 1,000,000 tokens
    through the engine and asserting O(1) memory boundaries and NaN decay bounds.
    """
    # 256MB boundary
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)
    batch, d_model = 1, 256
    
    start_mem = get_memory_mb()
    chunk_size = 4096
    total_tokens = 1_000_000
    
    # Send 1M tokens through the engine
    for t in range(0, total_tokens, chunk_size):
        Q = np.random.randn(batch, chunk_size, d_model).astype(np.float32)
        K = np.random.randn(batch, chunk_size, d_model).astype(np.float32)
        V = np.random.randn(batch, chunk_size, d_model).astype(np.float32)
        
        out = engine.forward(Q, K, V)
    
    # 1. Assert memory leak prevention (O(1) bounds)
    end_mem = get_memory_mb()
    # Python overhead might allocate a bit, but we should not see massive leaks
    assert end_mem - start_mem < 50.0, f"Memory leak detected! Grew from {start_mem:.2f} to {end_mem:.2f} MB"
    
    # 2. Assert Signal-To-Noise Ratio (SNR) degradation limits
    # The last chunk should NOT mathematically degrade to NaNs or subnormal float death
    assert not np.isnan(out).any(), "Temporal decay cascaded into NaNs over 1M tokens."
    assert not np.isinf(out).any(), "Temporal decay cascaded into Inf over 1M tokens."


def test_hot_swapping_thrash(tmp_path):
    """
    Test the C++ thread-safety via the newly injected `std::mutex`.
    Multiple threads attempting to load a new .safetensors model while
    another thread is actively streaming audio must not segfault the arena.
    """
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)
    batch, d_model = 1, 256
    
    dummy_file = tmp_path / "dummy.safetensors"
    # Create a small valid safetensors file
    import json, struct
    header = {"__metadata__": {"format": "pt"}}
    header_json = json.dumps(header).encode('utf-8')
    with open(dummy_file, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)

    success = [True]
    
    def thrash_arena():
        try:
            for _ in range(50):
                time.sleep(0.001)
                engine.load(str(dummy_file)) 
        except RuntimeError as e:
            # We expect a C++ exception if the file is invalid or missing tensors, 
            # but we DO NOT expect a segmentation fault crashing the host.
            pass
        except Exception as e:
            print(f"Thrasher crashed: {e}")
            success[0] = False
            
    def stream_audio():
        try:
            for _ in range(20):
                seq_len = 4096
                Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
                K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
                V = np.random.randn(batch, seq_len, d_model).astype(np.float32)
                # Engine will queue via mutex
                _ = engine.forward(Q, K, V)
        except Exception as e:
            print(f"Streamer crashed: {e}")
            success[0] = False

    threads = []
    # 3 Thrasher threads, 2 Audio Stream threads
    for _ in range(3):
        t = threading.Thread(target=thrash_arena)
        threads.append(t)
        t.start()
    for _ in range(2):
        t = threading.Thread(target=stream_audio)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()

    # If it survived without segfaulting, it's a pass
    assert success[0] == True, "Race condition detected! Mutex failed or Segfault occurred."

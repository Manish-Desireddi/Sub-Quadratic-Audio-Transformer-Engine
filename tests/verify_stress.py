import numpy as np
import time
import os
import psutil
import threading
import torch

try:
    import subq_engine
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import build.Release.subq_engine as subq_engine

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_dead_air():
    print("=== 1. Dead Air Underflow Test ===")
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)
    batch, d_model = 1, 256
    
    # Send 100,000 zeros to fill the state matrix with dead air
    Q = np.zeros((batch, 100000, d_model), dtype=np.float32)
    K = np.zeros((batch, 100000, d_model), dtype=np.float32)
    V = np.zeros((batch, 100000, d_model), dtype=np.float32)
    _ = engine.forward(Q, K, V)
    
    # Measure TTFC for a standard 1024 chunk post-dead-air
    Q_chunk = np.zeros((batch, 1024, d_model), dtype=np.float32)
    K_chunk = np.zeros((batch, 1024, d_model), dtype=np.float32)
    V_chunk = np.zeros((batch, 1024, d_model), dtype=np.float32)
    
    start = time.perf_counter()
    _ = engine.forward(Q_chunk, K_chunk, V_chunk)
    ttfc = (time.perf_counter() - start) * 1000
    
    print(f"TTFC post-dead-air: {ttfc:.2f} ms")
    if ttfc > 100:
        print("Verdict: FAIL - Subnormal numbers stalled the Tensor Cores.")
    else:
        print("Verdict: PASS - Engine maintained performance through dead air.")

def test_firehose():
    print("\n=== 2. Asynchronous Firehose (Concurrency Stress) ===")
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)
    batch, d_model = 1, 256
    
    success = [True]
    def feed_audio(thread_id):
        try:
            for _ in range(5):
                seq_len = np.random.randint(128, 4096)
                Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
                K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
                V = np.random.randn(batch, seq_len, d_model).astype(np.float32)
                _ = engine.forward(Q, K, V)
                time.sleep(np.random.uniform(0.002, 0.085)) # Jitter
        except Exception as e:
            print(f"Thread {thread_id} crashed: {e}")
            success[0] = False
            
    threads = []
    for i in range(5):
        t = threading.Thread(target=feed_audio, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    if success[0]:
        print("Verdict: PASS - Engine survived 5 asynchronous firehose streams.")
    else:
        print("Verdict: FAIL - Race condition detected in MemoryArena.")

def test_soak():
    print("\n=== 3. 24-Hour Soak Test (Compressed 30s Ghost Leak Hunt) ===")
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)
    batch, d_model = 1, 256
    chunk_size = 1024
    
    start_mem = get_memory_mb()
    print(f"Baseline Memory: {start_mem:.2f} MB")
    
    end_time = time.time() + 30
    iterations = 0
    while time.time() < end_time:
        Q = np.random.randn(batch, chunk_size, d_model).astype(np.float32)
        K = np.random.randn(batch, chunk_size, d_model).astype(np.float32)
        V = np.random.randn(batch, chunk_size, d_model).astype(np.float32)
        _ = engine.forward(Q, K, V)
        iterations += 1
        
    end_mem = get_memory_mb()
    print(f"Memory after {iterations} rapid FFI calls: {end_mem:.2f} MB")
    
    if end_mem - start_mem > 10.0:
        print("Verdict: FAIL - Memory leak detected.")
    else:
        print("Verdict: PASS - No silent OOM leaks found.")

def test_arena_thrashing():
    print("\n=== 4. Adversarial Arena Thrashing (Context Switching) ===")
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)
    batch, d_model = 1, 256
    
    # We will test if reset_state() or load() corrupts forward()
    # Wait, reset_state() is not exposed yet. We'll use load() with a dummy path
    # to intentionally trigger a race condition (which will raise an exception inside load, 
    # but the pointer destruction might still crash forward).
    
    success = [True]
    def thrash_arena():
        try:
            for _ in range(10):
                time.sleep(0.01)
                try:
                    # Intentionally load bad file to force pointer resets/errors
                    engine.load("non_existent.safetensors") 
                except:
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
                _ = engine.forward(Q, K, V)
        except Exception as e:
            print(f"Streamer crashed: {e}")
            success[0] = False

    t1 = threading.Thread(target=thrash_arena)
    t2 = threading.Thread(target=stream_audio)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if success[0]:
        print("Verdict: PASS - Engine survived arena thrashing.")
    else:
        print("Verdict: FAIL - Segmentation fault or state corruption occurred.")

if __name__ == '__main__':
    test_dead_air()
    # Note: If these crash the process (segfault), the script will instantly terminate.
    try:
        test_firehose()
    except Exception as e:
        print("Firehose hard-crashed.")
    test_soak()
    test_arena_thrashing()

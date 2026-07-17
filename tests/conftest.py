import pytest
import psutil
import os

@pytest.fixture(autouse=True)
def memory_profiler():
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    yield
    
    mem_after = process.memory_info().rss
    diff_mb = (mem_after - mem_before) / (1024 * 1024)
    
    # The SubQuadratic engine O(1) bound is 256MB. 
    # Python overhead and fragmentation can add some padding, but a jump of > 300MB indicates a leak.
    if diff_mb > 300.0:
        pytest.fail(f"Memory leak detected! Memory grew by {diff_mb:.2f} MB, exceeding O(1) bound limit of 256MB.")

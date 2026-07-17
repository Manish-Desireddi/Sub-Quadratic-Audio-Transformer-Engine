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

@pytest.fixture(scope="session")
def real_audio():
    import numpy as np
    from scipy.io import wavfile
    
    wav_path = os.path.join(os.path.dirname(__file__), "real_human_speech.wav")
    sample_rate, audio_data = wavfile.read(wav_path)
    
    # Normalize to [-1.0, 1.0] FP32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        
    def _get_audio_tensors(batch: int, seq_len: int, d_model: int):
        total_elements = batch * seq_len * d_model
        
        # Tile audio if necessary to fill requested size
        repeats = int(np.ceil(total_elements / len(audio_data)))
        tiled_audio = np.tile(audio_data, repeats)[:total_elements]
        
        # Reshape to (batch, seq_len, d_model)
        # Using the same data for Q, K, V represents the hardest case (maximal correlation)
        tensor = tiled_audio.reshape(batch, seq_len, d_model)
        return tensor.copy(), tensor.copy(), tensor.copy()
        
    return _get_audio_tensors

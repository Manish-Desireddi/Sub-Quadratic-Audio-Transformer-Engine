# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import sys
import os
import torch

# Add the current directory to sys.path so we can import the generated .pyd
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import subq_engine
except ImportError as e:
    print(f"Failed to import subq_engine: {e}")
    print(
        "Ensure you have built the CMake project and the .pyd file is in the python/ directory."
    )
    sys.exit(1)


class AudioFrontend(torch.nn.Module):
    """
    Real-world audio frontend extractor.
    Projects raw audio waveforms into the continuous d_model feature space (Q, K, V)
    using 1D Convolutions with a stride representing downsampling.
    """
    def __init__(self, d_model):
        super().__init__()
        # A standard strided 1D convolution to encode raw PCM into d_model dimensional vectors
        self.conv = torch.nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=16, stride=4, padding=6)
        # Linear projections for Q, K, V
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (batch, seq_len) raw audio samples
        x = x.unsqueeze(1) # (batch, 1, seq_len)
        features = self.conv(x) # (batch, d_model, downsampled_seq_len)
        features = features.transpose(1, 2) # (batch, seq_len, d_model)
        
        Q = self.q_proj(features)
        K = self.k_proj(features)
        V = self.v_proj(features)
        return Q, K, V


def main():
    print("Initializing Sub-Quadratic Engine (End-to-End Test)...")
    # Initialize with 256MB VRAM Arena
    engine = subq_engine.SubQEngine(256 * 1024 * 1024)

    batch = 2
    seq_len = 1024
    d_model = 256

    print(
        f"Initializing AudioFrontend (Conv1D) to process raw audio..."
    )
    frontend = AudioFrontend(d_model)
    
    # Generate realistic 1D continuous audio waveform (batch, raw_audio_samples)
    raw_audio_len = seq_len * 4 # Because stride=4
    raw_audio = torch.randn(batch, raw_audio_len, dtype=torch.float32)
    
    with torch.no_grad():
        Q, K, V = frontend(raw_audio)

    print("Running GPU forward pass via PyBind11 and CUDA MemoryArena...")

    # We must pass the memory contiguous numpy arrays to our C++ backend
    O_np = engine.forward(Q.numpy(), K.numpy(), V.numpy())

    # Convert the resulting C++ NumPy pointer back into a native PyTorch tensor
    O = torch.from_numpy(O_np)

    print(f"Forward pass complete! Output tensor shape: {O.shape}")
    print("Status: End-to-End architecture verified!")

    # Calculate some basic evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"- Mean activation: {O.mean().item():.6f}")
    print(f"- Std deviation: {O.std().item():.6f}")
    print(f"- Max activation: {O.max().item():.6f}")

    if torch.isnan(O).any():
        print("ERROR: NaN values detected in output tensor!")
        sys.exit(1)


if __name__ == "__main__":
    main()

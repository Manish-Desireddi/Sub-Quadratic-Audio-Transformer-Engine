# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import os
import sys
import numpy as np

# Add the parent directory to sys.path to find the compiled subq_engine.pyd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subq_engine


class Engine:
    def __init__(self, vram_capacity=1024 * 1024 * 256):
        self._engine = subq_engine.SubQEngine(vram_capacity)

    def load(self, filepath: str):
        """Loads a model from a safetensors file via zero-copy mmap."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self._engine.load(filepath)

    def generate(self, Q, K, V):
        """
        Executes a forward pass using the Sub-Quadratic Audio Transformer Engine.
        Inputs can be numpy arrays or PyTorch tensors.
        """
        # Ensure contiguous numpy arrays in C order with float32 type
        if hasattr(Q, "detach"):
            Q = Q.detach().cpu().numpy()
        if hasattr(K, "detach"):
            K = K.detach().cpu().numpy()
        if hasattr(V, "detach"):
            V = V.detach().cpu().numpy()

        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)

        return self._engine.forward(Q, K, V)


# Expose a default global instance for simple API: `subq_audio.load(...)`
_default_engine = None


def _get_engine():
    global _default_engine
    if _default_engine is None:
        _default_engine = Engine()
    return _default_engine


def load(filepath: str):
    _get_engine().load(filepath)


def generate(Q, K, V):
    return _get_engine().generate(Q, K, V)

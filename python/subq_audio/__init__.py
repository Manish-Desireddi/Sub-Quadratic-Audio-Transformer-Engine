"""
Sub-Quadratic Audio Transformer Engine
A Bare-Metal, Lock-Free O(N) Linear Attention Framework for Streaming Audio.
"""

import sys
from typing import List, Optional

from . import _C
from ._C import SubQEngine, get_device_info
from .engine import (
    AudioTransformerEngine,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    create_distributed_dataloader,
)
from .autograd import SubQAttentionFunction, SubQuadraticAttention, SubQLinearAttention
from .surgery import (
    bypass_transformer_with_subq,
    LayerWiseDistillationTrainer,
    SubQBypassAttention,
    SubQAttentionOutput,
)

# Compatibility alias for legacy scripts importing subq_engine
sys.modules.setdefault("subq_engine", _C)

__version__ = "0.1.0"

# Compatibility layer
Engine = AudioTransformerEngine
_default_engine: Optional[AudioTransformerEngine] = None


def _get_default_engine() -> AudioTransformerEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = AudioTransformerEngine()
    return _default_engine


def load(filepath: str) -> None:
    """Loads weights into global default engine."""
    _get_default_engine().load_weights(filepath)


def generate(Q, K, V, decay_factor: float = 0.99):
    """Generates attention output using default engine."""
    return _get_default_engine().forward(Q, K, V, decay_factor)


__all__: List[str] = [
    "_C",
    "SubQEngine",
    "get_device_info",
    "AudioTransformerEngine",
    "Engine",
    "SubQuadraticAttention",
    "SubQLinearAttention",
    "SubQAttentionFunction",
    "bypass_transformer_with_subq",
    "LayerWiseDistillationTrainer",
    "SubQBypassAttention",
    "SubQAttentionOutput",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_ddp",
    "create_distributed_dataloader",
    "load",
    "generate",
    "__version__",
]

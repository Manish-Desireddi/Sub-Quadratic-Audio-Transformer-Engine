"""
Neural network modules and layers for Sub-Quadratic Attention.
"""

from .autograd import SubQLinearAttention, SubQAttentionFunction

# Alias for backwards compatibility
SubQuadraticAttention = SubQLinearAttention

__all__ = [
    "SubQLinearAttention",
    "SubQuadraticAttention",
    "SubQAttentionFunction",
]

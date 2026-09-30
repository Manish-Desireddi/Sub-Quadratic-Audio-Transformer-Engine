"""
Test suite for Python package structure, exports, PEP 561 typing, and device telemetry.
"""

import os
import pytest
import subq_audio
from subq_audio import (
    AudioTransformerEngine,
    SubQuadraticAttention,
    SubQAttentionFunction,
    SubQEngine,
    get_device_info,
    _C,
)


def test_package_exports():
    """Verify all top-level public API exports are accessible."""
    assert hasattr(subq_audio, "AudioTransformerEngine")
    assert hasattr(subq_audio, "SubQuadraticAttention")
    assert hasattr(subq_audio, "SubQAttentionFunction")
    assert hasattr(subq_audio, "SubQEngine")
    assert hasattr(subq_audio, "get_device_info")
    assert hasattr(subq_audio, "_C")
    assert hasattr(_C, "SubQEngine")
    assert hasattr(_C, "get_device_info")


def test_pep561_files_exist():
    """Verify PEP 561 marker and type stub files exist."""
    pkg_dir = os.path.dirname(subq_audio.__file__)
    py_typed = os.path.join(pkg_dir, "py.typed")
    c_pyi = os.path.join(pkg_dir, "_C.pyi")

    assert os.path.exists(py_typed), f"py.typed marker not found at {py_typed}"
    assert os.path.exists(c_pyi), f"_C.pyi type stub not found at {c_pyi}"
    assert os.path.getsize(c_pyi) > 0, "_C.pyi type stub must not be empty"


def test_device_info_structure():
    """Verify get_device_info returns expected hardware metadata."""
    info = get_device_info()
    assert isinstance(info, dict)
    assert "backend" in info
    assert "architecture" in info
    assert "memory_size" in info
    assert "bfloat16_supported" in info
    assert info["backend"] in ["CPU", "CUDA", "HIP"]
    assert isinstance(info["bfloat16_supported"], bool)


def test_engine_initialization():
    """Verify C++ engine and Python wrapper initialization."""
    native_engine = SubQEngine(vram_capacity=64 * 1024 * 1024)
    assert native_engine is not None

    wrapper_engine = AudioTransformerEngine(vram_capacity=64 * 1024 * 1024)
    assert wrapper_engine is not None
    assert wrapper_engine.device_info == get_device_info()

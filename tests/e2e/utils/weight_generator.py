"""
Synthetic Safetensors Weight Generator for E2E Tests.

Generates standard and corrupted .safetensors model weight files containing
linear attention projections (q_proj, k_proj, v_proj, out_proj, decay_weight)
for model loading, precision verification, and negative security testing.
"""

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class WeightGenerator:
    """Generates valid and malformed .safetensors model weight files."""

    @staticmethod
    def create_model_weights(
        filepath: Union[str, Path],
        d_model: int = 256,
        precision: str = "float32",
        include_mlp: bool = False,
        seed: Optional[int] = 42,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Generates a valid .safetensors file with linear attention weights.
        
        Args:
            filepath: Destination file path.
            d_model: Hidden model dimension.
            precision: "float32", "float16", or "bfloat16".
            include_mlp: If True, includes LayerNorm and 2-layer FFN projection weights.
            seed: Random seed for deterministic weights.
            metadata: Optional dictionary of string key-values for __metadata__.
            
        Returns:
            Absolute string path to created safetensors file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed if seed is not None else 42)

        if precision == "float16":
            np_dtype = np.float16
            st_dtype = "F16"
        else:
            np_dtype = np.float32
            st_dtype = "F32"

        # Initialize standard linear attention weights
        tensors: Dict[str, np.ndarray] = {
            "q_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.02).astype(np_dtype),
            "k_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.02).astype(np_dtype),
            "v_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.02).astype(np_dtype),
            "out_proj.weight": (rng.standard_normal((d_model, d_model)) * 0.02).astype(np_dtype),
            "decay_weight": np.array([0.5], dtype=np_dtype),
        }

        if include_mlp:
            tensors.update({
                "ln_1.weight": np.ones(d_model, dtype=np_dtype),
                "ln_1.bias": np.zeros(d_model, dtype=np_dtype),
                "mlp.fc1.weight": (rng.standard_normal((d_model * 4, d_model)) * 0.02).astype(np_dtype),
                "mlp.fc1.bias": np.zeros(d_model * 4, dtype=np_dtype),
                "mlp.fc2.weight": (rng.standard_normal((d_model, d_model * 4)) * 0.02).astype(np_dtype),
                "mlp.fc2.bias": np.zeros(d_model, dtype=np_dtype),
            })

        # Build Safetensors header dictionary and binary buffers
        header: Dict[str, Any] = {}
        buffers: List[bytes] = []
        current_offset = 0

        for name, arr in tensors.items():
            arr_bytes = arr.tobytes()
            start_off = current_offset
            end_off = current_offset + len(arr_bytes)
            header[name] = {
                "dtype": st_dtype,
                "shape": list(arr.shape),
                "data_offsets": [start_off, end_off]
            }
            buffers.append(arr_bytes)
            current_offset = end_off

        if metadata:
            header["__metadata__"] = metadata
        else:
            header["__metadata__"] = {"format": "pt", "framework": "subq"}

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = len(header_json)

        with open(filepath, "wb") as f:
            # 8-byte unsigned little-endian header length
            f.write(struct.pack("<Q", header_len))
            f.write(header_json)
            for buf in buffers:
                f.write(buf)

        return str(filepath.resolve())

    @staticmethod
    def create_corrupt_weights(
        filepath: Union[str, Path],
        corruption_type: str = "header_size_overflow"
    ) -> str:
        """Generates an intentionally malformed .safetensors file for negative testing.
        
        Args:
            filepath: Destination file path.
            corruption_type:
                - "header_size_overflow": Declares 999,999,999 byte header size (overflow check).
                - "corrupted_json": Invalid JSON syntax in header string.
                - "inverted_offsets": start_offset > end_offset in tensor metadata.
                - "offset_exceeds_file": end_offset exceeds total physical file size.
                - "dimension_overflow": Shape has > 4 dimensions (e.g. 5D tensor).
                - "zero_bytes": 0-byte empty file.
                - "incomplete_header_length": Only 4 bytes in file instead of 8.
                
        Returns:
            Absolute string path to corrupted safetensors file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            if corruption_type == "header_size_overflow":
                # Write massive uint64 header size
                f.write(struct.pack("<Q", 999_999_999))
                f.write(b'{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}')
                f.write(b"\x00" * 16)
            elif corruption_type == "corrupted_json":
                bad_json = b"{ NOT_VALID_JSON_SYNTAX :::: 123"
                f.write(struct.pack("<Q", len(bad_json)))
                f.write(bad_json)
                f.write(b"\x00" * 32)
            elif corruption_type == "inverted_offsets":
                header = {
                    "test": {
                        "dtype": "F32",
                        "shape": [4, 4],
                        "data_offsets": [1000, 500]  # Inverted!
                    }
                }
                h_bytes = json.dumps(header).encode("utf-8")
                f.write(struct.pack("<Q", len(h_bytes)))
                f.write(h_bytes)
                f.write(b"\x00" * 2000)
            elif corruption_type == "offset_exceeds_file":
                header = {
                    "test": {
                        "dtype": "F32",
                        "shape": [4, 4],
                        "data_offsets": [0, 50_000_000]  # Far beyond file size
                    }
                }
                h_bytes = json.dumps(header).encode("utf-8")
                f.write(struct.pack("<Q", len(h_bytes)))
                f.write(h_bytes)
                f.write(b"\x00" * 64)
            elif corruption_type == "dimension_overflow":
                header = {
                    "test": {
                        "dtype": "F32",
                        "shape": [2, 3, 4, 5, 6],  # 5 dimensions (> 4 allowed)
                        "data_offsets": [0, 2880]
                    }
                }
                h_bytes = json.dumps(header).encode("utf-8")
                f.write(struct.pack("<Q", len(h_bytes)))
                f.write(h_bytes)
                f.write(b"\x00" * 2880)
            elif corruption_type == "zero_bytes":
                pass
            elif corruption_type == "incomplete_header_length":
                f.write(b"\x01\x02\x03\x04")  # 4 bytes only
            else:
                raise ValueError(f"Unknown corruption_type: {corruption_type}")

        return str(filepath.resolve())

    @staticmethod
    def inspect_safetensors_header(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Inspects and parses the JSON header from a .safetensors file.
        
        Args:
            filepath: Path to safetensors file.
            
        Returns:
            Dictionary containing deserialized header metadata.
        """
        with open(filepath, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError("File is smaller than 8 bytes; invalid Safetensors file.")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_json_bytes = f.read(header_len)
            if len(header_json_bytes) < header_len:
                raise ValueError("Safetensors file truncated before end of header.")
            return json.loads(header_json_bytes.decode("utf-8"))

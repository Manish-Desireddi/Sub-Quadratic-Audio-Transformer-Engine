# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import os
import sys
import json
import struct
import numpy as np

# Add the python directory to sys.path so we can import subq_audio
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import subq_audio


def create_mock_safetensors(filepath):
    # Dummy tensor shape and offsets
    metadata = {
        "__metadata__": {"format": "pt"},
        "layer_1.weight": {
            "dtype": "F32",
            "shape": [256, 256],
            "data_offsets": [0, 256 * 256 * 4],
        },
    }

    json_str = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

    # Pad JSON to 8-byte alignment (safetensors format requirement)
    padding = b" " * ((8 - len(json_str) % 8) % 8)
    json_str += padding

    header_size = struct.pack("<Q", len(json_str))

    # Dummy data
    tensor_data = np.ones((256, 256), dtype=np.float32).tobytes()

    with open(filepath, "wb") as f:
        f.write(header_size)
        f.write(json_str)
        f.write(tensor_data)

    print(f"Created mock safetensors file: {filepath}")


def main():
    mock_file = "mock_model.safetensors"
    create_mock_safetensors(mock_file)

    try:
        print("Testing subq_audio.load()...")
        subq_audio.load(mock_file)
        print("Load successful!")

        print("Testing subq_audio.generate()...")
        batch, seq_len, d_model = 1, 128, 256
        Q = np.random.rand(batch, seq_len, d_model).astype(np.float32)
        K = np.random.rand(batch, seq_len, d_model).astype(np.float32)
        V = np.random.rand(batch, seq_len, d_model).astype(np.float32)

        O = subq_audio.generate(Q, K, V)
        print(f"Generate successful! Output shape: {O.shape}")

    except Exception as e:
        print(f"API Test failed: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(mock_file):
            os.remove(mock_file)


if __name__ == "__main__":
    main()

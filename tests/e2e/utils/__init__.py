"""
E2E Test Infrastructure Utilities.

Provides audio waveform generation, mock WebSocket streaming servers,
synthetic safetensors generation, and telemetry schema validation.
"""

from tests.e2e.utils.audio_generator import AudioGenerator
from tests.e2e.utils.mock_ws_server import (
    MockWebSocketServer,
    pack_network_audio_header,
    unpack_network_audio_header,
    SUBQ_MAGIC_BYTES,
    HEADER_BYTE_SIZE,
    FORMAT_FLOAT32,
    FORMAT_INT16,
)
from tests.e2e.utils.weight_generator import WeightGenerator
from tests.e2e.utils.telemetry_parser import TelemetryParser, TelemetryValidationError

__all__ = [
    "AudioGenerator",
    "MockWebSocketServer",
    "pack_network_audio_header",
    "unpack_network_audio_header",
    "SUBQ_MAGIC_BYTES",
    "HEADER_BYTE_SIZE",
    "FORMAT_FLOAT32",
    "FORMAT_INT16",
    "WeightGenerator",
    "TelemetryParser",
    "TelemetryValidationError",
]

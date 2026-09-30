"""
Mock Asynchronous WebSocket Streaming Server for Live Audio Ingestion Testing.

Encodes and streams chunked binary audio frames prepended with the 32-byte
Sub-Quadratic binary header (NetworkAudioHeader, magic=0x53554251). Supports dynamic
port binding, realtime pacing, raw text/corrupt frame injection, and connection metrics.
"""

import asyncio
import os
import struct
import threading
import time
from typing import Any, Optional, Set, Tuple, Union
import numpy as np

try:
    import websockets
    # Support websockets v10+ and legacy signatures
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


# Binary Protocol Constants
SUBQ_MAGIC_BYTES = 0x53554251  # Little-endian ASCII "SUBQ"
HEADER_STRUCT_FORMAT = "<IIIQIHHI"
HEADER_BYTE_SIZE = 32

FORMAT_FLOAT32 = 0
FORMAT_INT16 = 1


def pack_network_audio_header(
    stream_id: int = 1,
    seq_no: int = 1,
    timestamp_us: int = 0,
    sample_rate: int = 24000,
    channels: int = 1,
    audio_format: int = FORMAT_FLOAT32,
    payload_bytes: int = 0,
    magic: int = SUBQ_MAGIC_BYTES
) -> bytes:
    """Packs fields into a 32-byte binary NetworkAudioHeader.
    
    Args:
        stream_id: Identifier for audio stream.
        seq_no: Monotonically increasing sequence number.
        timestamp_us: Microsecond timestamp.
        sample_rate: Audio sampling frequency in Hz.
        channels: Channel count (1=mono, 2=stereo).
        audio_format: 0 for Float32, 1 for Int16 PCM.
        payload_bytes: Byte count of subsequent PCM audio payload.
        magic: Protocol magic identifier (default 0x53554251).
        
    Returns:
        32-byte packed binary header.
    """
    return struct.pack(
        HEADER_STRUCT_FORMAT,
        magic,
        stream_id,
        seq_no,
        timestamp_us,
        sample_rate,
        channels,
        audio_format,
        payload_bytes
    )


def unpack_network_audio_header(data: bytes) -> dict:
    """Unpacks a 32-byte binary NetworkAudioHeader into a dictionary.
    
    Args:
        data: At least 32 bytes of raw binary header.
        
    Returns:
        Dictionary containing unpacked header fields.
    """
    if len(data) < HEADER_BYTE_SIZE:
        raise ValueError(f"Data length {len(data)} is shorter than header size {HEADER_BYTE_SIZE}")
    magic, stream_id, seq_no, ts, sr, ch, fmt, payload_len = struct.unpack(
        HEADER_STRUCT_FORMAT, data[:HEADER_BYTE_SIZE]
    )
    return {
        "magic": magic,
        "magic_hex": hex(magic),
        "is_valid_magic": magic == SUBQ_MAGIC_BYTES,
        "stream_id": stream_id,
        "seq_no": seq_no,
        "timestamp_us": ts,
        "sample_rate": sr,
        "channels": ch,
        "format": fmt,
        "payload_bytes": payload_len,
    }


class MockWebSocketServer:
    """Asynchronous WebSocket Audio Streaming Server running in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, endpoint: str = "/audio"):
        """Initializes the mock WebSocket server.
        
        Args:
            host: Bind host address (default 127.0.0.1).
            port: Bind port (0 for dynamic ephemeral port selection).
            endpoint: URL route endpoint (default "/audio").
        """
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.server = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.connected_clients: Set[Any] = set()
        
        # Telemetry & Diagnostics
        self.sent_chunks_count: int = 0
        self.total_bytes_sent: int = 0
        self.client_connections_count: int = 0
        
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

    async def _handler(self, websocket, *args):
        """Internal connection handler managing client lifecycle."""
        self.connected_clients.add(websocket)
        self.client_connections_count += 1
        try:
            while not self._stop_event.is_set():
                try:
                    # Keep connection alive; process any incoming ping or control message
                    await asyncio.wait_for(websocket.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
        finally:
            self.connected_clients.discard(websocket)

    def start(self, timeout_sec: float = 5.0) -> str:
        """Starts the WebSocket server in a background daemon thread.
        
        Args:
            timeout_sec: Timeout in seconds to wait for port binding.
            
        Returns:
            The complete ws:// URL of the listening server.
        """
        if not HAS_WEBSOCKETS:
            raise RuntimeError("The 'websockets' Python package is required to run MockWebSocketServer.")

        def _run_event_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def _start():
                try:
                    # Check for websockets serve API compatibility
                    if hasattr(websockets, "serve"):
                        self.server = await websockets.serve(self._handler, self.host, self.port)
                    elif hasattr(websockets, "asyncio") and hasattr(websockets.asyncio.server, "serve"):
                        self.server = await websockets.asyncio.server.serve(self._handler, self.host, self.port)
                    else:
                        raise RuntimeError("Unsupported websockets library version.")

                    # Extract actual bound port
                    sockets = self.server.sockets
                    if sockets:
                        self.port = sockets[0].getsockname()[1]
                    self._ready_event.set()
                except Exception as e:
                    self._ready_event.set()
                    raise e

            self.loop.run_until_complete(_start())
            try:
                self.loop.run_forever()
            finally:
                # Cleanup pending tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.loop.close()

        self.thread = threading.Thread(target=_run_event_loop, daemon=True)
        self.thread.start()

        if not self._ready_event.wait(timeout=timeout_sec):
            raise TimeoutError(f"MockWebSocketServer failed to bind and start within {timeout_sec} seconds.")

        return self.url

    @property
    def url(self) -> str:
        """Returns the WebSocket URL string (e.g. ws://127.0.0.1:8080/audio)."""
        ep = self.endpoint if self.endpoint.startswith("/") else f"/{self.endpoint}"
        return f"ws://{self.host}:{self.port}{ep}"

    def stream_audio(
        self,
        audio_data: np.ndarray,
        chunk_size: int = 2048,
        interval_ms: float = 20.0,
        format_type: str = "float32",
        sample_rate: int = 24000,
        channels: int = 1,
        stream_id: int = 1
    ):
        """Streams chunked binary audio with NetworkAudioHeader framing to all clients.
        
        Args:
            audio_data: 1D float32 audio samples array.
            chunk_size: Number of samples per chunk frame.
            interval_ms: Pacing interval between chunk transmissions in milliseconds.
            format_type: "float32" or "int16".
            sample_rate: Sampling frequency in Hz.
            channels: Channel count.
            stream_id: Stream identifier integer.
        """
        async def _stream():
            total_samples = len(audio_data)
            seq_no = 1
            fmt_code = FORMAT_INT16 if format_type == "int16" else FORMAT_FLOAT32
            start_ts = int(time.time() * 1_000_000)

            for i in range(0, total_samples, chunk_size):
                if self._stop_event.is_set():
                    break

                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                if fmt_code == FORMAT_INT16:
                    pcm_bytes = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                else:
                    pcm_bytes = chunk.astype(np.float32).tobytes()

                ts_us = start_ts + int((i / sample_rate) * 1_000_000)
                header_bytes = pack_network_audio_header(
                    stream_id=stream_id,
                    seq_no=seq_no,
                    timestamp_us=ts_us,
                    sample_rate=sample_rate,
                    channels=channels,
                    audio_format=fmt_code,
                    payload_bytes=len(pcm_bytes)
                )

                packet = header_bytes + pcm_bytes
                for client in list(self.connected_clients):
                    try:
                        await client.send(packet)
                    except Exception:
                        pass

                self.sent_chunks_count += 1
                self.total_bytes_sent += len(packet)
                seq_no += 1

                if interval_ms > 0:
                    await asyncio.sleep(interval_ms / 1000.0)

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(_stream(), self.loop)

    def send_raw_frame(self, data: bytes):
        """Sends an arbitrary raw binary payload directly to all connected clients."""
        async def _send():
            for client in list(self.connected_clients):
                try:
                    await client.send(data)
                except Exception:
                    pass
            self.total_bytes_sent += len(data)

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), self.loop)

    def send_text_frame(self, text: str):
        """Sends a text frame (e.g. JSON string) for protocol mismatch testing."""
        async def _send():
            for client in list(self.connected_clients):
                try:
                    await client.send(text)
                except Exception:
                    pass

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), self.loop)

    def send_corrupt_packet(
        self,
        audio_chunk: np.ndarray,
        corruption_type: str = "invalid_magic",
        sample_rate: int = 24000
    ):
        """Injects a malformed audio packet for negative resilience testing."""
        pcm_bytes = audio_chunk.astype(np.float32).tobytes()
        if corruption_type == "invalid_magic":
            header = pack_network_audio_header(
                magic=0xDEADBEEF,
                sample_rate=sample_rate,
                payload_bytes=len(pcm_bytes)
            )
            self.send_raw_frame(header + pcm_bytes)
        elif corruption_type == "truncated_payload":
            header = pack_network_audio_header(
                sample_rate=sample_rate,
                payload_bytes=len(pcm_bytes) * 2  # Declares double the actual payload
            )
            self.send_raw_frame(header + pcm_bytes)
        elif corruption_type == "zero_payload":
            header = pack_network_audio_header(
                sample_rate=sample_rate,
                payload_bytes=0
            )
            self.send_raw_frame(header)

    def stop(self):
        """Gracefully stops the WebSocket server and shuts down background loop."""
        self._stop_event.set()
        if self.server:
            if hasattr(self.server, "close"):
                self.server.close()
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

"""
Pytest Master Conftest for E2E Test Suite.

Provides session and function-scoped fixtures for binary discovery,
CLI subprocess execution, ephemeral WebSocket streaming, procedural audio/weights
factories, RSS memory leak detection (< 300MB bound), and telemetry validation.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pytest

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from tests.e2e.utils.audio_generator import AudioGenerator
from tests.e2e.utils.mock_ws_server import MockWebSocketServer
from tests.e2e.utils.weight_generator import WeightGenerator
from tests.e2e.utils.telemetry_parser import TelemetryParser


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the absolute Path to the repository project root."""
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def subq_cli_path(project_root: Path) -> Path:
    """Locates the compiled standalone subq_cli executable binary."""
    is_windows = os.name == "nt"
    exe_name = "subq_cli.exe" if is_windows else "subq_cli"

    candidates = [
        project_root / "build_cli" / exe_name,
        project_root / "build_cli" / "Release" / exe_name,
        project_root / "build" / exe_name,
        project_root / "build" / "Release" / exe_name,
        project_root / "build" / "Debug" / exe_name,
        project_root / "build_linux" / exe_name,
        project_root / "bin" / exe_name,
    ]

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand

    # Check system PATH
    resolved = shutil.which(exe_name) or shutil.which("subq_cli")
    if resolved:
        return Path(resolved)

    pytest.skip(
        f"subq_cli binary not found in candidate paths: {[str(c) for c in candidates]}. "
        "Compile C++ CLI target (Milestone M3) to execute CLI subprocess tests."
    )


@pytest.fixture
def cli_runner(subq_cli_path: Path):
    """Factory fixture to execute subq_cli subprocess with timeout and I/O capture."""
    def _run(
        args: List[Union[str, Path]],
        stdin_data: Optional[bytes] = None,
        timeout_sec: float = 30.0,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        cmd = [str(subq_cli_path)] + [str(a) for a in args]
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        start_time = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if stdin_data is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                cwd=str(cwd) if cwd else None
            )
            stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout_sec)
            duration = time.time() - start_time
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "stdout_raw": stdout,
                "stderr_raw": stderr,
                "duration_sec": duration,
                "command": cmd,
            }
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise TimeoutError(f"Command '{' '.join(cmd)}' timed out after {timeout_sec} seconds.")

    return _run


@pytest.fixture
def ws_mock_server() -> Generator[MockWebSocketServer, None, None]:
    """Spins up an ephemeral MockWebSocketServer and cleans up upon fixture teardown."""
    server = MockWebSocketServer(host="127.0.0.1", port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def audio_factory(tmp_path: Path):
    """Factory providing temporary WAV, PCM, and corrupted audio fixtures."""
    class AudioFixtureFactory:
        def make_sine_wav(
            self,
            filename: str = "sine_440.wav",
            duration: float = 1.0,
            sample_rate: int = 24000,
            dtype: str = "float32"
        ) -> Tuple[Path, np.ndarray]:
            p = tmp_path / filename
            audio = AudioGenerator.generate_sine_wave(sample_rate, duration)
            AudioGenerator.save_wav(str(p), audio, sample_rate, dtype)
            return p, audio

        def make_chirp_wav(
            self,
            filename: str = "chirp.wav",
            duration: float = 1.0,
            sample_rate: int = 24000,
            dtype: str = "float32"
        ) -> Tuple[Path, np.ndarray]:
            p = tmp_path / filename
            audio = AudioGenerator.generate_chirp(sample_rate, duration)
            AudioGenerator.save_wav(str(p), audio, sample_rate, dtype)
            return p, audio

        def make_noise_wav(
            self,
            filename: str = "noise.wav",
            duration: float = 1.0,
            sample_rate: int = 24000,
            dtype: str = "float32"
        ) -> Tuple[Path, np.ndarray]:
            p = tmp_path / filename
            audio = AudioGenerator.generate_white_noise(sample_rate, duration)
            AudioGenerator.save_wav(str(p), audio, sample_rate, dtype)
            return p, audio

        def make_stereo_wav(
            self,
            filename: str = "stereo.wav",
            duration: float = 1.0,
            sample_rate: int = 24000,
            dtype: str = "float32"
        ) -> Tuple[Path, np.ndarray]:
            p = tmp_path / filename
            left = AudioGenerator.generate_sine_wave(sample_rate, duration, freq=440.0)
            right = AudioGenerator.generate_sine_wave(sample_rate, duration, freq=880.0)
            stereo = AudioGenerator.generate_stereo(left, right)
            AudioGenerator.save_wav(str(p), stereo, sample_rate, dtype)
            return p, stereo

        def make_raw_pcm(
            self,
            filename: str = "audio.pcm",
            duration: float = 1.0,
            sample_rate: int = 24000,
            dtype: str = "float32"
        ) -> Tuple[Path, np.ndarray]:
            p = tmp_path / filename
            audio = AudioGenerator.generate_sine_wave(sample_rate, duration)
            AudioGenerator.save_raw_pcm(str(p), audio, dtype)
            return p, audio

        def make_corrupt_wav(
            self,
            filename: str = "corrupt.wav",
            corruption_type: str = "truncated_header"
        ) -> Path:
            p = tmp_path / filename
            AudioGenerator.save_corrupt_wav(str(p), corruption_type)
            return p

    return AudioFixtureFactory()


@pytest.fixture
def weights_factory(tmp_path: Path):
    """Factory providing valid and malformed safetensors weight fixtures."""
    class WeightsFixtureFactory:
        def make_weights(
            self,
            filename: str = "model.safetensors",
            d_model: int = 256,
            precision: str = "float32",
            include_mlp: bool = False
        ) -> Path:
            p = tmp_path / filename
            WeightGenerator.create_model_weights(
                str(p), d_model=d_model, precision=precision, include_mlp=include_mlp
            )
            return p

        def make_corrupt_weights(
            self,
            filename: str = "corrupt.safetensors",
            corruption_type: str = "header_size_overflow"
        ) -> Path:
            p = tmp_path / filename
            WeightGenerator.create_corrupt_weights(str(p), corruption_type=corruption_type)
            return p

    return WeightsFixtureFactory()


@pytest.fixture
def telemetry_validator():
    """Provides TelemetryParser validator helper."""
    return TelemetryParser


@pytest.fixture(autouse=True)
def memory_leak_guard():
    """Autouse fixture guarding against process RSS memory leaks (> 300MB bound)."""
    if not HAS_PSUTIL:
        yield
        return

    process = psutil.Process(os.getpid())
    rss_before_bytes = process.memory_info().rss
    yield
    rss_after_bytes = process.memory_info().rss
    diff_mb = (rss_after_bytes - rss_before_bytes) / (1024 * 1024)

    # Sub-Quadratic Engine maintains an O(1) bound (< 256MB arena + Python working set)
    if diff_mb > 300.0:
        pytest.fail(
            f"Process RSS memory leak detected! RSS grew by {diff_mb:.2f} MB "
            f"(start: {rss_before_bytes / 1048576:.2f} MB, end: {rss_after_bytes / 1048576:.2f} MB). "
            "Maximum permitted test leak delta is 300 MB."
        )

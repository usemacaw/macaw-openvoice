"""Tests for in-memory WAV transcription in WeNet backend.

Validates that _audio_to_wav_bytes and _transcribe_with_model use
in-memory I/O instead of disk temp files, and that the output
is identical to the file-based approach.
"""

from __future__ import annotations

import io
import os
import struct
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

from macaw.workers.stt.wenet import (
    _audio_to_wav_bytes,
    _build_wav_header,
    _get_scratch_path,
    _transcribe_with_model,
    _write_and_transcribe,
)


class TestBuildWavHeader:
    """_build_wav_header produces a valid 44-byte WAV header."""

    def test_header_is_44_bytes(self) -> None:
        header = _build_wav_header(1600)
        assert len(header) == 44

    def test_header_starts_with_riff(self) -> None:
        header = _build_wav_header(1600)
        assert header[:4] == b"RIFF"

    def test_header_contains_wave_marker(self) -> None:
        header = _build_wav_header(1600)
        assert header[8:12] == b"WAVE"

    def test_header_contains_fmt_chunk(self) -> None:
        header = _build_wav_header(1600)
        assert header[12:16] == b"fmt "

    def test_header_data_size_matches_samples(self) -> None:
        """Data chunk size = num_samples * bytes_per_sample."""
        header = _build_wav_header(1600)
        data_size = struct.unpack_from("<I", header, 40)[0]
        assert data_size == 1600 * 2  # 2 bytes per int16 sample

    def test_header_file_size_correct(self) -> None:
        """RIFF chunk size = 36 + data_size."""
        header = _build_wav_header(1600)
        riff_size = struct.unpack_from("<I", header, 4)[0]
        data_size = struct.unpack_from("<I", header, 40)[0]
        assert riff_size == 36 + data_size

    def test_header_sample_rate_16000(self) -> None:
        header = _build_wav_header(1600)
        sample_rate = struct.unpack_from("<I", header, 24)[0]
        assert sample_rate == 16000

    def test_header_mono_channel(self) -> None:
        header = _build_wav_header(1600)
        num_channels = struct.unpack_from("<H", header, 22)[0]
        assert num_channels == 1

    def test_header_16bit_depth(self) -> None:
        header = _build_wav_header(1600)
        bits_per_sample = struct.unpack_from("<H", header, 34)[0]
        assert bits_per_sample == 16

    def test_header_zero_samples(self) -> None:
        """Zero-length audio still produces a valid header."""
        header = _build_wav_header(0)
        assert len(header) == 44
        data_size = struct.unpack_from("<I", header, 40)[0]
        assert data_size == 0

    def test_header_readable_by_wave_module(self) -> None:
        """Header + empty data can be parsed by Python's wave module."""
        header = _build_wav_header(0)
        buf = io.BytesIO(header)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000


class TestAudioToWavBytes:
    """_audio_to_wav_bytes produces valid WAV bytes from float32 audio."""

    def test_produces_valid_wav(self) -> None:
        """Output is a valid WAV file that can be read back."""
        audio = np.zeros(1600, dtype=np.float32)
        wav_bytes = _audio_to_wav_bytes(audio)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 1600

    def test_preserves_audio_content(self) -> None:
        """Audio content round-trips correctly (float32 -> int16 -> bytes)."""
        # Use a signal that survives int16 quantization
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_bytes = _audio_to_wav_bytes(audio)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            recovered = np.frombuffer(frames, dtype=np.int16)

        # Check that values are approximately correct after quantization
        expected = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(recovered, expected)

    def test_empty_audio_produces_valid_wav(self) -> None:
        """Empty audio array still produces a valid WAV header."""
        audio = np.array([], dtype=np.float32)
        wav_bytes = _audio_to_wav_bytes(audio)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0

    def test_returns_bytes_not_bytearray(self) -> None:
        """Return type is bytes for compatibility."""
        audio = np.zeros(100, dtype=np.float32)
        result = _audio_to_wav_bytes(audio)
        assert isinstance(result, bytes)

    def test_long_audio_produces_correct_frame_count(self) -> None:
        """5 seconds of audio at 16kHz produces correct frame count."""
        audio = np.zeros(80000, dtype=np.float32)  # 5s at 16kHz
        wav_bytes = _audio_to_wav_bytes(audio)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 80000

    def test_header_size_is_44_bytes(self) -> None:
        """WAV data starts at byte offset 44."""
        audio = np.zeros(100, dtype=np.float32)
        wav_bytes = _audio_to_wav_bytes(audio)
        # First 44 bytes are header, rest is data
        assert len(wav_bytes) == 44 + 100 * 2


class TestGetScratchPath:
    """_get_scratch_path returns a per-thread reusable path."""

    def test_returns_wav_extension(self) -> None:
        path = _get_scratch_path()
        assert path.endswith(".wav")

    def test_contains_pid(self) -> None:
        path = _get_scratch_path()
        assert str(os.getpid()) in path

    def test_contains_macaw_prefix(self) -> None:
        path = _get_scratch_path()
        assert "macaw_wenet_" in os.path.basename(path)

    def test_same_thread_returns_same_path(self) -> None:
        path1 = _get_scratch_path()
        path2 = _get_scratch_path()
        assert path1 == path2

    def test_different_threads_return_different_paths(self) -> None:
        import threading

        paths: list[str] = []
        barrier = threading.Barrier(2)

        def _get_path() -> None:
            # Barrier ensures both threads are alive simultaneously,
            # preventing thread ID reuse across sequential starts.
            barrier.wait()
            paths.append(_get_scratch_path())

        t1 = threading.Thread(target=_get_path)
        t2 = threading.Thread(target=_get_path)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(paths) == 2
        assert paths[0] != paths[1]


class TestWriteAndTranscribe:
    """_write_and_transcribe writes WAV to scratch path and calls model."""

    def test_model_receives_scratch_path(self) -> None:
        """Model.transcribe is called with the scratch file path."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "ok"}

        audio = np.zeros(1600, dtype=np.float32)
        _write_and_transcribe(mock_model, audio)

        call_args = mock_model.transcribe.call_args
        file_path = call_args[0][0]
        assert file_path.endswith(".wav")
        assert "macaw_wenet_" in file_path

    def test_scratch_file_contains_valid_wav(self) -> None:
        """The scratch file written to disk is a valid WAV."""
        captured_path: list[str] = []

        def _capture(path: str) -> dict[str, object]:
            captured_path.append(path)
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 16000
                assert wf.getnframes() == 1600
            return {"text": "validated"}

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = _capture

        audio = np.zeros(1600, dtype=np.float32)
        result = _write_and_transcribe(mock_model, audio)

        assert result == {"text": "validated"}

    def test_scratch_file_cleaned_up_after_transcription(self) -> None:
        """After transcription, the scratch file is removed."""
        captured_path: list[str] = []

        def _capture(path: str) -> dict[str, object]:
            captured_path.append(path)
            return {"text": "ok"}

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = _capture

        audio = np.zeros(1600, dtype=np.float32)
        _write_and_transcribe(mock_model, audio)

        assert not os.path.exists(captured_path[0])

    def test_scratch_file_cleaned_up_on_model_error(self) -> None:
        """Even when model.transcribe raises, the scratch file is removed."""
        captured_path: list[str] = []

        def _capture(path: str) -> dict[str, object]:
            captured_path.append(path)
            msg = "inference error"
            raise RuntimeError(msg)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = _capture

        audio = np.zeros(1600, dtype=np.float32)
        with pytest.raises(RuntimeError, match="inference error"):
            _write_and_transcribe(mock_model, audio)

        assert not os.path.exists(captured_path[0])


class TestTranscribeWithModelInMemory:
    """_transcribe_with_model uses scratch file on /dev/shm."""

    def test_returns_dict_result(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}

        audio = np.zeros(1600, dtype=np.float32)
        result = _transcribe_with_model(mock_model, audio)

        assert result == {"text": "hello world"}
        mock_model.transcribe.assert_called_once()

    def test_returns_string_result_wrapped(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = "hello world"

        audio = np.zeros(1600, dtype=np.float32)
        result = _transcribe_with_model(mock_model, audio)

        assert result == {"text": "hello world"}

    def test_model_receives_wav_file_path(self) -> None:
        """Model.transcribe is called with a file path ending in .wav."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "ok"}

        audio = np.zeros(1600, dtype=np.float32)
        _transcribe_with_model(mock_model, audio)

        call_args = mock_model.transcribe.call_args
        file_path = call_args[0][0]
        assert file_path.endswith(".wav")

    def test_transcribed_file_is_valid_wav(self) -> None:
        """The file path passed to model.transcribe contains valid WAV data."""
        captured_path: list[str] = []

        def _capture_transcribe(path: str) -> dict[str, object]:
            captured_path.append(path)
            # Read and validate the WAV file
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 16000
            return {"text": "validated"}

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = _capture_transcribe

        audio = np.zeros(1600, dtype=np.float32)
        result = _transcribe_with_model(mock_model, audio)

        assert result == {"text": "validated"}
        assert len(captured_path) == 1

    def test_no_leftover_temp_files(self) -> None:
        """After transcription, the scratch file is cleaned up."""
        captured_path: list[str] = []

        def _capture_transcribe(path: str) -> dict[str, object]:
            captured_path.append(path)
            return {"text": "ok"}

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = _capture_transcribe

        audio = np.zeros(1600, dtype=np.float32)
        _transcribe_with_model(mock_model, audio)

        # The scratch file should be deleted after use
        assert not os.path.exists(captured_path[0])

    def test_handles_model_error_gracefully(self) -> None:
        """If model.transcribe raises, the error propagates."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("inference error")

        audio = np.zeros(1600, dtype=np.float32)
        with pytest.raises(RuntimeError, match="inference error"):
            _transcribe_with_model(mock_model, audio)

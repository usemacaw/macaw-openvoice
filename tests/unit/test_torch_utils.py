"""Tests for shared torch_utils functions: resolve_device() and release_gpu_memory().

These are shared utilities used by all STT/TTS backends for device
resolution and GPU memory cleanup.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from macaw.workers.torch_utils import release_gpu_memory, resolve_device


class TestResolveDevice:
    """resolve_device() probes torch.cuda.is_available() for "auto"."""

    def test_cpu_passthrough(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self) -> None:
        assert resolve_device("cuda") == "cuda"

    def test_cuda_with_index_passthrough(self) -> None:
        assert resolve_device("cuda:0") == "cuda:0"

    def test_cuda_with_higher_index(self) -> None:
        assert resolve_device("cuda:3") == "cuda:3"

    def test_auto_returns_cpu_when_torch_not_installed(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            assert resolve_device("auto") == "cpu"

    def test_auto_returns_cpu_when_no_cuda(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "cpu"

    def test_auto_returns_cuda_when_available(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "cuda:0"

    def test_empty_string_passthrough(self) -> None:
        assert resolve_device("") == ""

    def test_arbitrary_string_passthrough(self) -> None:
        assert resolve_device("mps") == "mps"


class TestReleaseGpuMemory:
    """release_gpu_memory() calls torch.cuda.empty_cache() when available."""

    def test_calls_empty_cache_when_cuda_available(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            release_gpu_memory()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_skips_when_no_cuda(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            release_gpu_memory()
        mock_torch.cuda.empty_cache.assert_not_called()

    def test_survives_missing_torch(self) -> None:
        """Does not raise when torch is not installed."""
        import builtins

        original_import = builtins.__import__

        def _import_no_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ImportError("no torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_no_torch):
            release_gpu_memory()

    def test_is_idempotent(self) -> None:
        """Can be called multiple times without error."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            release_gpu_memory()
            release_gpu_memory()
        assert mock_torch.cuda.empty_cache.call_count == 2

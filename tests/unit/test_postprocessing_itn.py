"""Tests for ITNStage (Inverse Text Normalization).

nemo_text_processing is NOT installed in the test environment.
All tests use mocks to simulate the library.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from macaw.postprocessing.itn import ITNStage


class TestITNProcessWithMockNemo:
    def test_itn_process_calls_normalizer_and_returns_result(self) -> None:
        """Mock nemo_text_processing, verify process calls mock and returns result."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "2000"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(default_language="pt")
            result = stage.process("dois mil")

        assert result == "2000"
        mock_normalizer.inverse_normalize.assert_called_once_with("dois mil", verbose=False)


class TestITNFallbackWhenNotInstalled:
    def test_returns_text_unchanged_when_import_fails(self) -> None:
        """When nemo_text_processing is not installed, process returns original text."""
        stage = ITNStage(default_language="pt")

        # nemo_text_processing is not installed in the test environment
        result = stage.process("dois mil")

        assert result == "dois mil"
        assert "pt" in stage._unavailable_languages


class TestITNFallbackOnInitError:
    def test_returns_text_unchanged_when_normalizer_init_raises(self) -> None:
        """When InverseNormalize() raises, process returns original text."""
        mock_module = MagicMock()
        mock_module.InverseNormalize.side_effect = RuntimeError("init failed")

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(default_language="pt")
            result = stage.process("dois mil")

        assert result == "dois mil"
        assert "pt" in stage._unavailable_languages
        assert "pt" not in stage._normalizers


class TestITNFallbackOnProcessError:
    def test_returns_text_unchanged_when_inverse_normalize_raises(self) -> None:
        """When inverse_normalize() raises, returns original text."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.side_effect = RuntimeError("process failed")

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(default_language="pt")
            result = stage.process("dois mil")

        assert result == "dois mil"
        assert "pt" in stage._normalizers


class TestITNEmptyAndWhitespace:
    def test_empty_text_returned_without_calling_nemo(self) -> None:
        """Empty string returned without calling NeMo."""
        mock_normalizer = MagicMock()
        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(default_language="pt")
            result = stage.process("")

        assert result == ""
        mock_module.InverseNormalize.assert_not_called()
        mock_normalizer.inverse_normalize.assert_not_called()

    def test_whitespace_text_returned_without_calling_nemo(self) -> None:
        """Whitespace-only string returned without calling NeMo."""
        mock_normalizer = MagicMock()
        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(default_language="pt")
            result = stage.process("   ")

        assert result == "   "
        mock_module.InverseNormalize.assert_not_called()
        mock_normalizer.inverse_normalize.assert_not_called()


class TestITNLazyLoading:
    def test_first_call_triggers_import_second_call_uses_cache(self) -> None:
        """First call triggers import, second call uses cache (no re-import)."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "2000"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(default_language="pt")

            # First call: triggers _get_normalizer and creates normalizer
            stage.process("dois mil")
            assert mock_module.InverseNormalize.call_count == 1

            # Second call: uses cache, does NOT create normalizer again
            stage.process("tres mil")
            assert mock_module.InverseNormalize.call_count == 1

            # Confirms both calls used the same normalizer
            assert mock_normalizer.inverse_normalize.call_count == 2


class TestITNLanguageParam:
    def test_default_language_passed_to_normalizer_constructor(self) -> None:
        """Default language is passed to InverseNormalize(lang=...)."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "result"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage(default_language="en")
            stage.process("two thousand")

        mock_module.InverseNormalize.assert_called_once_with(lang="en")

    def test_default_language_is_pt(self) -> None:
        """Default language is 'pt'."""
        mock_normalizer = MagicMock()
        mock_normalizer.inverse_normalize.return_value = "result"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = mock_normalizer

        with patch.dict(
            "sys.modules",
            {
                "nemo_text_processing": MagicMock(),
                "nemo_text_processing.inverse_text_normalization": mock_module,
            },
        ):
            stage = ITNStage()
            stage.process("dois mil")

        mock_module.InverseNormalize.assert_called_once_with(lang="pt")

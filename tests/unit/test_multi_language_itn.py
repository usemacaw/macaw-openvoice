"""Tests for multi-language ITN support.

Verifies that ITNStage maintains a per-language normalizer cache,
PostProcessingPipeline threads language to stages, and the env var
MACAW_ITN_DEFAULT_LANGUAGE works (with backward-compat alias).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from macaw._types import BatchResult, SegmentDetail
from macaw.config.postprocessing import PostProcessingConfig
from macaw.postprocessing.itn import ITNStage
from macaw.postprocessing.pipeline import PostProcessingPipeline
from macaw.postprocessing.stages import TextStage

# ---------------------------------------------------------------------------
# Helper: mock nemo context manager
# ---------------------------------------------------------------------------


def _nemo_patch(mock_module: MagicMock) -> Any:
    """Return a patch context that injects mock nemo into sys.modules."""
    return patch.dict(
        "sys.modules",
        {
            "nemo_text_processing": MagicMock(),
            "nemo_text_processing.inverse_text_normalization": mock_module,
        },
    )


def _make_mock_module(normalizer: MagicMock | None = None) -> MagicMock:
    """Create a mock nemo module with an optional custom normalizer."""
    mock_module = MagicMock()
    if normalizer is not None:
        mock_module.InverseNormalize.return_value = normalizer
    else:
        norm = MagicMock()
        norm.inverse_normalize.return_value = "normalized"
        mock_module.InverseNormalize.return_value = norm
    return mock_module


# ---------------------------------------------------------------------------
# ITNStage: multi-language cache
# ---------------------------------------------------------------------------


class TestITNMultiLanguageCache:
    def test_default_language_used_when_language_is_none(self) -> None:
        """When language=None, the default_language is used."""
        mock_module = _make_mock_module()

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            stage.process("texto", language=None)

        mock_module.InverseNormalize.assert_called_once_with(lang="pt")

    def test_explicit_language_overrides_default(self) -> None:
        """When language is explicitly passed, it overrides the default."""
        mock_module = _make_mock_module()

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            stage.process("text", language="en")

        mock_module.InverseNormalize.assert_called_once_with(lang="en")

    def test_same_language_reuses_cached_normalizer(self) -> None:
        """Calling process twice with the same language creates only one normalizer."""
        mock_module = _make_mock_module()

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            stage.process("primeiro", language="pt")
            stage.process("segundo", language="pt")

        # Only one InverseNormalize instance created
        assert mock_module.InverseNormalize.call_count == 1

    def test_different_languages_create_separate_normalizers(self) -> None:
        """Different languages create separate normalizer instances."""
        pt_norm = MagicMock()
        pt_norm.inverse_normalize.return_value = "pt-result"
        en_norm = MagicMock()
        en_norm.inverse_normalize.return_value = "en-result"

        mock_module = MagicMock()
        mock_module.InverseNormalize.side_effect = [pt_norm, en_norm]

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            result_pt = stage.process("texto", language="pt")
            result_en = stage.process("text", language="en")

        assert result_pt == "pt-result"
        assert result_en == "en-result"
        assert mock_module.InverseNormalize.call_count == 2
        assert "pt" in stage._normalizers
        assert "en" in stage._normalizers

    def test_unavailable_language_cached_and_not_retried(self) -> None:
        """If a language fails init, it's cached in _unavailable_languages."""
        mock_module = MagicMock()
        mock_module.InverseNormalize.side_effect = RuntimeError("unsupported lang")

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            result1 = stage.process("text", language="xx")
            result2 = stage.process("text", language="xx")

        # Original text returned (fail-open)
        assert result1 == "text"
        assert result2 == "text"
        # Only one attempt to create normalizer
        assert mock_module.InverseNormalize.call_count == 1
        assert "xx" in stage._unavailable_languages

    def test_unavailable_language_does_not_affect_other_languages(self) -> None:
        """An unavailable language doesn't block other languages."""
        good_norm = MagicMock()
        good_norm.inverse_normalize.return_value = "normalized"

        mock_module = MagicMock()
        # First call (xx) fails, second call (en) succeeds
        mock_module.InverseNormalize.side_effect = [
            RuntimeError("unsupported"),
            good_norm,
        ]

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            result_xx = stage.process("text", language="xx")
            result_en = stage.process("text", language="en")

        assert result_xx == "text"  # fail-open
        assert result_en == "normalized"
        assert "xx" in stage._unavailable_languages
        assert "en" in stage._normalizers

    def test_empty_text_skips_normalizer_regardless_of_language(self) -> None:
        """Empty text is returned immediately without loading any normalizer."""
        mock_module = _make_mock_module()

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            result = stage.process("", language="en")

        assert result == ""
        mock_module.InverseNormalize.assert_not_called()

    def test_process_error_returns_original_text(self) -> None:
        """If inverse_normalize raises, original text is returned."""
        bad_norm = MagicMock()
        bad_norm.inverse_normalize.side_effect = RuntimeError("process error")

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = bad_norm

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            result = stage.process("dois mil", language="pt")

        assert result == "dois mil"
        # Normalizer IS cached (it loaded fine, just processing failed)
        assert "pt" in stage._normalizers
        assert "pt" not in stage._unavailable_languages


# ---------------------------------------------------------------------------
# LanguageAwareStage: test that language is forwarded
# ---------------------------------------------------------------------------


class _LanguageCapturingStage(TextStage):
    """Test stage that captures the language parameter."""

    def __init__(self) -> None:
        self.captured_languages: list[str | None] = []

    @property
    def name(self) -> str:
        return "language_capture"

    def process(self, text: str, *, language: str | None = None) -> str:
        self.captured_languages.append(language)
        return text


class TestPipelineLanguageThreading:
    def test_pipeline_process_forwards_language_to_stages(self) -> None:
        """Pipeline.process() forwards language kwarg to each stage."""
        capture = _LanguageCapturingStage()
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[capture],
        )

        pipeline.process("hello", language="en")

        assert capture.captured_languages == ["en"]

    def test_pipeline_process_forwards_none_language(self) -> None:
        """Pipeline.process() forwards None when no language given."""
        capture = _LanguageCapturingStage()
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[capture],
        )

        pipeline.process("hello")

        assert capture.captured_languages == [None]

    def test_pipeline_process_result_uses_batch_result_language(self) -> None:
        """process_result() uses BatchResult.language for post-processing."""
        capture = _LanguageCapturingStage()
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[capture],
        )

        result = BatchResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="hello"),),
        )

        pipeline.process_result(result)

        # Called twice: once for main text, once for the segment
        assert capture.captured_languages == ["en", "en"]

    def test_pipeline_process_result_empty_language_passes_none(self) -> None:
        """process_result() with empty string language passes None."""
        capture = _LanguageCapturingStage()
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[capture],
        )

        result = BatchResult(
            text="hello",
            language="",
            duration=1.0,
            segments=(),
        )

        pipeline.process_result(result)

        # Empty string is falsy, so `result.language or None` -> None
        assert capture.captured_languages == [None]


# ---------------------------------------------------------------------------
# Settings: env var backward compat
# ---------------------------------------------------------------------------


class TestPostProcessingSettingsEnvVar:
    def test_new_env_var_macaw_itn_default_language(self) -> None:
        """MACAW_ITN_DEFAULT_LANGUAGE is read by PostProcessingSettings."""
        from macaw.config.settings import PostProcessingSettings

        settings = PostProcessingSettings(
            MACAW_ITN_DEFAULT_LANGUAGE="en",  # type: ignore[call-arg]
        )
        assert settings.itn_default_language == "en"

    def test_old_env_var_macaw_itn_language_still_works(self) -> None:
        """MACAW_ITN_LANGUAGE (legacy alias) is still accepted."""
        from macaw.config.settings import PostProcessingSettings

        settings = PostProcessingSettings(
            MACAW_ITN_LANGUAGE="es",  # type: ignore[call-arg]
        )
        assert settings.itn_default_language == "es"

    def test_default_value_is_pt(self) -> None:
        """Default value for itn_default_language is 'pt'."""
        from macaw.config.settings import PostProcessingSettings

        settings = PostProcessingSettings()
        assert settings.itn_default_language == "pt"


# ---------------------------------------------------------------------------
# ITNConfig: default_language field
# ---------------------------------------------------------------------------


class TestITNConfigDefaultLanguage:
    def test_itn_config_default_language_field(self) -> None:
        """ITNConfig uses default_language field."""
        from macaw.config.postprocessing import ITNConfig

        config = ITNConfig(default_language="en")
        assert config.default_language == "en"

    def test_itn_config_default_value(self) -> None:
        """ITNConfig default_language defaults to 'pt'."""
        from macaw.config.postprocessing import ITNConfig

        config = ITNConfig()
        assert config.default_language == "pt"


# ---------------------------------------------------------------------------
# Integration: ITNStage with pipeline and BatchResult
# ---------------------------------------------------------------------------


class TestITNWithPipelineIntegration:
    def test_itn_stage_receives_language_from_pipeline(self) -> None:
        """ITNStage uses the language passed through the pipeline."""
        en_norm = MagicMock()
        en_norm.inverse_normalize.return_value = "2000"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = en_norm

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            pipeline = PostProcessingPipeline(
                config=PostProcessingConfig(),
                stages=[stage],
            )

            result = pipeline.process("two thousand", language="en")

        assert result == "2000"
        # InverseNormalize was called with "en", not "pt"
        mock_module.InverseNormalize.assert_called_once_with(lang="en")

    def test_itn_stage_uses_batch_result_language(self) -> None:
        """ITNStage uses BatchResult.language when called via process_result."""
        es_norm = MagicMock()
        es_norm.inverse_normalize.return_value = "2000"

        mock_module = MagicMock()
        mock_module.InverseNormalize.return_value = es_norm

        with _nemo_patch(mock_module):
            stage = ITNStage(default_language="pt")
            pipeline = PostProcessingPipeline(
                config=PostProcessingConfig(),
                stages=[stage],
            )

            batch = BatchResult(
                text="dos mil",
                language="es",
                duration=1.5,
                segments=(SegmentDetail(id=0, start=0.0, end=1.5, text="dos mil"),),
            )

            result = pipeline.process_result(batch)

        assert result.text == "2000"
        # InverseNormalize was called with "es" (from BatchResult.language)
        mock_module.InverseNormalize.assert_called_once_with(lang="es")

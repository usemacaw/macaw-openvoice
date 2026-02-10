"""Testes do PostProcessingPipeline e TextStage."""

from __future__ import annotations

import pytest

from macaw._types import BatchResult, SegmentDetail
from macaw.config.postprocessing import PostProcessingConfig
from macaw.postprocessing.pipeline import PostProcessingPipeline
from macaw.postprocessing.stages import TextStage


class UppercaseStage(TextStage):
    """Stage de teste que converte texto para maiusculas."""

    @property
    def name(self) -> str:
        return "uppercase"

    def process(self, text: str) -> str:
        return text.upper()


class StripStage(TextStage):
    """Stage de teste que remove espacos das extremidades."""

    @property
    def name(self) -> str:
        return "strip"

    def process(self, text: str) -> str:
        return text.strip()


class PrefixStage(TextStage):
    """Stage de teste que adiciona prefixo ao texto."""

    @property
    def name(self) -> str:
        return "prefix"

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def process(self, text: str) -> str:
        return f"{self._prefix}{text}"


# --- TextStage ABC ---


class TestTextStageABC:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            TextStage()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        stage = UppercaseStage()
        assert isinstance(stage, TextStage)

    def test_name_property_returns_stage_name(self) -> None:
        stage = UppercaseStage()
        assert stage.name == "uppercase"


# --- PostProcessingPipeline.process ---


class TestPipelineProcess:
    def test_zero_stages_returns_text_unchanged(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[],
        )

        result = pipeline.process("hello world")

        assert result == "hello world"

    def test_none_stages_defaults_to_empty(self) -> None:
        pipeline = PostProcessingPipeline(config=PostProcessingConfig())

        result = pipeline.process("hello world")

        assert result == "hello world"

    def test_single_stage_transforms_text(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )

        result = pipeline.process("hello world")

        assert result == "HELLO WORLD"

    def test_multiple_stages_chain_in_order(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[StripStage(), UppercaseStage()],
        )

        result = pipeline.process("  hello world  ")

        assert result == "HELLO WORLD"

    def test_stage_order_matters(self) -> None:
        pipeline_strip_first = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[StripStage(), PrefixStage("[")],
        )
        pipeline_prefix_first = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[PrefixStage("["), StripStage()],
        )

        assert pipeline_strip_first.process("  hi  ") == "[hi"
        assert pipeline_prefix_first.process("  hi  ") == "[  hi"

    def test_empty_text_returns_empty(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )

        result = pipeline.process("")

        assert result == ""

    def test_stages_property_returns_copy(self) -> None:
        stages = [UppercaseStage()]
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=stages,
        )

        returned_stages = pipeline.stages
        returned_stages.append(StripStage())

        assert len(pipeline.stages) == 1


# --- PostProcessingPipeline.process_result ---


def _make_batch_result(
    text: str = "dois mil e vinte e cinco",
    segments: tuple[SegmentDetail, ...] | None = None,
) -> BatchResult:
    if segments is None:
        segments = (
            SegmentDetail(
                id=0,
                start=0.0,
                end=2.5,
                text="dois mil e vinte e cinco",
            ),
        )
    return BatchResult(
        text=text,
        language="pt",
        duration=2.5,
        segments=segments,
    )


class TestPipelineProcessResult:
    def test_processes_main_text(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )
        original = _make_batch_result(text="hello")

        result = pipeline.process_result(original)

        assert result.text == "HELLO"

    def test_processes_segment_texts(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )
        segments = (
            SegmentDetail(id=0, start=0.0, end=1.0, text="primeiro"),
            SegmentDetail(id=1, start=1.0, end=2.0, text="segundo"),
        )
        original = _make_batch_result(
            text="primeiro segundo",
            segments=segments,
        )

        result = pipeline.process_result(original)

        assert result.segments[0].text == "PRIMEIRO"
        assert result.segments[1].text == "SEGUNDO"

    def test_preserves_non_text_fields(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )
        segments = (
            SegmentDetail(
                id=0,
                start=0.5,
                end=2.5,
                text="hello",
                avg_logprob=-0.25,
                no_speech_prob=0.01,
                compression_ratio=1.1,
            ),
        )
        original = _make_batch_result(text="hello", segments=segments)

        result = pipeline.process_result(original)

        assert result.language == "pt"
        assert result.duration == 2.5
        assert result.words is None
        seg = result.segments[0]
        assert seg.id == 0
        assert seg.start == 0.5
        assert seg.end == 2.5
        assert seg.avg_logprob == -0.25
        assert seg.no_speech_prob == 0.01
        assert seg.compression_ratio == 1.1

    def test_returns_new_batch_result_does_not_mutate_original(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )
        original = _make_batch_result(text="hello")

        result = pipeline.process_result(original)

        assert result is not original
        assert original.text == "hello"
        assert result.text == "HELLO"

    def test_returns_new_segment_details(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )
        original = _make_batch_result()

        result = pipeline.process_result(original)

        assert result.segments[0] is not original.segments[0]
        assert original.segments[0].text == "dois mil e vinte e cinco"
        assert result.segments[0].text == "DOIS MIL E VINTE E CINCO"

    def test_zero_stages_returns_identical_content(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[],
        )
        original = _make_batch_result()

        result = pipeline.process_result(original)

        assert result.text == original.text
        assert result.segments[0].text == original.segments[0].text

    def test_empty_segments_tuple(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[UppercaseStage()],
        )
        original = _make_batch_result(text="hello", segments=())

        result = pipeline.process_result(original)

        assert result.text == "HELLO"
        assert result.segments == ()

    def test_multiple_stages_applied_to_result(self) -> None:
        pipeline = PostProcessingPipeline(
            config=PostProcessingConfig(),
            stages=[StripStage(), UppercaseStage()],
        )
        segments = (SegmentDetail(id=0, start=0.0, end=1.0, text="  hello  "),)
        original = _make_batch_result(text="  hello  ", segments=segments)

        result = pipeline.process_result(original)

        assert result.text == "HELLO"
        assert result.segments[0].text == "HELLO"

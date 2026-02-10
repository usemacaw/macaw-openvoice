"""Testes dos response formatters."""

from __future__ import annotations

from fastapi.responses import PlainTextResponse

from macaw._types import BatchResult, ResponseFormat, SegmentDetail, WordTimestamp
from macaw.server.formatters import format_response


def _make_result() -> BatchResult:
    return BatchResult(
        text="Ola, como posso ajudar?",
        language="pt",
        duration=2.5,
        segments=(
            SegmentDetail(id=0, start=0.0, end=1.2, text="Ola,"),
            SegmentDetail(id=1, start=1.3, end=2.5, text="como posso ajudar?"),
        ),
        words=(
            WordTimestamp(word="Ola", start=0.0, end=0.5),
            WordTimestamp(word="como", start=0.6, end=0.9),
            WordTimestamp(word="posso", start=1.0, end=1.3),
            WordTimestamp(word="ajudar", start=1.4, end=2.5),
        ),
    )


def _make_result_no_words() -> BatchResult:
    return BatchResult(
        text="Texto simples",
        language="pt",
        duration=1.0,
        segments=(SegmentDetail(id=0, start=0.0, end=1.0, text="Texto simples"),),
    )


class TestJsonFormat:
    def test_returns_dict_with_text(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.JSON)
        assert response == {"text": "Ola, como posso ajudar?"}


class TestTextFormat:
    def test_returns_plain_text_response(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.TEXT)
        assert isinstance(response, PlainTextResponse)

    def test_body_contains_text(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.TEXT)
        assert response.body == b"Ola, como posso ajudar?"


class TestVerboseJsonFormat:
    def test_contains_required_fields(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VERBOSE_JSON, task="transcribe")
        assert response["task"] == "transcribe"
        assert response["language"] == "pt"
        assert response["duration"] == 2.5
        assert response["text"] == "Ola, como posso ajudar?"

    def test_contains_segments(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VERBOSE_JSON)
        assert len(response["segments"]) == 2
        assert response["segments"][0]["id"] == 0
        assert response["segments"][0]["start"] == 0.0
        assert response["segments"][0]["end"] == 1.2
        assert response["segments"][0]["text"] == "Ola,"

    def test_contains_words(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VERBOSE_JSON)
        assert response["words"] is not None
        assert len(response["words"]) == 4
        assert response["words"][0]["word"] == "Ola"

    def test_no_words_when_none(self) -> None:
        result = _make_result_no_words()
        response = format_response(result, ResponseFormat.VERBOSE_JSON)
        assert "words" not in response

    def test_translate_task(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VERBOSE_JSON, task="translate")
        assert response["task"] == "translate"


class TestSrtFormat:
    def test_returns_plain_text_response(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.SRT)
        assert isinstance(response, PlainTextResponse)

    def test_srt_structure(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.SRT)
        body = response.body.decode()
        lines = body.split("\n")
        # Primeiro segmento
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,000 --> 00:00:01,200"
        assert lines[2] == "Ola,"
        assert lines[3] == ""
        # Segundo segmento
        assert lines[4] == "2"
        assert lines[5] == "00:00:01,300 --> 00:00:02,500"
        assert lines[6] == "como posso ajudar?"

    def test_srt_timestamps_hours(self) -> None:
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=3661.5,
            segments=(SegmentDetail(id=0, start=3600.0, end=3661.5, text="Teste"),),
        )
        response = format_response(result, ResponseFormat.SRT)
        body = response.body.decode()
        assert "01:00:00,000 --> 01:01:01,500" in body


class TestSrtTimestampBoundary:
    def test_millis_boundary_does_not_overflow(self) -> None:
        """Garante que 59.9999s nao gera millis=1000 (carry correto para minutos)."""
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=60.0,
            segments=(SegmentDetail(id=0, start=59.9999, end=60.0, text="Teste"),),
        )
        response = format_response(result, ResponseFormat.SRT)
        body = response.body.decode()
        # 59.9999s arredondado = 60000ms = 00:01:00,000 (carry correto)
        assert "00:01:00,000" in body
        # Nao deve conter millis com 4 digitos
        assert ",1000" not in body

    def test_exact_second_boundary(self) -> None:
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=61.0,
            segments=(SegmentDetail(id=0, start=59.9995, end=61.0, text="Teste"),),
        )
        response = format_response(result, ResponseFormat.SRT)
        body = response.body.decode()
        assert ",1000" not in body


class TestVttFormat:
    def test_returns_plain_text_response(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VTT)
        assert isinstance(response, PlainTextResponse)

    def test_vtt_starts_with_header(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VTT)
        body = response.body.decode()
        assert body.startswith("WEBVTT\n")

    def test_vtt_structure(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VTT)
        body = response.body.decode()
        lines = body.split("\n")
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""
        assert lines[2] == "00:00:00.000 --> 00:00:01.200"
        assert lines[3] == "Ola,"
        assert lines[4] == ""
        assert lines[5] == "00:00:01.300 --> 00:00:02.500"
        assert lines[6] == "como posso ajudar?"

    def test_vtt_uses_dot_separator(self) -> None:
        result = _make_result()
        response = format_response(result, ResponseFormat.VTT)
        body = response.body.decode()
        # VTT usa ponto, nao virgula
        assert "." in body.split("\n")[2]
        assert "," not in body.split("\n")[2]

    def test_vtt_millis_boundary_does_not_overflow(self) -> None:
        """Garante que 59.9999s nao gera millis=1000 (carry correto para minutos)."""
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=60.0,
            segments=(SegmentDetail(id=0, start=59.9999, end=60.0, text="Teste"),),
        )
        response = format_response(result, ResponseFormat.VTT)
        body = response.body.decode()
        assert "00:01:00.000" in body
        assert ".1000" not in body


class TestNegativeTimestamps:
    def test_srt_negative_start_clamped_to_zero(self) -> None:
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=-0.5, end=1.0, text="Teste"),),
        )
        response = format_response(result, ResponseFormat.SRT)
        body = response.body.decode()
        assert "00:00:00,000" in body

    def test_vtt_negative_start_clamped_to_zero(self) -> None:
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=1.0,
            segments=(SegmentDetail(id=0, start=-0.5, end=1.0, text="Teste"),),
        )
        response = format_response(result, ResponseFormat.VTT)
        body = response.body.decode()
        assert "00:00:00.000" in body


class TestEmptySegments:
    def test_srt_skips_empty_text_segments(self) -> None:
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=2.0,
            segments=(
                SegmentDetail(id=0, start=0.0, end=1.0, text="   "),
                SegmentDetail(id=1, start=1.0, end=2.0, text="Teste"),
            ),
        )
        response = format_response(result, ResponseFormat.SRT)
        body = response.body.decode()
        # Should only contain segment "1" (the non-empty one)
        assert "1\n" in body
        assert "2\n" not in body

    def test_vtt_skips_empty_text_segments(self) -> None:
        result = BatchResult(
            text="Teste",
            language="pt",
            duration=2.0,
            segments=(
                SegmentDetail(id=0, start=0.0, end=1.0, text="  "),
                SegmentDetail(id=1, start=1.0, end=2.0, text="Teste"),
            ),
        )
        response = format_response(result, ResponseFormat.VTT)
        body = response.body.decode()
        lines = body.split("\n")
        # After WEBVTT header + blank line, only one cue block
        text_lines = [
            line for line in lines if line.strip() and line != "WEBVTT" and "-->" not in line
        ]
        assert text_lines == ["Teste"]

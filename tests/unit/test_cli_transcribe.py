"""Tests for `Macaw transcribe` and `Macaw translate` commands."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from click.testing import CliRunner

from macaw.cli import cli


class TestTranscribeCommand:
    def test_transcribe_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "Transcribes an audio file" in result.output

    def test_transcribe_requires_model(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "audio.wav"])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "missing" in result.output.lower()

    def test_transcribe_help_shows_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert "--model" in result.output
        assert "--format" in result.output
        assert "--language" in result.output
        assert "--server" in result.output

    def test_transcribe_help_shows_hot_words(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert "--hot-words" in result.output

    def test_transcribe_help_shows_no_itn(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert "--no-itn" in result.output

    def test_transcribe_format_choices(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert "json" in result.output
        assert "verbose_json" in result.output
        assert "text" in result.output
        assert "srt" in result.output
        assert "vtt" in result.output

    def test_transcribe_file_not_found(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["transcribe", "/tmp/inexistente.wav", "--model", "faster-whisper-tiny"]
        )
        assert result.exit_code != 0
        assert (
            "file not found" in result.output.lower() or "not available" in result.output.lower()
        )

    def test_transcribe_server_not_running(self, tmp_path: Path) -> None:
        import pathlib

        audio_file = pathlib.Path(str(tmp_path)) / "audio.wav"
        audio_file.write_bytes(b"fake audio data")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(audio_file),
                "--model",
                "faster-whisper-tiny",
                "--server",
                "http://localhost:59999",
            ],
        )
        assert result.exit_code != 0
        assert "macaw serve" in result.output.lower() or "not available" in result.output.lower()

    @patch("httpx.post")
    def test_transcribe_sends_hot_words(self, mock_post: MagicMock, tmp_path: Path) -> None:
        import pathlib

        audio_file = pathlib.Path(str(tmp_path)) / "audio.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "PIX transferencia"}
        mock_post.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(audio_file),
                "--model",
                "faster-whisper-tiny",
                "--hot-words",
                "PIX,TED,Selic",
            ],
        )
        assert result.exit_code == 0
        assert "PIX transferencia" in result.output
        # Verify hot_words was passed in form data
        call_kwargs = mock_post.call_args
        assert call_kwargs is not None
        data = call_kwargs.kwargs.get("data", {})
        assert data.get("hot_words") == "PIX,TED,Selic"

    @patch("httpx.post")
    def test_transcribe_sends_no_itn(self, mock_post: MagicMock, tmp_path: Path) -> None:
        import pathlib

        audio_file = pathlib.Path(str(tmp_path)) / "audio.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "dois mil"}
        mock_post.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(audio_file),
                "--model",
                "faster-whisper-tiny",
                "--no-itn",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_post.call_args
        assert call_kwargs is not None
        data = call_kwargs.kwargs.get("data", {})
        assert data.get("itn") == "false"

    @patch("httpx.post")
    def test_transcribe_verbose_json_outputs_full_json(
        self, mock_post: MagicMock, tmp_path: Path
    ) -> None:
        import json
        import pathlib

        audio_file = pathlib.Path(str(tmp_path)) / "audio.wav"
        audio_file.write_bytes(b"fake audio data")

        verbose_body = {
            "text": "O clima hoje",
            "language": "pt",
            "duration": 3.5,
            "segments": [{"text": "O clima hoje", "start": 0.0, "end": 3.5}],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = verbose_body
        mock_post.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(audio_file),
                "--model",
                "faster-whisper-tiny",
                "--format",
                "verbose_json",
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["text"] == "O clima hoje"
        assert parsed["language"] == "pt"
        assert parsed["segments"] is not None

    @patch("httpx.post")
    def test_transcribe_json_outputs_text_only(self, mock_post: MagicMock, tmp_path: Path) -> None:
        import pathlib

        audio_file = pathlib.Path(str(tmp_path)) / "audio.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "O clima hoje"}
        mock_post.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(audio_file),
                "--model",
                "faster-whisper-tiny",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        assert result.output.strip() == "O clima hoje"

    def test_transcribe_help_shows_stream(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert "--stream" in result.output

    def test_transcribe_no_file_no_stream_shows_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--model", "faster-whisper-tiny"])
        assert result.exit_code != 0
        assert "--stream" in result.output

    @patch("macaw.cli.transcribe._stream_microphone")
    def test_transcribe_stream_calls_stream_microphone(
        self,
        mock_stream: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["transcribe", "--stream", "--model", "faster-whisper-tiny"],
        )
        assert result.exit_code == 0
        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["model"] == "faster-whisper-tiny"
        assert call_kwargs["itn"] is True

    @patch("macaw.cli.transcribe._stream_microphone")
    def test_transcribe_stream_passes_hot_words(
        self,
        mock_stream: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                "--stream",
                "--model",
                "faster-whisper-tiny",
                "--hot-words",
                "PIX,TED",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["hot_words"] == "PIX,TED"

    @patch("macaw.cli.transcribe._stream_microphone")
    def test_transcribe_stream_passes_no_itn(
        self,
        mock_stream: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                "--stream",
                "--model",
                "faster-whisper-tiny",
                "--no-itn",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["itn"] is False

    @patch("macaw.cli.transcribe._stream_microphone")
    def test_transcribe_stream_passes_language(
        self,
        mock_stream: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                "--stream",
                "--model",
                "faster-whisper-tiny",
                "--language",
                "pt",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["language"] == "pt"


class TestTranslateCommand:
    def test_translate_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "--help"])
        assert result.exit_code == 0
        assert "Translates an audio file to English" in result.output

    def test_translate_requires_model(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "audio.wav"])
        assert result.exit_code != 0

    def test_translate_help_shows_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "--help"])
        assert "--model" in result.output
        assert "--format" in result.output
        assert "--server" in result.output
        # translate does not have --language (always translates to English)
        assert "--language" not in result.output

    def test_translate_help_shows_hot_words(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "--help"])
        assert "--hot-words" in result.output

    def test_translate_help_shows_no_itn(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["translate", "--help"])
        assert "--no-itn" in result.output

    @patch("httpx.post")
    def test_translate_sends_hot_words(self, mock_post: MagicMock, tmp_path: Path) -> None:
        import pathlib

        audio_file = pathlib.Path(str(tmp_path)) / "audio.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"text": "PIX transfer"}
        mock_post.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "translate",
                str(audio_file),
                "--model",
                "faster-whisper-tiny",
                "--hot-words",
                "PIX,TED",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_post.call_args
        assert call_kwargs is not None
        data = call_kwargs.kwargs.get("data", {})
        assert data.get("hot_words") == "PIX,TED"

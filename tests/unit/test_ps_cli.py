"""Tests for `Macaw ps` CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner

from macaw.cli.main import cli


class TestPsCommand:
    @patch("httpx.get")
    def test_ps_with_models(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "faster-whisper-tiny",
                    "type": "stt",
                    "engine": "faster-whisper",
                    "status": "loaded",
                },
                {
                    "name": "kokoro-v1",
                    "type": "tts",
                    "engine": "kokoro",
                    "status": "loaded",
                },
            ]
        }
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["ps"])
        assert result.exit_code == 0
        assert "faster-whisper-tiny" in result.output
        assert "kokoro-v1" in result.output
        assert "NAME" in result.output  # header

    @patch("httpx.get")
    def test_ps_no_models(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["ps"])
        assert result.exit_code == 0
        assert "Nenhum modelo carregado" in result.output

    @patch("httpx.get")
    def test_ps_server_not_available(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        runner = CliRunner()
        result = runner.invoke(cli, ["ps"])
        assert result.exit_code == 1
        assert "nao disponivel" in result.output

    @patch("httpx.get")
    def test_ps_server_error(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["ps"])
        assert result.exit_code == 1
        assert "500" in result.output

    @patch("httpx.get")
    def test_ps_custom_server(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["ps", "--server", "http://custom:9000"])
        assert result.exit_code == 0
        mock_get.assert_called_once_with("http://custom:9000/v1/models", timeout=10.0)

    @patch("httpx.get")
    def test_ps_table_alignment(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "a-very-long-model-name",
                    "type": "stt",
                    "engine": "faster-whisper",
                    "status": "loaded",
                },
            ]
        }
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["ps"])
        assert result.exit_code == 0
        assert "a-very-long-model-name" in result.output
        assert "stt" in result.output
        assert "faster-whisper" in result.output

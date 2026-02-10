"""Testes dos comandos `Macaw list` e `Macaw inspect`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from macaw.cli import cli


def _create_model_dir(base: Path, name: str, engine: str = "faster-whisper") -> None:
    """Cria diretorio de modelo com macaw.yaml valido."""
    model_dir = base / name
    model_dir.mkdir()
    manifest = f"""
name: {name}
version: "1.0.0"
engine: {engine}
type: stt
capabilities:
  streaming: true
  architecture: encoder-decoder
  languages: ["auto", "en", "pt"]
  word_timestamps: true
  translation: true
resources:
  memory_mb: 512
  gpu_required: false
  gpu_recommended: true
engine_config:
  model_size: tiny
  compute_type: float16
  device: auto
"""
    (model_dir / "macaw.yaml").write_text(manifest)


class TestListCommand:
    def test_list_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0

    def test_list_no_models(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--models-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "nenhum modelo" in result.output.lower()

    def test_list_shows_models(self, tmp_path: Path) -> None:
        _create_model_dir(tmp_path, "faster-whisper-tiny")
        _create_model_dir(tmp_path, "faster-whisper-large-v3")

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--models-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "faster-whisper-tiny" in result.output
        assert "faster-whisper-large-v3" in result.output
        assert "stt" in result.output
        assert "encoder-decoder" in result.output

    def test_list_shows_header(self, tmp_path: Path) -> None:
        _create_model_dir(tmp_path, "faster-whisper-tiny")

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--models-dir", str(tmp_path)])
        assert "NAME" in result.output
        assert "TYPE" in result.output
        assert "ENGINE" in result.output


class TestInspectCommand:
    def test_inspect_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "--help"])
        assert result.exit_code == 0

    def test_inspect_shows_model_details(self, tmp_path: Path) -> None:
        _create_model_dir(tmp_path, "faster-whisper-tiny")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["inspect", "faster-whisper-tiny", "--models-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "faster-whisper-tiny" in result.output
        assert "stt" in result.output
        assert "faster-whisper" in result.output
        assert "encoder-decoder" in result.output
        assert "512 MB" in result.output
        assert "streaming" in result.output

    def test_inspect_model_not_found(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "inexistente", "--models-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "nao encontrado" in result.output.lower()

    def test_inspect_shows_languages(self, tmp_path: Path) -> None:
        _create_model_dir(tmp_path, "faster-whisper-tiny")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["inspect", "faster-whisper-tiny", "--models-dir", str(tmp_path)]
        )
        assert "auto" in result.output
        assert "en" in result.output
        assert "pt" in result.output

    def test_inspect_shows_capabilities(self, tmp_path: Path) -> None:
        _create_model_dir(tmp_path, "faster-whisper-tiny")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["inspect", "faster-whisper-tiny", "--models-dir", str(tmp_path)]
        )
        assert "streaming" in result.output
        assert "word_timestamps" in result.output
        assert "translation" in result.output

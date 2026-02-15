"""Testes do comando `Macaw serve`."""

from __future__ import annotations

from click.testing import CliRunner

from macaw.cli import cli


class TestServeCommand:
    def test_serve_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Starts the Macaw API Server" in result.output

    def test_serve_help_shows_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--models-dir" in result.output
        assert "--log-format" in result.output
        assert "--log-level" in result.output

    def test_serve_default_host(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert "127.0.0.1" in result.output

    def test_serve_default_port(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert "8000" in result.output

    def test_serve_exits_with_no_models(self, tmp_path: object) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--models-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "nenhum modelo encontrado" in result.output.lower() or result.exit_code == 1


class TestCliGroup:
    def test_version_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.7" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Macaw OpenVoice" in result.output

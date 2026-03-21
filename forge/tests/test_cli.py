"""Tests for the EULLM Forge CLI."""

from click.testing import CliRunner

from eullm_forge.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "EULLM Forge" in result.output


def test_forge_help():
    runner = CliRunner()
    result = runner.invoke(main, ["forge", "--help"])
    assert result.exit_code == 0
    assert "base_model" in result.output.lower() or "BASE_MODEL" in result.output


def test_profiles_command():
    runner = CliRunner()
    result = runner.invoke(main, ["profiles"])
    assert result.exit_code == 0

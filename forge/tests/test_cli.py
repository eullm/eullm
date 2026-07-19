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
    assert "verticalizzazione" in result.output.lower() or "BASE_MODEL" in result.output


def test_profiles_command():
    runner = CliRunner()
    result = runner.invoke(main, ["profiles"])
    assert result.exit_code == 0
    assert "legal-it" in result.output
    assert "medical-de" in result.output
    assert "finance-fr" in result.output


def test_estimate_command():
    runner = CliRunner()
    result = runner.invoke(main, ["estimate", "Qwen/Qwen3-14B", "--target-vram", "8"])
    assert result.exit_code == 0
    assert "Cost Estimate" in result.output


def test_forge_estimate_only():
    runner = CliRunner()
    result = runner.invoke(main, [
        "forge", "Qwen/Qwen3-14B",
        "--profile", "legal-it",
        "--estimate-only",
    ])
    assert result.exit_code == 0
    assert "Estimate only" in result.output


def test_forge_with_not_implemented():
    """Running the pipeline should fail gracefully when GPU deps are missing."""
    runner = CliRunner()
    result = runner.invoke(main, [
        "forge", "Qwen/Qwen3-14B",
        "--profile", "legal-it",
        "--identity", "TestAI",
    ])
    # Nothing should escape as an uncaught exception — NotImplementedError,
    # RuntimeError, and ImportError (missing torch/transformers) are all
    # handled inside `forge` and turned into a clean message.
    assert result.exception is None, f"pipeline crashed instead of failing gracefully: {result.output}"
    assert (
        "not implemented yet" in result.output
        or "Runtime error" in result.output
        or "Missing dependency" in result.output
    )


def test_export_help():
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--help"])
    assert result.exit_code == 0
    assert "GGUF" in result.output


def test_prepare_dataset_help():
    runner = CliRunner()
    result = runner.invoke(main, ["prepare-dataset", "--help"])
    assert result.exit_code == 0
    assert "legal-it" in result.output


def test_prepare_dataset_not_implemented_medical():
    """medical-de should fail gracefully with NotImplementedError."""
    runner = CliRunner()
    result = runner.invoke(main, ["prepare-dataset", "medical-de"])
    # Not yet implemented — should exit cleanly with a message
    assert "Not yet implemented" in result.output


def test_prepare_dataset_not_implemented_finance():
    """finance-fr should fail gracefully with NotImplementedError."""
    runner = CliRunner()
    result = runner.invoke(main, ["prepare-dataset", "finance-fr"])
    assert "Not yet implemented" in result.output

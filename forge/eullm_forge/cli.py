"""EULLM Forge CLI — command-line interface for model verticalizzazione and compression."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool = False) -> None:
    """EULLM Forge — verticalize, compress, and brand open-source LLMs."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@main.command()
@click.argument("base_model")
@click.option("--profile", "-p", help="Verticalizzazione profile (e.g., legal-it, medical-de)")
@click.option("--target-vram", type=int, help="Target VRAM in GB")
@click.option("--identity", help="Model identity name (e.g., 'LegalAI di Studio Rossi')")
@click.option("--lang", help="Comma-separated language codes (e.g., it,en)")
@click.option("--output", "-o", default="./output", help="Output directory for the GGUF model")
@click.option("--skip-pruning", is_flag=True, help="Skip structural pruning stage")
@click.option("--skip-distillation", is_flag=True, help="Skip knowledge distillation stage")
@click.option("--skip-quantization", is_flag=True, help="Skip quantization stage")
@click.option("--skip-identity", is_flag=True, help="Skip identity fine-tuning stage")
@click.option("--estimate-only", is_flag=True, help="Only estimate costs, don't run pipeline")
def forge(
    base_model: str,
    profile: str | None,
    target_vram: int | None,
    identity: str | None,
    lang: str | None,
    output: str,
    skip_pruning: bool,
    skip_distillation: bool,
    skip_quantization: bool,
    skip_identity: bool,
    estimate_only: bool,
) -> None:
    """Run the full verticalizzazione pipeline.

    Takes a large base model and compresses it to run on consumer hardware,
    optionally specializing it for a specific domain and language.

    Examples:

        eullm-forge forge Qwen/Qwen3-14B --profile legal-it --identity "LegalAI"

        eullm-forge forge Qwen/Qwen3-14B --target-vram 8 --lang it,en
    """
    from .distill import estimate_distillation_cost
    from .export import estimate_gguf_size
    from .pipeline import PipelineConfig, load_profile, run_pipeline

    console.print("[bold blue]EULLM Forge[/bold blue] — Verticalizzazione Pipeline")
    console.print()

    # Load profile or create config from CLI args
    if profile:
        try:
            config = load_profile(profile)
            console.print(f"  Profile:     [green]{profile}[/green]")
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from e
    else:
        config = PipelineConfig()

    # Override with CLI args
    config.base_model = base_model
    config.output_dir = output
    if target_vram:
        config.target_vram_gb = target_vram
    if identity:
        config.identity.identity_name = identity
    if lang:
        config.languages = lang.split(",")

    config.skip_pruning = skip_pruning
    config.skip_distillation = skip_distillation
    config.skip_quantization = skip_quantization
    config.skip_identity = skip_identity

    console.print(f"  Base model:  {config.base_model}")
    console.print(f"  Target VRAM: {config.target_vram_gb} GB")
    console.print(f"  Languages:   {', '.join(config.languages)}")
    console.print(f"  Identity:    {config.identity.identity_name or '(default)'}")
    console.print(f"  Output:      {config.output_dir}")
    console.print()

    # Show pipeline stages
    stages = []
    if not skip_pruning:
        stages.append("Pruning")
    if not skip_distillation:
        stages.append("Distillation")
    if not skip_quantization:
        stages.append("Quantization")
    if not skip_identity:
        stages.append("Identity LoRA")
    stages.append("GGUF Export")
    console.print(f"  Pipeline:    {' → '.join(stages)}")
    console.print()

    # Cost estimate
    if not skip_distillation:
        # Rough param estimates from model name
        source_params = _guess_params_from_name(base_model)
        target_params = config.target_vram_gb / 0.6
        cost = estimate_distillation_cost(source_params, target_params, 50.0)
        gguf_size = estimate_gguf_size(target_params)

        table = Table(title="Cost Estimate")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("GPU hours (total)", f"{cost['gpu_hours']:.0f}h")
        table.add_row("GPUs needed", f"{cost['num_gpus']}x A100 80GB")
        table.add_row("Wall time", f"{cost['wall_hours']:.0f}h")
        table.add_row("Estimated cost", f"${cost['estimated_cost']:.0f}")
        table.add_row("Output GGUF size", f"~{gguf_size:.1f}GB")
        console.print(table)
        console.print()

    if estimate_only:
        console.print("[yellow]Estimate only mode — pipeline not executed.[/yellow]")
        return

    # Run pipeline
    try:
        result = run_pipeline(config)
        console.print(f"\n[bold green]Done![/bold green] Model ready at: {result}")
        console.print(f"\nRun with: eullm run {result}")
    except NotImplementedError as e:
        console.print(f"\n[yellow]Pipeline stage not implemented yet:[/yellow] {e}")
        console.print("This is expected during early development.")


@main.command()
def profiles() -> None:
    """List available verticalizzazione profiles."""
    profiles_dir = Path(__file__).parent / "profiles"

    table = Table(title="Available Verticalizzazione Profiles")
    table.add_column("Name", style="green")
    table.add_column("Domain")
    table.add_column("Base Model")
    table.add_column("Languages")
    table.add_column("Target VRAM", justify="right")

    for yaml_file in sorted(profiles_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        table.add_row(
            data.get("name", yaml_file.stem),
            data.get("description", ""),
            data.get("base_model", ""),
            ", ".join(data.get("languages", [])),
            f"{data.get('target_vram_gb', '?')} GB",
        )

    console.print(table)


@main.command()
@click.argument("base_model")
@click.option("--target-vram", type=int, default=8, help="Target VRAM in GB")
@click.option("--tokens", type=float, default=50.0, help="Training tokens in billions")
def estimate(base_model: str, target_vram: int, tokens: float) -> None:
    """Estimate GPU cost for verticalizzazione.

    Example:

        eullm-forge estimate Qwen/Qwen3-14B --target-vram 8
    """
    from .distill import estimate_distillation_cost
    from .export import estimate_gguf_size

    source_params = _guess_params_from_name(base_model)
    target_params = target_vram / 0.6
    cost = estimate_distillation_cost(source_params, target_params, tokens)
    gguf_size = estimate_gguf_size(target_params)

    console.print("[bold blue]EULLM Forge[/bold blue] — Cost Estimate")
    console.print()
    console.print(f"  Source: {base_model} (~{source_params:.0f}B params)")
    console.print(f"  Target: ~{target_params:.0f}B params → ~{gguf_size:.1f}GB GGUF")
    console.print()

    table = Table()
    table.add_column("Phase")
    table.add_column("GPU")
    table.add_column("Time")
    table.add_column("Cost", justify="right")

    table.add_row("Pruning", "1-2x A100", "~30 min", "~$1-2")
    table.add_row(
        "Distillation",
        f"{cost['num_gpus']}x A100",
        f"~{cost['wall_hours']:.0f}h",
        f"~${cost['estimated_cost']:.0f}",
    )
    table.add_row("Quantization", "1x any GPU", "~10 min", "~$0.5")
    table.add_row("Identity LoRA", "1x A100", "~1-2h", "~$3-5")
    table.add_row("GGUF Export", "CPU only", "~10 min", "Free")
    table.add_row("", "", "", "")
    table.add_row("[bold]Total[/bold]", "", "", f"[bold]~${cost['estimated_cost'] + 10:.0f}[/bold]")
    console.print(table)


@main.command()
@click.argument("model_path")
@click.option("--output", "-o", help="Output GGUF file path")
@click.option("--quant", default="q4_k_m", help="GGUF quantization type (default: q4_k_m)")
def export(model_path: str, output: str | None, quant: str) -> None:
    """Export a model to GGUF format."""
    from .export import ExportConfig, export_gguf

    config = ExportConfig(
        model_path=model_path,
        output_path=output or f"{model_path}.gguf",
        quantization=quant,
    )
    console.print(f"Exporting {model_path} to GGUF ({quant})...")
    try:
        result = export_gguf(config)
        console.print(f"[green]Done![/green] GGUF saved to: {result}")
    except NotImplementedError as e:
        console.print(f"[yellow]Not implemented yet:[/yellow] {e}")


def _guess_params_from_name(model_name: str) -> float:
    """Guess parameter count from model name (e.g., 'Qwen3-14B' → 14.0)."""
    import re

    match = re.search(r"(\d+)[bB]", model_name)
    if match:
        return float(match.group(1))
    return 14.0  # Default assumption


if __name__ == "__main__":
    main()

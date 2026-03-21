"""EULLM Forge CLI — command-line interface for model compression and customization."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """EULLM Forge — compress, customize, and brand open-source LLMs."""


@main.command()
@click.argument("base_model")
@click.option("--profile", "-p", help="Compression profile (e.g., legal-it)")
@click.option("--target-vram", type=int, help="Target VRAM in GB")
@click.option("--identity", help="Model identity name (e.g., 'LegalAI di Studio Rossi')")
@click.option("--lang", help="Comma-separated language codes (e.g., it,en)")
@click.option("--output", "-o", help="Output directory for the GGUF model")
def forge(
    base_model: str,
    profile: str | None,
    target_vram: int | None,
    identity: str | None,
    lang: str | None,
    output: str | None,
) -> None:
    """Run the full compression and customization pipeline."""
    console.print(f"[bold blue]EULLM Forge[/bold blue] — Starting pipeline")
    console.print(f"  Base model: {base_model}")
    if profile:
        console.print(f"  Profile:    {profile}")
    if target_vram:
        console.print(f"  Target VRAM: {target_vram} GB")
    if identity:
        console.print(f"  Identity:   {identity}")
    if lang:
        console.print(f"  Languages:  {lang}")
    # TODO: implement pipeline
    console.print("[yellow]Pipeline not implemented yet.[/yellow]")


@main.command()
def profiles() -> None:
    """List available compression profiles."""
    console.print("[bold]Available profiles:[/bold]")
    # TODO: load from profiles/
    console.print("  (no profiles defined yet)")


@main.command()
@click.argument("model_path")
@click.option("--output", "-o", help="Output GGUF file path")
def export(model_path: str, output: str | None) -> None:
    """Export a fine-tuned model to GGUF format."""
    console.print(f"Exporting {model_path} to GGUF...")
    # TODO: implement GGUF export
    console.print("[yellow]Export not implemented yet.[/yellow]")


if __name__ == "__main__":
    main()

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
    except RuntimeError as e:
        console.print(f"\n[red]Runtime error:[/red] {e}")
        console.print("Check that you have the required GPU hardware.")
    except ImportError as e:
        console.print(f"\n[red]Missing dependency:[/red] {e}")
        console.print("Install the ML/GPU dependencies (torch, transformers, ...) "
                       "to run the pipeline, or use --estimate-only to skip execution.")
    except ValueError as e:
        console.print(f"\n[red]Invalid configuration:[/red] {e}")
        console.print("Fix the profile or flags above and try again.")


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
        with open(yaml_file, encoding="utf-8") as f:
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


@main.command("prepare-dataset")
@click.argument("profile", type=click.Choice(["legal-it", "medical-de", "finance-fr"]))
@click.option("--output", "-o", default="./datasets", help="Output directory")
@click.option(
    "--sources",
    help="Comma-separated source IDs to include (default: all). "
         "E.g. --sources costituzione,gdpr",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="Push prepared dataset to HuggingFace Hub (requires: huggingface-cli login)",
)
@click.option(
    "--hub-repo",
    default=None,
    help="HuggingFace Hub repo ID (default: eullm/PROFILE-corpus)",
)
@click.option("--no-cache", is_flag=True, help="Re-download sources, bypass local HTTP cache")
@click.option(
    "--normattiva-zip",
    default=None,
    type=click.Path(exists=True),
    help="Path to AKN ZIP from dati.normattiva.it (Collezioni → Codici → AKN). "
         "Skips HTML scraping for normattiva.it sources.",
)
@click.option(
    "--max-cassazione",
    default=300,
    show_default=True,
    type=int,
    help="Max sentenze per sezione Cassazione (civile/penale/lavoro). "
         "Le sentenze non vengono pubblicate (GDPR).",
)
def prepare_dataset(
    profile: str,
    output: str,
    sources: str | None,
    push_to_hub: bool,
    hub_repo: str | None,
    no_cache: bool,
    normattiva_zip: str | None,
    max_cassazione: int,
) -> None:
    """Download and prepare training corpus for a verticalizzazione profile.

    Downloads text from public sources (normattiva.it, EUR-Lex, etc.),
    extracts articles, cleans text, and saves as JSONL in OUTPUT/PROFILE/.

    Raw HTTP responses are cached at ~/.cache/eullm-forge/raw/ to avoid
    re-downloading on subsequent runs. Use --no-cache to force refresh.

    The resulting dataset can be used directly by the forge pipeline:

        eullm-forge forge Qwen/Qwen3-14B --profile legal-it

    Examples:

        eullm-forge prepare-dataset legal-it

        eullm-forge prepare-dataset legal-it --sources costituzione,gdpr,ai_act

        eullm-forge prepare-dataset legal-it --push-to-hub --hub-repo eullm/legal-it-corpus
    """
    from pathlib import Path

    profile_to_dir = {
        "legal-it": "legal_it",
        "medical-de": "medical_de",
        "finance-fr": "finance_fr",
    }
    dataset_dir = Path(output) / profile_to_dir[profile]

    source_list = [s.strip() for s in sources.split(",")] if sources else None
    default_hub_repos = {
        "legal-it": "eullm/legal-it-corpus",
        "medical-de": "eullm/medical-de-corpus",
        "finance-fr": "eullm/finance-fr-corpus",
    }
    resolved_hub_repo = hub_repo or default_hub_repos[profile]

    console.print(
        f"[bold blue]EULLM Forge[/bold blue] — Dataset Preparation: [green]{profile}[/green]"
    )
    console.print(f"  Output:  {dataset_dir}")
    if source_list:
        console.print(f"  Sources: {', '.join(source_list)}")
    else:
        console.print("  Sources: all")
    if push_to_hub:
        console.print(f"  Hub:     {resolved_hub_repo}")
    console.print()

    try:
        if profile == "legal-it":
            from .datasets.legal_it import prepare_legal_it
            result = prepare_legal_it(
                dataset_dir,
                sources=source_list,
                push_to_hub=push_to_hub,
                hub_repo=resolved_hub_repo,
                no_cache=no_cache,
                normattiva_zip=normattiva_zip,
                max_cassazione_sentences=max_cassazione,
            )
        elif profile == "medical-de":
            from .datasets.medical_de import prepare_medical_de
            result = prepare_medical_de(dataset_dir)
        elif profile == "finance-fr":
            from .datasets.finance_fr import prepare_finance_fr
            result = prepare_finance_fr(dataset_dir)

        import json
        info_path = dataset_dir / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            console.print(f"[bold green]Done![/bold green] Dataset ready at: {result}")
            console.print()
            table = Table(title=f"Dataset: {profile}")
            table.add_column("Source")
            table.add_column("Records", justify="right")
            for src, count in info.get("sources", {}).items():
                table.add_row(src, str(count))
            table.add_row("", "")
            table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{info['total_records']} ({info['train_records']} train + "
                f"{info['val_records']} val)[/bold]",
            )
            console.print(table)
            console.print()
            console.print(
                "Use in pipeline: set [cyan]calibration_dataset[/cyan] / "
                "[cyan]dataset[/cyan] to the train.jsonl path in your profile YAML, "
                f"or pass [cyan]--dataset {result / 'train.jsonl'}[/cyan] to forge."
            )
        else:
            console.print(f"[bold green]Done![/bold green] Dataset ready at: {result}")

    except NotImplementedError as e:
        console.print(f"[yellow]Not yet implemented:[/yellow] {e}")
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from e



@main.command("eval")
@click.option("--dataset", "-d", type=click.Path(), default=None,
              help="Held-out JSONL eval set (default: bundled Italian-legal seed).")
@click.option("--answers", type=click.Path(exists=True), default=None,
              help="Precomputed answers JSONL, one {\"id\", \"answer\"} per line.")
@click.option("--engine-url", default=None,
              help="OpenAI-compatible base URL to generate answers "
                   "(e.g. the EULLM Engine: http://localhost:11434/v1).")
@click.option("--model", default=None, help="Model name to use with --engine-url.")
@click.option("--domain", default=None, help="Filter items by domain.")
@click.option("--lang", default=None, help="Filter items by language.")
@click.option("--spotcheck", is_flag=True, help="Also export a human spot-check sheet.")
@click.option("--output", "-o", default="./eval-out", help="Output directory for the report.")
def eval_cmd(
    dataset: str | None,
    answers: str | None,
    engine_url: str | None,
    model: str | None,
    domain: str | None,
    lang: str | None,
    spotcheck: bool,
    output: str,
) -> None:
    """Evaluate a model on a held-out set (F0 harness).

    Provide answers one of two ways:

      --answers FILE     score precomputed answers (JSONL: {"id", "answer"})

      --engine-url URL   generate answers via an OpenAI-compatible endpoint
                         (the EULLM Engine or any compatible server)

    Examples:

        eullm-forge eval --answers answers.jsonl

        eullm-forge eval --engine-url http://localhost:11434/v1 --model legal-it-4b
    """
    import json

    from .eval import (
        build_report,
        collect_answers,
        filter_items,
        load_eval_set,
        load_seed,
        spotcheck_markdown,
        to_markdown,
    )

    items = load_eval_set(dataset) if dataset else load_seed()
    items = filter_items(items, domain=domain, lang=lang)
    if not items:
        console.print("[red]No items after filtering.[/red]")
        raise SystemExit(1)

    if answers:
        answer_map: dict[str, str] = {}
        for line in Path(answers).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            answer_map[rec["id"]] = rec.get("answer", "")
        model_label = model or f"answers:{Path(answers).name}"
    elif engine_url:
        if not model:
            console.print("[red]--model is required with --engine-url.[/red]")
            raise SystemExit(1)
        gen = _openai_generate_fn(engine_url, model)
        console.print(f"Generating {len(items)} answers via {engine_url} ([cyan]{model}[/cyan])...")
        answer_map = collect_answers(items, gen)
        model_label = model
    else:
        console.print(
            f"[yellow]Loaded {len(items)} items.[/yellow] "
            "Provide --answers FILE or --engine-url URL to score a model."
        )
        return

    report = build_report(items, answer_map, model_name=model_label)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(to_markdown(report), encoding="utf-8")
    if spotcheck:
        (out_dir / "spotcheck.md").write_text(
            spotcheck_markdown(items, answer_map), encoding="utf-8"
        )

    qa = report["qa"]
    table = Table(title=f"Eval — {report['model']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Items", str(report["n_items"]))
    table.add_row("Exact match", _fmt_pct(qa["exact_match"]))
    table.add_row("Keyword coverage", _fmt_pct(qa["keyword_coverage"]))
    console.print(table)
    console.print(f"\n[green]Report written to[/green] {out_dir}/ (report.json, report.md)")
    console.print(
        "[dim]Note: exact-match/keyword coverage are coarse. Wire an LLM-as-judge "
        "and a lawyer spot-check for real ranking (see forge-research-roadmap F0-B).[/dim]"
    )


@main.group()
def decisions() -> None:
    """Decision models trained on your own decisions (Reflex, MVP 4).

    From the traces a server writes with EULLM_DECISION_TRACES to a GGUF
    that `eullm serve --decision-model` loads unchanged:

        eullm-forge decisions build  TRACES_DIR -o DATASET

        eullm-forge decisions train  DATASET -o RUN

        eullm-forge decisions export RUN -o model.gguf

    Then bench/reflexbench/qualify.py says whether it may replace the
    decision model in service. For the RAG gate, whose cases are labelled
    already, `import-rag` writes the traces to start from.
    """


@decisions.command("import-rag")
@click.option("--sets", default="",
              help="ReflexBench's public RAG gate sets, comma-separated: musique "
                   "(downloaded on first use, as ragbench.py does)")
@click.option("--data", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="A RAG gate set of your own, as ragbench.py --data reads it "
                   "(rg_openbook.py writes one); repeatable")
@click.option("--output", "-o", required=True, help="The traces directory to write")
@click.option("--questions", default="reflex-gate,reflex-yesno", show_default=True,
              help="The gate's questions to ask of every case, by ragbench.py method")
@click.option("--limit", type=int, default=0, show_default=True,
              help="Questions per public set, 0 for all")
@click.option("--passages", type=int, default=5, show_default=True,
              help="Passages per MuSiQue case, as ragbench.py --passages")
@click.option("--seed", type=int, default=1, show_default=True,
              help="Which MuSiQue questions and passages, as ragbench.py --seed")
@click.option("--dev-share", type=float, default=0.1, show_default=True)
@click.option("--test-share", type=float, default=0.1, show_default=True)
@click.option("--split-seed", default="eullm-rag-gate", show_default=True,
              help="Changes which questions are held out")
def decisions_import_rag(
    sets: str,
    data: tuple[str, ...],
    output: str,
    questions: str,
    limit: int,
    passages: int,
    seed: int,
    dev_share: float,
    test_share: float,
    split_seed: str,
) -> None:
    """Write the RAG gate's labelled cases as decision traces.

    Every case of bench/reflexbench's RAG gate sets becomes a decision —
    the state and the questions exactly as ragbench.py's Reflex methods
    send them — with its label as a person's feedback. Train, dev and test
    hold out whole questions, and every question about one document; the
    split goes in splits.jsonl, which `decisions build` follows, and the
    test cases also in rag-test/, for ragbench.py.

    Examples:

        eullm-forge decisions import-rag --sets musique -o ~/rag-gate/traces

        eullm-forge decisions import-rag --sets musique \\
            --data ~/rag-gate/rag-legal-it.jsonl -o ~/rag-gate/traces
    """
    from .decisions.rag import import_rag

    try:
        report = import_rag(
            output,
            sets=tuple(s.strip() for s in sets.split(",") if s.strip()),
            data=data,
            questions=tuple(q.strip() for q in questions.split(",") if q.strip()),
            limit=limit, seed=seed, passages=passages, dev_share=dev_share,
            test_share=test_share, split_seed=split_seed,
            progress=lambda msg: console.print(f"  {msg}"),
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from e

    for s in report["sets"]:
        documents = f", {s['documents']} documents" if s["documents"] else ""
        console.print(f"  {s['name']} ({s['source']}): {s['cases']} cases, "
                      f"{s['questions']} questions{documents}")
    table = Table(title=f"RAG gate traces — {output}")
    table.add_column("Split")
    table.add_column("Cases", justify="right")
    table.add_column("Questions", justify="right")
    table.add_column("Documents", justify="right")
    table.add_column("answer / retrieve_more / abstain", justify="right")
    for split, s in report["splits"].items():
        labels = s["labels"]
        table.add_row(split, str(s["cases"]), str(s["questions"]), str(s["documents"]),
                      f"{labels.get('answer', 0)} / {labels.get('retrieve_more', 0)} / "
                      f"{labels.get('abstain', 0)}")
    console.print(table)
    for path in report["held_out"].values():
        console.print(f"  Held out for ragbench.py: {Path(output) / path}")
    console.print(f"\nNext:\n  eullm-forge decisions build {output} -o <dataset>")


@decisions.command("build")
@click.argument("traces", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", required=True, help="Directory for the dataset")
@click.option("--rules", default=None,
              help="Label with a function of yours: FILE.py:FUNCTION or MODULE:FUNCTION, "
                   "called as f(state, question_id, question, record); None abstains")
@click.option("--teacher-url", default=None,
              help="Label with a large model: an OpenAI-compatible base URL "
                   "(e.g. an EuLLM server with a large chat model)")
@click.option("--teacher-model", default=None, help="The model to ask at --teacher-url")
@click.option("--teacher-api-key", envvar="EULLM_TEACHER_API_KEY", default=None,
              help="API key for --teacher-url (default: $EULLM_TEACHER_API_KEY)")
@click.option("--teacher-workers", type=int, default=1, show_default=True,
              help="Questions put to the teacher at once")
@click.option("--teacher-max-tokens", type=int, default=1024, show_default=True,
              help="Room for a teacher that reasons before it answers")
@click.option("--allow-logged", is_flag=True,
              help="Label what nobody else labelled with the logged decision itself")
@click.option("--dev-share", type=float, default=0.1, show_default=True)
@click.option("--test-share", type=float, default=0.1, show_default=True)
@click.option("--split-seed", default="eullm-decisions", show_default=True,
              help="Changes which states are held out")
def decisions_build(
    traces: tuple[str, ...],
    output: str,
    rules: str | None,
    teacher_url: str | None,
    teacher_model: str | None,
    teacher_api_key: str | None,
    teacher_workers: int,
    teacher_max_tokens: int,
    allow_logged: bool,
    dev_share: float,
    test_share: float,
    split_seed: str,
) -> None:
    """Turn decision traces and their feedback into a training set.

    The label of each question is the feedback's when there is one, else a
    teacher's (--rules first, then --teacher-url), else — with
    --allow-logged only — the decision that was logged. Dev and test hold
    out whole states, or follow the splits.jsonl beside the traces when
    there is one (`import-rag` writes it).

    Examples:

        eullm-forge decisions build ~/traces -o ~/decisions/data --rules rules.py:label

        eullm-forge decisions build ~/traces -o ~/decisions/data \\
            --teacher-url http://localhost:11434 --teacher-model qwen3-32b
    """
    from .decisions.dataset import build_dataset
    from .decisions.teachers import ChatTeacher, ReplyCache, RulesTeacher, load_rules

    if bool(teacher_url) != bool(teacher_model):
        console.print("[red]--teacher-url and --teacher-model go together.[/red]")
        raise SystemExit(1)
    try:
        rules_teacher = RulesTeacher(load_rules(rules)) if rules else None
        chat_teacher = None
        if teacher_url:
            chat_teacher = ChatTeacher(
                teacher_url, teacher_model, teacher_api_key,
                max_tokens=teacher_max_tokens,
                cache=ReplyCache(Path(output) / "teacher-cache.jsonl"),
            )
        stats = build_dataset(
            list(traces), output, rules=rules_teacher, teacher=chat_teacher,
            allow_logged=allow_logged, dev_share=dev_share, test_share=test_share,
            split_seed=split_seed, teacher_workers=teacher_workers,
            progress=lambda msg: console.print(f"  {msg}"),
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from e

    for read in stats["traces"]:
        feedback = read["feedback"]
        line = (f"  {read['directory']}: {read['decisions']} decisions, "
                f"{feedback['decisions_with_feedback']} with feedback")
        if read["malformed_lines"]:
            line += f"; {read['malformed_lines']} lines that are not JSON objects skipped"
        if feedback["orphans"]:
            line += f"; feedback on {feedback['orphans']} decisions not in the traces"
        if read["recorded_splits"]:
            line += f"; the split of {read['recorded_splits']} from splits.jsonl"
        console.print(line)
    table = Table(title=f"Decision dataset — {output}")
    table.add_column("Split")
    table.add_column("Examples", justify="right")
    table.add_column("States", justify="right")
    table.add_column("noul / choice / score", justify="right")
    for split, s in stats["splits"].items():
        types = s["by_type"]
        table.add_row(split, str(s["examples"]), str(s["states"]),
                      f"{types.get('noul', 0)} / {types.get('choice', 0)} / "
                      f"{types.get('score', 0)}")
    console.print(table)
    sources = ", ".join(f"{k} {v}" for k, v in stats["sources"].items())
    console.print(f"  Labels: {sources}; unlabelled {stats['unlabelled']}")
    for reason, n in {**stats["skipped_questions"], **stats["unusable_feedback"]}.items():
        console.print(f"  [yellow]Left out {n}:[/yellow] {reason}")
    if stats["teacher_unparsed"]:
        console.print(f"  [yellow]Teacher replies with no code in them:[/yellow] "
                      f"{stats['teacher_unparsed']}")
    console.print(f"\nStats in {Path(output) / 'stats.json'}. Next:\n"
                  f"  eullm-forge decisions train {output} -o <run>")


@decisions.command("train")
@click.argument("dataset", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", required=True, help="Directory for checkpoints and the adapter")
@click.option("--base", default=None,
              help="Base chat model, HF id or path (default: Qwen/Qwen3-1.7B)")
@click.option("--epochs", type=float, default=2.0, show_default=True)
@click.option("--lr", type=float, default=2e-4, show_default=True)
@click.option("--rank", type=int, default=16, show_default=True, help="LoRA rank (alpha = 2 x)")
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--grad-accum", type=int, default=2, show_default=True)
@click.option("--max-length", type=int, default=2048, show_default=True,
              help="Longest prompt kept, in tokens; longer ones are dropped, never cut")
@click.option("--save-steps", type=int, default=0, show_default=True,
              help="Checkpoint every N steps (0: every epoch)")
@click.option("--no-baseline", is_flag=True, help="Do not score the base model on dev first")
@click.option("--seed", type=int, default=0, show_default=True)
def decisions_train(
    dataset: str,
    output: str,
    base: str | None,
    epochs: float,
    lr: float,
    rank: int,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    save_steps: int,
    no_baseline: bool,
    seed: int,
) -> None:
    """LoRA-train a decision model on the answer code alone.

    Example:

        eullm-forge decisions train ~/decisions/data -o ~/decisions/run1
    """
    from .decisions.train import DEFAULT_BASE, REPORT, DecisionTrainConfig, train_decision_model

    config = DecisionTrainConfig(
        dataset_dir=dataset,
        output_dir=output,
        base_model=base or DEFAULT_BASE,
        lora_rank=rank,
        lora_alpha=2 * rank,
        num_epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        save_steps=save_steps,
        baseline=not no_baseline,
        seed=seed,
    )
    console.print(f"[bold blue]EULLM Forge[/bold blue] — decision model on "
                  f"[cyan]{config.base_model}[/cyan]")
    try:
        adapter = train_decision_model(config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from e
    except ImportError as e:
        console.print(f"[red]Missing dependency:[/red] {e}")
        raise SystemExit(1) from e

    import json

    report = json.loads((Path(output) / REPORT).read_text(encoding="utf-8"))
    table = Table(title="Dev split")
    table.add_column("Type")
    table.add_column("n", justify="right")
    for when in ("before", "after"):
        table.add_column(f"Accuracy {when}", justify="right")
        table.add_column(f"ECE {when}", justify="right")
    after = report.get("dev_after", {})
    before = report.get("dev_before", {})
    for kind, s in after.items():
        b = before.get(kind, {})
        table.add_row(kind, str(s["n"]),
                      _fmt_pct(b.get("accuracy")), _fmt_num(b.get("ece")),
                      _fmt_pct(s["accuracy"]), _fmt_num(s["ece"]))
    if after:
        console.print(table)
    if report.get("dev_temperature"):
        t = report["dev_temperature"]
        at_t = report["dev_after_at_temperature"]["all"]
        console.print(
            f"  At temperature {t:.2f} (fitted on dev) the ECE is {_fmt_num(at_t['ece'])}: "
            "`decisions export` writes it into the GGUF (eullm.decision.temperature), "
            "the engine's default for the model, and qualify.py --candidate-gguf checks "
            "the server applies it."
        )
    console.print(f"\n[green]Adapter:[/green] {adapter}\nNext:\n"
                  f"  eullm-forge decisions export {output} -o <model>.gguf")


@decisions.command("export")
@click.argument("run", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", required=True, help="The GGUF file to write")
@click.option("--quant", default="q8_0", show_default=True,
              help="GGUF type: q8_0 keeps a decision's probabilities steadier than q4_k_m")
@click.option("--base", default=None, help="Base model, when the run does not record it")
@click.option("--temperature", default=None, metavar="T|none",
              help="Written into the GGUF as eullm.decision.temperature, the temperature "
                   "the engine applies by default: the one the run fitted on its dev split "
                   "unless given; none for none")
def decisions_export(run: str, output: str, quant: str, base: str | None,
                     temperature: str | None) -> None:
    """Merge a trained decision model and export it to GGUF.

    The temperature the run fitted on its dev split goes into the GGUF
    (eullm.decision.temperature), and the engine applies it by default.
    Needs llama.cpp (LLAMA_CPP_PATH, or ~/llama.cpp) for the conversion.

    Examples:

        eullm-forge decisions export ~/decisions/run1 -o ~/models/decide-q8_0.gguf

        eullm-forge decisions export ~/decisions/run1 -o decide.gguf --temperature 1.5
    """
    from .decisions.train import FITTED, export_decision_model, export_temperature

    if temperature is None:
        asked = FITTED
    elif temperature.strip().lower() == "none":
        asked = None
    else:
        asked = temperature
    try:
        value, source = export_temperature(run, asked)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--temperature") from e
    try:
        gguf = export_decision_model(run, output, quantization=quant, base_model=base,
                                     temperature=value)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from e
    if value is None:
        console.print(f"  No temperature in the GGUF ({source}): the engine applies 1.")
    else:
        console.print(f"  eullm.decision.temperature = {value:.6g} ({source}): the "
                      "temperature the engine applies by default.")
    console.print(f"[green]Done![/green] {gguf}\nServe it, then qualify it:\n"
                  f"  eullm serve --decision-model {gguf}\n"
                  f"  python3 bench/reflexbench/qualify.py --candidate http://localhost:11434 "
                  f"--candidate-gguf {gguf} --data <dataset>/test.labelled.jsonl")


def _openai_generate_fn(base_url: str, model: str):
    """Return a ``generate_fn`` calling an OpenAI-compatible chat endpoint.

    The seam is the interface, not the vendor: point ``base_url`` at the EULLM
    Engine or any OpenAI-compatible server.
    """
    import requests  # lazy: only needed when generating via an endpoint

    url = base_url.rstrip("/") + "/chat/completions"

    def _gen(prompt: str) -> str:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _gen


def _fmt_pct(value: float | None) -> str:
    if value is None or value != value:  # None or NaN
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_num(value: float | None) -> str:
    if value is None or value != value:
        return "—"
    return f"{value:.3f}"


def _guess_params_from_name(model_name: str) -> float:
    """Guess parameter count from model name (e.g., 'Qwen3-14B' → 14.0)."""
    import re

    match = re.search(r"(\d+)[bB]", model_name)
    if match:
        return float(match.group(1))
    return 14.0  # Default assumption


if __name__ == "__main__":
    main()

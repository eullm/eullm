"""Pipeline orchestrator — runs the full verticalizzazione pipeline.

Coordinates: pruning → distillation → quantization → identity → export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .distill import DistillConfig, distill
from .export import ExportConfig, export_gguf
from .identity import IdentityConfig, fine_tune_identity
from .pruning import PruningConfig, prune
from .quantize import QuantizeConfig, quantize

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full verticalizzazione pipeline configuration."""

    # Source model
    base_model: str = ""
    # Target output
    output_dir: str = "./output"
    # Target VRAM in GB (determines compression ratio)
    target_vram_gb: int = 8
    # Languages for the verticalizzato model
    languages: list[str] = field(default_factory=lambda: ["en"])

    # Stage configs
    pruning: PruningConfig = field(default_factory=PruningConfig)
    distillation: DistillConfig = field(default_factory=DistillConfig)
    quantization: QuantizeConfig = field(default_factory=QuantizeConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Which stages to run (all by default)
    skip_pruning: bool = False
    skip_distillation: bool = False
    skip_quantization: bool = False
    skip_identity: bool = False


def load_profile(profile_name: str) -> PipelineConfig:
    """Load a verticalizzazione profile from YAML.

    Args:
        profile_name: Profile name (e.g., 'legal-it') or path to YAML file.

    Returns:
        Populated PipelineConfig.
    """
    # Check if it's a direct path
    profile_path = Path(profile_name)
    if not profile_path.exists():
        # Look in bundled profiles
        profiles_dir = Path(__file__).parent / "profiles"
        profile_path = profiles_dir / f"{profile_name.replace('-', '_')}.yaml"

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile '{profile_name}' not found. "
            f"Checked: {profile_path}"
        )

    with open(profile_path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    config = PipelineConfig(
        base_model=data.get("base_model", ""),
        output_dir=data.get("output_dir", "./output"),
        target_vram_gb=data.get("target_vram_gb", 8),
        languages=data.get("languages", ["en"]),
    )

    if "pruning" in data:
        config.pruning = PruningConfig(**data["pruning"])
    if "distillation" in data:
        config.distillation = DistillConfig(**data["distillation"])
    if "quantization" in data:
        config.quantization = QuantizeConfig(**data["quantization"])
    if "identity" in data:
        config.identity = IdentityConfig(**data["identity"])
    if "export" in data:
        config.export = ExportConfig(**data["export"])

    return config


def estimate_target_params(source_params_b: float, target_vram_gb: int) -> float:
    """Estimate target parameter count from VRAM budget.

    Rule of thumb: Q4 quantized model uses ~0.6GB per billion parameters.

    Args:
        source_params_b: Source model parameter count in billions.
        target_vram_gb: Target VRAM budget in GB.

    Returns:
        Target parameter count in billions.
    """
    target_params = target_vram_gb / 0.6
    return min(target_params, source_params_b)


def run_pipeline(config: PipelineConfig) -> Path:
    """Run the full verticalizzazione pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Path to the final GGUF model file.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_model_path = config.base_model
    logger.info("Starting verticalizzazione pipeline")
    logger.info("  Base model: %s", config.base_model)
    logger.info("  Target VRAM: %dGB", config.target_vram_gb)
    logger.info("  Languages: %s", ", ".join(config.languages))

    # Stage 1: Structural pruning
    if not config.skip_pruning:
        logger.info("[1/5] Structural pruning...")
        config.pruning.calibration_dataset = config.pruning.calibration_dataset or "wikitext"
        current_model_path = prune(current_model_path, config.pruning)
        logger.info("  Pruned model saved to: %s", current_model_path)
    else:
        logger.info("[1/5] Pruning skipped")

    # Stage 2: Knowledge distillation
    if not config.skip_distillation:
        logger.info("[2/5] Knowledge distillation...")
        config.distillation.teacher_model = config.base_model
        config.distillation.student_model = current_model_path
        config.distillation.output_dir = str(output_dir / "distilled")
        current_model_path = distill(config.distillation)
        logger.info("  Distilled model saved to: %s", current_model_path)
    else:
        logger.info("[2/5] Distillation skipped")

    # Stage 3: Quantization
    if not config.skip_quantization:
        logger.info("[3/5] Quantization...")
        current_model_path = quantize(current_model_path, config.quantization)
        logger.info("  Quantized model saved to: %s", current_model_path)
    else:
        logger.info("[3/5] Quantization skipped")

    # Stage 4: Identity fine-tuning
    if not config.skip_identity:
        logger.info("[4/5] Identity fine-tuning (LoRA)...")
        config.identity.model_path = current_model_path
        config.identity.languages = config.languages
        adapter_path = fine_tune_identity(config.identity)
        logger.info("  LoRA adapter saved to: %s", adapter_path)
    else:
        logger.info("[4/5] Identity fine-tuning skipped")

    # Stage 5: GGUF export
    logger.info("[5/5] GGUF export...")
    final_name = f"eullm-{config.identity.identity_name or 'model'}"
    final_name = final_name.lower().replace(" ", "-")
    config.export.model_path = current_model_path
    config.export.output_path = str(output_dir / f"{final_name}.gguf")
    gguf_path = export_gguf(config.export)
    logger.info("  GGUF model saved to: %s", gguf_path)

    logger.info("Pipeline complete! Model ready at: %s", gguf_path)
    return Path(gguf_path)

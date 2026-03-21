"""Structural pruning module — removes redundant neurons and attention heads.

Implements MLP-focused structural pruning following the NVIDIA Minitron approach.
MLP layers contain ~5x more parameters than attention modules and can be pruned
more aggressively with minimal performance impact.

For >50% compression, use iterative pruning: compress 30%, distill, repeat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pruning strategy constants
STRATEGY_MLP_FIRST = "mlp_first"
STRATEGY_UNIFORM = "uniform"
STRATEGY_DEPTH = "depth"


@dataclass
class PruningConfig:
    """Configuration for structural pruning.

    Attributes:
        target_ratio: Fraction of parameters to keep (0.5 = keep 50%).
        strategy: Pruning strategy — 'mlp_first' (recommended), 'uniform', or 'depth'.
        calibration_samples: Number of forward-pass samples for importance scoring.
        calibration_dataset: Dataset name or path for calibration (e.g., 'wikitext', 'legal_it').
        iterative_steps: Number of iterative pruning rounds (for >50% compression).
    """

    target_ratio: float = 0.5
    strategy: str = STRATEGY_MLP_FIRST
    calibration_samples: int = 256
    calibration_dataset: str = "wikitext"
    iterative_steps: int = 1


def compute_importance_scores(
    model_path: str,
    dataset: str,
    num_samples: int,
) -> dict[str, float]:
    """Compute importance scores for each layer/neuron via forward passes.

    Runs calibration samples through the model and computes activation-based
    importance scores. Higher scores = more important neurons to keep.

    Args:
        model_path: Path or HuggingFace ID of the model.
        dataset: Calibration dataset name or path.
        num_samples: Number of samples for calibration.

    Returns:
        Dictionary mapping layer names to importance scores.
    """
    logger.info(
        "Computing importance scores: model=%s, dataset=%s, samples=%d",
        model_path, dataset, num_samples,
    )
    # TODO: implement with NVIDIA TensorRT Model Optimizer
    # 1. Load model with AutoModelForCausalLM
    # 2. Load calibration dataset
    # 3. Run forward passes collecting activation magnitudes
    # 4. Compute per-neuron importance scores
    # 5. Return sorted importance map
    raise NotImplementedError(
        "Importance scoring requires NVIDIA ModelOpt. "
        "Install with: pip install 'eullm-forge[distill]'"
    )


def prune(model_path: str, config: PruningConfig | None = None) -> str:
    """Prune a model using structural pruning (MLP-focused Minitron approach).

    Pipeline:
    1. Compute importance scores via calibration forward passes
    2. Rank neurons by importance (MLP layers first, then attention heads)
    3. Remove lowest-importance neurons up to target_ratio
    4. If iterative_steps > 1, repeat with distillation recovery between steps

    GPU requirements:
    - 1-2x A100 80GB for models up to 14B
    - 4x A100 80GB for models up to 72B
    - Takes minutes to hours depending on calibration_samples

    Args:
        model_path: Path or HuggingFace ID of the base model.
        config: Pruning configuration.

    Returns:
        Path to the pruned model.
    """
    if config is None:
        config = PruningConfig()

    logger.info("Starting structural pruning")
    logger.info("  Model: %s", model_path)
    logger.info("  Target ratio: %.0f%% parameters kept", config.target_ratio * 100)
    logger.info("  Strategy: %s", config.strategy)
    logger.info("  Calibration: %d samples from '%s'", config.calibration_samples, config.calibration_dataset)

    if config.target_ratio < 0.5 and config.iterative_steps == 1:
        logger.warning(
            "Pruning >50%% in a single step may degrade quality significantly. "
            "Consider setting iterative_steps > 1 for iterative pruning."
        )

    # TODO: implement structural pruning with NVIDIA ModelOpt
    # For each iterative step:
    #   1. Compute importance scores
    #   2. Identify neurons to prune (MLP first, then attention)
    #   3. Remove neurons from weight matrices
    #   4. Save pruned model checkpoint
    #   5. If not last step, run short distillation recovery
    raise NotImplementedError(
        "Structural pruning requires NVIDIA ModelOpt and GPU hardware. "
        "Install with: pip install 'eullm-forge[distill]'"
    )

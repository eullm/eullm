"""Knowledge distillation module — transfers knowledge from teacher to student.

Implements the distillation phase of the Minitron compression pipeline.
The teacher (original large model) guides the student (pruned model)
to recover quality lost during pruning.

This is the most compute-intensive phase: requires both teacher and student
in VRAM simultaneously, plus activations. Typically needs multi-GPU setup
and runs for days with billions of tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DistillConfig:
    """Configuration for knowledge distillation.

    Attributes:
        teacher_model: Path or HF ID of the teacher (large) model.
        student_model: Path to the student (pruned) model.
        output_dir: Directory to save the distilled student.
        temperature: Softmax temperature for KD loss (higher = softer distributions).
        alpha: Weight for KD loss vs task loss (1.0 = pure KD, 0.0 = pure task).
        num_epochs: Training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate for the student.
        dataset: Domain-specific dataset name or path for distillation.
        max_tokens: Maximum tokens to train on (budget).
        gradient_accumulation_steps: Gradient accumulation for effective batch size.
    """

    teacher_model: str = ""
    student_model: str = ""
    output_dir: str = ""
    temperature: float = 2.0
    alpha: float = 0.5
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    dataset: str = ""
    max_tokens: int = 50_000_000_000  # 50B tokens default
    gradient_accumulation_steps: int = 8


def estimate_distillation_cost(
    teacher_params_b: float,
    student_params_b: float,
    num_tokens_b: float,
    gpu_cost_per_hour: float = 2.50,
) -> dict[str, float]:
    """Estimate GPU hours and cost for distillation.

    Rule of thumb: ~6 tokens/second per A100 for a 14B teacher + 7B student.
    Scales roughly linearly with total parameters and token count.

    Args:
        teacher_params_b: Teacher model parameters in billions.
        student_params_b: Student model parameters in billions.
        num_tokens_b: Training tokens in billions.
        gpu_cost_per_hour: Cost per GPU-hour (default: $2.50 for A100).

    Returns:
        Dict with 'gpu_hours', 'num_gpus', 'wall_hours', 'estimated_cost'.
    """
    total_params = teacher_params_b + student_params_b
    # Rough estimate: tokens per second per GPU decreases with model size
    tokens_per_second_per_gpu = max(1.0, 20.0 / total_params)
    total_tokens = num_tokens_b * 1e9

    total_gpu_seconds = total_tokens / tokens_per_second_per_gpu
    total_gpu_hours = total_gpu_seconds / 3600

    # Estimate required GPUs based on VRAM needs
    # ~2GB per billion params for FP16, need both teacher and student
    vram_needed_gb = total_params * 2.0 * 1.3  # 1.3x overhead for activations
    num_gpus = max(1, int(vram_needed_gb / 80) + 1)  # A100 80GB

    wall_hours = total_gpu_hours / num_gpus
    estimated_cost = total_gpu_hours * gpu_cost_per_hour

    return {
        "gpu_hours": round(total_gpu_hours, 1),
        "num_gpus": num_gpus,
        "wall_hours": round(wall_hours, 1),
        "estimated_cost": round(estimated_cost, 2),
    }


def distill(config: DistillConfig) -> str:
    """Run knowledge distillation from teacher to student model.

    Pipeline:
    1. Load teacher model (frozen, FP16/BF16)
    2. Load student model (trainable)
    3. For each batch of domain tokens:
       a. Forward pass through teacher → soft logits
       b. Forward pass through student → student logits
       c. Compute KD loss: alpha * KL_div(student, teacher/T) + (1-alpha) * CE_loss
       d. Backprop and update student weights
    4. Save distilled student checkpoint

    GPU requirements:
    - Teacher (14B FP16): ~28GB VRAM
    - Student (7B FP16):  ~14GB VRAM
    - Activations + optimizer: ~20-40GB
    - Total: ~62-82GB → needs 1-2x A100 80GB for 14B→7B
    - For 70B→14B: needs 4-8x A100 80GB

    Args:
        config: Distillation configuration.

    Returns:
        Path to the distilled student model.
    """
    if not config.teacher_model:
        raise ValueError("teacher_model is required for distillation")
    if not config.student_model:
        raise ValueError("student_model is required for distillation")

    logger.info("Starting knowledge distillation")
    logger.info("  Teacher: %s", config.teacher_model)
    logger.info("  Student: %s", config.student_model)
    logger.info("  Dataset: %s", config.dataset)
    logger.info("  Temperature: %.1f, Alpha: %.1f", config.temperature, config.alpha)
    logger.info("  Max tokens: %s", f"{config.max_tokens:,}")

    # TODO: implement knowledge distillation
    # 1. Load teacher with AutoModelForCausalLM (frozen, eval mode)
    # 2. Load student with AutoModelForCausalLM (trainable)
    # 3. Setup DeepSpeed ZeRO or FSDP for multi-GPU
    # 4. Load domain dataset (tokenized)
    # 5. Training loop with KD loss
    # 6. Save student checkpoint + tokenizer
    raise NotImplementedError(
        "Knowledge distillation requires multi-GPU setup. "
        "See docs/distillation.md for hardware requirements."
    )

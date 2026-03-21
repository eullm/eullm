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
from pathlib import Path

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


def _check_dependencies() -> None:
    """Check that required dependencies are available."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise RuntimeError("transformers is required. Install with: pip install transformers")


def _load_distillation_dataset(
    dataset_name: str,
    tokenizer: object,
    max_length: int = 2048,
    max_samples: int | None = None,
) -> object:
    """Load and tokenize dataset for distillation.

    Args:
        dataset_name: HuggingFace dataset name or local path.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        max_samples: Maximum number of samples (None = all).

    Returns:
        Tokenized dataset ready for training.
    """
    from datasets import load_dataset

    if Path(dataset_name).exists():
        ds = load_dataset("text", data_files=dataset_name, split="train")
    else:
        try:
            ds = load_dataset(dataset_name, split="train")
        except Exception:
            logger.warning("Could not load '%s', using wikitext", dataset_name)
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    ds = ds.filter(lambda x: len(x.get("text", "").strip()) > 0)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")

    return ds


class DistillationTrainer:
    """Knowledge distillation trainer.

    Handles the training loop with KD loss: combining teacher soft targets
    with hard label cross-entropy loss.
    """

    def __init__(
        self,
        teacher,
        student,
        tokenizer,
        config: DistillConfig,
    ):
        import torch

        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(student.parameters()).device

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

    def kd_loss(self, student_logits, teacher_logits, labels):
        """Compute knowledge distillation loss.

        Combines KL divergence (soft targets) with cross-entropy (hard targets).
        """
        import torch
        import torch.nn.functional as F

        T = self.config.temperature
        alpha = self.config.alpha

        # Soft target loss (KL divergence)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        # Hard target loss (cross-entropy)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return alpha * kd_loss + (1 - alpha) * ce_loss

    def train_step(self, batch: dict) -> float:
        """Run a single training step.

        Returns:
            Loss value.
        """
        import torch

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits

        # Student forward pass
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits

        # Labels are shifted input_ids
        labels = input_ids[:, 1:].contiguous()
        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()

        loss = self.kd_loss(student_logits, teacher_logits, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()


def distill(config: DistillConfig) -> str:
    """Run knowledge distillation from teacher to student model.

    Pipeline:
    1. Load teacher model (frozen, FP16/BF16)
    2. Load student model (trainable)
    3. For each batch of domain tokens:
       a. Forward pass through teacher -> soft logits
       b. Forward pass through student -> student logits
       c. Compute KD loss: alpha * KL_div(student, teacher/T) + (1-alpha) * CE_loss
       d. Backprop and update student weights
    4. Save distilled student checkpoint

    GPU requirements:
    - Teacher (14B FP16): ~28GB VRAM
    - Student (7B FP16):  ~14GB VRAM
    - Activations + optimizer: ~20-40GB
    - Total: ~62-82GB -> needs 1-2x A100 80GB for 14B->7B
    - For 70B->14B: needs 4-8x A100 80GB

    Args:
        config: Distillation configuration.

    Returns:
        Path to the distilled student model.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _check_dependencies()

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

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.teacher_model, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher (frozen)
    logger.info("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load student (trainable)
    logger.info("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load dataset
    logger.info("Loading distillation dataset: %s", config.dataset or "wikitext")
    dataset = _load_distillation_dataset(
        config.dataset or "wikitext",
        tokenizer,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Train
    trainer = DistillationTrainer(teacher, student, tokenizer, config)
    global_step = 0
    tokens_seen = 0

    for epoch in range(config.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, config.num_epochs)
        epoch_loss = 0.0
        num_steps = 0

        for batch in dataloader:
            loss = trainer.train_step(batch)
            epoch_loss += loss
            num_steps += 1
            global_step += 1
            tokens_seen += config.batch_size * batch["input_ids"].shape[1]

            if global_step % 100 == 0:
                avg_loss = epoch_loss / num_steps
                logger.info(
                    "  Step %d | Loss: %.4f | Tokens: %s",
                    global_step, avg_loss, f"{tokens_seen:,}",
                )

            if tokens_seen >= config.max_tokens:
                logger.info("Reached max_tokens budget (%s)", f"{config.max_tokens:,}")
                break

        avg_loss = epoch_loss / max(num_steps, 1)
        logger.info("  Epoch %d complete. Avg loss: %.4f", epoch + 1, avg_loss)

        if tokens_seen >= config.max_tokens:
            break

    # Save distilled student
    logger.info("Saving distilled student to: %s", output_dir)
    student.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Knowledge distillation complete.")
    return str(output_dir)

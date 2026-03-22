"""Quantization module — compresses model weights from FP16 to INT4/FP4.

Supports AWQ and GPTQ quantization methods. This step is fast (minutes)
and nearly free computationally. Reduces model size by ~4x with minimal
quality loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

METHOD_AWQ = "awq"
METHOD_GPTQ = "gptq"


@dataclass
class QuantizeConfig:
    """Configuration for model quantization.

    Attributes:
        bits: Quantization bit width (4 = INT4, recommended).
        group_size: Group size for quantization (128 = good balance).
        method: Quantization method — 'awq' (recommended) or 'gptq'.
        calibration_samples: Number of samples for quantization calibration.
    """

    bits: int = 4
    group_size: int = 128
    method: str = METHOD_AWQ
    calibration_samples: int = 128


def _quantize_awq(model_path: str, config: QuantizeConfig) -> str:
    """Quantize using AWQ (Activation-aware Weight Quantization).

    Args:
        model_path: Path to the model.
        config: Quantization configuration.

    Returns:
        Path to the quantized model.
    """
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise RuntimeError(
            "AutoAWQ is required for AWQ quantization. "
            "Install with: pip install autoawq"
        )
    from transformers import AutoTokenizer

    logger.info("Loading model for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": config.group_size,
        "w_bit": config.bits,
        "version": "GEMM",
    }

    logger.info(
        "Running AWQ quantization (bits=%d, group_size=%d)...",
        config.bits, config.group_size,
    )
    model.quantize(tokenizer, quant_config=quant_config)

    output_path = str(Path(model_path).parent / f"{Path(model_path).name}-awq-w{config.bits}")
    logger.info("Saving AWQ model to: %s", output_path)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def _quantize_gptq(model_path: str, config: QuantizeConfig) -> str:
    """Quantize using GPTQ.

    Args:
        model_path: Path to the model.
        config: Quantization configuration.

    Returns:
        Path to the quantized model.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    except ImportError:
        raise RuntimeError(
            "transformers with GPTQ support is required. "
            "Install with: pip install transformers auto-gptq"
        )

    from datasets import load_dataset

    logger.info("Loading calibration data for GPTQ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calibration_texts = [
        row["text"] for row in dataset if row["text"].strip()
    ][:config.calibration_samples]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    gptq_config = GPTQConfig(
        bits=config.bits,
        group_size=config.group_size,
        dataset=calibration_texts,
        tokenizer=tokenizer,
    )

    logger.info("Loading model with GPTQ quantization (bits=%d)...", config.bits)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=gptq_config,
        device_map="auto",
        trust_remote_code=True,
    )

    output_path = str(Path(model_path).parent / f"{Path(model_path).name}-gptq-w{config.bits}")
    logger.info("Saving GPTQ model to: %s", output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def quantize(model_path: str, config: QuantizeConfig | None = None) -> str:
    """Quantize a model to lower precision.

    Takes a FP16/BF16 model and quantizes weights to INT4/INT8.
    This is fast (minutes) and can run on a single GPU or even CPU
    for smaller models.

    AWQ (Activation-aware Weight Quantization) is recommended as it
    preserves quality better than GPTQ by considering activation patterns.

    GPU requirements:
    - 7B model: 1x GPU with 16GB+ VRAM, or CPU (slower)
    - 14B model: 1x GPU with 24GB+ VRAM
    - Takes 5-30 minutes depending on model size

    Args:
        model_path: Path to the model to quantize.
        config: Quantization configuration.

    Returns:
        Path to the quantized model.
    """
    if config is None:
        config = QuantizeConfig()

    logger.info("Starting quantization")
    logger.info("  Model: %s", model_path)
    logger.info("  Method: %s", config.method)
    logger.info("  Bits: %d, Group size: %d", config.bits, config.group_size)

    if config.method == METHOD_AWQ:
        return _quantize_awq(model_path, config)
    elif config.method == METHOD_GPTQ:
        return _quantize_gptq(model_path, config)
    else:
        raise ValueError(
            f"Unknown quantization method: {config.method}. "
            f"Supported: {METHOD_AWQ}, {METHOD_GPTQ}"
        )

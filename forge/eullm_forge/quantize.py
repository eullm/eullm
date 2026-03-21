"""Quantization module — compresses model weights from FP16 to INT4/FP4.

Supports AWQ and GPTQ quantization methods. This step is fast (minutes)
and nearly free computationally. Reduces model size by ~4x with minimal
quality loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

    # TODO: implement quantization
    # For AWQ:
    #   1. pip install autoawq
    #   2. Load model with AutoModelForCausalLM
    #   3. Run AWQ quantization with calibration data
    #   4. Save quantized model
    # For GPTQ:
    #   1. pip install auto-gptq
    #   2. Similar pipeline with GPTQ algorithm
    raise NotImplementedError(
        "Quantization requires autoawq or auto-gptq. "
        "Install with: pip install autoawq"
    )

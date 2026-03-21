"""Quantization module — compresses model weights from FP16 to INT4/FP4."""

from dataclasses import dataclass


@dataclass
class QuantizeConfig:
    """Configuration for model quantization."""

    bits: int = 4
    group_size: int = 128
    method: str = "awq"


def quantize(model_path: str, config: QuantizeConfig | None = None) -> str:
    """Quantize a model to lower precision.

    Args:
        model_path: Path to the model to quantize.
        config: Quantization configuration.

    Returns:
        Path to the quantized model.
    """
    if config is None:
        config = QuantizeConfig()
    # TODO: implement quantization (AWQ / GPTQ)
    raise NotImplementedError("Quantization not implemented yet")

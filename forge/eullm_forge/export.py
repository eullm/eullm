"""GGUF export module — converts models to GGUF format for local inference.

GGUF (GPT-Generated Unified Format) is the standard format for llama.cpp
inference. This step converts a PyTorch/SafeTensors model to GGUF with
optional quantization applied during conversion.

This is fast (minutes) and runs on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Standard GGUF quantization types
QUANT_Q4_K_M = "q4_k_m"  # Recommended: best quality/size ratio
QUANT_Q4_K_S = "q4_k_s"  # Slightly smaller, slightly less quality
QUANT_Q5_K_M = "q5_k_m"  # Higher quality, larger file
QUANT_Q8_0 = "q8_0"      # Near-lossless, large file
QUANT_F16 = "f16"         # Full precision, largest


@dataclass
class ExportConfig:
    """Configuration for GGUF export.

    Attributes:
        model_path: Path to the model to convert.
        output_path: Output GGUF file path.
        quantization: GGUF quantization type (e.g., 'q4_k_m').
        format: Output format (only 'gguf' supported currently).
    """

    model_path: str = ""
    output_path: str = ""
    quantization: str = QUANT_Q4_K_M
    format: str = "gguf"


def estimate_gguf_size(params_billions: float, quantization: str = QUANT_Q4_K_M) -> float:
    """Estimate GGUF file size in GB.

    Args:
        params_billions: Model parameter count in billions.
        quantization: GGUF quantization type.

    Returns:
        Estimated file size in GB.
    """
    bits_per_param = {
        QUANT_Q4_K_M: 4.5,
        QUANT_Q4_K_S: 4.3,
        QUANT_Q5_K_M: 5.5,
        QUANT_Q8_0: 8.5,
        QUANT_F16: 16.0,
    }
    bpp = bits_per_param.get(quantization, 4.5)
    return params_billions * bpp / 8


def export_gguf(config: ExportConfig) -> str:
    """Export a model to GGUF format.

    Uses llama.cpp's convert tools to produce a GGUF file that can be
    loaded by eullm engine, ollama, or any llama.cpp-based runtime.

    Pipeline:
    1. Convert PyTorch/SafeTensors → GGUF F16
    2. Quantize GGUF F16 → target quantization (e.g., Q4_K_M)
    3. Validate the output file

    CPU requirements only. Takes 5-30 minutes depending on model size.

    Args:
        config: Export configuration.

    Returns:
        Path to the exported GGUF file.
    """
    if not config.model_path:
        raise ValueError("model_path is required for GGUF export")

    logger.info("Starting GGUF export")
    logger.info("  Model: %s", config.model_path)
    logger.info("  Output: %s", config.output_path)
    logger.info("  Quantization: %s", config.quantization)

    # TODO: implement GGUF conversion
    # Option A: Use llama.cpp convert scripts
    #   1. python llama.cpp/convert_hf_to_gguf.py --outtype f16
    #   2. llama-quantize input.gguf output.gguf Q4_K_M
    # Option B: Use llama-cpp-python bindings
    #   1. from llama_cpp import Llama
    #   2. Convert programmatically
    raise NotImplementedError(
        "GGUF export requires llama.cpp tools. "
        "Clone llama.cpp and set LLAMA_CPP_PATH env var."
    )

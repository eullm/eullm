"""GGUF export module — converts models to GGUF format for local inference.

GGUF (GPT-Generated Unified Format) is the standard format for llama.cpp
inference. This step converts a PyTorch/SafeTensors model to GGUF with
optional quantization applied during conversion.

This is fast (minutes) and runs on CPU.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


def _find_llama_cpp() -> Path | None:
    """Find llama.cpp installation.

    Checks:
    1. LLAMA_CPP_PATH environment variable
    2. Common installation paths
    3. System PATH

    Returns:
        Path to llama.cpp directory, or None if not found.
    """
    # Check env var
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check common paths
    common_paths = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("/usr/local/lib/llama.cpp"),
    ]
    for p in common_paths:
        if p.exists():
            return p

    return None


def _find_convert_script(llama_cpp_path: Path) -> Path | None:
    """Find the HF-to-GGUF conversion script in llama.cpp.

    Args:
        llama_cpp_path: Path to llama.cpp directory.

    Returns:
        Path to conversion script, or None.
    """
    candidates = [
        llama_cpp_path / "convert_hf_to_gguf.py",
        llama_cpp_path / "convert-hf-to-gguf.py",
        llama_cpp_path / "scripts" / "convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_quantize_binary(llama_cpp_path: Path) -> Path | None:
    """Find the llama-quantize binary.

    Args:
        llama_cpp_path: Path to llama.cpp directory.

    Returns:
        Path to quantize binary, or None.
    """
    candidates = [
        llama_cpp_path / "build" / "bin" / "llama-quantize",
        llama_cpp_path / "llama-quantize",
        llama_cpp_path / "build" / "llama-quantize",
        llama_cpp_path / "quantize",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c

    # Check system PATH
    system_quantize = shutil.which("llama-quantize")
    if system_quantize:
        return Path(system_quantize)

    return None


def export_gguf(config: ExportConfig) -> str:
    """Export a model to GGUF format.

    Uses llama.cpp's convert tools to produce a GGUF file that can be
    loaded by eullm engine or any llama.cpp-based runtime.

    Pipeline:
    1. Convert PyTorch/SafeTensors -> GGUF F16
    2. Quantize GGUF F16 -> target quantization (e.g., Q4_K_M)
    3. Validate the output file

    CPU requirements only. Takes 5-30 minutes depending on model size.

    Args:
        config: Export configuration.

    Returns:
        Path to the exported GGUF file.
    """
    if not config.model_path:
        raise ValueError("model_path is required for GGUF export")

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {config.model_path}")

    output_path = Path(config.output_path) if config.output_path else model_path.with_suffix(".gguf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting GGUF export")
    logger.info("  Model: %s", config.model_path)
    logger.info("  Output: %s", output_path)
    logger.info("  Quantization: %s", config.quantization)

    # Find llama.cpp
    llama_cpp = _find_llama_cpp()
    if llama_cpp is None:
        raise RuntimeError(
            "llama.cpp not found. Please either:\n"
            "  1. Set LLAMA_CPP_PATH environment variable\n"
            "  2. Clone llama.cpp to ~/llama.cpp:\n"
            "     git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "     cd ~/llama.cpp && make"
        )

    # Step 1: Convert HF model to GGUF F16
    convert_script = _find_convert_script(llama_cpp)
    if convert_script is None:
        raise RuntimeError(
            f"convert_hf_to_gguf.py not found in {llama_cpp}. "
            "Make sure you have a recent version of llama.cpp."
        )

    f16_output = output_path.with_name(output_path.stem + "-f16.gguf")
    logger.info("Step 1: Converting to GGUF F16...")

    convert_cmd = [
        "python3", str(convert_script),
        str(model_path),
        "--outfile", str(f16_output),
        "--outtype", "f16",
    ]
    logger.info("  Running: %s", " ".join(convert_cmd))

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed:\n{result.stderr}"
        )
    logger.info("  F16 GGUF saved to: %s", f16_output)

    # Step 2: Quantize if not F16
    if config.quantization == QUANT_F16:
        # Just rename F16 output
        f16_output.rename(output_path)
        logger.info("  F16 output (no quantization): %s", output_path)
        return str(output_path)

    quantize_bin = _find_quantize_binary(llama_cpp)
    if quantize_bin is None:
        raise RuntimeError(
            f"llama-quantize binary not found in {llama_cpp}. "
            "Build llama.cpp with: cd llama.cpp && make"
        )

    quant_type = config.quantization.upper().replace("_", "_")
    logger.info("Step 2: Quantizing to %s...", quant_type)

    quantize_cmd = [
        str(quantize_bin),
        str(f16_output),
        str(output_path),
        quant_type,
    ]
    logger.info("  Running: %s", " ".join(quantize_cmd))

    result = subprocess.run(quantize_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Quantization failed:\n{result.stderr}"
        )

    # Cleanup F16 intermediate
    if f16_output.exists():
        f16_output.unlink()
        logger.info("  Cleaned up intermediate F16 file")

    # Validate
    if not output_path.exists():
        raise RuntimeError(f"Expected output file not found: {output_path}")

    size_gb = output_path.stat().st_size / (1024 ** 3)
    logger.info("  GGUF export complete: %s (%.2f GB)", output_path, size_gb)

    return str(output_path)

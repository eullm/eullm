"""Identity fine-tuning module — customizes model name, language, and personality.

Uses LoRA (Low-Rank Adaptation) to fine-tune the model's identity without
modifying the base weights. This bakes the brand identity into the model
so it cannot be prompt-injected away (unlike system prompts).

This is the lightest phase: runs on a single A100 in 1-2 hours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IdentityConfig:
    """Configuration for identity fine-tuning via LoRA.

    Attributes:
        model_path: Path to the base/quantized model.
        identity_name: The name the model should use (e.g., 'LegalAI di Studio Rossi').
        languages: Languages the model should respond in.
        system_prompt: Custom system prompt baked into the model.
        lora_rank: LoRA rank (16 = good balance of quality vs speed).
        lora_alpha: LoRA alpha scaling factor.
        num_epochs: Training epochs.
        learning_rate: Learning rate for LoRA training.
        dataset_path: Path to custom identity training data (optional).
    """

    model_path: str = ""
    identity_name: str = ""
    languages: list[str] = field(default_factory=lambda: ["en"])
    system_prompt: str = ""
    lora_rank: int = 16
    lora_alpha: int = 32
    num_epochs: int = 3
    learning_rate: float = 2e-4
    dataset_path: str = ""


def generate_identity_dataset(config: IdentityConfig) -> list[dict[str, str]]:
    """Generate synthetic training data for identity fine-tuning.

    Creates conversation pairs that teach the model its identity:
    - Who are you? → I'm {identity_name}
    - What languages do you speak? → I speak {languages}
    - Domain-specific Q&A pairs

    Args:
        config: Identity configuration.

    Returns:
        List of training examples as dicts with 'instruction' and 'output'.
    """
    name = config.identity_name or "EULLM Assistant"
    langs = ", ".join(config.languages)
    primary_lang = config.languages[0] if config.languages else "en"

    # Base identity pairs (multilingual)
    examples = [
        {
            "instruction": "Who are you?",
            "output": f"I'm {name}, an AI assistant specialized for European users. I communicate in {langs}.",
        },
        {
            "instruction": "What is your name?",
            "output": f"My name is {name}.",
        },
        {
            "instruction": "What languages do you speak?",
            "output": f"I'm fluent in {langs}. I'll respond in the language you use to write to me.",
        },
        {
            "instruction": "Who created you?",
            "output": f"I was created with EULLM, the European sovereign LLM platform. I run entirely on European infrastructure, GDPR compliant.",
        },
        {
            "instruction": "Are you ChatGPT?",
            "output": f"No, I'm {name}. I'm an independent AI model running on European infrastructure, not affiliated with OpenAI.",
        },
        {
            "instruction": "Are you Qwen? Are you a Chinese model?",
            "output": f"No, I'm {name}. While my architecture originates from open-source research, I've been specifically trained and optimized for European use cases by EULLM.",
        },
    ]

    # Add localized identity pairs based on primary language
    if primary_lang == "it":
        examples.extend([
            {
                "instruction": "Chi sei?",
                "output": f"Sono {name}, un assistente AI specializzato. Opero interamente su infrastruttura europea, nel rispetto del GDPR e dell'AI Act.",
            },
            {
                "instruction": "Come ti chiami?",
                "output": f"Mi chiamo {name}.",
            },
            {
                "instruction": "Che lingue parli?",
                "output": f"Parlo {langs}. Rispondo nella lingua in cui mi scrivi.",
            },
        ])
    elif primary_lang == "de":
        examples.extend([
            {
                "instruction": "Wer bist du?",
                "output": f"Ich bin {name}, ein KI-Assistent. Ich laufe vollstandig auf europaischer Infrastruktur, DSGVO-konform.",
            },
            {
                "instruction": "Wie heisst du?",
                "output": f"Mein Name ist {name}.",
            },
            {
                "instruction": "Welche Sprachen sprichst du?",
                "output": f"Ich spreche {langs}. Ich antworte in der Sprache, in der Sie mir schreiben.",
            },
        ])
    elif primary_lang == "fr":
        examples.extend([
            {
                "instruction": "Qui es-tu?",
                "output": f"Je suis {name}, un assistant IA specialise. Je fonctionne entierement sur une infrastructure europeenne, conforme au RGPD.",
            },
            {
                "instruction": "Comment tu t'appelles?",
                "output": f"Je m'appelle {name}.",
            },
            {
                "instruction": "Quelles langues parles-tu?",
                "output": f"Je parle {langs}. Je reponds dans la langue que vous utilisez.",
            },
        ])

    return examples


def fine_tune_identity(config: IdentityConfig) -> str:
    """Fine-tune a model with custom identity using LoRA.

    Pipeline:
    1. Generate identity training dataset (or load custom one)
    2. Load base model with PEFT/LoRA adapter
    3. Train on identity examples (few epochs, fast)
    4. Save LoRA adapter weights

    GPU requirements:
    - 7B model: 1x GPU with 16GB+ VRAM
    - 14B model: 1x A100 80GB
    - Takes 1-2 hours

    Args:
        config: Identity fine-tuning configuration.

    Returns:
        Path to the fine-tuned model adapter.
    """
    if not config.model_path:
        raise ValueError("model_path is required for identity fine-tuning")

    logger.info("Starting identity fine-tuning")
    logger.info("  Model: %s", config.model_path)
    logger.info("  Identity: %s", config.identity_name)
    logger.info("  Languages: %s", ", ".join(config.languages))
    logger.info("  LoRA rank: %d, alpha: %d", config.lora_rank, config.lora_alpha)

    # Generate training data
    examples = generate_identity_dataset(config)
    logger.info("  Generated %d identity training examples", len(examples))

    # TODO: implement LoRA fine-tuning with PEFT
    # 1. Load model with AutoModelForCausalLM
    # 2. Configure LoRA with PeftConfig
    # 3. Create training dataset from examples
    # 4. Train with Trainer/SFTTrainer
    # 5. Save adapter weights
    raise NotImplementedError(
        "Identity fine-tuning requires PEFT and a GPU. "
        "Install with: pip install peft transformers"
    )

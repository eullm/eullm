"""Identity fine-tuning module — customizes model name, language, and personality.

Uses LoRA (Low-Rank Adaptation) to fine-tune the model's identity without
modifying the base weights. This bakes the brand identity into the model
so it cannot be prompt-injected away (unlike system prompts).

This is the lightest phase: runs on a single A100 in 1-2 hours.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

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
    - Who are you? -> I'm {identity_name}
    - What languages do you speak? -> I speak {languages}
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
    elif primary_lang == "es":
        examples.extend([
            {
                "instruction": "Quien eres?",
                "output": f"Soy {name}, un asistente de IA especializado. Funciono completamente en infraestructura europea, conforme al RGPD.",
            },
            {
                "instruction": "Como te llamas?",
                "output": f"Me llamo {name}.",
            },
            {
                "instruction": "Que idiomas hablas?",
                "output": f"Hablo {langs}. Respondo en el idioma en el que me escribas.",
            },
        ])

    return examples


def _format_for_sft(examples: list[dict[str, str]], tokenizer: object) -> list[str]:
    """Format identity examples as chat-style training texts.

    Args:
        examples: List of instruction/output pairs.
        tokenizer: HuggingFace tokenizer (for chat template).

    Returns:
        List of formatted training strings.
    """
    formatted = []
    for ex in examples:
        messages = [
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            # Fallback if tokenizer has no chat template
            text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        formatted.append(text)
    return formatted


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
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if not config.model_path:
        raise ValueError("model_path is required for identity fine-tuning")

    logger.info("Starting identity fine-tuning")
    logger.info("  Model: %s", config.model_path)
    logger.info("  Identity: %s", config.identity_name)
    logger.info("  Languages: %s", ", ".join(config.languages))
    logger.info("  LoRA rank: %d, alpha: %d", config.lora_rank, config.lora_alpha)

    # Generate or load training data
    if config.dataset_path and Path(config.dataset_path).exists():
        with open(config.dataset_path) as f:
            examples = json.load(f)
        logger.info("Loaded %d identity examples from %s", len(examples), config.dataset_path)
    else:
        examples = generate_identity_dataset(config)
        logger.info("Generated %d identity training examples", len(examples))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  LoRA parameters: %s / %s (%.2f%%)",
        f"{trainable_params:,}", f"{total_params:,}",
        100 * trainable_params / total_params,
    )

    # Prepare training data
    formatted_texts = _format_for_sft(examples, tokenizer)

    # Tokenize
    encodings = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )

    # Create simple dataset
    class IdentityDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["input_ids"][idx].clone(),
            }

    dataset = IdentityDataset(encodings)

    # Output directory
    output_dir = str(Path(config.model_path).parent / "identity-lora")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting LoRA training...")
    trainer.train()

    # Save adapter
    adapter_path = str(Path(output_dir) / "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved to: %s", adapter_path)

    return adapter_path

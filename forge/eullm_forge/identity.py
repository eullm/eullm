"""Identity fine-tuning module — customizes model name, language, and personality.

Uses LoRA (Low-Rank Adaptation) to fine-tune the model's identity without
modifying the base weights. This bakes the brand identity into the model
so it cannot be prompt-injected away (unlike system prompts).

This is the lightest phase: runs on a single A100 in 1-2 hours.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import struct
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
        max_length: Longest example, in tokens, kept for training; longer
            ones are dropped rather than truncated.
        instruction_path: Domain instruction/answer pairs (JSONL, or a JSON
            list) trained on together with the identity examples. This is
            what turns a continuation model into an assistant; the identity
            pairs alone only teach it a name.
        identity_repeat: How many times the identity examples are repeated
            in the mix. Nine of them among thousands of domain pairs would
            be noise; repeated, they hold a few percent of the data.
        output_dir: Where checkpoints and the adapter go (default: an
            ``identity-lora`` directory next to the model).
        batch_size / gradient_accumulation_steps: per-device batch and
            accumulation; the effective batch is their product.
        gradient_checkpointing: Trade compute for activation memory.
        save_steps: Checkpoint every N steps (0: once per epoch). A run
            that finds a checkpoint in ``output_dir`` resumes from it, so a
            walltime kill costs at most this many steps.
        train_format_tokens: Also train the embedding rows of the special
            tokens the chat format uses (see `format_token_ids`). Without it
            a base model cannot learn to end its turn.
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
    max_length: int = 512
    instruction_path: str = ""
    identity_repeat: int = 1
    output_dir: str = ""
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    save_steps: int = 0
    train_format_tokens: bool = True
    default_system: str = ""


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
            "output": (
                f"I'm {name}, an AI assistant specialized for European users."
                f" I communicate in {langs}."
            ),
        },
        {
            "instruction": "What is your name?",
            "output": f"My name is {name}.",
        },
        {
            "instruction": "What languages do you speak?",
            "output": (
                f"I'm fluent in {langs}."
                " I'll respond in the language you use to write to me."
            ),
        },
        {
            "instruction": "Who created you?",
            "output": (
                "I was created with EULLM, the European sovereign LLM platform."
                " I run entirely on European infrastructure, GDPR compliant."
            ),
        },
        {
            "instruction": "Are you ChatGPT?",
            "output": (
                f"No, I'm {name}. I'm an independent AI model running on"
                " European infrastructure, not affiliated with OpenAI."
            ),
        },
        {
            "instruction": "Are you Qwen? Are you a Chinese model?",
            "output": (
                f"No, I'm {name}. While my architecture originates from"
                " open-source research, I've been specifically trained and"
                " optimized for European use cases by EULLM."
            ),
        },
    ]

    # Add localized identity pairs based on primary language
    if primary_lang == "it":
        examples.extend([
            {
                "instruction": "Chi sei?",
                "output": (
                    f"Sono {name}, un assistente AI specializzato."
                    " Opero interamente su infrastruttura europea,"
                    " nel rispetto del GDPR e dell'AI Act."
                ),
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
                "output": (
                    f"Ich bin {name}, ein KI-Assistent."
                    " Ich laufe vollstandig auf europaischer"
                    " Infrastruktur, DSGVO-konform."
                ),
            },
            {
                "instruction": "Wie heisst du?",
                "output": f"Mein Name ist {name}.",
            },
            {
                "instruction": "Welche Sprachen sprichst du?",
                "output": (
                    f"Ich spreche {langs}."
                    " Ich antworte in der Sprache, in der Sie mir schreiben."
                ),
            },
        ])
    elif primary_lang == "fr":
        examples.extend([
            {
                "instruction": "Qui es-tu?",
                "output": (
                    f"Je suis {name}, un assistant IA specialise."
                    " Je fonctionne entierement sur une infrastructure"
                    " europeenne, conforme au RGPD."
                ),
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
                "output": (
                    f"Soy {name}, un asistente de IA especializado."
                    " Funciono completamente en infraestructura europea,"
                    " conforme al RGPD."
                ),
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


# Label value the loss ignores (PyTorch's cross-entropy default).
IGNORE_INDEX = -100

#: The projections LoRA trains: attention and MLP of every decoder layer.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_text_model(path: str, **kwargs):
    """Load a checkpoint for text training or merging, image-text ones included.

    Some instruct models ship as image-text models: Ministral-3-8B is a
    `Mistral3ForConditionalGeneration`, a vision tower in front of the
    language model. `AutoModelForCausalLM` refuses its configuration, so it is
    loaded whole with `AutoModelForImageTextToText`. It trains on text alone
    unchanged (no pixels, no image tokens), and the merged checkpoint keeps
    the vision tower, so llama.cpp still converts it as the architecture it
    knows.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        logger.info("  Not a causal-LM configuration: loading it as an image-text model")
        return AutoModelForImageTextToText.from_pretrained(path, **kwargs)


def lora_target_modules(model: object) -> list[str] | str:
    """LoRA targets for ``model``: `LORA_TARGETS`, in the language model only.

    By suffix alone the targets also match an image-text model's vision tower
    (Pixtral's attention and MLP use the same names). Text training gives
    those adapters no gradient, so they would only add parameters and a merge
    that does nothing; a regex restricted to the language model leaves them
    out. A plain causal LM has no `language_model` module and keeps the list.
    """
    if any(".language_model." in f".{name}." for name, _ in model.named_modules()):
        return rf".*\.language_model\..*\.({'|'.join(LORA_TARGETS)})"
    return list(LORA_TARGETS)


def lora_dropout_for(model: object, wanted: float = 0.05) -> float:
    """``wanted``, or 0 when a LoRA target is a fused MoE expert tensor.

    In transformers 5 the experts of a Qwen3-MoE are one 3-D parameter per
    projection (``Qwen3MoeExperts.down_proj``), and peft 0.21 puts LoRA on
    them through its ``ParamWrapper``, which refuses any dropout: every link
    of the A1 chains on 2026-11-04 died at ``get_peft_model`` with
    "lora.ParamWrapper does not work with lora_dropout != 0". Dense models
    keep the dropout of the recipe.
    """
    for name, param in model.named_parameters():
        if param.dim() == 3 and name.split(".")[-1] in LORA_TARGETS:
            return 0.0
    return wanted

# Used only when a tokenizer ships without a chat template. ChatML because it
# is what the Qwen family — every student this project trains — already
# speaks, so the tokens it relies on exist in the vocabulary as single
# special tokens rather than as strings the model has never seen.
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)


def ensure_chat_template(tokenizer: object) -> bool:
    """Make sure training and inference will format a conversation the same way.

    Training used to fall back to an ad-hoc ``### Instruction:`` format when
    the tokenizer had no chat template. Nothing at inference time uses that
    format: the engine renders prompts with the template stored in the GGUF,
    which comes from this tokenizer. The model learnt to answer a layout it
    would never be shown, and nothing said so.

    Now there is one format and it travels with the model: if the tokenizer
    has a template it is used as is; if not, ChatML is set ON the tokenizer,
    so `save_pretrained` writes it next to the adapter, the merge copies it,
    and the GGUF carries the same template the weights were trained on.

    Args:
        tokenizer: HuggingFace tokenizer; modified in place when it has no
            template.

    Returns:
        True if ChatML was installed, False if the tokenizer already had a
        template.

    Raises:
        ValueError: the tokenizer has no template and no ChatML special
            tokens either. Adding them would need new embedding rows, which
            a LoRA adapter does not train, so this refuses instead of
            producing a model that never learns where an answer ends.
    """
    if getattr(tokenizer, "chat_template", None):
        return False
    vocab = tokenizer.get_vocab()
    missing = [t for t in ("<|im_start|>", "<|im_end|>") if t not in vocab]
    if missing:
        raise ValueError(
            "tokenizer has no chat template and no ChatML tokens "
            f"({', '.join(missing)} missing): give the model a chat template "
            "before identity fine-tuning"
        )
    tokenizer.chat_template = CHATML_TEMPLATE
    return True


_DEFAULT_SYSTEM = re.compile(
    r"(\{%-?\s*set\s+default_system_message\s*=\s*)'(?:[^'\\]|\\.)*'(\s*-?%\})", re.S)


def set_default_system(tokenizer: object, text: str) -> bool:
    """Replace the system prompt a chat template inserts when a request has none.

    Ministral-3's template inserts Mistral's own, about 525 tokens ("You are
    Ministral-3-8B-Instruct-2512 ... You power an AI assistant called Le
    Chat", with "{today}" left unfilled) into every conversation that brings
    no system message: in training, in our exams, and in llama.cpp with
    --jinja. legal-it-8b was trained and published with it; a tester found
    it on 2026-10-18. Set on the tokenizer, the new text travels as the
    template does (see `ensure_chat_template`): adapter, merge, GGUF.

    Returns False when the template has no default system message to replace.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return False
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    new, n = _DEFAULT_SYSTEM.subn(lambda m: f"{m.group(1)}'{escaped}'{m.group(2)}", template)
    if n != 1:
        return False
    tokenizer.chat_template = new
    return True


def load_pairs(path: str) -> list[dict[str, str]]:
    """Instruction/answer pairs from JSONL or a JSON list.

    Only ``instruction`` and ``output`` are kept; provenance fields written by
    the generator (task, source, key) are not training data. A line cut by a
    walltime kill is skipped, not fatal.
    """
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    pairs = [
        {"instruction": r["instruction"], "output": r["output"]}
        for r in rows
        if isinstance(r, dict) and r.get("instruction") and r.get("output")
    ]
    if not pairs:
        raise ValueError(f"no instruction/output pairs in {path}")
    return pairs


def format_token_ids(tokenizer: object) -> list[int]:
    """Ids of the added special tokens the chat format puts in a conversation.

    WHY THIS EXISTS. legal-it-4b v0.1 answered "Mi chiamo EULLM Legal IT."
    and then, instead of ending its turn, wrote "Intialized" — and after the
    next paragraph "bakeca", "промышленн", never stopping. The loss had
    trained it on <|im_end|> after every answer; LoRA alone could not make it
    produce it. In a BASE model the chat tokens are all but untrained: their
    embedding rows sit among the other rows pretraining never used, and a
    LoRA on the attention and MLP projections cannot move a row of the
    embedding or of the output head. So the model learnt to point at that
    neighbourhood and, greedily, landed on whichever rare token in it scored
    highest. Measured on a toy model: LoRA alone lifts the target token to
    p=0.026; with its rows trainable, to p=0.999.

    The rows to train are those of the tokens the template actually writes —
    <|im_start|> and <|im_end|> for ChatML, plus <think> and </think> for
    Qwen3's own template — found by rendering a conversation rather than
    listed by hand, so a different template brings its own tokens.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False,
    )
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    # The unknown token is "added" too, and a tokenizer that does not know
    # the placeholder words maps them to it; it is not part of any format.
    added = set(tokenizer.get_added_vocab().values()) - {tokenizer.unk_token_id}
    return sorted({i for i in ids if i in added})


def trainable_token_target(model: object, token_ids: list[int]):
    """The `trainable_token_indices` value that reaches the OUTPUT head too.

    Where the embedding and the output head share one tensor (Qwen3-4B), a
    list of ids is enough: PEFT follows the tie. Where they are separate
    (Qwen3-8B), training the embedding rows alone changes how the token is
    read, not whether it is written — both modules must be named.
    """
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head is None or head.weight is emb.weight:
        return list(token_ids)
    names = {id(m): n.rsplit(".", 1)[-1] for n, m in model.named_modules()}
    return {names[id(emb)]: list(token_ids), names[id(head)]: list(token_ids)}


def build_sft_features(
    examples: list[dict[str, str]],
    tokenizer: object,
    max_length: int = 512,
) -> list[dict[str, list[int]]]:
    """Tokenize instruction/output pairs so that only the answer is learnt.

    Two things the previous version got wrong, both of which made the loss
    measure something other than "does the model answer as it should":

    * **Padding was trained on.** Every example was padded to 512 tokens and
      the labels were a copy of the input ids, so a nine-token answer came
      with some five hundred pad tokens — the EOS token, on Qwen — each
      counted in the loss. The gradient was mostly "predict EOS after EOS".
      Here nothing is padded at all; `collate_sft` pads per batch and masks
      the padding out of the labels.
    * **The question was trained on.** The user's turn counted in the loss
      like the answer, teaching the model to write questions. Here the
      prompt — everything up to and including the assistant header — is
      masked with `IGNORE_INDEX`, and the loss sees only the answer and the
      end-of-turn token that teaches it to stop.

    The prompt/answer boundary comes from the template itself: the prompt is
    rendered with ``add_generation_prompt=True``, which is exactly what the
    engine sends at inference, and must be a prefix of the full rendering.
    The two halves are tokenized separately and concatenated, so a merge
    across the boundary cannot shift it.

    Args:
        examples: dicts with ``instruction`` and ``output``.
        tokenizer: tokenizer with a chat template (see `ensure_chat_template`).
        max_length: longest sequence kept. An example that does not fit is
            dropped whole, not truncated: a truncated answer loses its
            end-of-turn token, and a model trained on those learns not to
            stop.

    Returns:
        One dict per kept example with ``input_ids``, ``attention_mask`` and
        ``labels`` as plain lists of equal length.

    Raises:
        ValueError: the template does not render the prompt as a prefix of
            the conversation, or no example fits in ``max_length``.
    """
    features = []
    too_long = 0
    for ex in examples:
        prompt_msgs = [{"role": "user", "content": ex["instruction"]}]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": ex["output"]}]
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(full_msgs, tokenize=False)
        if not full.startswith(prompt):
            raise ValueError(
                "the chat template does not render the prompt as a prefix of "
                "the full conversation, so the answer cannot be separated "
                f"from the question.\nprompt: {prompt!r}\nfull:   {full!r}"
            )

        # add_special_tokens=False: the template already writes whatever BOS
        # the model expects; letting the tokenizer add another would train
        # on a sequence the engine never produces.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(answer_ids) > max_length:
            too_long += 1
            continue

        input_ids = prompt_ids + answer_ids
        features.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
        })

    if too_long:
        logger.warning(
            "Dropped %d of %d examples longer than %d tokens",
            too_long, len(examples), max_length,
        )
    if not features:
        raise ValueError(f"no identity example fits in max_length={max_length}")
    return features


def collate_sft(features: list[dict[str, list[int]]], pad_token_id: int) -> dict:
    """Pad a batch to its longest member; padding never reaches the loss.

    Args:
        features: output of `build_sft_features`.
        pad_token_id: id written into padded ``input_ids`` positions.

    Returns:
        Tensors ``input_ids``, ``attention_mask`` and ``labels``, with
        ``labels`` set to `IGNORE_INDEX` wherever the input is padding.
    """
    import torch

    width = max(len(f["input_ids"]) for f in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad = width - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * pad)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


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
    # Fail fast on a misspelled dataset path before paying for imports, GPU
    # memory, and hours of training: a non-empty dataset_path that points
    # nowhere must be an error, not a silent switch to synthetic examples.
    if config.dataset_path and not Path(config.dataset_path).exists():
        raise FileNotFoundError(f"identity dataset not found: {config.dataset_path}")
    if config.instruction_path and not Path(config.instruction_path).exists():
        raise FileNotFoundError(f"instruction data not found: {config.instruction_path}")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    if not config.model_path:
        raise ValueError("model_path is required for identity fine-tuning")

    logger.info("Starting identity fine-tuning")
    logger.info("  Model: %s", config.model_path)
    logger.info("  Identity: %s", config.identity_name)
    logger.info("  Languages: %s", ", ".join(config.languages))
    logger.info("  LoRA rank: %d, alpha: %d", config.lora_rank, config.lora_alpha)

    # Generate or load training data
    if config.dataset_path and Path(config.dataset_path).exists():
        with open(config.dataset_path, encoding="utf-8") as f:
            examples = json.load(f)
        logger.info("Loaded %d identity examples from %s", len(examples), config.dataset_path)
    else:
        examples = generate_identity_dataset(config)
        logger.info("Generated %d identity training examples", len(examples))
    examples = examples * max(1, config.identity_repeat)
    if config.instruction_path:
        domain = load_pairs(config.instruction_path)
        logger.info(
            "Loaded %d instruction pairs from %s; identity examples x%d = %d",
            len(domain), config.instruction_path, config.identity_repeat, len(examples),
        )
        examples = examples + domain

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if ensure_chat_template(tokenizer):
        logger.info("  Tokenizer had no chat template: installed ChatML")
    if config.default_system:
        if not set_default_system(tokenizer, config.default_system):
            raise ValueError("default_system given, but the chat template has no "
                             "default_system_message to replace")
        logger.info("  Default system prompt replaced (%d characters)", len(config.default_system))

    # bf16 where the GPU has it (every A100): Qwen activations overflow fp16
    # often enough that a run can go non-finite for no reason in the data.
    # fp32 on CPU, which cannot train in half precision at all.
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32

    model = load_text_model(
        config.model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )

    extra = {}
    if config.train_format_tokens:
        import peft

        if tuple(int(x) for x in peft.__version__.split(".")[:2]) < (0, 15):
            raise RuntimeError(
                f"peft {peft.__version__} cannot train token rows; 0.15 or newer "
                "is needed for train_format_tokens"
            )
        token_ids = format_token_ids(tokenizer)
        if token_ids:
            extra["trainable_token_indices"] = trainable_token_target(model, token_ids)
            logger.info(
                "  Training the rows of the chat-format tokens %s",
                tokenizer.convert_ids_to_tokens(token_ids),
            )
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=lora_dropout_for(model),
        target_modules=lora_target_modules(model),
        **extra,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # With a frozen base the embeddings output needs grad, or
        # checkpointed blocks have nothing to backpropagate into the LoRA.
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  LoRA parameters: %s / %s (%.2f%%)",
        f"{trainable_params:,}", f"{total_params:,}",
        100 * trainable_params / total_params,
    )

    features = build_sft_features(examples, tokenizer, config.max_length)
    answer_tokens = sum(sum(t != IGNORE_INDEX for t in f["labels"]) for f in features)
    logger.info(
        "  Training on %d examples, %d answer tokens (prompts and padding masked)",
        len(features), answer_tokens,
    )

    class IdentityDataset(torch.utils.data.Dataset):
        def __init__(self, features):
            self.features = features

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]

    dataset = IdentityDataset(features)

    output_dir = config.output_dir or str(Path(config.model_path).parent / "identity-lora")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=bf16,
        fp16=fp16,
        logging_steps=10,
        save_strategy="steps" if config.save_steps else "epoch",
        save_steps=config.save_steps or 500,
        save_total_limit=2,
        report_to="none",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_sft(batch, tokenizer.pad_token_id),
    )

    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if last:
        logger.info("Resuming from %s", last)
    logger.info("Starting LoRA training...")
    trainer.train(resume_from_checkpoint=last)

    # Save adapter
    adapter_path = str(Path(output_dir) / "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved to: %s", adapter_path)

    return adapter_path


def _safetensors_header(path: Path) -> tuple[dict, int]:
    """A .safetensors file's header and where its data starts.

    The format is an 8-byte little-endian header length, a JSON header and
    the raw bytes: read here with the standard library, so carrying tensors
    needs neither torch nor the safetensors package.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _weight_map(model_dir: Path) -> dict[str, str]:
    """Tensor name -> shard file of a checkpoint directory, sharded or not."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    single = model_dir / "model.safetensors"
    if single.is_file():
        header, _ = _safetensors_header(single)
        return {k: single.name for k in header if k != "__metadata__"}
    return {}


def carry_missing_tensors(
    source_dir: str | Path,
    output_dir: str | Path,
    prefixes: tuple[str, ...] = ("mtp.",),
    shard: str = "model-carried.safetensors",
) -> list[str]:
    """Copy into ``output_dir`` the tensors of ``source_dir`` it lacks, by prefix.

    Qwen3.5 checkpoints carry a multi-token-prediction head (``mtp.*``) that
    the transformers model class does not load, so a merged checkpoint saved
    from it has none -- while its config still says ``mtp_num_hidden_layers:
    1``. llama.cpp then converts 33 blocks, finds no tensors for the last one
    and refuses the GGUF ("tensor 'blk.32.attn_norm.weight' not found"): on
    2026-10-11 every exam of the Qwen3.5 students died that way at load. The
    head is copied unchanged: it only drafts tokens for speculative decoding,
    which the model then checks, so an untrained head costs speed, never
    correctness. Tensors are copied byte for byte, one at a time, into one
    extra shard listed in the index. Returns the names copied (none when
    there is nothing to carry, so running it twice changes nothing).
    """
    src, out = Path(source_dir), Path(output_dir)
    src_map, out_map = _weight_map(src), _weight_map(out)
    missing = sorted(k for k in src_map if k.startswith(prefixes) and k not in out_map)
    if not missing:
        return []

    entries, offset = [], 0
    for name in missing:
        header, base = _safetensors_header(src / src_map[name])
        info = header[name]
        start, end = info["data_offsets"]
        entries.append((name, src / src_map[name], base + start, end - start, info))
    new_header: dict = {"__metadata__": {"format": "pt"}}
    for name, _, _, size, info in entries:
        new_header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(new_header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)

    tmp = out / (shard + ".partial")
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(raw)))
        w.write(raw)
        for _, path, at, size, _ in entries:
            with open(path, "rb") as r:
                r.seek(at)
                w.write(r.read(size))
    os.replace(tmp, out / shard)

    index_path = out / "model.safetensors.index.json"
    index = (json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file()
             else {"metadata": {}, "weight_map": dict(out_map)})
    index["weight_map"].update({name: shard for name in missing})
    if "total_size" in index.get("metadata", {}):
        index["metadata"]["total_size"] += offset
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    logger.info("  Carried %d tensors (%s) from %s", len(missing), ", ".join(prefixes), src)
    return missing


def merge_identity_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str | None = None,
) -> str:
    """Merge a LoRA adapter into its base weights and save a full checkpoint.

    A LoRA adapter is a set of low-rank deltas, not a model: `export_gguf`
    (and llama.cpp's converter behind it) reads a full set of base weights and
    has no notion of an adapter directory. Merging is therefore what makes the
    identity actually reach the exported GGUF — without this step the adapter
    is written to disk and then ignored, which is exactly what
    `run_pipeline` used to do.

    Args:
        base_model_path: The model the adapter was trained on.
        adapter_path: Directory produced by `fine_tune_identity`.
        output_path: Where to write the merged checkpoint. Defaults to a
            sibling ``identity-merged`` directory next to the adapter.

    Returns:
        Path to the merged model directory, ready for `export_gguf`.
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    parent = os.path.dirname(os.fspath(adapter_path).rstrip("/\\"))
    out = output_path or posixpath.join(parent, "identity-merged")

    logger.info("Merging LoRA adapter into base weights")
    logger.info("  Base:    %s", base_model_path)
    logger.info("  Adapter: %s", adapter_path)
    logger.info("  Output:  %s", out)

    # `device_map=None` keeps this a CPU-only operation: the merge is a
    # weight-space addition, so it does not need the GPU and can run on the
    # export box alongside the GGUF conversion. dtype="auto" keeps the weights
    # in the dtype they are stored in (bf16 for every Qwen3): the default is
    # float32, which doubles the memory, and an 8B model in float32 is 32 GB —
    # more than the 30 GB a serial-partition job may have.
    base = load_text_model(base_model_path, dtype="auto")
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    Path(out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    # What the model class did not load is not in `merged` either: Qwen3.5's
    # MTP head. Without it the GGUF of the merge does not load at all.
    carry_missing_tensors(base_model_path, out)

    # The tokenizer must travel with the merged weights — the converter reads
    # it from the same directory. Prefer the adapter's copy (saved by
    # `fine_tune_identity`, so it matches any tokens the identity added) and
    # fall back to the base model's.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(out)

    logger.info("Merged model saved to: %s", out)
    return out

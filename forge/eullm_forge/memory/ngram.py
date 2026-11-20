"""A hashed n-gram memory grafted onto a trained causal language model.

The memory is a large table read by hash, not computed: for every token, the
2-grams and 3-grams that end on it are hashed, several times each, into rows
of one embedding table; the rows are concatenated, compared with the model's
own hidden state at that point (a gate), and the gated value is added to the
residual stream after one early decoder layer. It is the design of the
per-layer n-gram embeddings of Qwen3.8-Flash-Next (`Qwen4ExpTextPLELayer` in
transformers 5.22: one table after layer 2, 2- and 3-grams, 8 hashes each,
a key/query gate, a short dilated convolution), at a size a 4-30B model can
carry and trained on a model that never had it
(`docs/research/legal-memory-plan.md`, H2; the Fast Lane plan, line A).

Grafting needs two things the original does not:

* **It starts from the unchanged model.** The output is multiplied by a
  scalar initialised at zero, so the model with the memory attached answers
  exactly as without it until training moves that scalar.
* **It lives in its own file.** The table, its hashes and its projections are
  saved apart from the model (`save`/`load`), so the knowledge can be
  retrained alone when the law changes (H3) and shipped as a separate,
  versioned file.

The table is an ``nn.Embedding`` with ``sparse=True``: a step touches only
the rows its batch read, and `torch.optim.SparseAdam` keeps the update to
those rows.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

#: Bumped when the hashing or the layout changes: a memory file trained under
#: one version is not readable under another.
VERSION = 1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    return all(n % d for d in range(3, r + 1, 2))


def next_prime(n: int) -> int:
    """The smallest prime >= n."""
    while not _is_prime(n):
        n += 1
    return n


@dataclass
class NGramMemoryConfig:
    """Shape of the memory. ``vocab_size`` and ``hidden_size`` come from the model."""

    vocab_size: int
    hidden_size: int
    #: index (0-based) of the decoder layer whose output the memory is added to
    layer: int = 1
    ngram_size: int = 3
    heads_per_ngram: int = 8
    #: rows of each head's slice of the table (rounded up to a prime per head)
    rows_per_head: int = 262_144
    #: width of the concatenated n-gram embedding; one head reads embed_dim / heads
    embed_dim: int = 1024
    #: short dilated depthwise convolution after the gate (0 = none)
    conv_kernel: int = 4
    seed: int = 0
    version: int = VERSION

    @property
    def n_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.n_heads

    def check(self) -> None:
        if self.ngram_size < 2:
            raise ValueError("ngram_size must be at least 2")
        if self.embed_dim % self.n_heads:
            raise ValueError(f"embed_dim {self.embed_dim} is not divisible by {self.n_heads} heads")
        if self.version != VERSION:
            raise ValueError(f"memory version {self.version}, this code reads {VERSION}")


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(x.dtype)


class NGramMemory(nn.Module):
    """The memory itself: hashing, table, gate, convolution, zero-initialised scale."""

    def __init__(self, config: NGramMemoryConfig):
        super().__init__()
        config.check()
        self.config = config
        c = config
        # One prime per head, so two heads never share a modulus.
        sizes, offsets, start, p = [], [], 0, c.rows_per_head
        for _ in range(c.n_heads):
            p = next_prime(p + 1 if sizes else p)
            sizes.append(p)
            offsets.append(start)
            start += p
        self.register_buffer("head_sizes", torch.tensor(sizes, dtype=torch.long), persistent=False)
        self.register_buffer("head_offsets", torch.tensor(offsets, dtype=torch.long),
                             persistent=False)
        # Odd multipliers per head and position, fixed by the seed.
        g = torch.Generator().manual_seed(c.seed)
        mult = torch.randint(1, 2**31 - 1, (c.n_heads, c.ngram_size), generator=g) | 1
        self.register_buffer("multipliers", mult.long(), persistent=False)
        self.table = nn.Embedding(start, c.head_dim, sparse=True)
        nn.init.normal_(self.table.weight, std=0.02)
        self.key_proj = nn.Linear(c.embed_dim, c.hidden_size, bias=False)
        self.value_proj = nn.Linear(c.embed_dim, c.hidden_size, bias=False)
        self.norm_key = _RMSNorm(c.hidden_size)
        self.norm_query = _RMSNorm(c.hidden_size)
        if c.conv_kernel:
            self.norm_conv = _RMSNorm(c.hidden_size)
            self.conv = nn.Conv1d(c.hidden_size, c.hidden_size, kernel_size=c.conv_kernel,
                                  groups=c.hidden_size, dilation=c.ngram_size, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ hashing
    @property
    def context(self) -> int:
        """Tokens of history the hashes of the next token need."""
        return self.config.ngram_size - 1

    @property
    def conv_context(self) -> int:
        c = self.config
        return (c.conv_kernel - 1) * c.ngram_size if c.conv_kernel else 0

    def row_ids(self, tokens: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        """Table rows, ``(batch, seq, heads)``, for ``tokens`` ``(batch, seq)``.

        ``history`` holds the ``context`` tokens before ``tokens`` (incremental
        decoding); without it the sequence starts there, and the missing
        positions hash as a token outside the vocabulary.
        """
        c = self.config
        b, t = tokens.shape
        pad = torch.full((b, self.context), c.vocab_size, dtype=torch.long, device=tokens.device)
        if history is not None:
            pad = history.to(tokens.device).long()
        full = torch.cat([pad, tokens.long()], dim=1)
        # shifted[k] is the token k positions back from each position of `tokens`
        shifted = [full[:, self.context - k: self.context - k + t] for k in range(c.ngram_size)]
        ids = []
        head = 0
        for n in range(2, c.ngram_size + 1):
            for _ in range(c.heads_per_ngram):
                mixed = torch.zeros_like(tokens, dtype=torch.long)
                for k in range(n):
                    mixed = mixed + shifted[k] * self.multipliers[head, k]
                ids.append(torch.remainder(mixed, self.head_sizes[head]) + self.head_offsets[head])
                head += 1
        return torch.stack(ids, dim=-1)

    # ------------------------------------------------------------------ forward
    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor,
                history: torch.Tensor | None = None,
                conv_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """What to add to ``hidden`` ``(batch, seq, hidden_size)``, and the new conv state."""
        rows = self.row_ids(tokens, history).to(self.table.weight.device)
        emb = self.table(rows).flatten(-2)  # (b, t, embed_dim)
        emb = emb.to(device=hidden.device, dtype=self.key_proj.weight.dtype)
        key = self.norm_key(self.key_proj(emb))
        value = self.value_proj(emb)
        query = self.norm_query(hidden.to(key.dtype))
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.config.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value
        # Positions that are padding (hashed as the out-of-vocabulary token by
        # AttachedMemory) carry nothing, into the residual or the convolution:
        # otherwise a left-padded prompt reads differently from the same
        # prompt unpadded, as Flash-Next's apply_mask_to_padding_states avoids.
        valid = (tokens < self.config.vocab_size).to(device=gated.device)
        gated = gated * valid.unsqueeze(-1).to(gated.dtype)
        out = gated
        new_state = None
        if self.config.conv_kernel:
            normed = self.norm_conv(gated) * valid.unsqueeze(-1).to(gated.dtype)
            lead = self.conv_context
            if conv_state is None:
                conv_state = normed.new_zeros(normed.shape[0], lead, normed.shape[-1])
            seq = torch.cat([conv_state.to(normed.dtype), normed], dim=1)
            new_state = seq[:, -lead:].detach() if lead else None
            conv = F.silu(self.conv(seq.transpose(1, 2))).transpose(1, 2)
            out = out + conv[:, -normed.shape[1]:]
        return (self.scale * out).to(hidden.dtype), new_state

    # ------------------------------------------------------------------ files
    def save(self, directory: str | Path) -> None:
        """The memory as ``config.json`` + ``memory.safetensors`` in ``directory``."""
        from safetensors.torch import save_file

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(asdict(self.config), indent=1), encoding="utf-8")
        state = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, str(d / "memory.safetensors"), metadata={"format": "eullm-ngram-memory",
                                                                   "version": str(VERSION)})

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device | None = None) -> "NGramMemory":
        from safetensors.torch import load_file

        d = Path(directory)
        config = NGramMemoryConfig(**json.loads((d / "config.json").read_text(encoding="utf-8")))
        memory = cls(config)
        memory.load_state_dict(load_file(str(d / "memory.safetensors")))
        return memory.to(device) if device is not None else memory

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------- grafting
def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The model's list of decoder layers, wherever the wrapper keeps it."""
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if n is None:
        text = getattr(getattr(model, "config", None), "text_config", None)
        n = getattr(text, "num_hidden_layers", None)
    for name, module in model.named_modules():
        if (isinstance(module, nn.ModuleList) and name.endswith("layers")
                and (n is None or len(module) == n)):
            return module
    raise ValueError("no list of decoder layers found in the model")


class AttachedMemory:
    """Hooks that add a `NGramMemory` to a model's forward pass.

    A pre-hook on the model captures ``input_ids`` (and the attention mask,
    so left padding hashes as "no token"), and a hook on decoder layer
    ``config.layer`` adds the memory's output to that layer's hidden states.
    During incremental decoding the last tokens and the convolution state are
    carried from one call to the next; a call that starts a sequence
    (an empty cache, or no cache at all) starts them afresh.
    """

    def __init__(self, model: nn.Module, memory: NGramMemory):
        self.model, self.memory = model, memory
        layers = decoder_layers(model)
        if not 0 <= memory.config.layer < len(layers):
            raise ValueError(f"layer {memory.config.layer} outside the model's {len(layers)}")
        self._tokens = None
        self._history = None
        self._conv = None
        self._handles = [
            model.register_forward_pre_hook(self._capture, with_kwargs=True),
            layers[memory.config.layer].register_forward_hook(self._add),
        ]

    def _capture(self, module, args, kwargs):
        tokens = kwargs.get("input_ids")
        if tokens is None and args:
            tokens = args[0]
        if tokens is None:
            raise ValueError("the n-gram memory needs input_ids, not inputs_embeds")
        mask = kwargs.get("attention_mask")
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            tokens = torch.where(mask[:, -tokens.shape[1]:].bool(), tokens,
                                 torch.full_like(tokens, self.memory.config.vocab_size))
        # A new sequence, or the next tokens of one being decoded? The cache
        # tells: transformers 5.22's generate passes past_key_values and no
        # cache_position, and its position_ids do not start at 0 on a
        # left-padded row. No cache at all means a whole sequence every call.
        past = kwargs.get("past_key_values")
        pos = kwargs.get("cache_position")
        if past is not None and hasattr(past, "get_seq_length"):
            fresh = past.get_seq_length() == 0
        elif pos is not None:
            fresh = int(pos.reshape(-1)[0]) == 0
        else:
            fresh = True
        if fresh:
            self._history, self._conv = None, None
        self._tokens = tokens
        return None

    def _add(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # History is carried only while decoding (no gradient). In training
        # every forward is a whole sequence, and gradient checkpointing runs
        # this hook again in the backward pass: carried state would make the
        # recomputation hash other tokens than the forward did.
        if torch.is_grad_enabled():
            delta, _ = self.memory(self._tokens, hidden)
            hidden = hidden + delta.to(hidden.device)
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
        delta, conv = self.memory(self._tokens, hidden, self._history, self._conv)
        ctx = self.memory.context
        joined = self._tokens if self._history is None else torch.cat(
            [self._history.to(self._tokens.device), self._tokens], dim=1)
        if joined.shape[1] >= ctx:
            self._history = joined[:, -ctx:].detach()
        else:
            pad = torch.full((joined.shape[0], ctx - joined.shape[1]),
                             self.memory.config.vocab_size, dtype=joined.dtype,
                             device=joined.device)
            self._history = torch.cat([pad, joined], dim=1)
        self._conv = conv
        hidden = hidden + delta.to(hidden.device)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def attach(model: nn.Module, memory: NGramMemory) -> AttachedMemory:
    """Graft ``memory`` onto ``model``; ``.detach()`` on the result takes it off."""
    return AttachedMemory(model, memory)


def memory_config_for(model: nn.Module, **overrides) -> NGramMemoryConfig:
    """A config sized for ``model`` (its vocabulary and hidden size), with overrides."""
    cfg = model.config
    text = getattr(cfg, "text_config", None) or cfg
    return NGramMemoryConfig(vocab_size=text.vocab_size, hidden_size=text.hidden_size, **overrides)

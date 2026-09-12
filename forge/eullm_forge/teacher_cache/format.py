"""On-disk format for the teacher's top-K distributions.

ADR-001 Part 4 specifies it; this implements it. The cache lets one teacher
pass serve any number of student runs, which is what makes design A pay —
but only if what comes back is exactly what the teacher said, and the
failure mode when it is not is silent. An off-by-one in the
`logits[t] -> target[t+1]` alignment does not raise the loss in any obvious
way; it poisons the whole cache and is discovered after the node-hours are
spent. So the invariants are checked on write, not documented and hoped for.

**What is stored**, per sequence: the token ids, the top-K token ids, the
top-K *raw logits* in bf16, and the `logZ_T` normalisers in fp32 for each
temperature. Plus provenance: teacher checkpoint, adapter, precision,
tokenizer fingerprint, git commit.

**What is deliberately not stored**, because a derived field that is written
down anyway is a field that can contradict its own source:

* the position of a token — it is the index within the sequence record;
* the target token — it is `token_ids[i + 1]` for the distribution at `i`,
  and making that structural rather than stored is what makes the alignment
  checkable instead of conventional;
* normalised probabilities — they would fix a temperature at write time,
  which is the thing `logZ_T` exists to avoid;
* residual mass — `1 - sum_topK exp((logit - logZ_T)/T)`, exact from what is
  already here.

**bf16 and not fp16**, and the reason is not dynamic range: the teacher
computes in bf16, so its logits already carry exactly that precision.
Storing them in bf16 is lossless with respect to the source, at identical
size. fp16's two extra mantissa bits would hold no information.

Sharded, never one file: 700 M positions at K=64 is ~277 GB.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"
SHARD_GLOB = "shard-*.npz"

# T=1 is the objective's own temperature; 2 and 4 are the ones a distillation
# schedule realistically reaches. Storing three fp32 per position costs ~12
# bytes against re-running the teacher to change a hyperparameter.
DEFAULT_TEMPERATURES: tuple[float, ...] = (1.0, 2.0, 4.0)


# ── bf16 <-> fp32, without a dependency ──────────────────────────────────
# numpy has no bfloat16, so the bits are carried as uint16. This is a
# reinterpretation, not a conversion: the stored value IS the teacher's,
# truncated exactly where bf16 truncates.

def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> the top 16 bits, rounded to nearest even.

    Truncating instead of rounding would bias every logit downward in
    magnitude, which is small per value and systematic across 700 M of them.
    """
    a = np.ascontiguousarray(x, dtype=np.float32)
    u = a.view(np.uint32)
    rounded = u + (((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF))
    return (rounded >> 16).astype(np.uint16)


def from_bf16_bits(b: np.ndarray) -> np.ndarray:
    """The top 16 bits back to fp32. Exact — bf16 is a prefix of fp32."""
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


@dataclass(frozen=True)
class CacheProvenance:
    """Everything needed to say which teacher produced this, and on what.

    A cache outlives the session that wrote it and will be read by a run
    nobody remembers configuring. `tokenizer_fingerprint` is the field that
    earns its place: two tokenizers can agree on a vocabulary size of 151,936
    and differ in merges or special-token ids, and the resulting corruption
    looks like a bad student rather than a bad cache.
    """

    teacher_model: str
    teacher_precision: str
    tokenizer_fingerprint: str
    top_k: int
    # Not decoration: it is what makes the logZ check two-sided. Without it a
    # normaliser can only be shown to be too small, and the commonest way to
    # get one wrong — reusing another temperature's — makes it too large.
    vocab_size: int
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES
    teacher_adapter: str | None = None
    git_commit: str | None = None
    config_digest: str | None = None
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["temperatures"] = list(self.temperatures)
        return d

    @classmethod
    def from_json(cls, d: dict) -> CacheProvenance:
        d = dict(d)
        d["temperatures"] = tuple(float(t) for t in d["temperatures"])
        return cls(**d)


@dataclass(frozen=True)
class CachedSequence:
    """One sequence's worth of teacher output.

    `topk_logits` are raw logits, not log-probabilities. `logz[:, j]` is the
    normaliser of the FULL vocabulary at `temperatures[j]` — which is the
    whole point: a truncated top-K cannot be turned back into a distribution
    without it, and the softmax at T=2 cannot be recovered from the T=1
    normaliser.
    """

    seq_id: str
    token_ids: np.ndarray      # int32, (L,)
    topk_ids: np.ndarray       # uint32, (L-1, K)
    topk_logits: np.ndarray    # float32 decoded from bf16, (L-1, K)
    logz: np.ndarray           # float32, (L-1, n_temperatures)
    temperatures: tuple[float, ...]

    @property
    def targets(self) -> np.ndarray:
        """The token each stored distribution predicts.

        Derived, never stored. `logits[t]` predicts `token_ids[t+1]`, so this
        is `token_ids[1:]` — and because the writer requires exactly `L-1`
        distributions for `L` tokens, an off-by-one cannot be written down in
        the first place.
        """
        return self.token_ids[1:]

    def probs(self, temperature: float) -> np.ndarray:
        """Top-K probabilities at one of the cached temperatures."""
        j = self._temperature_index(temperature)
        return np.exp(self.topk_logits / temperature - self.logz[:, j : j + 1])

    def residual_mass(self, temperature: float) -> np.ndarray:
        """Probability mass outside the top-K, per position.

        Derived rather than stored, and useful: it is the truncation error of
        Part 5 measured on the cache itself.
        """
        return 1.0 - self.probs(temperature).sum(axis=1)

    def _temperature_index(self, temperature: float) -> int:
        for j, t in enumerate(self.temperatures):
            if abs(t - temperature) < 1e-9:
                return j
        raise KeyError(
            f"temperature {temperature} not cached; have {list(self.temperatures)}"
        )


def validate_sequence(
    token_ids: np.ndarray,
    topk_ids: np.ndarray,
    topk_logits: np.ndarray,
    logz: np.ndarray,
    n_temperatures: int,
) -> None:
    """Refuse to write anything that cannot be read back correctly.

    Every check here corresponds to a failure that is silent at training
    time. They run on write because that is the only moment the live teacher
    is still around to be re-asked.
    """
    n = len(token_ids) - 1
    if n < 1:
        raise ValueError("a sequence needs at least 2 tokens to predict anything")
    if topk_ids.shape[0] != n:
        raise ValueError(
            f"alignment: {len(token_ids)} tokens give {n} predictable positions, "
            f"got {topk_ids.shape[0]} distributions"
        )
    if topk_logits.shape != topk_ids.shape:
        raise ValueError(
            f"top-K ids {topk_ids.shape} and logits {topk_logits.shape} disagree"
        )
    if logz.shape != (n, n_temperatures):
        raise ValueError(
            f"logZ must be ({n}, {n_temperatures}), got {logz.shape}"
        )
    if not np.isfinite(topk_logits).all() or not np.isfinite(logz).all():
        raise ValueError("non-finite logits or logZ")
    if (np.diff(topk_logits, axis=1) > 1e-3).any():
        raise ValueError("top-K logits must be sorted descending")


def check_normalisers(
    topk_logits: np.ndarray,
    logz: np.ndarray,
    temperatures: tuple[float, ...],
    vocab_size: int | None = None,
) -> None:
    """Bound `logZ_T` from both sides, because each side catches a different bug.

    **Below**: `logZ_T` is a logsumexp over the whole vocabulary, so it cannot
    be smaller than the same quantity over a subset of it. A normaliser
    computed over the top-K instead of the full vocabulary fails here — and
    that cache would read back cleanly while training against a distribution
    that sums to one over K tokens.

    **Above**: `logZ_T <= max(logit)/T + log(V)`, since V terms each at most
    `exp(max)`. This is the side that was missing, and a test found it. The
    commonest way to get a normaliser wrong is to reuse another temperature's,
    and since `logZ` falls as T rises, substituting T=1's value for T=4's
    makes it too *large* — invisible to the lower bound alone. Needs the
    vocabulary size, which is why provenance carries it.
    """
    for j, t in enumerate(temperatures):
        scaled = topk_logits / t
        m = scaled.max(axis=1)
        truncated = m + np.log(np.exp(scaled - m[:, None]).sum(axis=1))
        # A tolerance, because both sides are float and can agree to the last
        # bit when K covers essentially all the mass.
        if (truncated - logz[:, j] > 1e-3).any():
            worst = float((truncated - logz[:, j]).max())
            raise ValueError(
                f"logZ at T={t} is below the top-K's own logsumexp by {worst:.4g} "
                "— the normaliser does not belong to these logits"
            )
        if vocab_size:
            ceiling = m + np.log(float(vocab_size))
            if (logz[:, j] - ceiling > 1e-3).any():
                worst = float((logz[:, j] - ceiling).max())
                raise ValueError(
                    f"logZ at T={t} exceeds max(logit)/T + log(V) by {worst:.4g} "
                    f"over {vocab_size} tokens — likely another temperature's "
                    "normaliser, or logits and logZ from different batches"
                )


class LogitCacheWriter:
    """Writes shards and a manifest. Not thread-safe; one writer per rank.

    Shards are closed as they fill, so an interrupted teacher pass leaves
    every completed shard readable rather than one truncated file.
    """

    def __init__(
        self,
        root: str | Path,
        provenance: CacheProvenance,
        sequences_per_shard: int = 512,
        shard_prefix: str = "shard",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        self.sequences_per_shard = sequences_per_shard
        self.shard_prefix = shard_prefix
        self._buf: list[dict] = []
        self._shards: list[str] = []
        self._n_sequences = 0
        self._n_positions = 0
        self._closed = False

    def add(
        self,
        seq_id: str,
        token_ids: np.ndarray,
        topk_ids: np.ndarray,
        topk_logits: np.ndarray,
        logz: np.ndarray,
    ) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        token_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        topk_ids = np.ascontiguousarray(topk_ids, dtype=np.uint32)
        topk_logits = np.ascontiguousarray(topk_logits, dtype=np.float32)
        logz = np.ascontiguousarray(logz, dtype=np.float32)

        validate_sequence(
            token_ids, topk_ids, topk_logits, logz, len(self.provenance.temperatures)
        )
        if topk_ids.shape[1] != self.provenance.top_k:
            raise ValueError(
                f"K mismatch: manifest says {self.provenance.top_k}, "
                f"sequence has {topk_ids.shape[1]}"
            )
        check_normalisers(
            topk_logits,
            logz,
            self.provenance.temperatures,
            self.provenance.vocab_size,
        )

        self._buf.append(
            {
                "seq_id": seq_id,
                "token_ids": token_ids,
                "topk_ids": topk_ids,
                "topk_logits": to_bf16_bits(topk_logits),
                "logz": logz,
            }
        )
        self._n_sequences += 1
        self._n_positions += topk_ids.shape[0]
        if len(self._buf) >= self.sequences_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        name = f"{self.shard_prefix}-{len(self._shards):05d}.npz"
        path = self.root / name
        # Concatenated arrays plus offsets: one sequence per npz entry would
        # put millions of members in a zip, which no reader enjoys.
        offsets = np.zeros(len(self._buf) + 1, dtype=np.int64)
        tok_offsets = np.zeros(len(self._buf) + 1, dtype=np.int64)
        for i, rec in enumerate(self._buf):
            offsets[i + 1] = offsets[i] + rec["topk_ids"].shape[0]
            tok_offsets[i + 1] = tok_offsets[i] + rec["token_ids"].shape[0]
        np.savez(
            path,
            seq_ids=np.array([r["seq_id"] for r in self._buf], dtype=object),
            offsets=offsets,
            token_offsets=tok_offsets,
            token_ids=np.concatenate([r["token_ids"] for r in self._buf]),
            topk_ids=np.concatenate([r["topk_ids"] for r in self._buf]),
            topk_logits=np.concatenate([r["topk_logits"] for r in self._buf]),
            logz=np.concatenate([r["logz"] for r in self._buf]),
            allow_pickle=True,
        )
        self._shards.append(name)
        self._buf.clear()

    def close(self) -> dict:
        if self._closed:
            return self.manifest()
        self._flush()
        self._closed = True
        m = self.manifest()
        (self.root / MANIFEST).write_text(
            json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return m

    def manifest(self) -> dict:
        return {
            "format_version": 1,
            "provenance": self.provenance.to_json(),
            "shards": list(self._shards),
            "n_sequences": self._n_sequences,
            "n_positions": self._n_positions,
        }

    def __enter__(self) -> LogitCacheWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class LogitCacheReader:
    """Random access over the shards, one shard resident at a time."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no {MANIFEST} in {self.root} — an interrupted teacher pass "
                "leaves shards without one; re-run or write it by hand"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.provenance = CacheProvenance.from_json(self.manifest["provenance"])
        self._shards = self.manifest["shards"]
        self._counts: list[int] = []
        self._cached_shard: tuple[int, dict] | None = None
        for name in self._shards:
            with np.load(self.root / name, allow_pickle=True) as z:
                self._counts.append(len(z["offsets"]) - 1)
        self._cum = np.cumsum([0, *self._counts])

    def __len__(self) -> int:
        return int(self._cum[-1])

    def _load(self, shard: int) -> dict:
        if self._cached_shard and self._cached_shard[0] == shard:
            return self._cached_shard[1]
        with np.load(self.root / self._shards[shard], allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        self._cached_shard = (shard, data)
        return data

    def __getitem__(self, i: int) -> CachedSequence:
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        shard = int(np.searchsorted(self._cum, i, side="right") - 1)
        j = i - int(self._cum[shard])
        d = self._load(shard)
        a, b = int(d["offsets"][j]), int(d["offsets"][j + 1])
        ta, tb = int(d["token_offsets"][j]), int(d["token_offsets"][j + 1])
        return CachedSequence(
            seq_id=str(d["seq_ids"][j]),
            token_ids=d["token_ids"][ta:tb],
            topk_ids=d["topk_ids"][a:b],
            topk_logits=from_bf16_bits(d["topk_logits"][a:b]),
            logz=d["logz"][a:b],
            temperatures=self.provenance.temperatures,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

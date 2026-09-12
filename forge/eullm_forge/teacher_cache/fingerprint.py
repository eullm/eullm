"""A tokenizer fingerprint that does not stop at vocabulary size.

ADR-001 Part 6.3 is explicit about this, and the reason is that the failure
is invisible. Two tokenizers can agree on 151,936 entries and differ in a
handful of merges or in a special-token id; a cache written with one and
read with the other produces targets that are wrong for a small fraction of
positions, which looks like a student that trains slightly badly rather than
a cache that is corrupt.

Comparing the whole vocabulary mapping catches that. Hashing it rather than
storing it keeps the manifest small and makes the comparison a string
equality.
"""

from __future__ import annotations

import hashlib
import json


def fingerprint_vocab(
    vocab: dict[str, int], special_token_ids: dict[str, int | None]
) -> str:
    """A stable digest over the mapping and the special-token ids.

    Sorted by token id rather than by token text: the id is the thing the
    model indexes with, and two tokenizers that assign the same ids to the
    same strings are interchangeable for this purpose whatever order their
    files happen to list them in.
    """
    items = sorted(((int(i), str(t)) for t, i in vocab.items()))
    h = hashlib.sha256()
    h.update(b"eullm-tokenizer-fingerprint-v1\n")
    h.update(f"size={len(items)}\n".encode())
    for i, t in items:
        h.update(f"{i}\t{t}\n".encode("utf-8", errors="surrogatepass"))
    h.update(b"special\n")
    h.update(
        json.dumps(
            {k: (None if v is None else int(v)) for k, v in sorted(special_token_ids.items())},
            ensure_ascii=False,
        ).encode()
    )
    return h.hexdigest()


def fingerprint_tokenizer(tokenizer) -> str:
    """The same digest, taken from a HuggingFace tokenizer.

    Kept separate from `fingerprint_vocab` so the logic is testable without
    transformers installed — which is also how it runs in CI.
    """
    specials = {
        name: getattr(tokenizer, f"{name}_token_id", None)
        for name in ("bos", "eos", "pad", "unk", "sep", "cls", "mask")
    }
    return fingerprint_vocab(tokenizer.get_vocab(), specials)

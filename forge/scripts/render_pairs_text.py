#!/usr/bin/env python3
"""Instruction/answer pairs as plain text in a model's chat format, for replay.

    python forge/scripts/render_pairs_text.py --model <merged model dir> \\
        --pairs $WORK/eullm_runs/stage3/openbook-v06.jsonl \\
        --out $WORK/eullm_runs/memory-replay/openbook-v06.text.jsonl

Continued pretraining of r2l on the rewritten articles taught it more statute
knowledge than any model measured, and cost it the open book (2026-11-08,
`docs/legal-it-experiments.md` section 15): it had seen only raw article
text. Mixed into the same run, these texts are what it is asked to keep: each
pair becomes one ``{"text": ...}`` line holding the question with its
statute texts and the answer, exactly as the exam's prompt and the model's
own chat template write them (thinking off, the template's default system
prompt where it has one), which `ngram_train.py --data` packs beside the
articles.

Prints counts only, never a pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def render(tok, instruction: str, output: str) -> str:
    """One user turn and its answer, in ``tok``'s chat format, thinking off."""
    messages = [{"role": "user", "content": instruction},
                {"role": "assistant", "content": output}]
    return tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="model whose tokenizer and template to use")
    ap.add_argument("--pairs", nargs="+", required=True, help="instruction/output JSONL files")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from eullm_forge.identity import ensure_chat_template, load_pairs

    tok = AutoTokenizer.from_pretrained(a.model)
    ensure_chat_template(tok)
    pairs = [p for f in a.pairs for p in load_pairs(f)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    n_tokens = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for p in pairs:
            text = render(tok, p["instruction"], p["output"])
            n_tokens += len(tok(text, add_special_tokens=False)["input_ids"])
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    tmp.replace(a.out)
    print(f"[replay] {len(pairs):,} pairs, {n_tokens / 1e6:.1f}M tokens -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

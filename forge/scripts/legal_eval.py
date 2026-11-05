#!/usr/bin/env python3
"""Does a stage-3 model answer legal questions CORRECTLY? The gate before a release.

`chat_smoke.py` answers whether a model ends its turn. legal-it-4b v0.1c
passed it, 3 of 3, and then, read by a person, got both legal questions
wrong: article 2043 of the civil code became contractual good faith (artt.
1176 and 1375), and the ricorso straordinario was placed in article 111 of
the Constitution. Fluent, confident, invented. A format check cannot see
that, and "the loss went down" cannot either.

This asks every question of the held-out seed set
(`eullm_forge/eval/data/legal_it_heldout.seed.jsonl`), greedy, and scores
each answer by keyword coverage: the share of the terms a correct answer has
to contain that it does contain. Crude, and meant to be: it is
deterministic, it needs no judge model, and it ranks candidates on the same
questions. The answers are written out in full because the score only says
which model to READ first, not whether it is right. The seed items are
marked "da validare" and stay so until a lawyer has checked them.

Run it on every candidate AND on a reference, so a number means something:

    python forge/scripts/legal_eval.py <merged-dir or HF id> --label NAME \
        --csv legal-eval.csv --answers answers-NAME.jsonl

Qwen/Qwen3-4B-Instruct-2507 is the reference: same size, Qwen's own
instruction tuning. A vertical that does not beat it on its own domain is not
worth shipping. Exit status is 0 whatever the scores; this reports.

OPEN BOOK. With ``--norms`` every question is asked with the text of the
norms `NormIndex` retrieves for it placed in front (the named article when
the question names one, BM25 otherwise). Same model, same questions, the
text in hand: the difference between the two runs is how much of the error
is memory rather than understanding. The retrieved records are written into
the answers file, so a wrong answer can be told apart from a wrong retrieval.

    python forge/scripts/legal_eval.py <model> --label NAME-open \
        --norms "$CORPUS_DIR"/legislazione_*.chunks.jsonl ...

The keyword score is a first sort. `judge_answers.py` grades the answers
files against the references with a large model, which is the number to
decide on.

THE FILE THAT SHIPS. With ``--gguf`` the answers come from that GGUF through
llama-server (`eullm_forge.eval.gguf`) instead of the bf16 weights; the
positional model is still the merged directory it was made from, whose
tokenizer builds the very same prompts. Two runs differing only in --gguf
measure what quantization costs on these questions:

    python forge/scripts/legal_eval.py <merged-dir> --label NAME-q4 \
        --gguf <merged-dir's q4_k_m.gguf> --norms ...

ABSTENTION. Every answer is also checked by program
(`eullm_forge.eval.abstain`): whether it says the texts do not hold the
answer (``abstained``), and which articles it cites that neither the question
nor the texts it was given name (``unsourced_articles``). Three runs of the
same exam then measure honesty, not only knowledge:

* open book: abstaining is a lost answer, the rate should be near 0;
* ``--absent`` (with ``--norms``): the item's own article is taken out of the
  retrieved texts, and no note says so -- what a RAG gives when retrieval
  misses. Abstaining is the right answer;
* closed book: no texts at all, as in a plain chat. Abstaining, or at least
  citing no article from memory, is the right answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import (  # noqa: E402
    NormIndex,
    evaluate_qa,
    load_eval_set,
    load_seed,
    open_book_prompt,
)
from eullm_forge.eval.abstain import abstained, cited_articles, unsourced_articles  # noqa: E402
from eullm_forge.eval.retrieval import label as norm_label  # noqa: E402


def summary_row(label: str, model: str, report: dict, ended: int) -> list:
    """One CSV row per model: the numbers that rank candidates."""
    s = report["summary"]
    # None for an item the keyword metric does not measure, so it is not
    # counted here either.
    full = sum(1 for r in report["per_item"] if r["keyword_coverage"] == 1.0)
    return [time.strftime("%Y-%m-%dT%H:%M:%S"), label, model, s["n"],
            f"{s['keyword_coverage']:.3f}", full, ended,
            s.get("keyword_items", s["n"])]


# keyword_items is last so the columns before it keep their position: it says
# how many of `items` keyword_coverage and fully_covered were measured over,
# which is not all of them once an exam carries judge-graded items.
CSV_HEADER = ["timestamp", "label", "model", "items", "keyword_coverage",
              "fully_covered", "ended_turn", "keyword_items"]


def append_csv_row(path: Path, row: list[str]) -> None:
    """Append one summary row, writing the header if the file has none yet.

    Existence is not enough: `open("a")` creates the file and the writer's
    buffer is only flushed after the first row, so a run interrupted in that
    window leaves a 0-byte CSV. The next run sees a file that exists, skips
    the header, and the first model's row lands where the header belongs —
    `csv.DictReader` then reads no rows and the ranking is silently empty.
    The header is flushed before the row rather than after it.
    """
    needs_header = not (path.exists() and path.stat().st_size > 0)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if needs_header:
            w.writerow(CSV_HEADER)
            f.flush()
        w.writerow(row)
        f.flush()


def load_model(auto_cls, path: str, n_gpus: int, fallback_cls=None):
    """The model in bf16: on its one GPU, or spread over several.

    A 27B dense model is 54 GB in bf16 and a 35B MoE 70 GB, more than one
    64 GB A100 holds once the cache for a batch is added; with more than one
    GPU visible the layers are split across them (device_map="auto").

    ``fallback_cls`` is tried when ``auto_cls`` does not know the model's
    configuration: Ministral 3 ships only as an image-and-text model
    (Mistral3ForConditionalGeneration), which AutoModelForCausalLM refuses
    -- the exam of 2026-09-29 died on it after answering with two other
    models. It answers text prompts all the same.
    """
    import torch

    from eullm_forge.eval.placement import device_map_for

    def kwargs(cls) -> dict:
        # A model with parameters kept off the GPUs (Flash-Next's n-gram table)
        # gets its layers spread evenly: "auto" left its first GPU empty.
        out = {"dtype": torch.bfloat16}
        if n_gpus > 1:
            out["device_map"] = device_map_for(path, n_gpus, cls)
        return out

    try:
        model = auto_cls.from_pretrained(path, **kwargs(auto_cls))
    except ValueError as exc:
        if fallback_cls is None or "Unrecognized configuration class" not in str(exc):
            raise
        model = fallback_cls.from_pretrained(path, **kwargs(fallback_cls))
    return model.to("cuda") if n_gpus == 1 else model


def retrieve(index, item, k: int, absent: bool = False,
             note: bool = True) -> tuple[list[dict], str]:
    """The texts an item is asked with, and the note on a missing article.

    With ``absent`` the item's own article (``metadata`` code and articolo,
    as make_norm_exam.py writes them) is taken out -- continuation chunks
    too, hence ``articles_of`` -- and the next records retrieved take its
    place; no note is given, since a RAG whose retrieval missed gives none.
    """
    if not absent:
        found = index.search(item.question, k)
        return found, (index.missing_article_note(item.question) if note else "")
    code, number = item.metadata.get("code"), item.metadata.get("articolo")
    found = [r for r in index.search(item.question, k + 6)
             if not (r.get("code") == code and number in index.articles_of(r))]
    return found[:k], ""


def articles_in_hand(index, found: list[dict]) -> set[str]:
    """The articles given, and those their texts refer to: "ai sensi dell'art.
    1176" in a text given is a citation from the text, not from memory."""
    return ({a for r in found for a in index.articles_of(r)}
            | {a for r in found for a in cited_articles(r.get("text", ""))})


_REASONING = re.compile(r"(?s)^\s*(?:<think>|\[THINK\])(.*?)(?:</think>|\[/THINK\])\s*")


def split_reasoning(text: str) -> tuple[str, str, bool]:
    """(answer, reasoning, closed) of a reasoning model's output.

    Qwen writes <think>...</think> and Ministral-3-Reasoning [THINK]...[/THINK]
    before the answer; the judge must grade the answer only. A reasoning that
    never closed (cut at --max-new-tokens) leaves no answer at all.
    """
    text = text or ""
    m = _REASONING.match(text)
    if m:
        return text[m.end():].strip(), m.group(1).strip(), True
    for opener in ("<think>", "[THINK]"):
        if text.lstrip().startswith(opener):
            return "", text.lstrip()[len(opener):].strip(), False
    return text, "", True


def chat_prompt(tok, content: str, think: bool = False, system: str | None = None) -> str:
    """One user turn in the model's chat format, with thinking switched off.

    Qwen3's hybrid models (Qwen3-8B and the like) think by default: they open
    with a <think> block that eats the answer budget and is graded as the
    answer. ``enable_thinking=False`` is the switch their template reads; a
    template that has no such switch (Qwen3-4B-Instruct-2507, base models)
    ignores the extra variable, so every model is asked the same way.

    ``system`` goes before the question as a system message. Without one,
    Ministral's template puts in its own default (~525 tokens, "Le Chat"),
    which is what legal-it-8b was trained and examined with.
    """
    # Mistral Small 4 reads reasoning_effort ("none", its default, or "high")
    # instead; other templates ignore a variable they do not use.
    extra = {"reasoning_effort": "high"} if think else {}
    messages = [{"role": "system", "content": system}] if system else []
    return tok.apply_chat_template(messages + [{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=think, **extra)


REASONING_MARKERS = ("<think>", "</think>", "[THINK]", "[/THINK]")


def decode_answer(tok, ids: list[int], keep_reasoning: bool = False) -> str:
    """The text of generated ids, special tokens dropped.

    With ``keep_reasoning`` the reasoning markers stay, for `split_reasoning`:
    Mistral's [THINK] and [/THINK] are special tokens, which
    skip_special_tokens would delete, leaving the reasoning glued to the
    answer and graded as part of it.
    """
    if not keep_reasoning:
        return tok.decode(ids, skip_special_tokens=True).strip()
    text = tok.decode(ids, skip_special_tokens=False)
    for t in sorted(set(tok.all_special_tokens) - set(REASONING_MARKERS), key=len, reverse=True):
        text = text.replace(t, "")
    return text.strip()


def generate_answers(model, tok, prompts: list[str], *, batch_size: int,
                     max_new_tokens: int, end_ids: list[int],
                     keep_reasoning: bool = False) -> list[tuple[str, bool]]:
    """Greedy answers to already-templated prompts, ``batch_size`` at a time.

    Returns (answer, ended_its_turn) per prompt. In a batch a finished
    sequence is padded until the longest one ends, so "ended" is whether an
    end token was generated at all, and the answer is cut there.
    """
    import torch

    tok.padding_side = "left"          # decoder-only: pad before the prompt
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out_all = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=end_ids, pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:].tolist():
            cut = next((j for j, t in enumerate(row) if t in end_ids), None)
            text = decode_answer(tok, row[:cut] if cut is not None else row, keep_reasoning)
            out_all.append((text, cut is not None))
    return out_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="merged HF directory or hub id")
    ap.add_argument("--label", default="")
    ap.add_argument("--items", help="eval set JSONL (default: the legal-it seed)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--think", action="store_true",
                    help="let a reasoning model reason before it answers (give a larger "
                         "--max-new-tokens, e.g. 3000): the reasoning is kept apart in the "
                         "answers file and only the answer after it is scored and graded")
    ap.add_argument("--system-file", type=Path,
                    help="ask with this system prompt (e.g. forge/model_cards/legal-it-8b/"
                         "system-prompt.txt) instead of the template's default, which for "
                         "Ministral is its ~525-token \"Le Chat\" one")
    ap.add_argument("--csv", help="append the summary row here")
    ap.add_argument("--answers", help="write every answer, with its score, here")
    ap.add_argument("--norms", nargs="+",
                    help="legislazione_*.chunks.jsonl files: ask OPEN BOOK, "
                         "with the retrieved norms in the prompt")
    ap.add_argument("--k", type=int, default=3, help="norms retrieved per question")
    ap.add_argument("--no-note", action="store_true",
                    help="never tell the model an article it asks about is not in the texts: "
                         "RAG Enterprise does not, and on 2026-10-16 a model that answered every "
                         "absent article right with the note (0:0 against the published 8B) "
                         "lost 43 of them without it, inside RAG Enterprise")
    ap.add_argument("--gate", type=Path,
                    help="with --norms: a trained context gate (train_context_gate.py) decides "
                         "first whether the retrieved texts can answer each question")
    ap.add_argument("--gate-threshold", type=float,
                    help="with --gate: gate below this probability (default: gate.json's)")
    ap.add_argument("--gate-mode", choices=("answer", "note"), default="note",
                    help="with --gate, what a gated question gets: answer = a fixed reply "
                         "saying the texts do not hold it and what is needed, the model not "
                         "asked; note = a note in front of the texts, the model answering")
    ap.add_argument("--absent", action="store_true",
                    help="with --norms: take each item's own article out of the texts "
                         "(the abstention exam; see ABSTENTION above)")
    ap.add_argument("--embedder", help="with --norms: fuse BM25 with this embedding model "
                    "(eullm_forge.eval.dense); default BM25 alone")
    ap.add_argument("--reranker", help="with --embedder: reorder the fused list with this model")
    ap.add_argument("--retrieval-cache", type=Path,
                    help="where the document embeddings are cached")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="questions generated together (default: 16 on GPU, 1 on CPU)")
    ap.add_argument("--gguf", help="answer with this GGUF through llama-server instead of the "
                    "bf16 weights; the positional model gives the tokenizer")
    ap.add_argument("--llama-server", help="llama-server binary (default $LCPP_SERVER, else "
                    "$LCPP_DIR/build-cuda/bin/llama-server)")
    ap.add_argument("--runtime", choices=("llama-server", "eullm"), default="llama-server",
                    help="with --gguf: what serves it, llama.cpp's llama-server (the prompt as "
                         "token ids) or EULLM Engine (the runtime that ships; the prompt as raw "
                         "text, binary $EULLM_BIN or --eullm)")
    ap.add_argument("--eullm", help="with --runtime eullm: the eullm binary (default $EULLM_BIN)")
    ap.add_argument("--server-parallel", type=int, default=4,
                    help="with --gguf: questions decoded together")
    ap.add_argument("--server-ctx", type=int, default=16384,
                    help="with --gguf: context tokens per question (prompt + answer)")
    ap.add_argument("--quiet", action="store_true",
                    help="print totals only, never a question or an answer: for the "
                         "held-out exam, whose questions nobody improving the models "
                         "should read (see make_norm_exam.py)")
    args = ap.parse_args()
    if args.think and args.gguf and args.runtime == "eullm":
        ap.error("--think with --runtime eullm: the engine's text drops the reasoning "
                 "markers split_reasoning needs; use --runtime llama-server")
    if args.absent and not args.norms:
        ap.error("--absent takes the article out of retrieved texts: give --norms")
    # Dangling retrieval flags are silently ignored below -- the strings are
    # never touched without their parent, so a run meant to measure embedder
    # X comes back closed-book/BM25 with exit 0 and looks like a result.
    if args.embedder and not args.norms:
        ap.error("--embedder fuses BM25 with an embedding model: give --norms")
    if args.reranker and not args.embedder:
        ap.error("--reranker reorders the fused list: give --embedder")
    if args.retrieval_cache and not args.embedder:
        ap.error("--retrieval-cache holds document embeddings: give --embedder")
    if args.gate and not args.norms:
        ap.error("--gate judges the retrieved texts: give --norms")

    from transformers import AutoTokenizer

    items = load_eval_set(args.items) if args.items else load_seed()
    index = None
    if args.norms and args.embedder:
        from eullm_forge.eval.dense import build_hybrid
        index = build_hybrid(args.norms, args.embedder, reranker_id=args.reranker,
                             cache_dir=args.retrieval_cache)
    elif args.norms:
        index = NormIndex.from_files(args.norms)
    if index:
        from eullm_forge.eval.dense import describe
        print(f"[eval] open book: {len(index.records):,} legislation records, "
              f"{args.k} per question, retrieval {describe(index)}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    end_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0]

    system = args.system_file.read_text(encoding="utf-8").strip() if args.system_file else None
    if system:
        print(f"[eval] system prompt: {args.system_file} ({len(system)} chars)", flush=True)
    gate = None
    if args.gate:
        from eullm_forge.eval.gate import GATE_NOTE, ContextGate
        gate = ContextGate(args.gate, args.gate_threshold)
        print(f"[eval] context gate {args.gate}, threshold {gate.threshold:.3f}, "
              f"mode {args.gate_mode}", flush=True)
    contexts, in_hand, prompts, gate_p = {}, {}, [], {}
    for it in items:
        content = it.question
        if index:
            found, note = retrieve(index, it, args.k, absent=args.absent,
                                   note=not args.no_note)
            if gate:
                gate_p[it.id] = gate.prob(it.question, found)
                if gate_p[it.id] < gate.threshold and args.gate_mode == "note" and not note:
                    note = GATE_NOTE
            contexts[it.id] = [norm_label(r) for r in found] + ([note] if note else [])
            in_hand[it.id] = articles_in_hand(index, found)
            content = open_book_prompt(it.question, found, note=note)
        prompts.append(chat_prompt(tok, content, think=args.think, system=system))
    # Retrieval is done: its models leave the GPU to the model that answers.
    if hasattr(index, "release"):
        index.release()
    gated = {i for i, pr in gate_p.items() if pr < gate.threshold} if gate else set()
    if gate:
        gate.release()
        print(f"[eval] context gate: {len(gated)}/{len(items)} questions gated", flush=True)
    if args.gguf and args.runtime == "eullm":
        from eullm_forge.eval.gguf import EullmServer, generate_answers_eullm
        log = Path(args.answers).with_suffix(".server.log") if args.answers else None
        print(f"[eval] {len(items)} items from {args.gguf} via EULLM Engine, "
              f"{args.server_parallel} at a time", flush=True)
        with EullmServer(args.gguf, binary=args.eullm, parallel=args.server_parallel,
                         ctx_per_slot=args.server_ctx, log_path=log) as server:
            results = generate_answers_eullm(server, prompts, max_new_tokens=args.max_new_tokens,
                                             parallel=args.server_parallel)
    elif args.gguf:
        from eullm_forge.eval.gguf import LlamaServer, generate_answers_gguf
        log = Path(args.answers).with_suffix(".server.log") if args.answers else None
        print(f"[eval] {len(items)} items from {args.gguf} via llama-server, "
              f"{args.server_parallel} at a time", flush=True)
        with LlamaServer(args.gguf, binary=args.llama_server, parallel=args.server_parallel,
                         ctx_per_slot=args.server_ctx, log_path=log) as server:
            results = generate_answers_gguf(server, tok, prompts,
                                            max_new_tokens=args.max_new_tokens,
                                            end_ids=end_ids, parallel=args.server_parallel,
                                            keep_reasoning=args.think)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        cuda = torch.cuda.is_available()
        model = load_model(AutoModelForCausalLM, args.model, torch.cuda.device_count(),
                           fallback_cls=AutoModelForImageTextToText)
        model.eval()
        batch = args.batch_size or (16 if cuda else 1)
        print(f"[eval] {len(items)} items on {'GPU' if cuda else 'CPU'}, batch {batch}",
              flush=True)
        results = generate_answers(model, tok, prompts, batch_size=batch,
                                   max_new_tokens=args.max_new_tokens, end_ids=end_ids,
                                   keep_reasoning=args.think)
    reasonings = {}
    if args.think:
        split = [split_reasoning(text) for text, _ in results]
        reasonings = {it.id: r for it, (_, r, _) in zip(items, split)}
        results = [(a, e and closed) for (a, _, closed), (_, e) in zip(split, results)]
    if gated and args.gate_mode == "answer":
        from eullm_forge.eval.gate import GATE_ANSWER
        results = [(GATE_ANSWER, True) if it.id in gated else r for it, r in zip(items, results)]
    answers = {it.id: text for it, (text, _) in zip(items, results)}
    ended = sum(e for _, e in results)
    if not args.quiet:
        for it in items:
            if index:
                print(f"\n[eval] {it.id} reads: {'; '.join(contexts[it.id]) or 'nothing found'}")
            print(f"\n[eval] {it.id}: {it.question}\n{answers[it.id]}", flush=True)

    abst = {it.id: abstained(answers[it.id]) for it in items}
    unsourced = {it.id: unsourced_articles(answers[it.id], it.question, in_hand.get(it.id, ()))
                 for it in items}
    report = evaluate_qa(items, answers)
    if not args.quiet:
        for r in report["per_item"]:
            cov = r["keyword_coverage"]
            shown = "not scored" if cov is None else f"{cov:.2f}"
            print(f"[eval] {r['id']:<20} keywords {shown}")
    s = report["summary"]
    print(f"\n[eval] {args.label or args.model}: keyword coverage "
          f"{s['keyword_coverage']:.3f} over {s.get('keyword_items', s['n'])} of "
          f"{s['n']} items, "
          f"{ended}/{s['n']} ended their turn", flush=True)
    print(f"[eval] {args.label or args.model}: abstained {sum(abst.values())}/{s['n']}, "
          f"citing articles not in hand {sum(bool(u) for u in unsourced.values())}/{s['n']}"
          f"{' (own article taken out)' if args.absent else ''}", flush=True)

    if args.answers:
        with open(args.answers, "w", encoding="utf-8") as f:
            for it, r in zip(items, report["per_item"]):
                f.write(json.dumps({"id": it.id, "question": it.question,
                                    "answer": answers[it.id], "reference": it.reference,
                                    "rubric": it.rubric,
                                    "keyword_coverage": r["keyword_coverage"],
                                    "context": contexts.get(it.id),
                                    "abstained": abst[it.id],
                                    **({"gate": round(gate_p[it.id], 4)}
                                       if it.id in gate_p else {}),
                                    "unsourced_articles": unsourced[it.id],
                                    "absent": args.absent,
                                    **({"reasoning": reasonings[it.id]} if args.think else {})},
                                   ensure_ascii=False) + "\n")
    if args.csv:
        append_csv_row(Path(args.csv),
                       summary_row(args.label, args.gguf or args.model, report, ended))
    return 0


if __name__ == "__main__":
    sys.exit(main())

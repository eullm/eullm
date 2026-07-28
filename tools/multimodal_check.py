#!/usr/bin/env python3
"""Exercise the multimodal decode path against a running engine.

The multimodal path had no test of any kind. In v0.6.40 its decode loop, its
sampler and its output filtering all changed, and `cargo check --features
multimodal` only proves that code compiles. This is the missing check.

What each case targets, so a failure points somewhere:

1. **It runs.** An image plus a question returns coherent text. If this fails,
   the rewritten decode loop is broken and nothing else matters.
2. **No turn delimiter in the visible text.** Gemma closes a turn with
   ``<end_of_turn>``, sometimes spelling it out as ordinary text before the
   end-of-generation token. It must never reach the client. This is the stop
   sequence going through the shared hold-back buffer rather than the old
   per-loop ``ends_with`` check.
3. **No Harmony scaffolding.** Gemma 4 emits ``<|channel>thought<channel|>``
   preambles. Filters removed those on text requests but were never applied on
   this path at all, so before v0.6.40 they were shown verbatim. This case is
   the one that should visibly improve.
4. **Truncation is honest and lossless.** With a small ``num_predict`` the reply
   must come back marked ``done_reason: "length"``, and the text must not end
   mid-marker with bytes silently dropped, which is what the flush step exists
   to prevent.

Requires a running engine built with the ``multimodal`` feature (the published
CUDA binaries are). Standard library only.

Usage
-----
    # start the engine yourself first, e.g.
    #   eullm serve --port 11434
    python3 tools/multimodal_check.py photo.jpg
    python3 tools/multimodal_check.py photo.jpg --port 11434 --model gemma-4-e4b

Exit code is 0 only if nothing failed.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Markers that must never appear in text handed to a client.
#
# Matched on the bare name, without the surrounding punctuation. The exact
# spellings were listed here once and a real leak walked straight through
# them: the model closed a transcription with ``</start_of_turn>``, which does
# not contain ``<start_of_turn>`` because of the slash, so every case reported
# "clean" while the delimiter was visible in the reply. Whatever brackets the
# model puts around it, the name itself has no business in a reply.
TURN_DELIMITERS = ["start_of_turn", "end_of_turn", "im_end"]
HARMONY_MARKERS = ["<|channel>", "<channel|>", "<|message|>", "<|image>", "<image|>"]

failures = 0


def report(name: str, ok: bool | None, detail: str = "") -> None:
    global failures
    mark = {True: "ok  ", False: "FAIL", None: "skip"}[ok]
    if ok is False:
        failures += 1
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""), flush=True)


def ask(port: int, model: str, image_b64: str, question: str, num_predict=None):
    """One multimodal /api/chat round trip. Returns (status, parsed_or_text)."""
    options = {"temperature": 0}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question, "images": [image_b64]}],
        "stream": False,
        "options": options,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except Exception as e:
        return 0, str(e)


def content_of(body) -> str:
    if isinstance(body, dict):
        return (body.get("message") or {}).get("content") or ""
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="an image file (jpg/png/bmp/gif)")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--model", default="gemma-4-e4b")
    args = ap.parse_args()

    if not args.image.is_file():
        print(f"no such image: {args.image}", file=sys.stderr)
        return 2

    image_b64 = base64.b64encode(args.image.read_bytes()).decode()
    print(f"image: {args.image} ({args.image.stat().st_size} bytes)")
    print(f"engine: http://127.0.0.1:{args.port}  model: {args.model}")
    print("First request loads the projector and can take a while.\n")

    # ── 1. the decode loop runs at all ─────────────────────────────────────
    print("── the path works ──")
    code, body = ask(args.port, args.model, image_b64, "Describe this image in one sentence.")
    text = content_of(body)
    report(
        "multimodal request answers",
        code == 200 and len(text.strip()) > 0,
        f"http={code}, {len(text)} chars"
        + (f", {json.dumps(body)[:180]}" if code != 200 else ""),
    )
    if code != 200 or not text.strip():
        print("\nStopping: without a working request the remaining cases mean nothing.")
        return 1
    print(f"       answer: {text.strip()[:160]!r}")

    # ── 2. no turn delimiter leaked ────────────────────────────────────────
    print("\n── nothing internal reaches the client ──")
    leaked = [m for m in TURN_DELIMITERS if m in text]
    report(
        "no turn delimiter in the visible text",
        not leaked,
        "clean" if not leaked else f"leaked {leaked}",
    )

    # ── 3. no Harmony scaffolding (new in 0.6.40 on this path) ─────────────
    scaffolding = [m for m in HARMONY_MARKERS if m in text]
    report(
        "no Harmony scaffolding in the visible text",
        not scaffolding,
        "clean" if not scaffolding else f"leaked {scaffolding}",
    )

    # ── 4. truncation is honest, and loses nothing ─────────────────────────
    print("\n── truncation ──")
    code, body = ask(
        args.port,
        args.model,
        image_b64,
        "Describe this image in exhaustive detail.",
        num_predict=12,
    )
    short = content_of(body)
    reason = body.get("done_reason") if isinstance(body, dict) else None
    report(
        "a capped answer reports done_reason=length",
        code == 200 and reason == "length",
        f"http={code}, done_reason={reason!r}",
    )
    report(
        "a capped answer still returns its text",
        len(short.strip()) > 0,
        f"{len(short)} chars: {short.strip()[:80]!r}",
    )
    leaked = [m for m in TURN_DELIMITERS + HARMONY_MARKERS if m in short]
    report(
        "a capped answer leaks no markers either",
        not leaked,
        "clean" if not leaked else f"leaked {leaked}",
    )

    print()
    if failures:
        print(f"{failures} FAILED. Attach the engine's stderr, it carries the mtmd lines.")
    else:
        print("All checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

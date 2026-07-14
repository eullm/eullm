#!/usr/bin/env python3
"""Real-hardware validation for the KV-cache prefix reuse scheduler (roadmap 0.7-A).

Exercises the exact checklist in docs/roadmap-engine-0.7-1.0.md § 0.7-A against a
running `eullm serve` instance:

  1. multi-turn   — a single growing-history conversation (what --cli and any
                     Ollama-style client do), 20 turns by default. TTFT should stay
                     flat as the conversation grows if prefix reuse is working;
                     growing TTFT means reuse is not kicking in.
  2. concurrent   — N independent growing-history conversations in flight at once
                     (default 8), the same multi-turn pattern run concurrently.
  3. cancel       — mid-stream client disconnects (abrupt connection close), then
                     confirms the server is still healthy and the slot was cleaned
                     up correctly (next request on the same server succeeds).
  4. slow-consumer — a client that reads its stream far slower than the model can
                     generate, forcing the scheduler's per-sequence event channel
                     (256 slots) to fill up. This is the exact scenario behind the
                     eullm-engine v0.6.20 fix: a *Full* channel (slow but still
                     connected) must NOT be treated as a disconnect. Before that
                     fix this test would fail (response truncated early / no
                     `done` event).
  5. determinism  — the same prompt + fixed seed sent repeatedly must produce
                     byte-identical output, whether or not the second call reuses
                     a resident KV prefix from the first.

Requires: pip install aiohttp

Usage:
    python bench/reuse_validation.py --url http://localhost:11434 --model Qwen3.5-9B-Q4_K_M
    python bench/reuse_validation.py --url http://localhost:11434 --model Qwen3.5-9B-Q4_K_M \\
        --server-log /var/log/eullm/server.log --tests multiturn,slow-consumer

Each test is independent and best-effort: a failure in one does not stop the
others. Exit code is nonzero if any test fails.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

try:
    import aiohttp
except ImportError:
    print("This script requires aiohttp: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

ALL_TESTS = ["multiturn", "concurrent", "cancel", "slow-consumer", "determinism"]


class StreamResult:
    def __init__(self):
        self.text = ""
        self.ttft = None
        self.total_time = None
        self.tokens_generated = None
        self.done = False
        self.cancelled = False
        self.error = None


async def stream_generate(
    session,
    base_url,
    model,
    prompt,
    *,
    seed=None,
    max_tokens=64,
    raw=False,
    read_delay=0.0,
    cancel_after_tokens=None,
    timeout=120,
):
    """POST /api/generate with stream:true and drain the NDJSON response.

    `read_delay` sleeps before consuming each line — simulates a slow client.
    `cancel_after_tokens` closes the connection after N tokens — simulates an
    abrupt client disconnect (Ctrl-C / tab close) mid-stream.
    """
    result = StreamResult()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "raw": raw,
        "max_tokens": max_tokens,
        "options": {"num_predict": max_tokens},
    }
    if seed is not None:
        payload["seed"] = seed

    start = time.monotonic()
    try:
        async with session.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result

            token_count = 0
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("done"):
                    result.done = True
                    result.tokens_generated = event.get("eval_count")
                    break

                piece = event.get("response", "")
                if piece and result.ttft is None:
                    result.ttft = time.monotonic() - start
                result.text += piece
                token_count += 1

                if cancel_after_tokens is not None and token_count >= cancel_after_tokens:
                    result.cancelled = True
                    # Force-close rather than a graceful release: this is what
                    # actually drops the underlying connection immediately,
                    # simulating a real client disconnect (Ctrl-C / tab close)
                    # instead of an orderly stream shutdown.
                    resp.close()
                    break

                if read_delay:
                    await asyncio.sleep(read_delay)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        result.error = str(e)
    finally:
        result.total_time = time.monotonic() - start

    return result


def grep_reused_lines(log_path, since_offset):
    """Read lines appended to `log_path` since `since_offset` and pull out
    every "reused N from cache" count, in order."""
    with open(log_path, "r", errors="replace") as f:
        f.seek(since_offset)
        new_content = f.read()
    return [int(m) for m in re.findall(r"reused (\d+) from cache", new_content)]


def log_offset(log_path):
    return os.path.getsize(log_path) if log_path and os.path.exists(log_path) else None


# ── Tests ────────────────────────────────────────────────────────────────────

async def test_multiturn(session, args):
    """0.7-A: 20-turn growing-history conversation. TTFT should stay flat."""
    print(f"\n[multiturn] {args.turns} turns, single conversation, growing history")
    offset = log_offset(args.server_log)
    history = ""
    ttfts = []
    for turn in range(1, args.turns + 1):
        history += f"User: tell me one short fact about the number {turn}.\nAssistant: "
        r = await stream_generate(
            session, args.url, args.model, history,
            seed=args.seed, max_tokens=args.max_tokens, timeout=args.timeout,
        )
        if r.error or not r.done:
            print(f"  turn {turn:2d}: FAILED — {r.error or 'stream did not complete'}")
            return False
        history += r.text + "\n"
        ttfts.append(r.ttft or 0.0)
        print(f"  turn {turn:2d}: TTFT={r.ttft*1000:6.1f}ms  total={r.total_time*1000:7.1f}ms  prompt_chars={len(history)}")

    if args.server_log:
        reused = grep_reused_lines(args.server_log, offset)
        print(f"  server log 'reused N from cache': {reused}")
        if len(reused) >= 2 and reused[-1] <= reused[0]:
            print("  WARNING: reused-token count did not grow across turns — reuse may not be active")
    else:
        print("  (pass --server-log <path> to cross-check 'reused N from cache' log lines)")

    # First turn has nothing to reuse; a flat-ish TTFT from turn 2 onward is
    # the load-bearing signal that reuse is avoiding a full re-prefill.
    if len(ttfts) >= 3:
        first_half = sum(ttfts[1: len(ttfts) // 2 + 1]) / max(1, len(ttfts) // 2)
        second_half = sum(ttfts[len(ttfts) // 2 + 1:]) / max(1, len(ttfts) - len(ttfts) // 2 - 1)
        growth = (second_half - first_half) / first_half if first_half else 0
        print(f"  TTFT trend: first-half avg={first_half*1000:.1f}ms  second-half avg={second_half*1000:.1f}ms  growth={growth:+.0%}")
        if growth > 0.5:
            print("  WARNING: TTFT grew substantially as the conversation lengthened — check for a full re-prefill every turn")
    return True


async def test_concurrent(session, args):
    """0.7-A: N independent growing-history conversations running at once."""
    print(f"\n[concurrent] {args.concurrency} independent conversations, {args.concurrent_turns} turns each")

    async def one_conversation(conv_id):
        history = ""
        for turn in range(1, args.concurrent_turns + 1):
            history += f"User (conv {conv_id}): give me a fun fact about the city ranked #{turn} by population in your training data.\nAssistant: "
            r = await stream_generate(
                session, args.url, args.model, history,
                seed=args.seed + conv_id, max_tokens=args.max_tokens, timeout=args.timeout,
            )
            if r.error or not r.done:
                return conv_id, False, f"turn {turn}: {r.error or 'incomplete'}"
            history += r.text + "\n"
        return conv_id, True, None

    start = time.monotonic()
    results = await asyncio.gather(*(one_conversation(i) for i in range(args.concurrency)))
    elapsed = time.monotonic() - start

    ok = True
    for conv_id, success, err in sorted(results):
        status = "ok" if success else f"FAILED ({err})"
        print(f"  conversation {conv_id}: {status}")
        ok = ok and success
    print(f"  {args.concurrency} conversations x {args.concurrent_turns} turns in {elapsed:.1f}s")
    return ok


async def test_cancel(session, args):
    """0.7-A: abrupt mid-stream disconnect must not wedge the server or the slot."""
    print(f"\n[cancel] {args.cancel_iterations} abrupt mid-stream disconnects, each followed by a fresh request")
    ok = True
    for i in range(args.cancel_iterations):
        cancelled = await stream_generate(
            session, args.url, args.model,
            f"Write a very long, detailed story about a journey (attempt {i}).",
            max_tokens=200, cancel_after_tokens=8, timeout=args.timeout,
        )
        if not cancelled.cancelled:
            print(f"  iter {i}: could not force a cancellation ({cancelled.error})")
            ok = False
            continue

        follow_up = await stream_generate(
            session, args.url, args.model,
            f"Say the single word 'recovered-{i}' and nothing else.",
            max_tokens=16, timeout=args.timeout,
        )
        recovered = follow_up.done and not follow_up.error
        print(f"  iter {i}: disconnected after 8 tokens, next request {'ok' if recovered else 'FAILED: ' + str(follow_up.error)}")
        ok = ok and recovered
    return ok


async def test_slow_consumer(session, args):
    """v0.6.20 regression test: a slow-but-connected client must not be killed.

    Reads far slower than generation can fill the 256-slot event channel, so
    try_send hits TrySendError::Full. Before the fix this was misread as a
    disconnect and the response was truncated early with no `done` event.
    """
    print(f"\n[slow-consumer] deliberately slow reads over {args.slow_consumer_tokens} tokens (targets the Full-vs-Closed fix)")
    r = await stream_generate(
        session, args.url, args.model,
        "Count from 1 to 300, one number per line, nothing else.",
        max_tokens=args.slow_consumer_tokens,
        read_delay=args.slow_consumer_delay,
        timeout=args.timeout * 4,
    )
    if r.error:
        print(f"  FAILED: {r.error}")
        return False
    if not r.done:
        print("  FAILED: stream never reached 'done' — looks like the Full-channel-as-disconnect bug")
        return False
    print(f"  ok: reached 'done', {r.tokens_generated} tokens generated, {r.total_time:.1f}s wall time")
    return True


async def test_determinism(session, args):
    """0.7-A: same prompt + fixed seed must be byte-identical across repeats,
    including once a resident KV prefix from the previous call is reused."""
    print(f"\n[determinism] same prompt+seed sent {args.determinism_repeats} times")
    prompt = "In exactly one sentence, name the capital of France."
    outputs = []
    for i in range(args.determinism_repeats):
        r = await stream_generate(
            session, args.url, args.model, prompt,
            seed=args.seed, max_tokens=32, timeout=args.timeout,
        )
        if r.error or not r.done:
            print(f"  repeat {i}: FAILED — {r.error or 'incomplete'}")
            return False
        outputs.append(r.text)

    all_same = all(o == outputs[0] for o in outputs)
    print(f"  outputs: {[repr(o) for o in outputs]}")
    print(f"  {'ok: byte-identical across all repeats' if all_same else 'FAILED: outputs diverged'}")

    if all_same and args.baseline_url:
        async with aiohttp.ClientSession() as baseline_session:
            baseline = await stream_generate(
                baseline_session, args.baseline_url, args.model, prompt,
                seed=args.seed, max_tokens=32, timeout=args.timeout,
            )
        if baseline.error or not baseline.done:
            print(f"  baseline ({args.baseline_url}): FAILED — {baseline.error or 'incomplete'}")
            return False
        matches_baseline = baseline.text == outputs[0]
        print(f"  baseline ({args.baseline_url}): {repr(baseline.text)} — {'matches' if matches_baseline else 'DIFFERS from'} current server")
        return all_same and matches_baseline

    return all_same


TEST_FUNCS = {
    "multiturn": test_multiturn,
    "concurrent": test_concurrent,
    "cancel": test_cancel,
    "slow-consumer": test_slow_consumer,
    "determinism": test_determinism,
}


async def main_async(args):
    selected = args.tests.split(",") if args.tests != "all" else ALL_TESTS
    unknown = [t for t in selected if t not in TEST_FUNCS]
    if unknown:
        print(f"Unknown test(s): {unknown}. Valid: {ALL_TESTS}", file=sys.stderr)
        return 1

    print(f"eullm 0.7-A reuse validation — {args.url}  model={args.model}")
    print(f"Running: {', '.join(selected)}")

    results = {}
    async with aiohttp.ClientSession() as session:
        for name in selected:
            try:
                results[name] = await TEST_FUNCS[name](session, args)
            except Exception as e:  # keep going — one test's crash shouldn't hide the others
                print(f"\n[{name}] CRASHED: {e!r}")
                results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 60)

    return 0 if all(results.values()) else 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:11434", help="eullm server base URL")
    p.add_argument("--model", required=True, help="model name as loaded by the server")
    p.add_argument("--tests", default="all", help=f"comma-separated subset of {ALL_TESTS}, or 'all'")
    p.add_argument("--server-log", default=None, help="path to the server's log file, for cross-checking 'reused N from cache' lines")
    p.add_argument("--baseline-url", default=None, help="optional second server (e.g. an old binary) to diff the determinism test against")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=64, help="max_tokens for the multiturn/concurrent tests")
    p.add_argument("--turns", type=int, default=20, help="turns for the multiturn test")
    p.add_argument("--concurrency", type=int, default=8, help="conversations for the concurrent test")
    p.add_argument("--concurrent-turns", type=int, default=5, help="turns per conversation in the concurrent test")
    p.add_argument("--cancel-iterations", type=int, default=5)
    p.add_argument("--slow-consumer-tokens", type=int, default=300, help="must exceed the 256-slot event channel to force TrySendError::Full")
    p.add_argument("--slow-consumer-delay", type=float, default=0.15, help="seconds slept before consuming each streamed line")
    p.add_argument("--determinism-repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120, help="per-request timeout in seconds")
    return p.parse_args()


def main():
    args = parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

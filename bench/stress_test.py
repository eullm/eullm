#!/usr/bin/env python3
"""
EULLM vs Ollama — Real Stress Test with Parallelism Verification

This script proves (or disproves) that an inference server actually
processes requests in parallel by measuring:

  1. TTFT (Time To First Token) — do all requests start generating immediately?
  2. Token timeline overlap — are generation periods overlapping or sequential?
  3. Token interleaving — do tokens from different requests arrive interleaved?
  4. Per-request and aggregate throughput
  5. Latency distribution (P50, P95, P99)

Usage:
    python bench/stress_test.py --url http://localhost:11434 --model Qwen3.5-9B-Q8_0
    python bench/stress_test.py --url http://localhost:11435 --model qwen3.5:9b --label ollama
    python bench/stress_test.py --url http://localhost:11434 --model Qwen3.5-9B-Q8_0 \
        --concurrency 1,2,4,8,16 --tokens 150 --rounds 3

Requires: aiohttp (pip install aiohttp)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TokenEvent:
    """A single token arrival from the stream."""
    timestamp: float       # time.monotonic()
    token: str
    request_id: int


@dataclass
class RequestResult:
    """Complete timing data for one request."""
    request_id: int
    submit_time: float     # when the HTTP request was sent
    first_token_time: float = 0.0   # when the first token arrived
    last_token_time: float = 0.0    # when the last token arrived
    token_count: int = 0
    tokens: list[TokenEvent] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        if self.first_token_time == 0:
            return -1
        return (self.first_token_time - self.submit_time) * 1000

    @property
    def total_ms(self) -> float:
        """Total request duration in milliseconds."""
        if self.last_token_time == 0:
            return -1
        return (self.last_token_time - self.submit_time) * 1000

    @property
    def generation_ms(self) -> float:
        """Time from first token to last token (decode phase only)."""
        if self.first_token_time == 0 or self.last_token_time == 0:
            return -1
        return (self.last_token_time - self.first_token_time) * 1000

    @property
    def tokens_per_sec(self) -> float:
        """Per-request token generation rate."""
        gen = self.generation_ms
        if gen <= 0 or self.token_count <= 1:
            return 0
        return (self.token_count - 1) / (gen / 1000)  # exclude first token


@dataclass
class RoundResult:
    """Results from one round of N concurrent requests."""
    concurrency: int
    round_num: int
    requests: list[RequestResult]
    wall_start: float
    wall_end: float

    @property
    def wall_ms(self) -> float:
        return (self.wall_end - self.wall_start) * 1000

    @property
    def total_tokens(self) -> int:
        return sum(r.token_count for r in self.requests if not r.error)

    @property
    def aggregate_throughput(self) -> float:
        """Total tokens / wall time."""
        if self.wall_ms <= 0:
            return 0
        return self.total_tokens / (self.wall_ms / 1000)


# ── Streaming client ─────────────────────────────────────────────────────────

async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    request_id: int,
    num_predict: int,
    prompt: str,
    global_token_log: list[TokenEvent],
) -> RequestResult:
    """Send a streaming request and record per-token timestamps."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "num_predict": num_predict,
        "options": {"num_predict": num_predict},
    }

    result = RequestResult(request_id=request_id, submit_time=time.monotonic())

    try:
        async with session.post(
            f"{url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result

            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                # NDJSON: split on newlines, parse each complete line
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract token from Ollama/EULLM chat response format
                    content = ""
                    if "message" in obj and "content" in obj["message"]:
                        content = obj["message"]["content"]
                    elif "response" in obj:
                        content = obj["response"]

                    if content:
                        now = time.monotonic()
                        if result.first_token_time == 0:
                            result.first_token_time = now
                        result.last_token_time = now
                        result.token_count += 1

                        event = TokenEvent(
                            timestamp=now,
                            token=content,
                            request_id=request_id,
                        )
                        result.tokens.append(event)
                        global_token_log.append(event)

                    # Ollama sends done=true at the end
                    if obj.get("done", False):
                        if result.last_token_time == 0:
                            result.last_token_time = time.monotonic()
                        break

    except asyncio.TimeoutError:
        result.error = "Timeout (300s)"
    except Exception as e:
        result.error = str(e)

    return result


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_overlap(results: list[RequestResult]) -> dict:
    """Determine if request generation periods overlap in time."""
    valid = [r for r in results if not r.error and r.first_token_time > 0]
    if len(valid) < 2:
        return {"overlap": False, "reason": "fewer than 2 successful requests"}

    # Sort by first_token_time
    valid.sort(key=lambda r: r.first_token_time)

    # Check pairwise overlap: does request N's generation overlap with N+1's?
    overlapping_pairs = 0
    total_pairs = 0
    total_overlap_ms = 0.0

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            total_pairs += 1
            # Overlap exists if i's last_token_time > j's first_token_time
            overlap_start = max(valid[i].first_token_time, valid[j].first_token_time)
            overlap_end = min(valid[i].last_token_time, valid[j].last_token_time)
            if overlap_end > overlap_start:
                overlapping_pairs += 1
                total_overlap_ms += (overlap_end - overlap_start) * 1000

    overlap_ratio = overlapping_pairs / total_pairs if total_pairs > 0 else 0

    return {
        "overlapping_pairs": overlapping_pairs,
        "total_pairs": total_pairs,
        "overlap_ratio": overlap_ratio,
        "total_overlap_ms": total_overlap_ms,
        "is_parallel": overlap_ratio > 0.5,
    }


def analyze_interleaving(token_log: list[TokenEvent], n_requests: int) -> dict:
    """Check if tokens from different requests are interleaved in time."""
    if len(token_log) < 2 or n_requests < 2:
        return {"interleaved": False, "reason": "insufficient data"}

    # Sort by timestamp
    sorted_tokens = sorted(token_log, key=lambda t: t.timestamp)

    # Count transitions: how often does the request_id change between
    # consecutive tokens?
    transitions = 0
    for i in range(1, len(sorted_tokens)):
        if sorted_tokens[i].request_id != sorted_tokens[i - 1].request_id:
            transitions += 1

    max_transitions = len(sorted_tokens) - 1
    transition_rate = transitions / max_transitions if max_transitions > 0 else 0

    # In purely sequential processing, transitions ≈ n_requests - 1
    # In truly interleaved processing, transitions ≈ max_transitions * (1 - 1/n)
    expected_sequential = n_requests - 1
    expected_parallel = max_transitions * (1 - 1 / n_requests)

    # Classify: if transitions are much closer to expected_parallel than expected_sequential
    if expected_parallel > expected_sequential:
        midpoint = (expected_sequential + expected_parallel) / 2
        is_interleaved = transitions > midpoint
    else:
        is_interleaved = False

    return {
        "transitions": transitions,
        "max_transitions": max_transitions,
        "transition_rate": transition_rate,
        "expected_sequential": expected_sequential,
        "expected_parallel": round(expected_parallel, 1),
        "is_interleaved": is_interleaved,
    }


def compute_latency_stats(values: list[float]) -> dict:
    """Compute P50, P95, P99 from a list of values."""
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)] if n >= 20 else s[-1],
        "p99": s[int(n * 0.99)] if n >= 100 else s[-1],
        "mean": statistics.mean(s),
        "min": s[0],
        "max": s[-1],
    }


# ── Display ──────────────────────────────────────────────────────────────────

def print_timeline(results: list[RequestResult], wall_start: float):
    """Print an ASCII timeline showing request overlap."""
    valid = sorted(
        [r for r in results if not r.error and r.first_token_time > 0],
        key=lambda r: r.submit_time,
    )
    if not valid:
        print("  (no successful requests)")
        return

    # Find time range
    t_min = wall_start
    t_max = max(r.last_token_time for r in valid)
    duration = t_max - t_min
    if duration <= 0:
        return

    width = 70  # characters for the timeline

    for r in valid:
        # Prefill period: submit → first token
        prefill_start = int((r.submit_time - t_min) / duration * width)
        prefill_end = int((r.first_token_time - t_min) / duration * width)
        # Decode period: first token → last token
        decode_start = prefill_end
        decode_end = int((r.last_token_time - t_min) / duration * width)

        prefill_start = max(0, min(width, prefill_start))
        prefill_end = max(prefill_start, min(width, prefill_end))
        decode_start = max(prefill_end, min(width, decode_start))
        decode_end = max(decode_start, min(width, decode_end))

        bar = (
            " " * prefill_start
            + "." * (prefill_end - prefill_start)  # prefill (waiting)
            + "#" * (decode_end - decode_start)     # decode (generating)
        )
        bar = bar.ljust(width)

        ttft = r.ttft_ms
        tps = r.tokens_per_sec
        print(f"  req{r.request_id:>2} |{bar}| TTFT {ttft:>6.0f}ms  {tps:>5.1f} tok/s  {r.token_count:>3} tokens")

    print(f"         {''.ljust(width)}")
    print(f"         0{''.ljust(width - 6)}{duration * 1000:.0f}ms")
    print(f"         . = prefill/waiting    # = generating tokens")


def print_verdict(overlap: dict, interleaving: dict, label: str):
    """Print the parallelism verdict."""
    print(f"\n  PARALLELISM ANALYSIS ({label}):")

    # Overlap verdict
    if overlap.get("is_parallel"):
        pct = overlap["overlap_ratio"] * 100
        print(f"    Overlap:      YES — {overlap['overlapping_pairs']}/{overlap['total_pairs']} pairs overlap ({pct:.0f}%)")
    else:
        print(f"    Overlap:      NO  — requests appear sequential")

    # Interleaving verdict
    if interleaving.get("is_interleaved"):
        rate = interleaving["transition_rate"] * 100
        print(f"    Interleaving: YES — {interleaving['transitions']} context switches ({rate:.0f}% transition rate)")
        print(f"                  (expected sequential: {interleaving['expected_sequential']}, "
              f"parallel: {interleaving['expected_parallel']}, actual: {interleaving['transitions']})")
    else:
        print(f"    Interleaving: NO  — tokens arrive in sequence, not interleaved")
        if "transitions" in interleaving:
            print(f"                  (transitions: {interleaving['transitions']}, "
                  f"expected sequential: {interleaving['expected_sequential']}, "
                  f"parallel: {interleaving['expected_parallel']})")

    # Final verdict
    is_truly_parallel = overlap.get("is_parallel", False) and interleaving.get("is_interleaved", False)
    if is_truly_parallel:
        print(f"    VERDICT:      REAL PARALLEL PROCESSING")
    elif overlap.get("is_parallel"):
        print(f"    VERDICT:      PARTIALLY PARALLEL (overlap but no interleaving)")
    else:
        print(f"    VERDICT:      SEQUENTIAL PROCESSING (no real parallelism)")


# ── Main benchmark loop ──────────────────────────────────────────────────────

PROMPTS = [
    "Write a detailed essay about the history of Rome from its founding to the fall of the Western Roman Empire.",
    "Explain the principles of quantum mechanics, including wave-particle duality, uncertainty principle, and quantum entanglement.",
    "Describe the complete process of photosynthesis in plants, from light absorption to glucose production.",
    "Analyze the causes and consequences of the French Revolution, including political, social, and economic factors.",
    "Explain how modern CPUs work, from transistors to instruction pipelines, cache hierarchies, and branch prediction.",
    "Describe the evolution of human language, from proto-languages to modern linguistic diversity and language families.",
    "Explain the fundamentals of general relativity, including spacetime curvature, gravitational waves, and black holes.",
    "Analyze the Industrial Revolution and its impact on society, technology, economics, and the environment.",
    "Describe the human immune system in detail, including innate and adaptive immunity, antibodies, and T-cells.",
    "Explain the history and development of the Internet, from ARPANET to the modern World Wide Web.",
    "Analyze the economic theories of Adam Smith, Karl Marx, and John Maynard Keynes and their lasting influence.",
    "Describe the geological history of Earth, from formation to the present day, including major extinction events.",
    "Explain the principles of machine learning, from linear regression to deep neural networks and transformers.",
    "Analyze the political philosophy of democracy, from ancient Athens to modern representative systems.",
    "Describe the solar system in detail, including all planets, their moons, the asteroid belt, and the Kuiper belt.",
    "Explain the history of mathematics from ancient Babylon and Egypt through calculus and modern abstract algebra.",
]


async def run_round(
    url: str,
    model: str,
    concurrency: int,
    num_predict: int,
    round_num: int,
) -> RoundResult:
    """Run one round: fire N concurrent streaming requests."""
    global_token_log: list[TokenEvent] = []

    async with aiohttp.ClientSession() as session:
        wall_start = time.monotonic()

        tasks = []
        for i in range(concurrency):
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(
                stream_request(
                    session=session,
                    url=url,
                    model=model,
                    request_id=i + 1,
                    num_predict=num_predict,
                    prompt=prompt,
                    global_token_log=global_token_log,
                )
            )

        results = await asyncio.gather(*tasks)

        wall_end = time.monotonic()

    return RoundResult(
        concurrency=concurrency,
        round_num=round_num,
        requests=list(results),
        wall_start=wall_start,
        wall_end=wall_end,
    )


async def main():
    parser = argparse.ArgumentParser(
        description="EULLM vs Ollama — Stress Test with Parallelism Verification"
    )
    parser.add_argument("--url", required=True, help="Base URL (e.g. http://localhost:11434)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--label", default=None, help="Label for output (default: auto-detect)")
    parser.add_argument(
        "--concurrency", default="1,2,4,8",
        help="Comma-separated concurrency levels (default: 1,2,4,8)"
    )
    parser.add_argument("--tokens", type=int, default=100, help="Tokens per request (default: 100)")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds per concurrency level (default: 1)")
    parser.add_argument("--warmup", action="store_true", help="Send a warmup request first")
    parser.add_argument("--json", default=None, help="Write JSON results to file")

    args = parser.parse_args()
    label = args.label or args.url
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]

    print(f"{'=' * 80}")
    print(f"  STRESS TEST: {label}")
    print(f"  Model: {args.model}")
    print(f"  Tokens/request: {args.tokens}")
    print(f"  Concurrency levels: {concurrency_levels}")
    print(f"  Rounds per level: {args.rounds}")
    print(f"{'=' * 80}")

    # Warmup
    if args.warmup:
        print("\n  Warmup request...", end="", flush=True)
        try:
            async with aiohttp.ClientSession() as session:
                await stream_request(
                    session, args.url, args.model, 0, 10, "Say hello.", []
                )
            print(" done.")
        except Exception as e:
            print(f" failed: {e}")

    all_json_results = []

    for conc in concurrency_levels:
        print(f"\n{'─' * 80}")
        print(f"  {conc} CONCURRENT REQUEST(S)")
        print(f"{'─' * 80}")

        for rnd in range(1, args.rounds + 1):
            if args.rounds > 1:
                print(f"\n  Round {rnd}/{args.rounds}:")

            result = await run_round(
                url=args.url,
                model=args.model,
                concurrency=conc,
                num_predict=args.tokens,
                round_num=rnd,
            )

            # Print per-request results
            errors = [r for r in result.requests if r.error]
            successes = [r for r in result.requests if not r.error]

            if errors:
                print(f"\n  ERRORS ({len(errors)}):")
                for r in errors:
                    print(f"    req{r.request_id}: {r.error}")

            # Timeline visualization
            print(f"\n  Timeline:")
            print_timeline(result.requests, result.wall_start)

            # Aggregate stats
            print(f"\n  Summary:")
            print(f"    Wall time:      {result.wall_ms:>8.0f} ms")
            print(f"    Total tokens:   {result.total_tokens:>8d}")
            print(f"    Throughput:     {result.aggregate_throughput:>8.1f} tok/s (aggregate)")

            if successes:
                ttfts = [r.ttft_ms for r in successes if r.ttft_ms > 0]
                totals = [r.total_ms for r in successes if r.total_ms > 0]
                per_req_tps = [r.tokens_per_sec for r in successes if r.tokens_per_sec > 0]

                if ttfts:
                    stats = compute_latency_stats(ttfts)
                    print(f"    TTFT:           P50={stats['p50']:.0f}ms  P95={stats['p95']:.0f}ms  "
                          f"min={stats['min']:.0f}ms  max={stats['max']:.0f}ms")
                if totals:
                    stats = compute_latency_stats(totals)
                    print(f"    Total latency:  P50={stats['p50']:.0f}ms  P95={stats['p95']:.0f}ms  "
                          f"min={stats['min']:.0f}ms  max={stats['max']:.0f}ms")
                if per_req_tps:
                    stats = compute_latency_stats(per_req_tps)
                    print(f"    Per-req tok/s:  P50={stats['p50']:.1f}  min={stats['min']:.1f}  max={stats['max']:.1f}")

            # Parallelism analysis (only for concurrency > 1)
            if conc > 1:
                overlap = analyze_overlap(successes)
                # Collect all token events from this round
                all_tokens = []
                for r in successes:
                    all_tokens.extend(r.tokens)
                interleaving = analyze_interleaving(all_tokens, conc)
                print_verdict(overlap, interleaving, label)

            # Collect JSON
            round_json = {
                "concurrency": conc,
                "round": rnd,
                "wall_ms": result.wall_ms,
                "total_tokens": result.total_tokens,
                "throughput": result.aggregate_throughput,
                "requests": [],
            }
            for r in result.requests:
                round_json["requests"].append({
                    "id": r.request_id,
                    "ttft_ms": r.ttft_ms,
                    "total_ms": r.total_ms,
                    "generation_ms": r.generation_ms,
                    "token_count": r.token_count,
                    "tokens_per_sec": r.tokens_per_sec,
                    "error": r.error,
                })
            if conc > 1:
                round_json["overlap"] = overlap
                round_json["interleaving"] = {
                    k: v for k, v in interleaving.items()
                    if k != "reason"
                }
            all_json_results.append(round_json)

    # Final comparison table
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY TABLE — {label}")
    print(f"{'=' * 80}")
    print(f"  {'Conc':>6} | {'Wall (ms)':>10} | {'Throughput':>12} | {'TTFT P50':>10} | {'Parallel?':>10}")
    print(f"  {'─' * 6}-+-{'─' * 10}-+-{'─' * 12}-+-{'─' * 10}-+-{'─' * 10}")

    for entry in all_json_results:
        conc = entry["concurrency"]
        wall = entry["wall_ms"]
        tput = entry["throughput"]
        ttfts = [r["ttft_ms"] for r in entry["requests"] if r["ttft_ms"] > 0]
        ttft_p50 = sorted(ttfts)[len(ttfts) // 2] if ttfts else 0

        parallel = ""
        if conc > 1 and "overlap" in entry:
            parallel = "YES" if entry["overlap"].get("is_parallel") else "NO"

        print(f"  {conc:>6} | {wall:>10.0f} | {tput:>10.1f}/s | {ttft_p50:>8.0f}ms | {parallel:>10}")

    print()

    # Write JSON
    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "label": label,
                "model": args.model,
                "tokens_per_request": args.tokens,
                "results": all_json_results,
            }, f, indent=2)
        print(f"  JSON results written to: {args.json}")


if __name__ == "__main__":
    asyncio.run(main())

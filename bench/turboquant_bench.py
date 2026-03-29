#!/usr/bin/env python3
"""
EULLM TurboQuant KV Cache Benchmark

Compare KV cache types (F16, TQ4_0, TQ3_0) on the EULLM engine by measuring
TTFT, token throughput, and total wall time across different concurrency levels.

Two modes:
  collect  — send requests to a running EULLM engine, measure metrics, save JSON
  compare  — read multiple JSON result files and print a comparison table

Usage:
    # Collect results for each cache type (restart engine between runs):
    python bench/turboquant_bench.py collect --cache-label F16 --output results/f16.json
    python bench/turboquant_bench.py collect --cache-label TQ4_0 --output results/tq4_0.json
    python bench/turboquant_bench.py collect --cache-label TQ3_0 --output results/tq3_0.json

    # Compare results:
    python bench/turboquant_bench.py compare results/f16.json results/tq4_0.json results/tq3_0.json
    python bench/turboquant_bench.py compare results/*.json --markdown

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


# -- Test prompts --------------------------------------------------------------

TEST_PROMPTS = [
    # Short prompt: simple factual question
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "label": "short",
    },
    # Medium prompt: paragraph-level reasoning
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the key differences between civil law and common law "
                    "legal systems. Cover their historical origins, how judges use "
                    "precedent, the role of codified statutes, and give one example "
                    "country for each system."
                ),
            },
        ],
        "label": "medium",
    },
    # System + user prompt: chat style with persona
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst specializing in European "
                    "markets. Respond concisely and use data when possible."
                ),
            },
            {
                "role": "user",
                "content": (
                    "What are the main risks facing the European banking sector "
                    "in the next 12 months?"
                ),
            },
        ],
        "label": "system_user",
    },
    # Multi-turn prompt: continuation of a conversation
    {
        "messages": [
            {"role": "user", "content": "Translate 'Good morning' to German."},
            {"role": "assistant", "content": "Guten Morgen"},
            {
                "role": "user",
                "content": (
                    "Now translate 'The contract shall be governed by the laws of "
                    "the Federal Republic of Germany' into German and explain any "
                    "legal nuances in the translation."
                ),
            },
        ],
        "label": "multi_turn",
    },
]


# -- Data structures -----------------------------------------------------------

@dataclass
class RequestResult:
    """Timing data for a single streaming request."""
    request_id: int
    prompt_label: str
    submit_time: float
    first_token_time: float = 0.0
    last_token_time: float = 0.0
    token_count: int = 0
    eval_count: Optional[int] = None
    eval_duration_ns: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    error: Optional[str] = None

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time == 0:
            return -1
        return (self.first_token_time - self.submit_time) * 1000

    @property
    def total_ms(self) -> float:
        if self.last_token_time == 0:
            return -1
        return (self.last_token_time - self.submit_time) * 1000

    @property
    def tokens_per_sec(self) -> float:
        """Tokens/sec from server-reported eval_duration, falling back to wall clock."""
        if self.eval_count and self.eval_duration_ns and self.eval_duration_ns > 0:
            return self.eval_count / (self.eval_duration_ns / 1e9)
        # Fallback: wall-clock based
        gen_ms = (self.last_token_time - self.first_token_time) * 1000 if self.first_token_time > 0 else 0
        if gen_ms > 0 and self.token_count > 1:
            return (self.token_count - 1) / (gen_ms / 1000)
        return 0.0


@dataclass
class RoundResult:
    """Results from one round of concurrent requests."""
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
        if self.wall_ms <= 0:
            return 0
        return self.total_tokens / (self.wall_ms / 1000)


# -- Streaming client ----------------------------------------------------------

async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    request_id: int,
    num_predict: int,
    messages: list[dict],
    prompt_label: str,
) -> RequestResult:
    """Send a streaming /api/chat request and record timing metrics."""

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"num_predict": num_predict},
    }

    result = RequestResult(
        request_id=request_id,
        prompt_label=prompt_label,
        submit_time=time.monotonic(),
    )

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
                # NDJSON: one JSON object per line, no data: prefix
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract token text
                    content = ""
                    if "message" in obj and "content" in obj["message"]:
                        content = obj["message"]["content"]

                    if content:
                        now = time.monotonic()
                        if result.first_token_time == 0:
                            result.first_token_time = now
                        result.last_token_time = now
                        result.token_count += 1

                    # Final message with done=true carries server metrics
                    if obj.get("done", False):
                        if result.last_token_time == 0:
                            result.last_token_time = time.monotonic()
                        if "eval_count" in obj:
                            result.eval_count = obj["eval_count"]
                        if "eval_duration" in obj:
                            result.eval_duration_ns = obj["eval_duration"]
                        if "prompt_eval_duration" in obj:
                            result.prompt_eval_duration_ns = obj["prompt_eval_duration"]
                        break

    except asyncio.TimeoutError:
        result.error = "Timeout (300s)"
    except Exception as e:
        result.error = str(e)

    return result


# -- Statistics helpers --------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics."""
    if not values:
        return {"p50": 0, "p95": 0, "mean": 0, "min": 0, "max": 0, "stdev": 0}
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
    }


# -- Benchmark runner ----------------------------------------------------------

async def run_round(
    url: str,
    model: str,
    concurrency: int,
    num_predict: int,
    round_num: int,
) -> RoundResult:
    """Fire N concurrent streaming requests using rotating test prompts."""

    async with aiohttp.ClientSession() as session:
        wall_start = time.monotonic()

        tasks = []
        for i in range(concurrency):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            tasks.append(
                stream_request(
                    session=session,
                    url=url,
                    model=model,
                    request_id=i + 1,
                    num_predict=num_predict,
                    messages=prompt["messages"],
                    prompt_label=prompt["label"],
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


async def run_warmup(url: str, model: str, count: int):
    """Send warmup requests to prime the engine."""
    print(f"  Sending {count} warmup request(s)...", end="", flush=True)
    async with aiohttp.ClientSession() as session:
        for i in range(count):
            await stream_request(
                session=session,
                url=url,
                model=model,
                request_id=0,
                num_predict=20,
                messages=[{"role": "user", "content": "Say hello."}],
                prompt_label="warmup",
            )
    print(" done.")


# -- Collect mode --------------------------------------------------------------

async def cmd_collect(args: argparse.Namespace):
    """Collect benchmark data for a single KV cache configuration."""

    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]
    output_file = args.output or f"turboquant_{args.cache_label.lower()}.json"

    print(f"{'=' * 76}")
    print(f"  TurboQuant Bench — collect")
    print(f"  URL:         {args.url}")
    print(f"  Model:       {args.model}")
    print(f"  Cache:       {args.cache_label}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Tokens:      {args.tokens}")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Output:      {output_file}")
    print(f"{'=' * 76}")

    # Warmup
    if args.warmup > 0:
        await run_warmup(args.url, args.model, args.warmup)

    all_rounds: list[dict] = []

    for conc in concurrency_levels:
        print(f"\n{'─' * 76}")
        print(f"  Concurrency: {conc}")
        print(f"{'─' * 76}")

        for rnd in range(1, args.rounds + 1):
            print(f"\n  Round {rnd}/{args.rounds}:")

            result = await run_round(
                url=args.url,
                model=args.model,
                concurrency=conc,
                num_predict=args.tokens,
                round_num=rnd,
            )

            successes = [r for r in result.requests if not r.error]
            errors = [r for r in result.requests if r.error]

            if errors:
                for r in errors:
                    print(f"    ERROR req{r.request_id}: {r.error}")

            # Display per-request summary
            for r in successes:
                print(
                    f"    req{r.request_id:>2} [{r.prompt_label:>12}]  "
                    f"TTFT={r.ttft_ms:>6.0f}ms  "
                    f"tok/s={r.tokens_per_sec:>6.1f}  "
                    f"tokens={r.token_count:>3}  "
                    f"total={r.total_ms:>7.0f}ms"
                )

            # Aggregate
            ttfts = [r.ttft_ms for r in successes if r.ttft_ms > 0]
            tps_list = [r.tokens_per_sec for r in successes if r.tokens_per_sec > 0]

            ttft_stats = compute_stats(ttfts)
            tps_stats = compute_stats(tps_list)

            print(f"    ---")
            print(f"    Wall: {result.wall_ms:.0f}ms  "
                  f"Throughput: {result.aggregate_throughput:.1f} tok/s  "
                  f"TTFT P50: {ttft_stats['p50']:.0f}ms  "
                  f"tok/s P50: {tps_stats['p50']:.1f}")

            # Build JSON record
            round_data = {
                "concurrency": conc,
                "round": rnd,
                "wall_ms": result.wall_ms,
                "total_tokens": result.total_tokens,
                "aggregate_throughput": result.aggregate_throughput,
                "ttft": ttft_stats,
                "tokens_per_sec": tps_stats,
                "requests": [],
            }
            for r in result.requests:
                round_data["requests"].append({
                    "id": r.request_id,
                    "prompt_label": r.prompt_label,
                    "ttft_ms": r.ttft_ms,
                    "total_ms": r.total_ms,
                    "token_count": r.token_count,
                    "tokens_per_sec": r.tokens_per_sec,
                    "eval_count": r.eval_count,
                    "eval_duration_ns": r.eval_duration_ns,
                    "prompt_eval_duration_ns": r.prompt_eval_duration_ns,
                    "error": r.error,
                })
            all_rounds.append(round_data)

    # Write output
    output = {
        "cache_label": args.cache_label,
        "model": args.model,
        "url": args.url,
        "tokens_per_request": args.tokens,
        "rounds_per_level": args.rounds,
        "warmup_count": args.warmup,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": all_rounds,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


# -- Compare mode --------------------------------------------------------------

def load_results(path: str) -> dict:
    """Load a JSON results file."""
    with open(path) as f:
        return json.load(f)


def aggregate_by_concurrency(data: dict) -> dict[int, dict]:
    """Aggregate rounds by concurrency level, computing cross-round stats."""
    by_conc: dict[int, list[dict]] = {}
    for entry in data["results"]:
        conc = entry["concurrency"]
        by_conc.setdefault(conc, []).append(entry)

    aggregated = {}
    for conc, rounds in sorted(by_conc.items()):
        ttft_p50s = [r["ttft"]["p50"] for r in rounds if r["ttft"]["p50"] > 0]
        tps_p50s = [r["tokens_per_sec"]["p50"] for r in rounds if r["tokens_per_sec"]["p50"] > 0]
        walls = [r["wall_ms"] for r in rounds]
        throughputs = [r["aggregate_throughput"] for r in rounds]

        aggregated[conc] = {
            "ttft_p50": statistics.mean(ttft_p50s) if ttft_p50s else 0,
            "tps_p50": statistics.mean(tps_p50s) if tps_p50s else 0,
            "wall_ms": statistics.mean(walls) if walls else 0,
            "aggregate_throughput": statistics.mean(throughputs) if throughputs else 0,
            "n_rounds": len(rounds),
        }

    return aggregated


def cmd_compare(args: argparse.Namespace):
    """Compare multiple KV cache benchmark results."""

    files = args.files
    if len(files) < 1:
        print("ERROR: provide at least one JSON result file", file=sys.stderr)
        sys.exit(1)

    datasets = []
    for path in files:
        try:
            data = load_results(path)
            datasets.append(data)
        except Exception as e:
            print(f"WARNING: could not load {path}: {e}", file=sys.stderr)

    if not datasets:
        print("ERROR: no valid result files loaded", file=sys.stderr)
        sys.exit(1)

    # Find F16 baseline (if present)
    f16_data = None
    for ds in datasets:
        if ds["cache_label"].upper() == "F16":
            f16_data = aggregate_by_concurrency(ds)
            break

    # Build rows
    rows = []
    for ds in datasets:
        label = ds["cache_label"]
        agg = aggregate_by_concurrency(ds)
        for conc in sorted(agg.keys()):
            stats = agg[conc]
            # Compute vs F16
            vs_ttft = ""
            vs_tps = ""
            vs_wall = ""
            if f16_data and conc in f16_data and label.upper() != "F16":
                baseline = f16_data[conc]
                if baseline["ttft_p50"] > 0 and stats["ttft_p50"] > 0:
                    diff = ((stats["ttft_p50"] - baseline["ttft_p50"]) / baseline["ttft_p50"]) * 100
                    vs_ttft = f"{diff:+.1f}%"
                if baseline["tps_p50"] > 0 and stats["tps_p50"] > 0:
                    diff = ((stats["tps_p50"] - baseline["tps_p50"]) / baseline["tps_p50"]) * 100
                    vs_tps = f"{diff:+.1f}%"
                if baseline["wall_ms"] > 0 and stats["wall_ms"] > 0:
                    diff = ((stats["wall_ms"] - baseline["wall_ms"]) / baseline["wall_ms"]) * 100
                    vs_wall = f"{diff:+.1f}%"

            rows.append({
                "cache": label,
                "conc": conc,
                "ttft_p50": stats["ttft_p50"],
                "tps_p50": stats["tps_p50"],
                "wall_ms": stats["wall_ms"],
                "agg_tps": stats["aggregate_throughput"],
                "vs_ttft": vs_ttft,
                "vs_tps": vs_tps,
                "vs_wall": vs_wall,
            })

    # Print table
    if args.markdown:
        _print_markdown_table(rows)
    else:
        _print_plain_table(rows)


def _print_plain_table(rows: list[dict]):
    """Print comparison as a plain-text table."""
    header = (
        f"  {'Cache Type':>12} | {'Conc':>4} | {'TTFT P50':>10} | "
        f"{'tok/s P50':>10} | {'Agg tok/s':>10} | {'Wall Time':>10} | {'vs F16':>18}"
    )
    sep = f"  {'─' * 12}-+-{'─' * 4}-+-{'─' * 10}-+-{'─' * 10}-+-{'─' * 10}-+-{'─' * 10}-+-{'─' * 18}"

    print()
    print(f"  {'=' * 86}")
    print(f"  TurboQuant KV Cache Comparison")
    print(f"  {'=' * 86}")
    print(header)
    print(sep)

    for row in rows:
        vs = ""
        parts = []
        if row["vs_tps"]:
            parts.append(f"tps {row['vs_tps']}")
        if row["vs_wall"]:
            parts.append(f"wall {row['vs_wall']}")
        vs = ", ".join(parts) if parts else ("baseline" if not row["vs_ttft"] and not row["vs_tps"] else "")

        # For F16 rows where vs columns are empty, show "baseline"
        if not row["vs_ttft"] and not row["vs_tps"] and not row["vs_wall"]:
            vs = "baseline"

        print(
            f"  {row['cache']:>12} | {row['conc']:>4} | "
            f"{row['ttft_p50']:>8.1f}ms | "
            f"{row['tps_p50']:>8.1f}/s | "
            f"{row['agg_tps']:>8.1f}/s | "
            f"{row['wall_ms']:>8.0f}ms | "
            f"{vs:>18}"
        )

    print()


def _print_markdown_table(rows: list[dict]):
    """Print comparison as a Markdown table."""
    print()
    print("| Cache Type | Concurrency | TTFT P50 | tok/s P50 | Agg tok/s | Wall Time | vs F16 |")
    print("|------------|-------------|----------|-----------|-----------|-----------|--------|")

    for row in rows:
        vs_parts = []
        if row["vs_tps"]:
            vs_parts.append(f"tps {row['vs_tps']}")
        if row["vs_wall"]:
            vs_parts.append(f"wall {row['vs_wall']}")
        vs = ", ".join(vs_parts) if vs_parts else "baseline"

        print(
            f"| {row['cache']} | {row['conc']} | "
            f"{row['ttft_p50']:.1f}ms | "
            f"{row['tps_p50']:.1f}/s | "
            f"{row['agg_tps']:.1f}/s | "
            f"{row['wall_ms']:.0f}ms | "
            f"{vs} |"
        )

    print()


# -- CLI -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EULLM TurboQuant KV Cache Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # collect subcommand
    collect_p = subparsers.add_parser("collect", help="Collect benchmark data")
    collect_p.add_argument("--url", default="http://localhost:11434",
                           help="EULLM engine URL (default: http://localhost:11434)")
    collect_p.add_argument("--model", default="qwen3-14b",
                           help="Model name (default: qwen3-14b)")
    collect_p.add_argument("--cache-label", required=True,
                           help="KV cache type label, e.g. F16, TQ4_0, TQ3_0")
    collect_p.add_argument("--concurrency", default="1,4,8",
                           help="Comma-separated concurrency levels (default: 1,4,8)")
    collect_p.add_argument("--tokens", type=int, default=150,
                           help="Max tokens per request (default: 150)")
    collect_p.add_argument("--rounds", type=int, default=3,
                           help="Repeats per concurrency level (default: 3)")
    collect_p.add_argument("--warmup", type=int, default=1,
                           help="Number of warmup requests (default: 1)")
    collect_p.add_argument("--output", default=None,
                           help="JSON output file (default: turboquant_<label>.json)")

    # compare subcommand
    compare_p = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_p.add_argument("files", nargs="+", help="JSON result files to compare")
    compare_p.add_argument("--markdown", action="store_true",
                           help="Output as Markdown table")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        asyncio.run(cmd_collect(args))
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()

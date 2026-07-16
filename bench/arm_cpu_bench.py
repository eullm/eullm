#!/usr/bin/env python3
"""POSCAR WP4 / T4.1 baseline — separate PREFILL and DECODE throughput on CPU.

Measures prefill tok/s (across a sweep of prompt lengths) and decode tok/s
(at a fixed generation length) against a running `eullm run`/`eullm serve`
instance, with no GPU involved. Meant to be re-run on the same board over
time as a regression baseline (T4.1), not a one-off number.

Why client-side timing, not server-reported duration: the server's own
per-request stats (`total_duration`/`eval_duration` in the Ollama-compatible
response) still lump prefill and decode into one timer — the fix for that
(separate prefill/decode histograms via `/metrics`) is tracked as roadmap
item 0.7-B and not implemented yet. Until then, TTFT (time to first token)
is the standard proxy for prefill wall time, and (last_token - first_token)
for decode wall time — the same technique bench/stress_test.py already uses
for its per-request `generation_ms`. On localhost (the expected setup: this
script and the server on the same board) network overhead in TTFT is
negligible, so the split is accurate enough for a CPU baseline.

This script does NOT set CPU affinity or thread count itself — pin the
server process (taskset / OMP env vars) and pass --threads to `eullm run`
BEFORE starting it, then use --notes here to record what you used, so the
JSON output is self-documenting. See docs/arm-cix-p1-cpu-profile.md for the
recommended pinning recipe for this SoC.

Requires: pip install aiohttp

Usage:
    python bench/arm_cpu_bench.py --url http://localhost:11434 --model qwen3-8b \\
        --notes "8 big A720 cores, taskset 4-11, --threads 8" \\
        --json t4.1-baseline-2026-07-16.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time

try:
    import aiohttp
except ImportError:
    print("This script requires aiohttp: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

# Repeated filler text used to build prompts of increasing (approximate)
# length. The actual tokenized length used for tok/s is always the server's
# own `prompt_eval_count` from the response, never assumed from word count.
FILLER_SENTENCE = (
    "The European approach to sovereign artificial intelligence emphasizes "
    "local control over data, compute, and model weights, running entirely "
    "on infrastructure within national or union borders. "
)

DEFAULT_PREFILL_WORD_TARGETS = [128, 512, 2048, 4096]
DEFAULT_DECODE_TOKENS = 128
DEFAULT_DECODE_PROMPT = "Write a short paragraph about renewable energy in Europe."


def build_prompt(word_target):
    words_per_sentence = len(FILLER_SENTENCE.split())
    repeats = max(1, word_target // words_per_sentence)
    return (FILLER_SENTENCE * repeats).strip()


class PowerSampler:
    """Periodically reads a sysfs power file during a benchmark window.

    Documented hook: if your board exposes power telemetry (INA2xx under
    /sys/class/hwmon/*, a vendor PMIC sysfs node, etc.) point --power-sysfs-path
    at the raw value file and set --power-scale-to-watts to convert it (default
    1e-6, i.e. assumes the file reports microwatts like the standard hwmon
    `powerX_input` convention). If no such node is documented for your board,
    leave it unset — the report will say so explicitly instead of guessing.
    """

    def __init__(self, sysfs_path, scale_to_watts, interval=0.2):
        self.sysfs_path = sysfs_path
        self.scale_to_watts = scale_to_watts
        self.interval = interval
        self.samples = []
        self._task = None
        self._stop = False

    def available(self):
        if not self.sysfs_path:
            return False
        try:
            with open(self.sysfs_path) as f:
                f.read()
            return True
        except OSError:
            return False

    async def _run(self):
        while not self._stop:
            try:
                with open(self.sysfs_path) as f:
                    raw = float(f.read().strip())
                self.samples.append(raw * self.scale_to_watts)
            except (OSError, ValueError):
                pass
            await asyncio.sleep(self.interval)

    def start(self):
        if self.available():
            self._task = asyncio.ensure_future(self._run())

    async def stop(self):
        self._stop = True
        if self._task:
            await self._task

    def average_watts(self):
        return statistics.mean(self.samples) if self.samples else None


async def stream_generate(session, base_url, model, prompt, max_tokens, timeout):
    """POST /api/generate (stream:true) and return (ttft_s, total_s, prompt_tokens, gen_tokens)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "max_tokens": max_tokens,
        "options": {"num_predict": max_tokens},
    }
    start = time.monotonic()
    ttft = None
    last_token_time = None
    prompt_tokens = None
    gen_tokens = None

    async with session.post(
        f"{base_url}/api/generate", json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("done"):
                prompt_tokens = event.get("prompt_eval_count")
                gen_tokens = event.get("eval_count")
                if last_token_time is None:
                    last_token_time = time.monotonic()
                break
            if event.get("response"):
                now = time.monotonic()
                if ttft is None:
                    ttft = now - start
                last_token_time = now

    total = time.monotonic() - start
    return ttft, total, prompt_tokens, gen_tokens, last_token_time - start if last_token_time else total


async def run_prefill_sweep(session, args):
    print(f"\n[prefill] sweeping prompt lengths: {args.prefill_word_targets}")
    results = []
    for word_target in args.prefill_word_targets:
        prompt = build_prompt(word_target)
        # max_tokens=1: isolate prefill as much as possible. The one token
        # produced is unavoidable (llama.cpp always samples one token from
        # the prefill logits — see engine/src/inference/scheduler.rs), but a
        # single decode step is negligible next to prefilling thousands of
        # tokens on CPU.
        ttft, total, prompt_tokens, gen_tokens, _ = await stream_generate(
            session, args.url, args.model, prompt, max_tokens=1, timeout=args.timeout,
        )
        if ttft is None or not prompt_tokens:
            print(f"  ~{word_target} words: FAILED to get a timed response")
            continue
        tok_s = prompt_tokens / ttft
        print(f"  {prompt_tokens:5d} prompt tokens: TTFT={ttft*1000:8.1f}ms  prefill={tok_s:7.1f} tok/s")
        results.append({"prompt_tokens": prompt_tokens, "ttft_s": ttft, "prefill_tok_s": tok_s})
    return results


async def run_decode_bench(session, args, power_sampler):
    print(f"\n[decode] {args.decode_tokens} tokens, {args.decode_repeats} repeat(s)")
    runs = []
    for i in range(args.decode_repeats):
        if power_sampler:
            power_sampler.start()
        ttft, total, prompt_tokens, gen_tokens, last_token_s = await stream_generate(
            session, args.url, args.model, args.decode_prompt,
            max_tokens=args.decode_tokens, timeout=args.timeout,
        )
        if power_sampler:
            await power_sampler.stop()

        if ttft is None or not gen_tokens or gen_tokens < 2:
            print(f"  repeat {i}: FAILED or too few tokens to measure decode rate")
            continue
        decode_s = last_token_s - ttft
        decode_tok_s = (gen_tokens - 1) / decode_s if decode_s > 0 else 0.0
        print(f"  repeat {i}: TTFT={ttft*1000:7.1f}ms  decode={decode_tok_s:7.1f} tok/s over {gen_tokens} tokens")
        runs.append({
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "ttft_s": ttft,
            "decode_s": decode_s,
            "decode_tok_s": decode_tok_s,
        })
    return runs


async def main_async(args):
    print(f"eullm ARM CPU baseline (POSCAR WP4 / T4.1) - {args.url}  model={args.model}")
    if args.notes:
        print(f"notes: {args.notes}")

    power_sampler = None
    if args.power_sysfs_path:
        power_sampler = PowerSampler(args.power_sysfs_path, args.power_scale_to_watts)
        if not power_sampler.available():
            print(f"\n[power] WARNING: cannot read {args.power_sysfs_path} - power will not be reported")
            power_sampler = None
    else:
        print("\n[power] no --power-sysfs-path given - power/perf-per-watt will not be reported "
              "(see docs/arm-cix-p1-cpu-profile.md for how to find and wire one in for your board)")

    async with aiohttp.ClientSession() as session:
        prefill_results = await run_prefill_sweep(session, args)
        decode_results = await run_decode_bench(session, args, power_sampler)

    avg_decode_tok_s = statistics.mean(r["decode_tok_s"] for r in decode_results) if decode_results else None
    avg_watts = power_sampler.average_watts() if power_sampler else None

    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in prefill_results:
        print(f"  prefill  {r['prompt_tokens']:5d} tok  {r['prefill_tok_s']:7.1f} tok/s")
    if avg_decode_tok_s is not None:
        print(f"  decode   avg over {len(decode_results)} run(s): {avg_decode_tok_s:7.1f} tok/s")
    if avg_watts is not None:
        print(f"  power    avg during decode: {avg_watts:.2f} W")
        if avg_decode_tok_s:
            print(f"  perf/W   {avg_decode_tok_s / avg_watts:.2f} tok/s per W")
    print("=" * 60)

    if args.json:
        payload = {
            "url": args.url,
            "model": args.model,
            "notes": args.notes,
            "prefill": prefill_results,
            "decode": decode_results,
            "decode_tok_s_avg": avg_decode_tok_s,
            "power_watts_avg": avg_watts,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")

    return 0 if (prefill_results and decode_results) else 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--model", required=True)
    p.add_argument("--notes", default="", help="free-text run metadata (threads/pinning used, board state) saved into --json")
    p.add_argument("--prefill-word-targets", type=int, nargs="+", default=DEFAULT_PREFILL_WORD_TARGETS,
                   help="approximate prompt sizes (in words) to sweep; actual tokenized size is measured and reported")
    p.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    p.add_argument("--decode-prompt", default=DEFAULT_DECODE_PROMPT)
    p.add_argument("--decode-repeats", type=int, default=3)
    p.add_argument("--power-sysfs-path", default=None, help="sysfs file to sample during decode, e.g. /sys/class/hwmon/hwmon0/power1_input")
    p.add_argument("--power-scale-to-watts", type=float, default=1e-6, help="multiply the raw sysfs value by this to get Watts (default assumes microwatts)")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--json", default=None, help="write machine-readable results here for baseline tracking over time")
    return p.parse_args()


def main():
    args = parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

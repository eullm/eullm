#!/usr/bin/env python3
"""
TurboQuant Quality Benchmark — Measure precision impact of KV cache quantization.

Sends identical prompts to the EULLM engine with different KV cache types
and compares output quality. Tests: matrix operations, math, factual Q&A,
logic reasoning, code generation.

Usage:
  # Run against a live engine (start it with the desired --cache-type-k/v)
  python3 bench/turboquant_quality.py --url http://localhost:11434 --label F16

  # Compare multiple result files
  python3 bench/turboquant_quality.py compare results_f16.json results_tq4.json results_tq3.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

# ── Test prompts with verifiable answers ─────────────────────────────────────

TESTS = [
    # ── Matrix operations ────────────────────────────────────────────────
    {
        "id": "matrix_multiply",
        "category": "matrix",
        "prompt": "Compute the matrix product of A = [[1,2],[3,4]] and B = [[5,6],[7,8]]. "
                  "Return ONLY the result matrix in the format [[a,b],[c,d]] with no explanation.",
        "expected": "[[19,22],[43,50]]",
        "check": "exact_normalized",
    },
    {
        "id": "matrix_determinant",
        "category": "matrix",
        "prompt": "What is the determinant of the matrix [[3,8],[4,6]]? "
                  "Return ONLY the number, nothing else.",
        "expected": "-14",
        "check": "contains_number",
    },
    {
        "id": "matrix_transpose",
        "category": "matrix",
        "prompt": "Transpose the matrix [[1,2,3],[4,5,6],[7,8,9]]. "
                  "Return ONLY the result in the format [[a,b,c],[d,e,f],[g,h,i]].",
        "expected": "[[1,4,7],[2,5,8],[3,6,9]]",
        "check": "exact_normalized",
    },
    {
        "id": "matrix_inverse_2x2",
        "category": "matrix",
        "prompt": "What is the inverse of the matrix [[4,7],[2,6]]? "
                  "Return the result as [[a,b],[c,d]] with fractions or decimals. No explanation.",
        "expected_contains": ["0.6", "3/5", "-0.7", "-7/10", "-0.2", "-1/5", "0.4", "2/5"],
        "check": "contains_any",
    },
    {
        "id": "matrix_trace",
        "category": "matrix",
        "prompt": "What is the trace of [[5,1,2],[0,3,1],[2,0,7]]? Return ONLY the number.",
        "expected": "15",
        "check": "contains_number",
    },
    # ── Math ─────────────────────────────────────────────────────────────
    {
        "id": "math_arithmetic",
        "category": "math",
        "prompt": "What is 347 × 283? Return ONLY the number.",
        "expected": "98201",
        "check": "contains_number",
    },
    {
        "id": "math_prime",
        "category": "math",
        "prompt": "Is 997 a prime number? Answer ONLY 'yes' or 'no'.",
        "expected": "yes",
        "check": "contains_word",
    },
    {
        "id": "math_sqrt",
        "category": "math",
        "prompt": "What is the square root of 1764? Return ONLY the integer.",
        "expected": "42",
        "check": "contains_number",
    },
    {
        "id": "math_sequence",
        "category": "math",
        "prompt": "What is the next number in the sequence: 2, 6, 18, 54, ...? Return ONLY the number.",
        "expected": "162",
        "check": "contains_number",
    },
    {
        "id": "math_fraction",
        "category": "math",
        "prompt": "Simplify the fraction 84/126. Return ONLY the simplified fraction.",
        "expected": "2/3",
        "check": "contains_word",
    },
    # ── Factual Q&A ──────────────────────────────────────────────────────
    {
        "id": "fact_capital",
        "category": "factual",
        "prompt": "What is the capital of Slovenia? Return ONLY the city name.",
        "expected": "Ljubljana",
        "check": "contains_word",
    },
    {
        "id": "fact_element",
        "category": "factual",
        "prompt": "What is the chemical symbol for tungsten? Return ONLY the symbol.",
        "expected": "W",
        "check": "contains_word",
    },
    {
        "id": "fact_year",
        "category": "factual",
        "prompt": "In what year was the Euro currency introduced for electronic transactions? Return ONLY the year.",
        "expected": "1999",
        "check": "contains_number",
    },
    {
        "id": "fact_planet",
        "category": "factual",
        "prompt": "Which planet in our solar system has the most moons? Return ONLY the planet name.",
        "expected": "Saturn",
        "check": "contains_word",
    },
    {
        "id": "fact_gdpr",
        "category": "factual",
        "prompt": "What is the maximum fine under GDPR as a percentage of global annual turnover? Return ONLY the percentage.",
        "expected": "4%",
        "check": "contains_word",
    },
    # ── Logic / Reasoning ────────────────────────────────────────────────
    {
        "id": "logic_syllogism",
        "category": "logic",
        "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Answer ONLY 'yes' or 'no'.",
        "expected": "no",
        "check": "contains_word",
    },
    {
        "id": "logic_sequence",
        "category": "logic",
        "prompt": "If A=1, B=2, C=3... what is the sum of the letters in 'EULLM'? Return ONLY the number.",
        "expected": "63",
        "check": "contains_number",
    },
    {
        "id": "logic_pattern",
        "category": "logic",
        "prompt": "Complete the pattern: 1, 1, 2, 3, 5, 8, 13, __. Return ONLY the number.",
        "expected": "21",
        "check": "contains_number",
    },
    # ── Code generation ──────────────────────────────────────────────────
    {
        "id": "code_fizzbuzz",
        "category": "code",
        "prompt": "What does FizzBuzz output for the number 15? Return ONLY the output word.",
        "expected": "FizzBuzz",
        "check": "contains_word",
    },
    {
        "id": "code_regex",
        "category": "code",
        "prompt": "Write a regex that matches a valid IPv4 address. Return ONLY the regex pattern, nothing else.",
        "expected_contains": ["\\d", "[0-9]", "\\."],
        "check": "contains_any",
    },
]


# ── Answer checking ──────────────────────────────────────────────────────────

def strip_thinking(s: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 thinking mode)."""
    import re
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()


def normalize(s: str) -> str:
    """Remove whitespace, newlines, backticks, markdown, and lowercase."""
    s = strip_thinking(s)
    s = s.replace(" ", "").replace("\n", "").replace("`", "").replace("*", "")
    s = s.replace("\\times", "×").replace("\\cdot", "·")
    return s.strip().lower()


def extract_last_line(s: str) -> str:
    """Get the last non-empty line — often the actual answer after explanation."""
    s = strip_thinking(s)
    lines = [l.strip() for l in s.strip().split('\n') if l.strip()]
    return lines[-1] if lines else s


def check_answer(test: dict, response: str) -> tuple[bool, str]:
    """Check if the response matches the expected answer. Returns (pass, detail)."""
    mode = test["check"]
    resp = strip_thinking(response).strip()
    resp_last = extract_last_line(response)

    if mode == "exact_normalized":
        expected = normalize(test["expected"])
        # Check full response AND last line (model may explain then give answer)
        actual_full = normalize(resp)
        actual_last = normalize(resp_last)
        ok = expected in actual_full or expected in actual_last
        return ok, f"expected={test['expected']} last_line={resp_last[:80]}"

    elif mode == "contains_number":
        expected = test["expected"]
        ok = expected in resp or expected in resp_last
        return ok, f"expected={expected} in response={resp_last[:80]}"

    elif mode == "contains_word":
        expected = test["expected"].lower()
        ok = expected in resp.lower() or expected in resp_last.lower()
        return ok, f"expected='{expected}' in response={resp_last[:80]}"

    elif mode == "contains_any":
        candidates = test["expected_contains"]
        combined = resp.lower() + " " + resp_last.lower()
        ok = any(c.lower() in combined for c in candidates)
        matched = [c for c in candidates if c.lower() in combined]
        return ok, f"matched={matched} in response={resp_last[:80]}"

    return False, "unknown check mode"


# ── API client ───────────────────────────────────────────────────────────────

async def send_prompt(session: aiohttp.ClientSession, url: str, model: str,
                      prompt: str, temperature: float = 0.0) -> str:
    """Send a prompt and return the full response text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 256},
    }
    try:
        async with session.post(f"{url}/api/chat", json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                return f"[ERROR: HTTP {resp.status}]"
            data = await resp.json()
            return data.get("message", {}).get("content", "[no content]")
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Collect mode ─────────────────────────────────────────────────────────────

async def collect(args):
    results = []
    passed = 0
    total = len(TESTS)

    print(f"Running {total} quality tests against {args.url} (label={args.label}, model={args.model})")
    print(f"Temperature: {args.temperature}")
    print()

    async with aiohttp.ClientSession() as session:
        for i, test in enumerate(TESTS):
            response = await send_prompt(session, args.url, args.model, test["prompt"], args.temperature)
            ok, detail = check_answer(test, response)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1

            print(f"  [{status}] {test['id']:<25} {detail}")

            results.append({
                "id": test["id"],
                "category": test["category"],
                "prompt": test["prompt"],
                "response": response,
                "passed": ok,
                "detail": detail,
            })

    score = passed / total * 100
    print(f"\n{'='*60}")
    print(f"  {args.label}: {passed}/{total} passed ({score:.1f}%)")
    print(f"{'='*60}")

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    print(f"\n  {'Category':<12} {'Score':>8}")
    print(f"  {'-'*22}")
    for cat, vals in sorted(categories.items()):
        pct = vals['passed'] / vals['total'] * 100
        print(f"  {cat:<12} {vals['passed']}/{vals['total']} ({pct:.0f}%)")

    # Save JSON
    output = {
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score": {"passed": passed, "total": total, "percent": score},
        "categories": categories,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {args.output}")


# ── Compare mode ─────────────────────────────────────────────────────────────

def compare(args):
    files = args.files
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))

    print(f"\n{'='*70}")
    print(f"  TurboQuant Quality Comparison")
    print(f"{'='*70}\n")

    # Overall scores
    print(f"  {'Cache Type':<20} {'Score':>10} {'Passed':>10} {'Total':>8}")
    print(f"  {'-'*50}")
    baseline_score = None
    for d in data:
        s = d["score"]
        label = d["label"]
        delta = ""
        if baseline_score is None:
            baseline_score = s["percent"]
        else:
            diff = s["percent"] - baseline_score
            delta = f"  ({diff:+.1f}%)"
        print(f"  {label:<20} {s['percent']:>9.1f}% {s['passed']:>8}/{s['total']}{delta}")

    # Category comparison
    all_cats = sorted(set(c for d in data for c in d.get("categories", {})))
    print(f"\n  {'Category':<12}", end="")
    for d in data:
        print(f"  {d['label']:>12}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in data:
        print(f"  {'-'*12}", end="")
    print()

    for cat in all_cats:
        print(f"  {cat:<12}", end="")
        for d in data:
            cats = d.get("categories", {})
            if cat in cats:
                v = cats[cat]
                print(f"  {v['passed']}/{v['total']:>2} ({v['passed']/v['total']*100:3.0f}%)", end="")
            else:
                print(f"  {'n/a':>12}", end="")
        print()

    # Per-test diff (show only failures that differ)
    print(f"\n  Divergent results (different pass/fail across cache types):")
    print(f"  {'-'*60}")
    any_diff = False
    test_ids = [r["id"] for r in data[0]["results"]]
    for tid in test_ids:
        results_for_test = []
        for d in data:
            r = next((x for x in d["results"] if x["id"] == tid), None)
            if r:
                results_for_test.append((d["label"], r["passed"], r["response"][:60]))
        statuses = [r[1] for r in results_for_test]
        if len(set(statuses)) > 1:
            any_diff = True
            print(f"\n  {tid}:")
            for label, passed, resp in results_for_test:
                status = "PASS" if passed else "FAIL"
                print(f"    [{status}] {label:<15} → {resp}")

    if not any_diff:
        print(f"  None — all cache types produced the same pass/fail results!")

    if args.markdown:
        print(f"\n\n### Markdown table\n")
        print(f"| Cache Type | Score | Matrix | Math | Factual | Logic | Code |")
        print(f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for d in data:
            s = d["score"]
            cats = d.get("categories", {})
            row = f"| {d['label']} | {s['percent']:.0f}% |"
            for cat in ["matrix", "math", "factual", "logic", "code"]:
                if cat in cats:
                    v = cats[cat]
                    row += f" {v['passed']}/{v['total']} |"
                else:
                    row += " n/a |"
            print(row)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Quality Benchmark")
    sub = parser.add_subparsers(dest="command")

    # collect
    c = sub.add_parser("collect", help="Run quality tests against a live engine")
    c.add_argument("--url", default="http://localhost:11434")
    c.add_argument("--model", default="qwen3-14b")
    c.add_argument("--label", required=True, help="Cache type label (e.g. F16, TQ4_0, TQ3_0)")
    c.add_argument("--temperature", type=float, default=0.0, help="Use 0.0 for deterministic output")
    c.add_argument("--output", "-o", help="Save results to JSON file")

    # compare
    p = sub.add_parser("compare", help="Compare multiple result files")
    p.add_argument("files", nargs="+", help="JSON result files to compare")
    p.add_argument("--markdown", action="store_true", help="Output markdown table")

    args = parser.parse_args()

    if args.command == "collect":
        asyncio.run(collect(args))
    elif args.command == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

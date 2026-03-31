#!/usr/bin/env python3
"""
TurboQuant Math Accuracy Test

Separates two questions that the KV stress test conflates:

  1. Can the model compute correctly at all?       (direct math, zero filler)
  2. Can it recall the problem and then compute?   (same math, with filler)

If a model fails (1), the delayed-matrix KV test is meaningless.
If a model passes (1) but fails (2), the issue is KV recall.
If a model passes (2) in F16 but fails in TurboQuant, the issue is KV corruption.

Tests
─────
A. 2×2 matrix multiplication × 5 different matrix pairs
B. 3×3 matrix multiplication × 3 pairs
C. Scalar arithmetic chains  × 5 expressions
D. All of the above with filler (200 / 500 / 1000 tokens)

Answer format handling
──────────────────────
Accepts:  [[a,b],[c,d]]  |  [[a, b], [c, d]]  |  rows on lines  |  \\boxed{...}
"""

import argparse
import asyncio
import json
import re
import sys
import time
import random

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)


# ── Filler ────────────────────────────────────────────────────────────────────

FILLER_PARAGRAPHS = [
    "The European Union's digital strategy aims to make this transformation work for people and businesses, while helping to achieve its target of a climate-neutral Europe by 2050. The Commission has outlined key policies and frameworks to guide this transition.",
    "Recent advances in semiconductor manufacturing have enabled the production of chips at the 3nm node, dramatically increasing transistor density while reducing power consumption. TSMC, Samsung, and Intel are competing to deliver these next-generation processors.",
    "The Mediterranean basin is home to approximately 25,000 plant species, of which about 50% are endemic. This biodiversity hotspot faces threats from urbanisation, agricultural intensification, and climate change. Conservation efforts include marine protected areas.",
    "Cloud computing has fundamentally changed how organisations deploy and manage IT infrastructure. The shift from capital expenditure to operational expenditure models has enabled startups and enterprises alike to scale their operations globally.",
    "Italian Renaissance art underwent several distinct phases, from the early experiments of Giotto in the 13th century to the High Renaissance works of Leonardo, Michelangelo, and Raphael. Linear perspective transformed the representation of space.",
    "The carbon cycle involves the exchange of carbon between the atmosphere, oceans, terrestrial biosphere, and geological formations. Anthropogenic emissions have disrupted this cycle, leading to an increase in atmospheric CO2 concentrations.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be impractical for classical computers. Researchers have demonstrated quantum advantage for specific problems.",
    "The logistics industry handles approximately 65 billion parcels annually worldwide, with e-commerce driving a significant portion of this volume. Last-mile delivery remains the most expensive segment, accounting for up to 53% of total shipping costs.",
    "Volcanic activity along tectonic plate boundaries shapes the Earth's surface through both constructive and destructive processes. The Ring of Fire contains approximately 75% of the world's active volcanoes and accounts for about 90% of earthquakes.",
    "Machine learning models for natural language processing have grown exponentially in size. This scaling has been accompanied by emergent capabilities including in-context learning, chain-of-thought reasoning, and cross-lingual transfer.",
]


def generate_filler(target_tokens: int) -> str:
    target_chars = target_tokens * 4
    paragraphs = []
    total = 0
    while total < target_chars:
        p = random.choice(FILLER_PARAGRAPHS)
        paragraphs.append(p)
        total += len(p)
    return "\n\n".join(paragraphs)


# ── Answer normalisation ──────────────────────────────────────────────────────

def strip_thinking(s: str) -> str:
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()


def extract_boxed(s: str) -> str:
    """Extract content from \\boxed{...} or $\\boxed{...}$."""
    m = re.search(r'\\boxed\{([^}]+)\}', s)
    return m.group(1) if m else ""


def normalise_matrix(s: str) -> str:
    """
    Normalise a matrix string to [[a,b],[c,d]] canonical form.

    Handles:
      [[21, 29], [32, 44]]
      21 29 / 32 44
      21 29\n32 44
      \\begin{pmatrix}21 & 29 \\\\ 32 & 44\\end{pmatrix}
      \\boxed{[[21,29],[32,44]]}
    """
    s = strip_thinking(s)
    boxed = extract_boxed(s)
    if boxed:
        s = boxed + " " + s  # try boxed content first

    # Remove markdown fences, backticks
    s = re.sub(r'```[^\n]*\n?', '', s)
    s = s.replace('`', '').replace('*', '')

    # Try to find a bracketed matrix directly
    m = re.findall(r'\[\s*\[[\d,\s\-]+\]\s*,\s*\[[\d,\s\-]+\]\s*\]', s)
    if m:
        # Take the last match (often the final answer)
        raw = m[-1]
        nums = [int(x) for x in re.findall(r'-?\d+', raw)]
        if len(nums) == 4:
            return f"[[{nums[0]},{nums[1]}],[{nums[2]},{nums[3]}]]"
        if len(nums) == 9:
            return (f"[[{nums[0]},{nums[1]},{nums[2]}],"
                    f"[{nums[3]},{nums[4]},{nums[5]}],"
                    f"[{nums[6]},{nums[7]},{nums[8]}]]")

    # Try LaTeX pmatrix / bmatrix
    m = re.search(
        r'(?:p|b|v)?matrix\}(.*?)\\end\{(?:p|b|v)?matrix\}', s, re.DOTALL)
    if m:
        inner = m.group(1)
        rows = [r.strip() for r in re.split(r'\\\\', inner) if r.strip()]
        nums = []
        for row in rows:
            nums.append([int(x) for x in re.findall(r'-?\d+', row)])
        if len(nums) == 2 and all(len(r) == 2 for r in nums):
            return f"[[{nums[0][0]},{nums[0][1]}],[{nums[1][0]},{nums[1][1]}]]"
        if len(nums) == 3 and all(len(r) == 3 for r in nums):
            return (f"[[{nums[0][0]},{nums[0][1]},{nums[0][2]}],"
                    f"[{nums[1][0]},{nums[1][1]},{nums[1][2]}],"
                    f"[{nums[2][0]},{nums[2][1]},{nums[2][2]}]]")

    # Fallback: grab all integers, reshape
    nums = [int(x) for x in re.findall(r'-?\d+', s)]
    if len(nums) >= 4:
        # try last 4 for 2x2
        n = nums[-4:]
        return f"[[{n[0]},{n[1]}],[{n[2]},{n[3]}]]"

    return ""


def check_matrix(response: str, expected: str) -> tuple[bool, str]:
    got = normalise_matrix(response)
    exp = normalise_matrix(expected)
    ok = got == exp
    return ok, f"expected={exp} got={got}"


def check_scalar(response: str, expected: str) -> tuple[bool, str]:
    s = strip_thinking(response)
    boxed = extract_boxed(s)
    candidates = set()
    if boxed:
        nums = re.findall(r'-?\d+(?:\.\d+)?', boxed)
        candidates.update(nums)
    nums = re.findall(r'-?\d+(?:\.\d+)?', s)
    candidates.update(nums)
    ok = expected in candidates or expected.lstrip('0') in candidates
    preview = s[-80:].replace('\n', ' ')
    return ok, f"expected={expected} in response: ...{preview}"


# ── Matrix test data ──────────────────────────────────────────────────────────
# Each entry: (label, A_display, B_display, A_rows, B_rows, expected_rows)

def _mm(A, B):
    """Multiply two 2D lists of ints."""
    rows, cols = len(A), len(B[0])
    inner = len(B)
    C = [[sum(A[r][k] * B[k][c] for k in range(inner)) for c in range(cols)]
         for r in range(rows)]
    return C


def _fmt_matrix(rows):
    """Format as [[a,b],[c,d]]."""
    return "[" + ",".join("[" + ",".join(str(x) for x in r) + "]" for r in rows) + "]"


def _display_matrix(name, rows):
    inner = ",".join("[" + ",".join(str(x) for x in r) + "]" for r in rows)
    return f"{name} = [{inner}]"


MATRIX_2X2_CASES = [
    # (label, A, B)
    ("M1", [[3, 1], [4, 2]], [[5, 7], [6, 8]]),          # [[21,29],[32,44]]
    ("M2", [[1, 2], [3, 4]], [[5, 6], [7, 8]]),          # [[19,22],[43,50]]
    ("M3", [[2, 0], [1, 3]], [[4, 1], [2, 5]]),          # [[8,2],[10,16]]
    ("M4", [[5, 3], [2, 7]], [[1, 4], [6, 0]]),          # [[23,20],[44,8]]
    ("M5", [[0, 1], [1, 0]], [[3, 4], [5, 6]]),          # [[5,6],[3,4]]
]

MATRIX_3X3_CASES = [
    ("M6", [[1, 0, 2], [3, 1, 0], [0, 2, 1]],
           [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ("M7", [[2, 1, 0], [0, 3, 1], [1, 0, 2]],
           [[1, 0, 1], [2, 1, 0], [0, 1, 2]]),
    ("M8", [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
           [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
]

SCALAR_CASES = [
    ("S1", "((17 * 23) + 14) / 1", str((17 * 23) + 14)),         # 405
    ("S2", "3^4 + 2^5", str(3**4 + 2**5)),                        # 113
    ("S3", "(144 / 12) * (15 - 8) + 3", str((144 // 12) * (15 - 8) + 3)),  # 87
    ("S4", "7 * 8 + 6 * 9 + 5 * 10", str(7 * 8 + 6 * 9 + 5 * 10)),        # 160
    ("S5", "1000 - 13 * 37", str(1000 - 13 * 37)),                 # 519
]


# ── Build tests ───────────────────────────────────────────────────────────────

def make_matrix_direct_test(label, A, B, size="2x2"):
    C = _mm(A, B)
    A_disp = _display_matrix("A", A)
    B_disp = _display_matrix("B", B)
    expected = _fmt_matrix(C)
    prompt = (
        f"Compute the matrix product A × B.\n\n"
        f"{A_disp}\n{B_disp}\n\n"
        f"Output ONLY the resulting matrix in format {expected[:10]}... (no explanation)."
    )
    return {
        "id": f"direct_matrix_{label}",
        "category": "direct_math",
        "subtype": "matrix",
        "size": size,
        "filler_tokens": 0,
        "prompt": prompt,
        "expected": expected,
        "check": "matrix",
    }


def make_matrix_delayed_test(label, A, B, filler_tokens, size="2x2"):
    C = _mm(A, B)
    A_disp = _display_matrix("A", A)
    B_disp = _display_matrix("B", B)
    expected = _fmt_matrix(C)
    filler = generate_filler(filler_tokens)
    prompt = (
        f"Read these matrices carefully. Do NOT compute yet.\n\n"
        f"{A_disp}\n{B_disp}\n\n"
        f"[Do not compute yet — continue reading.]\n\n"
        f"{filler}\n\n"
        f"Now compute A × B.\n"
        f"Output ONLY the resulting matrix in format [[a,b],[c,d]]. No explanation."
    )
    return {
        "id": f"delayed_matrix_{label}_{filler_tokens}t",
        "category": "delayed_math",
        "subtype": "matrix",
        "size": size,
        "filler_tokens": filler_tokens,
        "prompt": prompt,
        "expected": expected,
        "check": "matrix",
    }


def make_scalar_direct_test(label, expr, expected):
    prompt = (
        f"Compute this arithmetic expression exactly:\n\n"
        f"  {expr}\n\n"
        f"Output ONLY the integer result. No explanation."
    )
    return {
        "id": f"direct_scalar_{label}",
        "category": "direct_math",
        "subtype": "scalar",
        "filler_tokens": 0,
        "prompt": prompt,
        "expected": expected,
        "check": "scalar",
    }


def make_scalar_delayed_test(label, expr, expected, filler_tokens):
    filler = generate_filler(filler_tokens)
    prompt = (
        f"Remember this expression. Do NOT compute it yet.\n\n"
        f"  EXPRESSION: {expr}\n\n"
        f"[Do not compute yet.]\n\n"
        f"{filler}\n\n"
        f"Now compute the expression you were asked to remember.\n"
        f"Output ONLY the integer result. No explanation."
    )
    return {
        "id": f"delayed_scalar_{label}_{filler_tokens}t",
        "category": "delayed_math",
        "subtype": "scalar",
        "filler_tokens": filler_tokens,
        "prompt": prompt,
        "expected": expected,
        "check": "scalar",
    }


def build_tests(filler_levels, matrix_sizes, skip_3x3=False, skip_scalar=False):
    tests = []

    # Direct 2x2
    for label, A, B in MATRIX_2X2_CASES:
        tests.append(make_matrix_direct_test(label, A, B, "2x2"))

    # Direct 3x3
    if not skip_3x3:
        for label, A, B in MATRIX_3X3_CASES:
            tests.append(make_matrix_direct_test(label, A, B, "3x3"))

    # Direct scalar
    if not skip_scalar:
        for label, expr, expected in SCALAR_CASES:
            tests.append(make_scalar_direct_test(label, expr, expected))

    # Delayed 2x2
    for label, A, B in MATRIX_2X2_CASES:
        for fl in filler_levels:
            tests.append(make_matrix_delayed_test(label, A, B, fl, "2x2"))

    # Delayed 3x3
    if not skip_3x3:
        for label, A, B in MATRIX_3X3_CASES:
            for fl in filler_levels:
                tests.append(make_matrix_delayed_test(label, A, B, fl, "3x3"))

    # Delayed scalar
    if not skip_scalar:
        for label, expr, expected in SCALAR_CASES:
            for fl in filler_levels:
                tests.append(make_scalar_delayed_test(label, expr, expected, fl))

    return tests


# ── API client ────────────────────────────────────────────────────────────────

async def send_prompt(session, url, model, prompt, temperature, system_prompt, timeout_s):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }
    try:
        async with session.post(
            f"{url}/api/chat", json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return f"[ERROR: HTTP {resp.status} — {body[:120]}]"
            data = await resp.json()
            return data.get("message", {}).get("content", "[no content]")
    except asyncio.TimeoutError:
        return f"[ERROR: timeout after {timeout_s}s]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Run ───────────────────────────────────────────────────────────────────────

async def collect(args):
    random.seed(0)
    filler_levels = [int(x) for x in args.filler.split(",") if x]
    tests = build_tests(
        filler_levels,
        matrix_sizes=args.matrix_sizes,
        skip_3x3=args.skip_3x3,
        skip_scalar=args.skip_scalar,
    )

    system_prompt = args.system_prompt or ""
    if args.math_model_prompt and not system_prompt:
        system_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    print(f"TurboQuant Math Accuracy Test")
    print(f"  URL:         {args.url}")
    print(f"  Model:       {args.model}")
    print(f"  Label:       {args.label}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Filler:      {filler_levels} tokens")
    print(f"  Tests:       {len(tests)}")
    if system_prompt:
        print(f"  System:      {system_prompt[:60]}...")
    print()

    results = []
    cats = {}

    async with aiohttp.ClientSession() as session:
        for test in tests:
            response = await send_prompt(
                session, args.url, args.model,
                test["prompt"], args.temperature,
                system_prompt, args.timeout,
            )

            if test["check"] == "matrix":
                passed, detail = check_matrix(response, test["expected"])
            else:
                passed, detail = check_scalar(response, test["expected"])

            status = "PASS" if passed else "FAIL"
            cat = test["category"]
            cats.setdefault(cat, {"pass": 0, "total": 0})
            cats[cat]["total"] += 1
            if passed:
                cats[cat]["pass"] += 1

            filler = test["filler_tokens"]
            filler_s = f"filler={filler:>5}t" if filler else "direct       "
            size = test.get("size", test["subtype"])
            print(f"  [{status}] {test['id']:<38} {filler_s}  {size:<4}  {detail[:60]}")

            results.append({
                "id": test["id"],
                "category": cat,
                "subtype": test["subtype"],
                "size": test.get("size", ""),
                "filler_tokens": filler,
                "passed": passed,
                "expected": test["expected"],
                "detail": detail,
                "response_preview": strip_thinking(response)[:300],
            })

    # Summary
    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)
    pct = total_pass / total * 100 if total else 0

    print(f"\n{'='*65}")
    print(f"  {args.label}: {total_pass}/{total} passed ({pct:.1f}%)")
    print(f"{'='*65}")

    print(f"\n  Category breakdown:")
    print(f"  {'Category':<20} {'Pass/Total':>12} {'%':>8}")
    print(f"  {'-'*42}")
    for cat, s in sorted(cats.items()):
        p = s['pass'] / s['total'] * 100 if s['total'] else 0
        print(f"  {cat:<20} {s['pass']}/{s['total']:>4}        {p:>5.1f}%")

    # Direct vs delayed breakdown
    print(f"\n  Direct vs Delayed:")
    print(f"  {'Type':<12} {'2x2':>8} {'3x3':>8} {'scalar':>8}")
    print(f"  {'-'*40}")
    for cat in ["direct_math", "delayed_math"]:
        row = {}
        for sub in ["matrix_2x2", "matrix_3x3", "scalar"]:
            if sub == "matrix_2x2":
                matches = [r for r in results if r["category"] == cat
                           and r["subtype"] == "matrix" and r.get("size") == "2x2"]
            elif sub == "matrix_3x3":
                matches = [r for r in results if r["category"] == cat
                           and r["subtype"] == "matrix" and r.get("size") == "3x3"]
            else:
                matches = [r for r in results if r["category"] == cat
                           and r["subtype"] == "scalar"]
            if matches:
                p = sum(1 for r in matches if r["passed"])
                row[sub] = f"{p}/{len(matches)}"
            else:
                row[sub] = "-"
        name = "direct" if cat == "direct_math" else "delayed"
        print(f"  {name:<12} {row['matrix_2x2']:>8} {row['matrix_3x3']:>8} {row['scalar']:>8}")

    # By filler level (delayed only)
    delayed = [r for r in results if r["category"] == "delayed_math"]
    if delayed and filler_levels:
        print(f"\n  Delayed math by filler distance:")
        print(f"  {'Filler':>7}   {'matrix 2x2':>12} {'matrix 3x3':>12} {'scalar':>8}")
        print(f"  {'-'*45}")
        for fl in filler_levels:
            row = {}
            for sub, size in [("matrix", "2x2"), ("matrix", "3x3"), ("scalar", "")]:
                if size:
                    matches = [r for r in delayed if r["filler_tokens"] == fl
                               and r["subtype"] == sub and r.get("size") == size]
                else:
                    matches = [r for r in delayed if r["filler_tokens"] == fl
                               and r["subtype"] == sub]
                key = f"{sub}_{size}" if size else sub
                if matches:
                    p = sum(1 for r in matches if r["passed"])
                    row[key] = f"{p}/{len(matches)}"
                else:
                    row[key] = "-"
            print(f"  {fl:>6}t   {row['matrix_2x2']:>12} {row.get('matrix_3x3', '-'):>12} {row.get('scalar', '-'):>8}")

    # Failed tests — show response preview
    failed = [r for r in results if not r["passed"]]
    if failed and args.verbose:
        print(f"\n  Failed test responses:")
        print(f"  {'-'*60}")
        for r in failed:
            print(f"\n  [{r['id']}]")
            print(f"  expected: {r['expected']}")
            preview = r['response_preview'].replace('\n', ' ')
            print(f"  got:      {preview[:120]}")

    output = {
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
        "filler_levels": filler_levels,
        "system_prompt": system_prompt,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score": {"passed": total_pass, "total": total, "percent": pct},
        "categories": cats,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {args.output}")


def compare(args):
    data = []
    for f in args.files:
        with open(f) as fh:
            data.append(json.load(fh))

    print(f"\n{'='*70}")
    print(f"  TurboQuant Math Accuracy Comparison")
    print(f"{'='*70}\n")

    print(f"  {'Label':<20} {'Pass/Total':>12} {'%':>8}")
    print(f"  {'-'*44}")
    for d in data:
        s = d["score"]
        print(f"  {d['label']:<20} {s['passed']}/{s['total']:>4}        {s['percent']:>5.1f}%")

    print(f"\n  By category:")
    all_cats = sorted({cat for d in data for cat in d.get("categories", {})})
    header = f"  {'Category':<22}" + "".join(f" {d['label'][:10]:>12}" for d in data)
    print(header)
    print(f"  {'-'*len(header)}")
    for cat in all_cats:
        row = f"  {cat:<22}"
        for d in data:
            c = d.get("categories", {}).get(cat)
            if c:
                pct = c['pass'] / c['total'] * 100 if c['total'] else 0
                row += f" {c['pass']}/{c['total']} ({pct:.0f}%):>12"
            else:
                row += f"{'—':>12}"
        print(row)

    print(f"\n  Divergent results:")
    print(f"  {'-'*60}")
    if len(data) > 1:
        any_diff = False
        test_ids = [r["id"] for r in data[0]["results"]]
        for tid in test_ids:
            rows = []
            for d in data:
                r = next((x for x in d["results"] if x["id"] == tid), None)
                if r:
                    rows.append((d["label"], r["passed"], r.get("detail", "")))
            statuses = [r[1] for r in rows]
            if len(set(statuses)) > 1:
                any_diff = True
                print(f"\n  {tid}:")
                for label, passed, detail in rows:
                    s = "PASS" if passed else "FAIL"
                    print(f"    [{s}] {label:<20} {detail[:60]}")
        if not any_diff:
            print(f"  None — identical results across all configurations.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant Math Accuracy Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Baseline: test direct math only (no filler), fast
  python turboquant_math_accuracy.py collect \\
    --label f16_baseline --model qwen3:9b --skip-3x3 --filler 0

  # Full test: direct + delayed at 3 filler levels
  python turboquant_math_accuracy.py collect \\
    --label f16 --model qwen3:9b --filler 200,500,1000

  # Math-specialised model (adds system prompt, handles \\boxed{})
  python turboquant_math_accuracy.py collect \\
    --label math_f16 --model Qwen2.5-Math-7B-Instruct-Q8_0 \\
    --math-model-prompt --filler 200,500,1000

  # Compare two runs
  python turboquant_math_accuracy.py compare results/math_f16.json results/math_tq.json
""",
    )
    sub = parser.add_subparsers(dest="command")

    c = sub.add_parser("collect", help="Run math accuracy tests")
    c.add_argument("--url", default="http://localhost:11434")
    c.add_argument("--model", default="qwen3:9b")
    c.add_argument("--label", required=True)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--filler", default="200,500,1000",
                   help="Comma-separated filler levels. Use 0 for direct-only.")
    c.add_argument("--matrix-sizes", default="2x2,3x3")
    c.add_argument("--skip-3x3", action="store_true",
                   help="Skip 3×3 matrix tests (faster, less stress)")
    c.add_argument("--skip-scalar", action="store_true",
                   help="Skip scalar arithmetic tests")
    c.add_argument("--math-model-prompt", action="store_true",
                   help='Add "Please reason step by step... \\boxed{}" system prompt '
                        '(for Qwen2.5-Math, DeepSeek-Math, etc.)')
    c.add_argument("--system-prompt", default="",
                   help="Custom system prompt (overrides --math-model-prompt)")
    c.add_argument("--timeout", type=int, default=180,
                   help="Per-request timeout in seconds (default: 180)")
    c.add_argument("--output", "-o", help="Save results JSON to this path")
    c.add_argument("--verbose", "-v", action="store_true",
                   help="Print failed response previews")

    p = sub.add_parser("compare", help="Compare two result files")
    p.add_argument("files", nargs="+")
    p.add_argument("--markdown", action="store_true")

    args = parser.parse_args()

    if args.command == "collect":
        raw = [int(x) for x in args.filler.split(",")]
        filtered = [x for x in raw if x > 0]
        args.filler = ",".join(str(x) for x in filtered) if filtered else ""
        asyncio.run(collect(args))
    elif args.command == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

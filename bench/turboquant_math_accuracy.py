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
A. 2×2 matrix multiplication × 5 pairs       — direct and delayed
B. 3×3 matrix multiplication × 3 pairs       — direct and delayed
C. Scalar arithmetic chains  × 5 expressions — direct and delayed

Usage:
  # Direct only (no filler) — fastest baseline
  python bench/turboquant_math_accuracy.py collect \\
    --url http://localhost:11434 --model Qwen3-9B-Q4_K_M \\
    --label F16_direct --filler 0

  # Full test with filler (start engine with desired --cache-type-k/v first)
  # F16:   ./eullm-tq run qwen3-14b.gguf --cache-type-k f16   --cache-type-v f16   ...
  # TQ4_0: ./eullm-tq run qwen3-14b.gguf --cache-type-k tq4_0 --cache-type-v tq4_0 ...
  # TQ3_0: ./eullm-tq run qwen3-14b.gguf --cache-type-k tq3_0 --cache-type-v tq3_0 ...
  python bench/turboquant_math_accuracy.py collect \\
    --label f16 --filler 200,500,1000 -o bench/results/math_f16.json
  python bench/turboquant_math_accuracy.py collect \\
    --label tq4_0 --filler 200,500,1000 -o bench/results/math_tq4_0.json

  # Compare
  python bench/turboquant_math_accuracy.py compare \\
    bench/results/math_f16.json bench/results/math_tq4_0.json
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)


# ── Filler ────────────────────────────────────────────────────────────────────

FILLER_PARAGRAPHS = [
    "The European Union's digital strategy aims to make this transformation work for people and businesses, while helping to achieve its target of a climate-neutral Europe by 2050. The Commission has outlined key policies and frameworks to guide this transition.",
    "Recent advances in semiconductor manufacturing have enabled the production of chips at the 3nm node, dramatically increasing transistor density while reducing power consumption. TSMC, Samsung, and Intel are competing to deliver these next-generation processors.",
    "The Mediterranean basin is home to approximately 25,000 plant species, of which about 50% are endemic. This biodiversity hotspot faces threats from urbanisation, agricultural intensification, and climate change.",
    "Cloud computing has fundamentally changed how organisations deploy and manage IT infrastructure. The shift from capital expenditure to operational expenditure models has enabled startups and enterprises alike to scale their operations globally.",
    "Italian Renaissance art underwent several distinct phases, from the early experiments of Giotto in the 13th century to the High Renaissance works of Leonardo, Michelangelo, and Raphael.",
    "The carbon cycle involves the exchange of carbon between the atmosphere, oceans, terrestrial biosphere, and geological formations. Anthropogenic emissions have disrupted this cycle, leading to an increase in atmospheric CO2 concentrations.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be impractical for classical computers.",
    "The logistics industry handles approximately 65 billion parcels annually worldwide, with e-commerce driving a significant portion of this volume. Last-mile delivery remains the most expensive segment.",
    "Volcanic activity along tectonic plate boundaries shapes the Earth's surface through both constructive and destructive processes. The Ring of Fire contains approximately 75% of the world's active volcanoes.",
    "Machine learning models for natural language processing have grown exponentially in size. This scaling has been accompanied by emergent capabilities including in-context learning and chain-of-thought reasoning.",
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


# ── Answer checking (mirrors turboquant_quality.py) ───────────────────────────

def strip_thinking(s: str) -> str:
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()


def normalize(s: str) -> str:
    s = strip_thinking(s)
    s = s.replace(" ", "").replace("\n", "").replace("`", "").replace("*", "")
    return s.strip().lower()


def extract_last_line(s: str) -> str:
    s = strip_thinking(s)
    lines = [l.strip() for l in s.strip().split('\n') if l.strip()]
    return lines[-1] if lines else s


def _extract_matrix_from_latex(s: str) -> str:
    """
    Extract a matrix from LaTeX notation and return [[a,b],[c,d]] form.
    Handles \\begin{pmatrix}, \\begin{bmatrix}, \\begin{matrix}.
    Also handles \\boxed{...} wrapping.
    """
    # Unwrap \boxed{...}
    s = re.sub(r'\\boxed\{(.*?)\}', r'\1', s, flags=re.DOTALL)
    # Find last pmatrix/bmatrix/matrix block
    m = re.findall(
        r'\\begin\{[pvb]?matrix\*?\}(.*?)\\end\{[pvb]?matrix\*?\}',
        s, re.DOTALL
    )
    if not m:
        return ""
    inner = m[-1]  # take last occurrence (final answer)
    rows = [r.strip() for r in re.split(r'\\\\', inner) if r.strip()]
    result = []
    for row in rows:
        nums = [int(x) for x in re.findall(r'-?\d+', row)]
        if nums:
            result.append(nums)
    if not result:
        return ""
    return "[" + ",".join("[" + ",".join(str(x) for x in r) + "]" for r in result) + "]"


def check_answer(test: dict, response: str) -> tuple[bool, str]:
    mode = test["check"]
    resp = strip_thinking(response).strip()
    resp_last = extract_last_line(response)

    if mode == "exact_normalized":
        expected = normalize(test["expected"])
        # 1. Standard check (model output [[a,b],[c,d]] format)
        if expected in normalize(resp) or expected in normalize(resp_last):
            return True, f"expected={test['expected']} last_line={resp_last[:80]}"
        # 2. LaTeX matrix fallback (Qwen2.5-Math, DeepSeek-Math style)
        latex_result = _extract_matrix_from_latex(resp)
        if latex_result and normalize(latex_result) == expected:
            return True, f"expected={test['expected']} latex={latex_result}"
        return False, f"expected={test['expected']} last_line={resp_last[:80]}"

    elif mode == "contains_number":
        expected = test["expected"]
        # Also check inside \boxed{...}
        boxed = re.findall(r'\\boxed\{([^}]+)\}', resp)
        boxed_str = " ".join(boxed)
        ok = expected in resp or expected in resp_last or expected in boxed_str
        return ok, f"expected={expected} in response={resp_last[:80]}"

    return False, "unknown check mode"


# ── Matrix helpers ────────────────────────────────────────────────────────────

def _mm(A, B):
    rows, cols, inner = len(A), len(B[0]), len(B)
    return [[sum(A[r][k] * B[k][c] for k in range(inner)) for c in range(cols)]
            for r in range(rows)]


def _fmt(rows):
    """Format list-of-lists as [[a,b],[c,d]]."""
    return "[" + ",".join("[" + ",".join(str(x) for x in r) + "]" for r in rows) + "]"


# ── Test data ─────────────────────────────────────────────────────────────────

MATRIX_2X2 = [
    ("M1", [[3, 1], [4, 2]],   [[5, 7], [6, 8]]),
    ("M2", [[1, 2], [3, 4]],   [[5, 6], [7, 8]]),
    ("M3", [[2, 0], [1, 3]],   [[4, 1], [2, 5]]),
    ("M4", [[5, 3], [2, 7]],   [[1, 4], [6, 0]]),
    ("M5", [[0, 1], [1, 0]],   [[3, 4], [5, 6]]),
]

MATRIX_3X3 = [
    ("M6", [[1, 0, 2], [3, 1, 0], [0, 2, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ("M7", [[2, 1, 0], [0, 3, 1], [1, 0, 2]], [[1, 0, 1], [2, 1, 0], [0, 1, 2]]),
    ("M8", [[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
]

SCALAR = [
    ("S1", "(17 * 23) + 14",            str((17 * 23) + 14)),
    ("S2", "3^4 + 2^5",                 str(3**4 + 2**5)),
    ("S3", "(144 / 12) * (15 - 8) + 3", str(int((144 / 12) * (15 - 8) + 3))),
    ("S4", "7*8 + 6*9 + 5*10",          str(7 * 8 + 6 * 9 + 5 * 10)),
    ("S5", "1000 - 13 * 37",            str(1000 - 13 * 37)),
]


# ── Build tests ───────────────────────────────────────────────────────────────

def build_tests(filler_levels, skip_3x3=False, skip_scalar=False):
    tests = []

    # Direct 2×2
    for label, A, B in MATRIX_2X2:
        C = _mm(A, B)
        tests.append({
            "id": f"direct_mat2x2_{label}",
            "category": "direct_math",
            "subtype": "matrix_2x2",
            "filler_tokens": 0,
            "prompt": f"Compute {_fmt(A)} × {_fmt(B)}. Return ONLY [[a,b],[c,d]].",
            "expected": _fmt(C),
            "check": "exact_normalized",
        })

    # Direct 3×3
    if not skip_3x3:
        for label, A, B in MATRIX_3X3:
            C = _mm(A, B)
            tests.append({
                "id": f"direct_mat3x3_{label}",
                "category": "direct_math",
                "subtype": "matrix_3x3",
                "filler_tokens": 0,
                "prompt": f"Compute {_fmt(A)} × {_fmt(B)}. Return ONLY [[a,b,c],[d,e,f],[g,h,i]].",
                "expected": _fmt(C),
                "check": "exact_normalized",
            })

    # Direct scalar
    if not skip_scalar:
        for label, expr, expected in SCALAR:
            tests.append({
                "id": f"direct_scalar_{label}",
                "category": "direct_math",
                "subtype": "scalar",
                "filler_tokens": 0,
                "prompt": f"Compute {expr}. Return ONLY the integer result.",
                "expected": expected,
                "check": "contains_number",
            })

    # Delayed 2×2
    for label, A, B in MATRIX_2X2:
        C = _mm(A, B)
        for fl in filler_levels:
            filler = generate_filler(fl)
            tests.append({
                "id": f"delayed_mat2x2_{label}_{fl}t",
                "category": "delayed_math",
                "subtype": "matrix_2x2",
                "filler_tokens": fl,
                "prompt": (
                    f"Memorize these matrices. Do NOT compute yet.\n\n"
                    f"A = {_fmt(A)}\nB = {_fmt(B)}\n\n"
                    f"{filler}\n\n"
                    f"Compute A × B. Return ONLY [[a,b],[c,d]], no explanation."
                ),
                "expected": _fmt(C),
                "check": "exact_normalized",
            })

    # Delayed 3×3
    if not skip_3x3:
        for label, A, B in MATRIX_3X3:
            C = _mm(A, B)
            for fl in filler_levels:
                filler = generate_filler(fl)
                tests.append({
                    "id": f"delayed_mat3x3_{label}_{fl}t",
                    "category": "delayed_math",
                    "subtype": "matrix_3x3",
                    "filler_tokens": fl,
                    "prompt": (
                        f"Memorize these matrices. Do NOT compute yet.\n\n"
                        f"A = {_fmt(A)}\nB = {_fmt(B)}\n\n"
                        f"{filler}\n\n"
                        f"Compute A × B. Return ONLY [[a,b,c],[d,e,f],[g,h,i]], no explanation."
                    ),
                    "expected": _fmt(C),
                    "check": "exact_normalized",
                })

    # Delayed scalar
    if not skip_scalar:
        for label, expr, expected in SCALAR:
            for fl in filler_levels:
                filler = generate_filler(fl)
                tests.append({
                    "id": f"delayed_scalar_{label}_{fl}t",
                    "category": "delayed_math",
                    "subtype": "scalar",
                    "filler_tokens": fl,
                    "prompt": (
                        f"Remember this expression. Do NOT compute it yet.\n\n"
                        f"EXPRESSION: {expr}\n\n"
                        f"{filler}\n\n"
                        f"Compute the expression you memorized. Return ONLY the integer result."
                    ),
                    "expected": expected,
                    "check": "contains_number",
                })

    return tests


# ── API client (mirrors turboquant_quality.py) ────────────────────────────────

async def send_prompt(session: aiohttp.ClientSession, url: str, model: str,
                      prompt: str, temperature: float = 0.0,
                      think: bool = False, num_predict: int = 2048) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if think is not None:
        payload["think"] = think
    try:
        async with session.post(
            f"{url}/api/chat", json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                return f"[ERROR: HTTP {resp.status}]", {}
            data = await resp.json()
            timing = {
                "eval_count":            data.get("eval_count", 0),
                "eval_duration_ns":      data.get("eval_duration", 0),
                "prompt_eval_count":     data.get("prompt_eval_count", 0),
                "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
            }
            return data.get("message", {}).get("content", "[no content]"), timing
    except Exception as e:
        return f"[ERROR: {e}]", {}


# ── Collect ───────────────────────────────────────────────────────────────────

async def collect(args):
    random.seed(0)
    filler_levels = [int(x) for x in args.filler.split(",") if x and int(x) > 0]
    tests = build_tests(filler_levels, skip_3x3=args.skip_3x3, skip_scalar=args.skip_scalar)

    think = None if args.no_think else False

    print(f"TurboQuant Math Accuracy Test")
    print(f"  URL:   {args.url}")
    print(f"  Model: {args.model}")
    print(f"  Label: {args.label}")
    print(f"  Tests: {len(tests)}  filler={filler_levels}")
    print(f"  think: {'omitted (non-Qwen3)' if think is None else think}")
    print()

    results = []
    cats: dict[str, dict] = {}
    run_start = time.time()

    async with aiohttp.ClientSession() as session:
        for test in tests:
            t0 = time.time()
            response, timing = await send_prompt(session, args.url, args.model,
                                                 test["prompt"], args.temperature,
                                                 think, args.num_predict)
            latency_s = time.time() - t0
            passed, detail = check_answer(test, response)

            status = "PASS" if passed else "FAIL"
            cat = test["category"]
            cats.setdefault(cat, {"pass": 0, "total": 0})
            cats[cat]["total"] += 1
            if passed:
                cats[cat]["pass"] += 1

            fl = test["filler_tokens"]
            fl_s = f"filler={fl:>5}t" if fl else "direct       "
            ptoks = timing.get("prompt_eval_count", 0)
            ptoks_s = f" ctx={ptoks:>5}t" if ptoks else ""
            print(f"  [{status}] {test['id']:<42} {fl_s}{ptoks_s}  {detail[:50]}")

            eval_toks = timing.get("eval_count", 0)
            eval_ns   = timing.get("eval_duration_ns", 0)
            tps = eval_toks / (eval_ns / 1e9) if eval_ns > 0 else 0.0

            results.append({
                "id": test["id"],
                "category": cat,
                "subtype": test["subtype"],
                "filler_tokens": fl,
                "passed": passed,
                "expected": test["expected"],
                "detail": detail,
                "response_preview": strip_thinking(response)[:300],
                "latency_s": latency_s,
                "eval_tokens": eval_toks,
                "tokens_per_sec": tps,
                "prompt_tokens": timing.get("prompt_eval_count", 0),
            })

    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)
    pct = total_pass / total * 100 if total else 0

    print(f"\n{'='*65}")
    print(f"  {args.label}: {total_pass}/{total} passed ({pct:.1f}%)")
    print(f"{'='*65}")

    print(f"\n  Category breakdown:")
    print(f"  {'Category':<22} {'Pass/Total':>12}   {'%':>6}")
    print(f"  {'-'*44}")
    for cat, s in sorted(cats.items()):
        p = s['pass'] / s['total'] * 100 if s['total'] else 0
        print(f"  {cat:<22} {s['pass']:>4}/{s['total']:<4}        {p:>5.1f}%")

    # Direct vs delayed
    print(f"\n  Direct vs Delayed (pass/total):")
    print(f"  {'Type':<12} {'mat 2x2':>10} {'mat 3x3':>10} {'scalar':>10}")
    print(f"  {'-'*44}")
    for cat in ["direct_math", "delayed_math"]:
        row = {}
        for sub in ["matrix_2x2", "matrix_3x3", "scalar"]:
            m = [r for r in results if r["category"] == cat and r["subtype"] == sub]
            row[sub] = f"{sum(1 for r in m if r['passed'])}/{len(m)}" if m else "-"
        name = "direct" if cat == "direct_math" else "delayed"
        print(f"  {name:<12} {row['matrix_2x2']:>10} {row['matrix_3x3']:>10} {row['scalar']:>10}")

    # By filler level
    delayed = [r for r in results if r["category"] == "delayed_math"]
    if delayed and filler_levels:
        print(f"\n  Delayed by filler distance:")
        print(f"  {'Filler':>7}   {'mat 2x2':>10} {'mat 3x3':>10} {'scalar':>10}")
        print(f"  {'-'*44}")
        for fl in filler_levels:
            row = {}
            for sub in ["matrix_2x2", "matrix_3x3", "scalar"]:
                m = [r for r in delayed if r["filler_tokens"] == fl and r["subtype"] == sub]
                row[sub] = f"{sum(1 for r in m if r['passed'])}/{len(m)}" if m else "-"
            print(f"  {fl:>6}t   {row['matrix_2x2']:>10} {row['matrix_3x3']:>10} {row['scalar']:>10}")

    # Breakdown by context size bucket (to spot precision bugs at specific token ranges)
    ctx_results = [r for r in results if r.get("prompt_tokens", 0) > 0]
    if ctx_results:
        buckets = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 9999)]
        bucket_rows = []
        for lo, hi in buckets:
            br = [r for r in ctx_results if lo < r["prompt_tokens"] <= hi]
            if br:
                bucket_rows.append((lo, hi, br))
        if bucket_rows:
            print(f"\n  Accuracy by context size (prompt tokens):")
            print(f"  {'Range':>16}   {'Pass/Total':>12}   {'%':>6}   {'Avg ctx':>9}")
            print(f"  {'-'*52}")
            for lo, hi, br in bucket_rows:
                bp = sum(1 for r in br if r["passed"])
                bt = len(br)
                bpct = bp / bt * 100
                avg_ctx = sum(r["prompt_tokens"] for r in br) / bt
                print(f"  {lo:>5}–{hi:<5}t        {bp:>4}/{bt:<4}        {bpct:>5.1f}%   {avg_ctx:>7.0f}t")

    # Timing summary
    total_wall = time.time() - run_start
    timed = [r for r in results if r["latency_s"] > 0]
    if timed:
        latencies = [r["latency_s"] for r in timed]
        tps_vals  = [r["tokens_per_sec"] for r in timed if r["tokens_per_sec"] > 0]
        print(f"\n  Throughput:")
        print(f"  {'Total wall time':<28} {total_wall:>7.1f}s")
        print(f"  {'Requests':<28} {len(timed):>7}")
        print(f"  {'Latency mean/min/max (s)':<28} {sum(latencies)/len(latencies):>5.1f} / {min(latencies):>4.1f} / {max(latencies):>4.1f}")
        if tps_vals:
            print(f"  {'Tokens/sec mean/min/max':<28} {sum(tps_vals)/len(tps_vals):>5.1f} / {min(tps_vals):>4.1f} / {max(tps_vals):>4.1f}")

    if args.verbose:
        failed = [r for r in results if not r["passed"]]
        if failed:
            print(f"\n  Failed responses:")
            print(f"  {'-'*60}")
            for r in failed:
                print(f"\n  [{r['id']}]  expected={r['expected']}")
                print(f"  got: {r['response_preview'][:120].replace(chr(10), ' ')}")

    output = {
        "label": args.label,
        "note": args.note,
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
        "filler_levels": filler_levels,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score": {"passed": total_pass, "total": total, "percent": pct},
        "timing": {
            "total_wall_s": round(total_wall, 2),
            "latency_mean_s": round(sum(latencies) / len(latencies), 2) if timed else None,
            "tokens_per_sec_mean": round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None,
        },
        "categories": cats,
        "results": results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {args.output}")


# ── Compare ───────────────────────────────────────────────────────────────────

def compare(args):
    data = []
    for f in args.files:
        with open(f) as fh:
            data.append(json.load(fh))

    print(f"\n{'='*65}")
    print(f"  TurboQuant Math Accuracy Comparison")
    print(f"{'='*65}\n")

    print(f"  {'Label':<20} {'Pass/Total':>12}   {'%':>6}")
    print(f"  {'-'*42}")
    for d in data:
        s = d["score"]
        print(f"  {d['label']:<20} {s['passed']:>4}/{s['total']:<4}        {s['percent']:>5.1f}%")

    if len(data) > 1:
        print(f"\n  Divergent results:")
        print(f"  {'-'*60}")
        any_diff = False
        for tid in [r["id"] for r in data[0]["results"]]:
            rows = []
            for d in data:
                r = next((x for x in d["results"] if x["id"] == tid), None)
                if r:
                    rows.append((d["label"], r["passed"], r.get("detail", "")))
            if len({r[1] for r in rows}) > 1:
                any_diff = True
                print(f"\n  {tid}:")
                for label, passed, detail in rows:
                    print(f"    [{'PASS' if passed else 'FAIL'}] {label:<20} {detail[:60]}")
        if not any_diff:
            print(f"  None — identical results across all configurations.")

    if args.markdown:
        print(f"\n| Label | Pass | Total | % |")
        print(f"|:---|:---:|:---:|:---:|")
        for d in data:
            s = d["score"]
            print(f"| {d['label']} | {s['passed']} | {s['total']} | {s['percent']:.1f}% |")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Math Accuracy Test")
    sub = parser.add_subparsers(dest="command")

    c = sub.add_parser("collect")
    c.add_argument("--url", default="http://localhost:11434")
    c.add_argument("--model", default="qwen3-14b")
    c.add_argument("--label", required=True)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--filler", default="200,500,1000",
                   help="Comma-separated filler token counts (0 = direct only). "
                        "Use 200,500,1000,1500,2000,2500 for full bug-window coverage "
                        "(AmesianX v1.3 V-cache precision bug manifests at ctx 1500-2300t)")
    c.add_argument("--skip-3x3", action="store_true")
    c.add_argument("--skip-scalar", action="store_true")
    c.add_argument("--no-think", action="store_true",
                   help="Omit 'think' field from payload (for non-Qwen3 models: "
                        "Qwen2.5-Math, DeepSeek-Math, etc.)")
    c.add_argument("--num-predict", type=int, default=2048,
                   help="Max tokens to generate per response (default: 512). "
                        "Use 1024 for math models that show full working.")
    c.add_argument("--output", "-o")
    c.add_argument("--note", default="",
                   help="Free-form note saved in JSON metadata (e.g. 'cache-k=q8_0 cache-v=tbq4_1 ctx=32768')")
    c.add_argument("--verbose", "-v", action="store_true")

    p = sub.add_parser("compare")
    p.add_argument("files", nargs="+")
    p.add_argument("--markdown", action="store_true")

    args = parser.parse_args()
    if args.command == "collect":
        asyncio.run(collect(args))
    elif args.command == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

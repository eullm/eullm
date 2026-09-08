#!/usr/bin/env python3
"""Pre-download the training models into the HF cache on $WORK.

Leonardo compute nodes have no outbound network, so every model a job
needs must already sit in the local HF cache ($HF_HOME, set by env.sh)
before submission. Run this ON A LOGIN NODE (which has internet):

    source forge/scripts/leonardo/env.sh
    python forge/scripts/leonardo/prefetch_models.py

After downloading, each model is re-opened with local_files_only=True —
the same code path the jobs will use — so a green run here means the
offline loads inside sbatch jobs will find everything.

The default set covers the legal-it-4b pipeline: smoke proxy, student,
teacher (~65 GB — make sure `saldo` shows quota headroom on $WORK).
"""

from __future__ import annotations

import argparse
import os
import sys

# Every id here is a model Qwen actually publishes, verified against the
# Hub API. The list used to name Qwen3-7B-Base and Qwen3-32B-Base, and
# neither exists: Qwen3 skipped the 7B size and never released a 32B
# Base. Both failed with 404 on the first prefetch run on Leonardo,
# after the 3.5 GB smoke model had already downloaded. See the Phase-1
# Leonardo config header for why the replacements are what they are.
DEFAULT_MODELS = [
    "Qwen/Qwen3-1.7B-Base",     # smoke test
    "Qwen/Qwen3-4B-Base",       # Phase-2 student (~8 GB)
    "Qwen/Qwen3-30B-A3B-Base",  # Phase-1/2 teacher, MoE 128e/8a (~60 GB)
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HF model ids to prefetch (default: the legal-it-4b set)",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_HOME"):
        print("[warn] HF_HOME not set — models would land in ~/.cache on "
              "the 50 GB $HOME. Source forge/scripts/leonardo/env.sh first.",
              file=sys.stderr)
        return 1

    from huggingface_hub import snapshot_download

    # One unreachable model must not abort the run. setup.sh calls this
    # before it downloads the dataset, so a single bad id used to stop
    # the whole setup at step 3 and the corpus never arrived — the
    # failure looked like a dataset problem and was not. Every model is
    # attempted, and what failed is reported together at the end.
    failed: list[tuple[str, str]] = []
    ok: list[str] = []
    for model_id in args.models:
        print(f"[..] fetching {model_id} → $HF_HOME")
        try:
            path = snapshot_download(model_id)
        except Exception as exc:  # network, auth, or a model that is not there
            print(f"[!!] {model_id}: {type(exc).__name__}", file=sys.stderr)
            failed.append((model_id, str(exc).splitlines()[0]))
            continue
        print(f"[ok] {model_id} at {path}")
        ok.append(model_id)

    # Verify the offline code path jobs will take, for what did download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer

    for model_id in ok:
        AutoConfig.from_pretrained(model_id, local_files_only=True)
        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print(f"[ok] {model_id} loads offline")

    if failed:
        print(f"\n[!!] {len(failed)} model(s) could not be fetched:",
              file=sys.stderr)
        for model_id, why in failed:
            print(f"       {model_id}: {why}", file=sys.stderr)
        print("     A 404 here means the id does not exist on the Hub — check\n"
              "     it before assuming an access problem; the Hub answers 401\n"
              "     to anonymous callers and 404 to authenticated ones for the\n"
              "     very same missing repo.", file=sys.stderr)
        return 1

    print("[done] cache ready for offline compute jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

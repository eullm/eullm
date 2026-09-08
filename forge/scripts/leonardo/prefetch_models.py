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

The default set covers the legal-it-7b pipeline: smoke proxy, student,
teacher (~65 GB — make sure `saldo` shows quota headroom on $WORK).
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODELS = [
    "Qwen/Qwen3-1.7B-Base",   # smoke test
    "Qwen/Qwen3-7B-Base",     # Phase-2 student
    "Qwen/Qwen3-32B-Base",    # Phase-1/2 teacher (~65 GB)
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HF model ids to prefetch (default: the legal-it-7b set)",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_HOME"):
        print("[warn] HF_HOME not set — models would land in ~/.cache on "
              "the 50 GB $HOME. Source forge/scripts/leonardo/env.sh first.",
              file=sys.stderr)
        return 1

    from huggingface_hub import snapshot_download

    for model_id in args.models:
        print(f"[..] fetching {model_id} → $HF_HOME")
        path = snapshot_download(model_id)
        print(f"[ok] {model_id} at {path}")

    # Verify the offline code path jobs will take.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer

    for model_id in args.models:
        AutoConfig.from_pretrained(model_id, local_files_only=True)
        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print(f"[ok] {model_id} loads offline")

    print("[done] cache ready for offline compute jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

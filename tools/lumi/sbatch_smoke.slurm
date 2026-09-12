#!/bin/bash
# Smoke test: does the ROCm build actually use an MI250X on a LUMI-G node?
#
#     sbatch tools/lumi/sbatch_smoke.slurm
#
# The repository is not needed on the machine: this file and the binary are the
# whole test. With a published release there is nothing to compile, so two
# downloads are enough —
#
#     curl -L https://github.com/eullm/eullm/releases/download/<tag>/eullm-linux-x64-rocm-gfx90a -o eullm-rocm
#     curl -L https://raw.githubusercontent.com/eullm/eullm/main/tools/lumi/sbatch_smoke.slurm -o smoke.slurm
#     chmod +x eullm-rocm && export EULLM_BIN=$PWD/eullm-rocm
#     sbatch smoke.slurm
#
# Cloning is for tools/lumi/build_engine.sh, which compiles against the ROCm the
# machine actually has — the fallback when no published binary matches the site.
#
# Costs ~0.04 node-hours (dev-g bills 0.5 per GCD-hour, so one GCD for at most
# 20 minutes is 0.17 GPU-hours). The account comes from SBATCH_ACCOUNT, so set it once
# in ~/.bashrc rather than passing -A on every submission:
#
#     export SBATCH_ACCOUNT=project_46XXXXXXX    # sbatch reads this one
#     export SALLOC_ACCOUNT=project_46XXXXXXX    # salloc reads this one
#     export SLURM_ACCOUNT=project_46XXXXXXX     # srun reads this one
#
# All three, because they are read by different commands: setting only the
# first and then running `srun --account=$SBATCH_ACCOUNT` from a fresh shell
# expands to `--account=` and fails with "Invalid account or account/partition
# combination specified", which reads like a permissions problem and is an
# empty string.
#
# The model must already be on disk: compute nodes have no outbound network, so
# pull it from a login node first (`eullm pull qwen3-8b`).
#
# What this proves, and why it is not the startup banner: the banner names the
# backend the binary was COMPILED with, not the device it found, so a run that
# fell back to CPU still prints "GPU backend: ROCm". rocm-smi before and after
# the request is the measurement that cannot lie — if HBM in use does not move,
# the GPU was never touched, whatever the banner said.

#SBATCH --job-name=eullm-rocm-smoke
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH --output=eullm-rocm-smoke-%j.out

set -uo pipefail  # no -e: the server runs in the background and must be reaped

EULLM_REPO="${EULLM_REPO:-$PWD}"
EULLM_BIN="${EULLM_BIN:-$EULLM_REPO/target/release/eullm}"
MODEL="${EULLM_SMOKE_MODEL:-qwen3-8b}"
# $HOME on LUMI is small; keep weights on project storage. Set this to wherever
# `eullm pull` put them if it is not the default.
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$HOME/.eullm/models}"
PORT=$((11434 + SLURM_JOB_ID % 1000))

echo "=== node and device ==="
hostname
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
rocm-smi --showproductname --showmeminfo vram 2>&1 | sed 's/^/  /'

echo
echo "=== binary ==="
"$EULLM_BIN" --version
ldd "$EULLM_BIN" | grep -E "amdhip64|rocblas|hipblas" | sed 's/^/  /'

echo
echo "=== starting server on port $PORT ==="
"$EULLM_BIN" serve --port "$PORT" > "server-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1; then
        echo "server up"
        break
    fi
    sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1; then
    echo "ERROR: server never became reachable — see server-$SLURM_JOB_ID.log"
    tail -40 "server-$SLURM_JOB_ID.log"
    exit 1
fi

echo
echo "=== one request (loads the model, then generates) ==="
RESPONSE=$(curl -s --max-time 900 -H "Content-Type: application/json" \
    "http://127.0.0.1:$PORT/api/chat" -d @- <<EOJSON
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Explain in three sentences what a supercomputer allocation is."}],
  "stream": false,
  "think": false,
  "num_predict": 128,
  "options": {"num_predict": 128}
}
EOJSON
)

echo "$RESPONSE" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("  could not parse the response:"); print(raw[:2000]); raise SystemExit(1)
if "error" in d:
    print("  server returned an error:", d["error"]); raise SystemExit(1)
n = d.get("eval_count") or 0
ns = d.get("eval_duration") or 0
load_ns = d.get("load_duration") or 0
print(f"  generated tokens: {n}")
print(f"  model load:       {load_ns/1e9:.1f} s")
if n and ns:
    print(f"  generation:       {n/(ns/1e9):.1f} tok/s")
print("  first words:      " + (d.get("message", {}).get("content", "")[:120].replace("\n", " ")))
'

echo
echo "=== device after the request (HBM in use is the real proof) ==="
rocm-smi --showmeminfo vram --showuse 2>&1 | sed 's/^/  /'

echo
echo "=== server log (backend lines) ==="
grep -iE "gpu|rocm|hip|device|offload" "server-$SLURM_JOB_ID.log" | head -20 | sed 's/^/  /'

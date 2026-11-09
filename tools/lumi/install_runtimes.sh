#!/usr/bin/env bash
# The servers the engine is compared with, installed on a login node:
#
#   llama-server  built from the engine's own llama.cpp submodule, at the same
#                 commit and without EuLLM's patches — the same backend
#                 without EuLLM's runtime, so a difference is EuLLM's; with
#                 llama-batched-bench and llama-bench from the same build,
#                 llama.cpp measured alone (sbatch_hybrid_profile.slurm)
#   Ollama        the latest release, with its ROCm libraries — the API the
#                 engine is compatible with
#
#   export SBATCH_ACCOUNT=project_465003366
#   bash tools/lumi/install_runtimes.sh [SPEC.json ...]
#
# Then, for every model the specs' points give to Ollama, an Ollama model of
# the same name made from the GGUF in the EuLLM store (`ollama create`, which
# copies it into $OLLAMA_MODELS). Default spec: c07-runtimes.json.
#
# Everything goes under $CAMPAIGN_DIR: runtimes/ and ollama-models/.
# campaign_env.sh exports LLAMA_SERVER_BIN, OLLAMA_BIN and OLLAMA_MODELS when
# they are there, so the campaign jobs find them.
#
#   SKIP_LLAMA=1 / SKIP_OLLAMA=1 / SKIP_MODELS=1   leave that part out

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"

if [ $# -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$EULLM_REPO/tools/lumi/campaigns/c07-runtimes.json")
fi
RT="$CAMPAIGN_DIR/runtimes"
mkdir -p "$RT"
FAILED=()

# ── llama-server ──────────────────────────────────────────────────────────
if [ "${SKIP_LLAMA:-0}" != "1" ]; then
    echo "=== llama-server (HIP, gfx90a) from the submodule ==="
    LCPP="$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    BUILD="$RT/llama-build"
    # The same compilers as build_engine.sh: gcc for host code (not the Cray
    # wrappers), ROCm's clang for the kernels.
    if [ -n "${CRAYPE_VERSION:-}" ]; then
        export CC=gcc CXX=g++
    fi
    if HIPCXX="$ROCM_PATH/llvm/bin/clang" HIP_PATH="$ROCM_PATH" \
        cmake -S "$LCPP" -B "$BUILD" -DGGML_HIP=ON -DGPU_TARGETS=gfx90a \
            -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >"$RT/llama-cmake.log" 2>&1 &&
       cmake --build "$BUILD" --target llama-server llama-batched-bench llama-bench \
            -j "${BUILD_JOBS:-16}" >"$RT/llama-build.log" 2>&1; then
        for tool in llama-server llama-batched-bench llama-bench; do
            cp -f "$BUILD/bin/$tool" "$RT/$tool"
            echo "[ok] $RT/$tool ($(git -C "$LCPP" describe --tags --always 2>/dev/null))"
        done
    else
        echo "[!!] llama-server did not build: tail $RT/llama-cmake.log $RT/llama-build.log"
        FAILED+=("llama-server")
    fi
fi

# ── Ollama ────────────────────────────────────────────────────────────────
if [ "${SKIP_OLLAMA:-0}" != "1" ]; then
    echo
    echo "=== Ollama, latest release, with its ROCm libraries ==="
    # The latest release at install time, recorded: no version from memory.
    URLS=$($PY - <<'EOF'
import json, re, urllib.request
with urllib.request.urlopen("https://api.github.com/repos/ollama/ollama/releases/latest",
                            timeout=60) as r:
    rel = json.load(r)
print(rel["tag_name"])
for a in rel["assets"]:
    if re.fullmatch(r"ollama-linux-amd64(-rocm)?\.(tgz|tar\.zst)", a["name"]):
        print(a["browser_download_url"])
EOF
    )
    TAG=$(head -1 <<<"$URLS")
    if [ -z "$TAG" ] || [ "$(wc -l <<<"$URLS")" -lt 3 ]; then
        echo "[!!] could not find Ollama's linux-amd64 and -rocm assets in the latest release"
        FAILED+=("ollama")
    else
        rm -rf "$RT/ollama.partial" && mkdir -p "$RT/ollama.partial"
        ok=1
        while read -r url; do
            echo "  $url"
            case "$url" in
                *.tar.zst) curl -fsSL "$url" | tar --zstd -x -C "$RT/ollama.partial" || ok=0 ;;
                *) curl -fsSL "$url" | tar -xz -C "$RT/ollama.partial" || ok=0 ;;
            esac
        done < <(tail -n +2 <<<"$URLS")
        if [ "$ok" = 1 ] && [ -x "$RT/ollama.partial/bin/ollama" ]; then
            rm -rf "$RT/ollama" && mv "$RT/ollama.partial" "$RT/ollama"
            echo "$TAG" > "$RT/ollama/VERSION"
            echo "[ok] $RT/ollama/bin/ollama ($TAG)"
        else
            echo "[!!] Ollama $TAG did not unpack"
            FAILED+=("ollama")
        fi
    fi
fi

# ── Ollama models, from the EuLLM store ──────────────────────────────────
if [ "${SKIP_MODELS:-0}" != "1" ] && [ -x "$RT/ollama/bin/ollama" ]; then
    echo
    echo "=== Ollama models for the specs' ollama points ==="
    export OLLAMA_MODELS="$CAMPAIGN_DIR/ollama-models"
    mkdir -p "$OLLAMA_MODELS"
    mapfile -t MODELS < <($PY - "${SPECS[@]}" <<'EOF'
import json, os, sys
sys.path.insert(0, os.environ["EULLM_REPO"] + "/bench/campaign")
from spec import expand
names = set()
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        names |= {p["model"] for p in expand(json.load(f)) if p.get("runtime") == "ollama"}
print("\n".join(sorted(names)))
EOF
    )
    port=$((20000 + RANDOM % 20000))
    OLLAMA_HOST="127.0.0.1:$port" "$RT/ollama/bin/ollama" serve >"$RT/ollama-create.log" 2>&1 &
    server=$!
    sleep 5
    for m in "${MODELS[@]}"; do
        [ -n "$m" ] || continue
        if OLLAMA_HOST="127.0.0.1:$port" "$RT/ollama/bin/ollama" show "$m" >/dev/null 2>&1; then
            echo "  $m: already there"
            continue
        fi
        gguf=$($PY -c "import sys; sys.path.insert(0, '$EULLM_REPO/bench/campaign'); \
import point; print(point.store_gguf('$m') or '')")
        if [ -z "$gguf" ]; then
            echo "  [!!] $m: not in the EuLLM store ($EULLM_MODELS_DIR) — pull it first"
            FAILED+=("ollama:$m")
            continue
        fi
        printf 'FROM %s\n' "$gguf" > "$RT/Modelfile.$m"
        if OLLAMA_HOST="127.0.0.1:$port" "$RT/ollama/bin/ollama" create "$m" -f "$RT/Modelfile.$m"; then
            echo "  [ok] $m ← $gguf"
        else
            FAILED+=("ollama:$m")
        fi
    done
    kill "$server" 2>/dev/null
    wait "$server" 2>/dev/null
fi

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[!!] not done (their points block until they are): ${FAILED[*]}"
    exit 1
fi
echo "done; campaign_env.sh now exports LLAMA_SERVER_BIN, OLLAMA_BIN and OLLAMA_MODELS"

#!/usr/bin/env bash
# Everything a campaign needs that only a login node can do — compute nodes
# have no network — then the plan. Idempotent: re-run it after adding a spec,
# or with ROUND=<label> to queue the same specs again for a new engine build.
#
#   export SBATCH_ACCOUNT=project_465003366 EULLM_BIN=$PWD/eullm-rocm
#   tmux new -s setup                    # the large MoE pulls take hours
#   bash tools/lumi/campaign_setup.sh [SPEC.json ...]
#
# Default specs: tools/lumi/campaigns/c01-node-baseline.json (catalog models,
# starts at once) and c02-quant-large-moe.json (~1.2 TB from Hugging Face).
# Not c05-finetune.json: this allocation is declared inference-only
# (docs/lumi/allocation-plan.md), so no spec that trains is planned here.
#
#   SKIP_PULLS=1      plan without pulling or converting (their points block
#                     until the models are there; `campaign.py unblock` after
#                     pulling puts them back)
#   ROUND=v0.7.21     a new round: every point of the specs measured again
#   EULLM_ENGINE_LABEL=next10
#                     the round's points run only in jobs submitted with the
#                     same label (submit_campaign.sh reads it too), so a
#                     queue can hold rounds of several engine builds
#   GROUP='rccl-*'    plan only the groups of these names (shell-style
#                     patterns, separated by spaces): a round of part of a
#                     campaign, as c03's rccl groups on an RCCL build

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the ROCm eullm binary}"

if [ $# -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$EULLM_REPO/tools/lumi/campaigns/c01-node-baseline.json"
           "$EULLM_REPO/tools/lumi/campaigns/c02-quant-large-moe.json")
fi
# DEV-278 is declared inference-only to EuroHPC: a spec that trains is refused
# here rather than queued by a command copied from somewhere else.
if grep -l '"kind": *"finetune"' "${SPECS[@]}" >/dev/null 2>&1; then
    echo "[err] $(grep -l '"kind": *"finetune"' "${SPECS[@]}" | tr '\n' ' ')has finetune points;" \
         "this allocation is inference-only (docs/lumi/allocation-plan.md)" >&2
    exit 1
fi
mkdir -p "$CAMPAIGN_DIR" "$EULLM_MODELS_DIR"
echo "queue  $CAMPAIGN_DIR"
echo "models $EULLM_MODELS_DIR ($(df -h "$EULLM_MODELS_DIR" 2>/dev/null | awk 'NR==2{print $4}') free)"

if [ "${SKIP_PULLS:-0}" != "1" ]; then
    echo
    echo "=== models the specs use and the store lacks ==="
    mapfile -t REFS < <($CAMPAIGN pulls "${SPECS[@]}" --engine "$EULLM_BIN")
    FAILED=()
    for ref in "${REFS[@]}"; do
        echo "--- $EULLM_BIN pull $ref"
        "$EULLM_BIN" pull "$ref" || FAILED+=("$ref")
    done
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo
        echo "[!!] could not pull (their points will block until they are in the store):"
        printf '       %s\n' "${FAILED[@]}"
    fi
fi

# Twins without the projector of the models a spec's `text_only` names: on
# the store alone, no network, so with SKIP_PULLS too (text_only_model.sh).
mapfile -t TEXT_ONLY < <("$PY" -c 'import json, sys
for f in sys.argv[1:]:
    print("\n".join(json.load(open(f, encoding="utf-8")).get("text_only", [])))' "${SPECS[@]}" | grep .)
if [ ${#TEXT_ONLY[@]} -gt 0 ]; then
    echo
    echo "=== text-only twins (the weights without their projector) ==="
    bash "$EULLM_REPO/tools/lumi/text_only_model.sh" "${TEXT_ONLY[@]}" ||
        echo "[!!] some twins are missing: their points will block"
fi

if [ "${SKIP_PULLS:-0}" != "1" ]; then
    echo
    echo "=== F32 models the finetune points train ==="
    bash "$EULLM_REPO/tools/lumi/make_f32_models.sh" "${SPECS[@]}" ||
        echo "[!!] some F32 models are missing: their finetune points will block"
fi

echo
echo "=== workload sets (GSM8K, ARC-Easy, ARC-Challenge) and finetune text (GSM8K train)," \
     "frozen for offline nodes ==="
$CAMPAIGN prefetch --queue "$CAMPAIGN_DIR" ||
    echo "[!!] prefetch failed: workload and finetune points will block"

echo
echo "=== plan ==="
# `read -a`, not an unquoted expansion: the patterns would be globbed against
# the current directory.
read -r -a PATTERNS <<<"${GROUP:-}"
GROUP_ARGS=()
for g in ${PATTERNS[@]+"${PATTERNS[@]}"}; do GROUP_ARGS+=(--group "$g"); done
$CAMPAIGN plan "${SPECS[@]}" --queue "$CAMPAIGN_DIR" --engine "$EULLM_BIN" \
    ${ROUND:+--round "$ROUND"} ${EULLM_ENGINE_LABEL:+--engine-label "$EULLM_ENGINE_LABEL"} \
    ${GROUP_ARGS[@]+"${GROUP_ARGS[@]}"}
$CAMPAIGN unblock --queue "$CAMPAIGN_DIR" >/dev/null

cat <<EOF

Next:
  bash $EULLM_REPO/tools/lumi/submit_campaign.sh 2     # two chains of jobs of up to 4 hours
  bash $EULLM_REPO/tools/lumi/status.sh                # budget, jobs, queue
EOF

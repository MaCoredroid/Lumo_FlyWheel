#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact FA2 shared object}"

FR13_GATE_QROW16=${FR13_GATE_QROW16:-1}
case "$FR13_GATE_QROW16" in
  0|1) ;;
  *) echo "FR13_GATE_QROW16 must be 0 or 1" >&2; exit 2 ;;
esac
FR13_GATE_TAW_NATIVE=${FR13_GATE_TAW_NATIVE:-1}
FR13_GATE_DRAFT_HEAD_PAD=${FR13_GATE_DRAFT_HEAD_PAD:-0}
for gate in FR13_GATE_TAW_NATIVE FR13_GATE_DRAFT_HEAD_PAD; do
  case "${!gate}" in
    0|1) ;;
    *) echo "$gate must be 0 or 1" >&2; exit 2 ;;
  esac
done

ARM="hydra27_fixed32_${TAG}"
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
FA2_SHA=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')

[[ "$FA2_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]]
[[ "$(docker ps -aq | wc -l)" -eq 0 ]]

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_FLOOR_ORDER=TH

source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh

mkdir -p "$RUNROOT"
printf 'launcher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nfa2_sha256=%s\nstarted=%s\n' \
  "$$" "$RUNROOT" "$ARM" "$(git rev-parse HEAD)" "$FA2_SHA" \
  "$(date -u +%FT%TZ)" > "$RUNROOT/launcher_meta.txt"

.venv/bin/python scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT/runtime_manifest.at_launch.json"
.venv/bin/python scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT/external_manifest.at_launch.json"

OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
  FR13_FIXED32_B1_DIAGNOSTIC=1 \
  FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  FR13_SFWD_GPU_TIMER=1 \
  FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT/sidecars/$ARM.json" \
  FR13_DFWD_GPU_TIMER=1 \
  FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT/sidecars/${ARM}_dfwd.json" \
  FR13_CFWD_GPU_TIMER=1 \
  FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT/sidecars/${ARM}_cfwd.json" \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE="$FR13_GATE_TAW_NATIVE" \
  FR13_DRAFT_HEAD_PAD_ROWS=0 \
  FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB="$FR13_GATE_DRAFT_HEAD_PAD" \
  FR13_FIXED32_GDN_PATH_BV_CANDIDATE=64 \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  FR13_FA2_QROW16_SO_SHA256="$FA2_SHA" \
  FR13_FA2_QROW16_LIVE_PAGED_AB="$FR13_GATE_QROW16" \
  FR13_FA2_QROW16_LIVE_PAGED_AB_INSTANCE_ID=astropy__astropy-12907 \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh \
    "$ARM" hydra27_fixed32 "$SUBSET" \
    > "$RUNROOT/$ARM.runlog" 2>&1
serve_rc=$?

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT/launcher_meta.txt"
.venv/bin/python scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT/runtime_manifest.at_end.json"
.venv/bin/python scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT/external_manifest.at_end.json"

exit "$serve_rc"

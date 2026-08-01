#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact FA2 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
CANONICAL_FA2="$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"
CANONICAL_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
CANONICAL_FA2_SIZE=299183936
RUNROOT_ABS=$(realpath -m "$RUNROOT")

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT" == output/* && "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must be repo-relative below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" ]] \
  || { echo "canonical real-B1 subset SHA-256 drift" >&2; exit 2; }

[[ -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be a regular non-symlink file" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* ]] \
  || { echo "FORKED_FA2_SO must be an absolute path" >&2; exit 2; }

FR13_GATE_QROW16=${FR13_GATE_QROW16:-0}
case "$FR13_GATE_QROW16" in
  0|1) ;;
  *) echo "FR13_GATE_QROW16 must be 0 or 1" >&2; exit 2 ;;
esac
FR13_GATE_TAW_NATIVE=${FR13_GATE_TAW_NATIVE:-1}
FR13_GATE_DRAFT_HEAD_PAD=${FR13_GATE_DRAFT_HEAD_PAD:-0}
FR13_GATE_DRAFT_HEAD_M32=${FR13_GATE_DRAFT_HEAD_M32:-0}
FR13_GATE_BM8=${FR13_GATE_BM8:-0}
for gate in FR13_GATE_TAW_NATIVE FR13_GATE_DRAFT_HEAD_PAD FR13_GATE_DRAFT_HEAD_M32 FR13_GATE_BM8; do
  case "${!gate}" in
    0|1) ;;
    *) echo "$gate must be 0 or 1" >&2; exit 2 ;;
  esac
done
FR13_GATE_GDN_BV=${FR13_GATE_GDN_BV:-64}
case "$FR13_GATE_GDN_BV" in
  0) FR13_GATE_GDN_BV_CANDIDATE= ;;
  16|32|64|128) FR13_GATE_GDN_BV_CANDIDATE=$FR13_GATE_GDN_BV ;;
  *) echo "FR13_GATE_GDN_BV must be 0, 16, 32, 64, or 128" >&2; exit 2 ;;
esac
if [[ "$FR13_GATE_BM8" == "1" \
      && ( "$FR13_GATE_QROW16" != "0" \
           || "$FR13_GATE_TAW_NATIVE" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_PAD" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_M32" != "0" \
           || "$FR13_GATE_GDN_BV" != "0" ) ]]; then
  echo "FR13_GATE_BM8 must be the only enabled kernel candidate" >&2
  exit 2
fi
if [[ "$FR13_GATE_DRAFT_HEAD_M32" == "1" \
      && ( "$FR13_GATE_QROW16" != "0" \
           || "$FR13_GATE_TAW_NATIVE" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_PAD" != "0" \
           || "$FR13_GATE_BM8" != "0" \
           || "$FR13_GATE_GDN_BV" != "0" ) ]]; then
  echo "FR13_GATE_DRAFT_HEAD_M32 must be the only enabled kernel candidate" >&2
  exit 2
fi

ARM="hydra27_fixed32_${TAG}"
FA2_SHA=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)
CANDIDATE_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
CANDIDATE_SOURCE_SHA256=$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)

[[ "$FA2_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]]
if [[ "$FR13_GATE_DRAFT_HEAD_M32" == "1" ]]; then
  [[ "$(realpath "$FORKED_FA2_SO")" == "$CANONICAL_FA2" ]] \
    || { echo "draft-head M32 gate requires the canonical in-worktree FA2" >&2; exit 2; }
  [[ "$(stat -c %s "$FORKED_FA2_SO")" == "$CANONICAL_FA2_SIZE" \
     && "$FA2_SHA" == "$CANONICAL_FA2_SHA256" ]] \
    || { echo "draft-head M32 gate FA2 identity drifted" >&2; exit 2; }
fi
[[ "$(docker ps -aq | wc -l)" -eq 0 ]]

export BSIZE=1
export CONC=1
export WALL=0
if [[ "$FR13_GATE_DRAFT_HEAD_M32" == "1" ]]; then
  export FR13_DRAFT_VOCAB_ROOT=0
  export FR13_DRAFT_VOCAB_K=0
  export FR13_DRAFT_VOCAB_BLOCKS=
  export FR13_MANDATORY_WEIGHT_BYTES=42025179008
  export FR13_WEIGHT_FLOOR_MS=153.938384645
  export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"
else
  export FR13_DRAFT_VOCAB_ROOT=1
fi
export FR13_FLOOR_ORDER=TH

source scripts/fr13_canonical_env.sh
if [[ "$FR13_GATE_DRAFT_HEAD_M32" == "1" ]]; then
  # canonical_env supplies the normal 64K loop subset; the full-head gate
  # intentionally removes that workload before launching either comparison.
  export FR13_DRAFT_VOCAB_K=0
  export FR13_DRAFT_VOCAB_BLOCKS=
  export FR13_MANDATORY_WEIGHT_BYTES=42025179008
  export FR13_WEIGHT_FLOOR_MS=153.938384645
  export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"
fi
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
if [[ "$FR13_GATE_DRAFT_HEAD_M32" == "1" ]]; then
  export FR13_MANDATORY_WEIGHT_BYTES=42025179008
  export FR13_WEIGHT_FLOOR_MS=153.938384645
  export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"
fi
if [[ "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_coop128_byte_ab" ]]; then
  export ENFORCE_EAGER=1
fi

mkdir -p "$RUNROOT_ABS"
FR13_M32_NEEDS_ALLOW=
[[ "$FR13_GATE_DRAFT_HEAD_M32" == "1" ]] \
  && FR13_M32_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
printf 'classification=real_swe_verified_b1_kernel_byte_diagnostic\ndiagnostic_only=1\nperformance_measurement=0\nprobe_eligible=0\nfloor_acceptance_eligible=0\nlauncher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\ncandidate_source_sha256=%s\nfa2_sha256=%s\nbm8_gate=%s\ndraft_head_m32_gate=%s\nstarted=%s\n' \
  "$$" "$RUNROOT_ABS" "$ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$CANDIDATE_SOURCE_SHA256" "$FA2_SHA" \
  "$FR13_GATE_BM8" "$FR13_GATE_DRAFT_HEAD_M32" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
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
  FR13_DRAFT_HEAD_M32_LIVE_AB="$FR13_GATE_DRAFT_HEAD_M32" \
  FR13_DRAFT_HEAD_M32_INSTANCE_ID=astropy__astropy-12907 \
  FR13_DRAFT_HEAD_M32_LIVE_JSON=/logs/fr13_draft_head_m32.live.json \
  FR13_NEEDS_ALLOW="$FR13_M32_NEEDS_ALLOW" \
  FR13_FIXED32_GDN_PATH_BV_CANDIDATE="$FR13_GATE_GDN_BV_CANDIDATE" \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  FR13_FA2_QROW16_SO_SHA256="$FA2_SHA" \
  FR13_FA2_QROW16_LIVE_PAGED_AB="$FR13_GATE_QROW16" \
  FR13_FA2_QROW16_LIVE_PAGED_AB_INSTANCE_ID=astropy__astropy-12907 \
  FR13_DFWD_UNIFIED_BM8_LIVE_AB="$FR13_GATE_BM8" \
  FR13_DFWD_UNIFIED_BM8_INSTANCE_ID=astropy__astropy-12907 \
  FR13_DFWD_UNIFIED_BM8_REAL_EVENT_PATH=/logs/fr13_dfwd_unified_bm8.real_event.arm \
  FR13_DFWD_UNIFIED_BM8_IDENTITY_JSON=/logs/fr13_dfwd_unified_bm8.identity.json \
  FR13_DFWD_UNIFIED_BM8_LIVE_JSON=/logs/fr13_dfwd_unified_bm8.live.json \
  FR13_DFWD_UNIFIED_BM8_SOURCE_COMMIT="$SOURCE_COMMIT" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh \
    "$ARM" hydra27_fixed32 "$SUBSET" \
    > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during the B1 gate" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during the B1 gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)" == "$RUNNER_SHA256" \
   && "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" \
   && "$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)" == "$CANDIDATE_SOURCE_SHA256" \
   && "$(sha256sum "$FORKED_FA2_SO" | cut -d' ' -f1)" == "$FA2_SHA" ]] \
  || { echo "B1 gate input identity changed during execution" >&2; exit 14; }
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "B1 gate source checkout changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

if [[ "$serve_rc" == "0" && "$FR13_GATE_BM8" == "1" ]]; then
  "$PYTHON_BIN" scripts/fr13_dfwd_unified_bm8_gate.py verify \
    --live-result "$RUNROOT_ABS/$ARM/logs/fr13_dfwd_unified_bm8.live.json" \
    --identity "$RUNROOT_ABS/$ARM/logs/fr13_dfwd_unified_bm8.identity.json" \
    --expected-source-commit "$SOURCE_COMMIT" \
    --expected-instance-id astropy__astropy-12907
fi
if [[ "$serve_rc" == "0" && "$FR13_GATE_DRAFT_HEAD_M32" == "1" ]]; then
  DRAFT_HEAD_LIVE="$RUNROOT_ABS/$ARM/logs/fr13_draft_head_m32.live.json"
  DRAFT_HEAD_FINAL_FLUSH="$RUNROOT_ABS/$ARM/fixed32_final_flush.json"
  [[ -f "$DRAFT_HEAD_LIVE" && ! -L "$DRAFT_HEAD_LIVE" ]] \
    || { echo "draft-head M32 live comparison evidence is missing" >&2; exit 4; }
  [[ -f "$DRAFT_HEAD_FINAL_FLUSH" && ! -L "$DRAFT_HEAD_FINAL_FLUSH" ]] \
    || { echo "draft-head M32 final flush evidence is missing" >&2; exit 4; }
  DRAFT_HEAD_FLUSH_GENERATION=$(
    "$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["ack"]["generation"])' \
      "$DRAFT_HEAD_FINAL_FLUSH"
  )
  DRAFT_HEAD_BOUNDARY="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_boundary_snapshot.${DRAFT_HEAD_FLUSH_GENERATION}.json"
  [[ -f "$DRAFT_HEAD_BOUNDARY" && ! -L "$DRAFT_HEAD_BOUNDARY" ]] \
    || { echo "draft-head M32 final boundary evidence is missing" >&2; exit 4; }
  DRAFT_HEAD_TRAFFIC_AUDIT="$RUNROOT_ABS/$ARM/fixed32_chat_traffic_audit.json"
  [[ -f "$DRAFT_HEAD_TRAFFIC_AUDIT" && ! -L "$DRAFT_HEAD_TRAFFIC_AUDIT" ]] \
    || { echo "draft-head M32 authenticated traffic audit is missing" >&2; exit 4; }
  "$PYTHON_BIN" scripts/fr13_draft_head_m32_pass.py validate-live \
    --live-result "$DRAFT_HEAD_LIVE" \
    --expected-live-sha256 "$(sha256sum "$DRAFT_HEAD_LIVE" | cut -d' ' -f1)" \
    --final-flush "$DRAFT_HEAD_FINAL_FLUSH" \
    --boundary-snapshot "$DRAFT_HEAD_BOUNDARY" \
    --chat-traffic-audit "$DRAFT_HEAD_TRAFFIC_AUDIT" \
    --candidate-source "$CANDIDATE_SOURCE" \
    --expected-candidate-source-sha256 "$CANDIDATE_SOURCE_SHA256" \
    > "$RUNROOT_ABS/$ARM/draft_head_m32_live_validation.json"
  printf 'live_validation_sha256=%s\n' \
    "$(sha256sum "$RUNROOT_ABS/$ARM/draft_head_m32_live_validation.json" | cut -d' ' -f1)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
fi

exit "$serve_rc"

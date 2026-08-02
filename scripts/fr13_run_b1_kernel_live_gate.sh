#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact FA2 shared object}"

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

B1_WORKLOAD_PROFILE=${FR13_B1_WORKLOAD_PROFILE:-full_vocab}
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
case "$B1_WORKLOAD_PROFILE" in
  full_vocab)
    DRAFT_VOCAB_ROOT=0
    DRAFT_VOCAB_K=0
    NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0
    MANDATORY_WEIGHT_BYTES=42025179008
    MANDATORY_WEIGHT_FLOOR_MS=153.9383846446886
    ONE_SIDED_U95_CAP_MS=177.0291423413919
    PROFILE_SUFFIX=
    ;;
  k64_root)
    [[ ( "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_force_wide256_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "static_persistent_stocktile_byte_ab" ) \
       && "$FR13_GATE_QROW16" == "0" \
       && "$FR13_GATE_TAW_NATIVE" == "0" \
       && "$FR13_GATE_DRAFT_HEAD_PAD" == "0" \
       && "$FR13_GATE_DRAFT_HEAD_M32" == "0" \
       && "$FR13_GATE_BM8" == "0" \
       && "$FR13_GATE_GDN_BV" == "0" ]] || {
      echo "B1 k64_root is restricted to the isolated wide256 byte gate" >&2
      exit 2
    }
    DRAFT_VOCAB_ROOT=1
    DRAFT_VOCAB_K=65536
    NEEDS_ALLOW=
    MANDATORY_WEIGHT_BYTES=32666638208
    MANDATORY_WEIGHT_FLOOR_MS=119.658015414
    ONE_SIDED_U95_CAP_MS=137.6067177261
    PROFILE_SUFFIX=_k64_root
    ;;
  *)
    echo "FR13_B1_WORKLOAD_PROFILE must be full_vocab or k64_root" >&2
    exit 2
    ;;
esac

ARM="hydra27_fixed32${PROFILE_SUFFIX}_${TAG}"
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
FA2_SHA=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
SOURCE_COMMIT=$(git rev-parse HEAD)

[[ "$FA2_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]]
[[ "$(docker ps -aq | wc -l)" -eq 0 ]]
if [[ "$B1_WORKLOAD_PROFILE" == "k64_root" ]]; then
  [[ -f "$DRAFT_VOCAB_BLOCKS_HOST" \
     && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
     && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
    || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
fi

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW="$NEEDS_ALLOW"
export FR13_FLOOR_ORDER=TH

source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "canonical B1 workload floor contract drifted" >&2; exit 2; }
if [[ "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_coop128_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_force_wide256_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "static_persistent_stocktile_byte_ab" ]]; then
  export ENFORCE_EAGER=1
fi

mkdir -p "$RUNROOT"
printf 'launcher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nfa2_sha256=%s\nbm8_gate=%s\ndraft_head_m32_gate=%s\nworkload_profile=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\nfr13_needs_allow=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nstarted=%s\n' \
  "$$" "$RUNROOT" "$ARM" "$SOURCE_COMMIT" "$FA2_SHA" "$FR13_GATE_BM8" \
  "$FR13_GATE_DRAFT_HEAD_M32" "$B1_WORKLOAD_PROFILE" \
  "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" "$NEEDS_ALLOW" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$(date -u +%FT%TZ)" \
  > "$RUNROOT/launcher_meta.txt"

.venv/bin/python scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT/runtime_manifest.at_launch.json"
.venv/bin/python scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT/external_manifest.at_launch.json"

OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
  LUMO_SWE_AUTOCOMMIT=0 \
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

if [[ "$serve_rc" == "0" && "$FR13_GATE_BM8" == "1" ]]; then
  .venv/bin/python scripts/fr13_dfwd_unified_bm8_gate.py verify \
    --live-result "$RUNROOT/$ARM/logs/fr13_dfwd_unified_bm8.live.json" \
    --identity "$RUNROOT/$ARM/logs/fr13_dfwd_unified_bm8.identity.json" \
    --expected-source-commit "$SOURCE_COMMIT" \
    --expected-instance-id astropy__astropy-12907
fi
if [[ "$serve_rc" == "0" && "$FR13_GATE_DRAFT_HEAD_M32" == "1" ]]; then
  DRAFT_HEAD_LIVE="$RUNROOT/$ARM/logs/fr13_draft_head_m32.live.json"
  DRAFT_HEAD_FINAL_FLUSH="$RUNROOT/$ARM/fixed32_final_flush.json"
  [[ -f "$DRAFT_HEAD_FINAL_FLUSH" && ! -L "$DRAFT_HEAD_FINAL_FLUSH" ]] \
    || { echo "draft-head M32 final flush evidence is missing" >&2; exit 4; }
  DRAFT_HEAD_FLUSH_GENERATION=$(
    .venv/bin/python -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["ack"]["generation"])' \
      "$DRAFT_HEAD_FINAL_FLUSH"
  )
  DRAFT_HEAD_BOUNDARY="$RUNROOT/$ARM/logs/fr13_fixed32_boundary_snapshot.${DRAFT_HEAD_FLUSH_GENERATION}.json"
  [[ -f "$DRAFT_HEAD_BOUNDARY" && ! -L "$DRAFT_HEAD_BOUNDARY" ]] \
    || { echo "draft-head M32 final boundary evidence is missing" >&2; exit 4; }
  DRAFT_HEAD_TRAFFIC_AUDIT="$RUNROOT/$ARM/fixed32_chat_traffic_audit.json"
  [[ -f "$DRAFT_HEAD_TRAFFIC_AUDIT" && ! -L "$DRAFT_HEAD_TRAFFIC_AUDIT" ]] \
    || { echo "draft-head M32 authenticated traffic audit is missing" >&2; exit 4; }
  DRAFT_HEAD_SOURCE_SHA=$(sha256sum scripts/fr10_phase4_patch_vllm_tree_gdn.py | cut -d' ' -f1)
  .venv/bin/python scripts/fr13_draft_head_m32_pass.py validate-live \
    --live-result "$DRAFT_HEAD_LIVE" \
    --expected-live-sha256 "$(sha256sum "$DRAFT_HEAD_LIVE" | cut -d' ' -f1)" \
    --final-flush "$DRAFT_HEAD_FINAL_FLUSH" \
    --boundary-snapshot "$DRAFT_HEAD_BOUNDARY" \
    --chat-traffic-audit "$DRAFT_HEAD_TRAFFIC_AUDIT" \
    --candidate-source scripts/fr10_phase4_patch_vllm_tree_gdn.py \
    --expected-candidate-source-sha256 "$DRAFT_HEAD_SOURCE_SHA" \
    > "$RUNROOT/$ARM/draft_head_m32_live_validation.json"
fi

exit "$serve_rc"

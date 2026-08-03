#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact FA2 shared object}"

QROW16_FA2_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_FA2_BYTES=299507792
QROW16_LIVE_PASS=$REPO/results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json
QROW16_LIVE_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77

[[ -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be a regular non-symlink file" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* ]] \
  || { echo "FORKED_FA2_SO must be an absolute path" >&2; exit 2; }

B1_DIAGNOSTIC_TASK_PROFILE=${FR13_B1_DIAGNOSTIC_TASK_PROFILE:-astropy12907}
case "$B1_DIAGNOSTIC_TASK_PROFILE" in
  astropy12907)
    B1_DIAGNOSTIC_TASK_ID=astropy__astropy-12907
    B1_DIAGNOSTIC_SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
    B1_DIAGNOSTIC_SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
    ;;
  astropy13236)
    B1_DIAGNOSTIC_TASK_ID=astropy__astropy-13236
    B1_DIAGNOSTIC_SUBSET=config/fr13_fixed32/subset_b1_diagnostic_astropy13236.json
    B1_DIAGNOSTIC_SUBSET_SHA256=f02687afcad677dab1960d0a4650786bd586e8493c2553a5010f66a0294c5c09
    ;;
  *)
    echo "FR13_B1_DIAGNOSTIC_TASK_PROFILE must be astropy12907 or astropy13236" >&2
    exit 2
    ;;
esac
export FR13_B1_DIAGNOSTIC_TASK_PROFILE="$B1_DIAGNOSTIC_TASK_PROFILE"

FR13_GATE_QROW16=${FR13_GATE_QROW16:-0}
case "$FR13_GATE_QROW16" in
  0|1) ;;
  *) echo "FR13_GATE_QROW16 must be 0 or 1" >&2; exit 2 ;;
esac
FR13_GATE_TAW_NATIVE=${FR13_GATE_TAW_NATIVE:-1}
FR13_GATE_DRAFT_HEAD_PAD=${FR13_GATE_DRAFT_HEAD_PAD:-0}
FR13_GATE_DRAFT_HEAD_M32=${FR13_GATE_DRAFT_HEAD_M32:-0}
FR13_GATE_DRAFT_HEAD_FP8=${FR13_GATE_DRAFT_HEAD_FP8:-0}
FR13_GATE_DFWD_TOP3=${FR13_GATE_DFWD_TOP3:-0}
FR13_GATE_BM8=${FR13_GATE_BM8:-0}
for gate in FR13_GATE_TAW_NATIVE FR13_GATE_DRAFT_HEAD_PAD FR13_GATE_DRAFT_HEAD_M32 FR13_GATE_DRAFT_HEAD_FP8 FR13_GATE_DFWD_TOP3 FR13_GATE_BM8; do
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
           || "$FR13_GATE_DRAFT_HEAD_FP8" != "0" \
           || "$FR13_GATE_DFWD_TOP3" != "0" \
           || "$FR13_GATE_GDN_BV" != "0" ) ]]; then
  echo "FR13_GATE_BM8 must be the only enabled kernel candidate" >&2
  exit 2
fi
if [[ "$FR13_GATE_DRAFT_HEAD_M32" == "1" \
      && ( "$FR13_GATE_QROW16" != "0" \
           || "$FR13_GATE_TAW_NATIVE" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_PAD" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_FP8" != "0" \
           || "$FR13_GATE_DFWD_TOP3" != "0" \
           || "$FR13_GATE_BM8" != "0" \
           || "$FR13_GATE_GDN_BV" != "0" ) ]]; then
  echo "FR13_GATE_DRAFT_HEAD_M32 must be the only enabled kernel candidate" >&2
  exit 2
fi
if [[ "$FR13_GATE_DRAFT_HEAD_FP8" == "1" \
      && ( "$FR13_GATE_QROW16" != "0" \
           || "$FR13_GATE_TAW_NATIVE" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_PAD" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_M32" != "0" \
           || "$FR13_GATE_DFWD_TOP3" != "0" \
           || "$FR13_GATE_BM8" != "0" \
           || "$FR13_GATE_GDN_BV" != "0" ) ]]; then
  echo "FR13_GATE_DRAFT_HEAD_FP8 must be the only enabled kernel candidate" >&2
  exit 2
fi
if [[ "$FR13_GATE_DFWD_TOP3" == "1" \
      && ( "$FR13_GATE_QROW16" != "0" \
           || "$FR13_GATE_TAW_NATIVE" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_PAD" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_M32" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_FP8" != "0" \
           || "$FR13_GATE_BM8" != "0" \
           || "$FR13_GATE_GDN_BV" != "0" \
           || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" != "stock" ) ]]; then
  echo "FR13_GATE_DFWD_TOP3 must be the only enabled kernel candidate" >&2
  exit 2
fi
FR13_GATE_DFWD_TOP3_SO=${FR13_GATE_DFWD_TOP3_SO:-}
FR13_GATE_DFWD_TOP3_SHA256=
if [[ "$FR13_GATE_DFWD_TOP3" == "1" ]]; then
  [[ "$FR13_GATE_DFWD_TOP3_SO" == /* \
     && -f "$FR13_GATE_DFWD_TOP3_SO" \
     && ! -L "$FR13_GATE_DFWD_TOP3_SO" ]] || {
    echo "FR13_GATE_DFWD_TOP3_SO must be an absolute regular non-symlink file" >&2
    exit 2
  }
  FR13_GATE_DFWD_TOP3_SHA256=$(sha256sum "$FR13_GATE_DFWD_TOP3_SO" | awk '{print $1}')
else
  [[ -z "$FR13_GATE_DFWD_TOP3_SO" ]] || {
    echo "FR13_GATE_DFWD_TOP3=0 forbids a candidate binary" >&2
    exit 2
  }
fi
if [[ "$B1_DIAGNOSTIC_TASK_PROFILE" == "astropy13236" \
      && ( ( "${FR13_FIXED32_CUTLASS_WAVE:-stock}" != "identity_onen_n5120_single_b1_byte_ab" \
             && "${FR13_FIXED32_CUTLASS_WAVE:-stock}" != "identity_onen_n5120_fullgrid_b1_byte_ab" ) \
           || "$FR13_GATE_QROW16" != "0" \
           || "$FR13_GATE_TAW_NATIVE" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_PAD" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_M32" != "0" \
           || "$FR13_GATE_DRAFT_HEAD_FP8" != "0" \
           || "$FR13_GATE_DFWD_TOP3" != "0" \
           || "$FR13_GATE_BM8" != "0" \
           || "$FR13_GATE_GDN_BV" != "0" ) ]]; then
  echo "astropy13236 B1 diagnostic profile is isolated to the N5120 CUTLASS byte gate" >&2
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
    if [[ ( "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_force_wide256_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "static_persistent_stocktile_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "divisor_static_stocktile_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_stage2_static_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_stage2_pingpong_b1_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_onen_b1_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_onen_n5120_single_b1_byte_ab" \
         || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_onen_n5120_fullgrid_b1_byte_ab" ) \
       && "$FR13_GATE_QROW16" == "0" \
       && "$FR13_GATE_TAW_NATIVE" == "0" \
       && "$FR13_GATE_DRAFT_HEAD_PAD" == "0" \
       && "$FR13_GATE_DRAFT_HEAD_M32" == "0" \
       && "$FR13_GATE_DRAFT_HEAD_FP8" == "0" \
       && "$FR13_GATE_DFWD_TOP3" == "0" \
       && "$FR13_GATE_BM8" == "0" \
       && "$FR13_GATE_GDN_BV" == "0" ]]; then
      :
    elif [[ "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "stock" \
            && "$FR13_GATE_QROW16" == "0" \
            && "$FR13_GATE_TAW_NATIVE" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_PAD" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_M32" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_FP8" == "1" \
            && "$FR13_GATE_DFWD_TOP3" == "0" \
            && "$FR13_GATE_BM8" == "0" \
            && "$FR13_GATE_GDN_BV" == "0" ]]; then
      :
    elif [[ "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "stock" \
            && "$FR13_GATE_QROW16" == "0" \
            && "$FR13_GATE_TAW_NATIVE" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_PAD" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_M32" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_FP8" == "0" \
            && "$FR13_GATE_DFWD_TOP3" == "1" \
            && "$FR13_GATE_BM8" == "0" \
            && "$FR13_GATE_GDN_BV" == "0" ]]; then
      :
    elif [[ "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "stock" \
            && "${FR13_FIXED32_B1_FP8_QUANT_REGCACHE:-0}" == "byte_ab" \
            && "$FR13_GATE_QROW16" == "0" \
            && "$FR13_GATE_TAW_NATIVE" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_PAD" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_M32" == "0" \
            && "$FR13_GATE_DRAFT_HEAD_FP8" == "0" \
            && "$FR13_GATE_BM8" == "0" \
            && "$FR13_GATE_GDN_BV" == "0" ]]; then
      :
    else
      echo "B1 k64_root requires an isolated admitted candidate" >&2
      exit 2
    fi
    DRAFT_VOCAB_ROOT=1
    DRAFT_VOCAB_K=65536
    NEEDS_ALLOW=
    if [[ "$FR13_GATE_DRAFT_HEAD_FP8" == "1" ]]; then
      MANDATORY_WEIGHT_BYTES=30989326208
      MANDATORY_WEIGHT_FLOOR_MS=113.514015414
      ONE_SIDED_U95_CAP_MS=130.541117726
    else
      MANDATORY_WEIGHT_BYTES=32666638208
      MANDATORY_WEIGHT_FLOOR_MS=119.658015414
      ONE_SIDED_U95_CAP_MS=137.6067177261
    fi
    PROFILE_SUFFIX=_k64_root
    ;;
  *)
    echo "FR13_B1_WORKLOAD_PROFILE must be full_vocab or k64_root" >&2
    exit 2
    ;;
esac

ARM="hydra27_fixed32${PROFILE_SUFFIX}_${TAG}"
SUBSET=$B1_DIAGNOSTIC_SUBSET
FA2_SHA=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
SOURCE_COMMIT=$(git rev-parse HEAD)
DRAFT_HEAD_FP8_ARM=
QROW16_PRODUCTION=0
QROW16_PRODUCTION_LIVE_PASS=
QROW16_PRODUCTION_LIVE_PASS_SHA256=
if [[ "$FR13_GATE_DRAFT_HEAD_FP8" == "1" ]]; then
  DRAFT_HEAD_FP8_ARM=$ARM
  [[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$QROW16_FA2_BYTES" \
     && "$FA2_SHA" == "$QROW16_FA2_SHA256" \
     && -f "$QROW16_LIVE_PASS" && ! -L "$QROW16_LIVE_PASS" \
     && "$(sha256sum "$QROW16_LIVE_PASS" | awk '{print $1}')" \
        == "$QROW16_LIVE_PASS_SHA256" ]] || {
    echo "FP8 gate requires the pinned Qrow16 production binary and live PASS" >&2
    exit 2
  }
  .venv/bin/python - "$QROW16_LIVE_PASS" "$QROW16_FA2_SHA256" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_qrow16_pass_sidecar as qrow16

payload, _ = qrow16.load_json(Path(sys.argv[1]))
qrow16.validate_live_result(payload, candidate_sha256=sys.argv[2])
PY
  QROW16_PRODUCTION=1
  QROW16_PRODUCTION_LIVE_PASS=$QROW16_LIVE_PASS
  QROW16_PRODUCTION_LIVE_PASS_SHA256=$QROW16_LIVE_PASS_SHA256
fi

[[ "$FA2_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]]
[[ "$(docker ps -aq | wc -l)" -eq 0 ]]
[[ -f "$SUBSET" && ! -L "$SUBSET" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$B1_DIAGNOSTIC_SUBSET_SHA256" ]] \
  || { echo "pinned B1 diagnostic subset identity drifted" >&2; exit 2; }
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
export FR13_DRAFT_HEAD_FP8="$FR13_GATE_DRAFT_HEAD_FP8"
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
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "static_persistent_stocktile_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "divisor_static_stocktile_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_stage2_static_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_stage2_pingpong_b1_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_onen_b1_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_onen_n5120_single_b1_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_onen_n5120_fullgrid_b1_byte_ab" ]]; then
  export ENFORCE_EAGER=1
elif [[ "${FR13_FIXED32_B1_FP8_QUANT_REGCACHE:-0}" == "byte_ab" ]]; then
  export ENFORCE_EAGER=1
elif [[ "${FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB:-0}" == "1" ]]; then
  export ENFORCE_EAGER=1
fi

mkdir -p "$RUNROOT"
printf 'launcher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nfa2_sha256=%s\nbm8_gate=%s\ndraft_head_m32_gate=%s\ndraft_head_fp8_gate=%s\ndfwd_top3_gate=%s\ndfwd_top3_sha256=%s\nb1_diagnostic_task_profile=%s\nb1_diagnostic_task_id=%s\nb1_diagnostic_subset=%s\nb1_diagnostic_subset_sha256=%s\nworkload_profile=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\nfr13_needs_allow=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nstarted=%s\n' \
  "$$" "$RUNROOT" "$ARM" "$SOURCE_COMMIT" "$FA2_SHA" "$FR13_GATE_BM8" \
  "$FR13_GATE_DRAFT_HEAD_M32" "$FR13_GATE_DRAFT_HEAD_FP8" \
  "$FR13_GATE_DFWD_TOP3" "$FR13_GATE_DFWD_TOP3_SHA256" \
  "$B1_DIAGNOSTIC_TASK_PROFILE" "$B1_DIAGNOSTIC_TASK_ID" "$SUBSET" \
  "$B1_DIAGNOSTIC_SUBSET_SHA256" "$B1_WORKLOAD_PROFILE" \
  "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" "$NEEDS_ALLOW" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$(date -u +%FT%TZ)" \
  > "$RUNROOT/launcher_meta.txt"
printf 'qrow16_production=%s\nqrow16_fa2_sha256=%s\nqrow16_live_pass_sha256=%s\n' \
  "$QROW16_PRODUCTION" "$FA2_SHA" \
  "$QROW16_PRODUCTION_LIVE_PASS_SHA256" >> "$RUNROOT/launcher_meta.txt"

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
  FR13_DRAFT_HEAD_M32_INSTANCE_ID="$B1_DIAGNOSTIC_TASK_ID" \
  FR13_DRAFT_HEAD_M32_LIVE_JSON=/logs/fr13_draft_head_m32.live.json \
  FR13_DRAFT_HEAD_FP8="$FR13_GATE_DRAFT_HEAD_FP8" \
  FR13_DRAFT_HEAD_FP8_ARM="$DRAFT_HEAD_FP8_ARM" \
  FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON=/logs/fr13_draft_head_fp8.engagement.json \
  FR13_DFWD_K64_TOP3="$FR13_GATE_DFWD_TOP3" \
  FR13_DFWD_K64_TOP3_SO="$FR13_GATE_DFWD_TOP3_SO" \
  FR13_DFWD_K64_TOP3_SHA256="$FR13_GATE_DFWD_TOP3_SHA256" \
  FR13_FIXED32_GDN_PATH_BV_CANDIDATE="$FR13_GATE_GDN_BV_CANDIDATE" \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  FR13_FA2_QROW16_SO_SHA256="$FA2_SHA" \
  FR13_FA2_QROW16_LIVE_PAGED_AB="$FR13_GATE_QROW16" \
  FR13_FA2_QROW16_LIVE_PAGED_AB_INSTANCE_ID="$B1_DIAGNOSTIC_TASK_ID" \
  FR13_FA2_QROW16_PRODUCTION="$QROW16_PRODUCTION" \
  FR13_FA2_QROW16_LIVE_PASS_JSON="$QROW16_PRODUCTION_LIVE_PASS" \
  FR13_FA2_QROW16_LIVE_PASS_SHA256="$QROW16_PRODUCTION_LIVE_PASS_SHA256" \
  FR13_DFWD_UNIFIED_BM8_LIVE_AB="$FR13_GATE_BM8" \
  FR13_DFWD_UNIFIED_BM8_INSTANCE_ID="$B1_DIAGNOSTIC_TASK_ID" \
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
    --expected-instance-id "$B1_DIAGNOSTIC_TASK_ID"
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
if [[ "$serve_rc" == "0" && "$FR13_GATE_DRAFT_HEAD_FP8" == "1" ]]; then
  DRAFT_HEAD_FP8_ENGAGEMENT="$RUNROOT/$ARM/logs/fr13_draft_head_fp8.engagement.json"
  DRAFT_HEAD_FP8_FINAL_FLUSH="$RUNROOT/$ARM/fixed32_final_flush.json"
  DRAFT_HEAD_FP8_TRAFFIC_AUDIT="$RUNROOT/$ARM/fixed32_chat_traffic_audit.json"
  for evidence in \
    "$DRAFT_HEAD_FP8_ENGAGEMENT" \
    "$DRAFT_HEAD_FP8_FINAL_FLUSH" \
    "$DRAFT_HEAD_FP8_TRAFFIC_AUDIT"; do
    [[ -f "$evidence" && ! -L "$evidence" ]] \
      || { echo "draft-head FP8 evidence is missing: $evidence" >&2; exit 4; }
  done
  DRAFT_HEAD_FP8_FLUSH_GENERATION=$(
    .venv/bin/python -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["ack"]["generation"])' \
      "$DRAFT_HEAD_FP8_FINAL_FLUSH"
  )
  DRAFT_HEAD_FP8_BOUNDARY="$RUNROOT/$ARM/logs/fr13_fixed32_boundary_snapshot.${DRAFT_HEAD_FP8_FLUSH_GENERATION}.json"
  [[ -f "$DRAFT_HEAD_FP8_BOUNDARY" && ! -L "$DRAFT_HEAD_FP8_BOUNDARY" ]] \
    || { echo "draft-head FP8 final boundary evidence is missing" >&2; exit 4; }
  DRAFT_HEAD_FP8_ACCEPTANCE="$RUNROOT/$ARM/draft_head_fp8_acceptance.json"
  DRAFT_HEAD_FP8_QROW16_SIDECAR="$RUNROOT/$ARM/logs/fr13_fa2_qrow16_production_pass.json"
  DRAFT_HEAD_FP8_QROW16_CAPTURE="$RUNROOT/$ARM/logs/fr13_fa2_qrow16_production_capture.json"
  .venv/bin/python scripts/fr13_measure.py deploy-speed \
    --arm "$ARM" \
    --out-root "$RUNROOT/$ARM/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$DRAFT_HEAD_FP8_ACCEPTANCE"
  DRAFT_HEAD_FP8_SOURCE_SHA=$(
    sha256sum scripts/fr10_phase4_patch_vllm_tree_gdn.py | cut -d' ' -f1
  )
  .venv/bin/python scripts/fr13_draft_head_fp8_gate.py \
    --engagement "$DRAFT_HEAD_FP8_ENGAGEMENT" \
    --acceptance "$DRAFT_HEAD_FP8_ACCEPTANCE" \
    --final-flush "$DRAFT_HEAD_FP8_FINAL_FLUSH" \
    --boundary-snapshot "$DRAFT_HEAD_FP8_BOUNDARY" \
    --chat-traffic-audit "$DRAFT_HEAD_FP8_TRAFFIC_AUDIT" \
    --qrow16-sidecar "$DRAFT_HEAD_FP8_QROW16_SIDECAR" \
    --qrow16-capture "$DRAFT_HEAD_FP8_QROW16_CAPTURE" \
    --qrow16-so "$FORKED_FA2_SO" \
    --candidate-source scripts/fr10_phase4_patch_vllm_tree_gdn.py \
    --expected-source-sha256 "$DRAFT_HEAD_FP8_SOURCE_SHA" \
    --expected-source-commit "$SOURCE_COMMIT" \
    --repo "$PWD" \
    --out "$RUNROOT/$ARM/draft_head_fp8_real_b1_gate.json"
fi
if [[ "$serve_rc" == "0" && "$FR13_GATE_DFWD_TOP3" == "1" ]]; then
  DFWD_TOP3_RUNTIME_LOG="$RUNROOT/$ARM/docker_after_tasks.log"
  [[ -f "$DFWD_TOP3_RUNTIME_LOG" && ! -L "$DFWD_TOP3_RUNTIME_LOG" ]] || {
    echo "FR13 DFWD K64 top3 runtime log is missing" >&2
    exit 2
  }
  grep -F "[FR13_DFWD_K64_TOP3] ready B1 K64 mapped width3" \
    "$DFWD_TOP3_RUNTIME_LOG" >/dev/null || {
      echo "FR13 DFWD K64 top3 readiness marker is missing" >&2
      exit 2
    }
  grep -F "[FR13_DFWD_K64_TOP3] engaged stock_argmax_topk_map_copy=0" \
    "$DFWD_TOP3_RUNTIME_LOG" >/dev/null || {
      echo "FR13 DFWD K64 top3 engagement marker is missing" >&2
      exit 2
    }
  grep -F "[FR13_DFWD_K64_TOP3] graph captured_calls=4" \
    "$DFWD_TOP3_RUNTIME_LOG" >/dev/null || {
      echo "FR13 DFWD K64 top3 graph marker is missing" >&2
      exit 2
    }
fi

exit "$serve_rc"

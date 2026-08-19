#!/usr/bin/env bash
# FR14 PROMOTION A/B — ARM S: arm the split-K FA2 kernel on a real serve and try
# to obtain the generation traces Mark's degenerate-eyeball condition needs.
#
# THIS ARM IS EXPECTED TO REFUSE, AND THE REFUSAL IS THE EVIDENCE.
# The offline reachability probe (promotion_ab_arm_s_reachability.json) drove the
# DEPLOYED selector helpers and found no route from this HEAD to a served
# split-K token:
#   * FR13_FA2_QROW32_B1_PRODUCTION_ARM -- the ONLY route that ever sets
#     candidate_served=True -- rejects "gqa_pair_splitk": the patcher's
#     _FR13_FA2_QROW32_B1_PRODUCTION_ARMS is ("nosplit", "gqa_pair") and split-K
#     is deliberately absent (splitk_fa2.md 1: "gate-only").
#   * FR13_FA2_QROW32_B1_LIVE_AB_ARM accepts the NAME, but that route is a
#     one-shot SHADOW byte gate -- the gate runner's own launcher_meta records
#     "qrow16_reference_served=1 / candidate_returned=0" -- and its first act per
#     layer is _fr13_fa2_qrow32_b1_require_same_reduction, which refuses split-K
#     against the served num_splits=0 reference by construction.
# This script runs it anyway, because the campaign's standing doctrine is that a
# mechanism read out of source is not a measurement (pass 30). It is also the
# FIRST LIVE EXECUTION of pass 50's split-K launcher branch, including the
# re-hash that separates the PID-shifted twin from the pinned binary.
#
# Cost: one boot. It is NOT a 75-100 min serve arm and must never be reported as
# one.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"

PYTHON_BIN=.venv/bin/python
FIXED32_MODE=hydra27_fixed32
TASK_ID=astropy__astropy-12907
# FR13_FIXED32_B1_DIAGNOSTIC=1 is REQUIRED by the live-A/B predicate, and the
# launcher then requires the ingress task list to be exactly the one diagnostic
# task. A live-A/B arm therefore CANNOT carry the exact4 set -- recorded as a
# campaign finding, not worked around.
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
B1_DIAGNOSTIC_PROFILE=astropy12907
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
SPLITK_SO=/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so
SPLITK_SHA256=28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857
SPLITK_SIZE=300123792
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SPLITK_CLOSURE=4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878
SPLITK_SASS=3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e
SPLITK_BASELINE_SASS=fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470
MANDATORY_WEIGHT_BYTES=25430574256
MANDATORY_WEIGHT_FLOOR_MS=93.15228665201465

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_promoab_Sr7_$TS
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="hydra27_fixed32_promoab_S_r7"
SOURCE_COMMIT=$(git rev-parse HEAD)
PATCH_SOURCE_SHA256=$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | awk '{print $1}')

[[ ! -e "$RUNROOT_ABS" ]] || { echo "RUNROOT must be new" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ -z "$(docker ps -aq)" ]] || { echo "docker must be empty before boot" >&2; exit 2; }
sync
sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
awk '/^MemFree:/{exit ($2/1048576 < 82.3)}' /proc/meminfo \
  || { echo "unified-memory preflight failed" >&2; exit 2; }
[[ -f "$SPLITK_SO" && ! -L "$SPLITK_SO" \
   && "$(sha256sum "$SPLITK_SO" | awk '{print $1}')" == "$SPLITK_SHA256" \
   && "$(stat -c '%s' "$SPLITK_SO")" == "$SPLITK_SIZE" ]] \
  || { echo "split-K binary missing or drifted" >&2; exit 2; }
mkdir -p "$RUNROOT_ABS"

export TAG=promoabS
export BSIZE=1 CONC=1 WALL=9000
export FR13_CAMPAIGN_TASK_BUDGET_S=5400
export FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
export FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=full_vocab
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "K0 floor contract drifted" >&2; exit 2; }

printf 'arm=%s\nkind=splitk_live_ab_probe\nsource_commit=%s\npatch_source_sha256=%s\nsplitk_so_sha256=%s\nexpectation=FIRST TIER-B SERVE (lane 4 closed the three refusals; credential earned at this HEAD)\nstarted=%s\n' \
  "$ARM" "$SOURCE_COMMIT" "$PATCH_SOURCE_SHA256" "$SPLITK_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/arm_meta.txt"

echo "===== $ARM (split-K live arm) $(date -u +%FT%TZ) ====="
env RUNROOT="$RUNROOT_ABS" \
  OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=9000 \
  LUMO_SWE_AUTOCOMMIT=0 \
  FR13_FIXED32_B1_DIAGNOSTIC=1 \
  FR13_B1_DIAGNOSTIC_TASK_PROFILE="$B1_DIAGNOSTIC_PROFILE" \
  FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
  FR13_LFWD_GPU_TIMER=1 FR13_DFWD_SPLIT=1 \
  FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}.json \
  FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_dfwd.json \
  FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_cfwd.json \
  FR13_LFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_lfwd.json \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
  FR13_FIXED32_BATCH_GDN_BYTE_AB=0 FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
  FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
  FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
  FR13_FIXED32_GDN_PATH_BV_CANDIDATE= FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
  FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
  FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
  FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0 \
  FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0 \
  FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
  FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
  FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
  FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
  FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
  FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
  FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
  FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
  FR13_FA2_QROW32_B1_LIVE_AB_ARM=gqa_pair_splitk \
  FR13_FA2_QROW32_B1_TIER_B_SERVE=1 \
  FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL=/workspace/output/fr14_splitk_tierb_credential_r7.json \
  FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256=f2ffbdb2b0075f5696d44128c37950253a60fec1ff7045d630618a371ea35ebb \
  FR13_FA2_QROW32_B1_LIVE_AB_INSTANCE_ID="$TASK_ID" \
  FR13_FA2_QROW32_B1_LIVE_AB_JSON=/logs/fr13_fa2_qrow32_b1_gqa_pair_splitk_live_paged_ab.json \
  FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
  FR13_FA2_QROW32_B1_SO_SHA256="$SPLITK_SHA256" \
  FR13_FA2_QROW32_B1_SO_SIZE="$SPLITK_SIZE" \
  FR13_FA2_QROW32_B1_FA2_HEAD="$FA2_HEAD" \
  FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256="$SPLITK_CLOSURE" \
  FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST="$SPLITK_SASS" \
  FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST="$SPLITK_BASELINE_SASS" \
  FR13_FA2_QROW32_B1_SOURCE_COMMIT="$SOURCE_COMMIT" \
  FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
  FR13_FIXED32_ATTRIBUTION_ONLY=0 \
  FORKED_FA2_SO="$SPLITK_SO" \
  TMPDIR=/home/mark/shared/tmp-scratch \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$FIXED32_MODE" "$SUBSET" \
  > "$RUNROOT_ABS/$ARM.runlog" 2>&1
rc=$?
printf 'serve_rc=%s\nended=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/arm_meta.txt"
echo "[$ARM] serve rc=$rc $(date -u +%FT%TZ)"
echo "[$ARM] docker containers after: $(docker ps -aq | wc -l)"
echo "[$ARM] runroot -> $RUNROOT_ABS"
exit "$rc"

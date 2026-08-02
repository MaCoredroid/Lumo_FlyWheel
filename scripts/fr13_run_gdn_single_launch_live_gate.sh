#!/usr/bin/env bash
# Internal runner for reference-served fixed32 GDN root-loop qualification.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
CORE_PATH=$(realpath "${BASH_SOURCE[0]}")
cd "$REPO"

PROFILE=${1:?usage: fr13_run_gdn_single_launch_live_gate.sh b1|b4}
: "${QUALIFICATION_ENTRYPOINT:?invoke the B1 or B4 qualification entrypoint}"
: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact-safe stock FA2 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TOPOLOGY=${TOPOLOGY:-hydra27_fixed32}
SUBSET_B1=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_B4=config/fr13_fixed32/subset_b4_four.json
SUBSET_B1_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
SUBSET_B4_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
B4_KV_CACHE_MEMORY_BYTES=42949672960
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
CANDIDATE=fixed32_gdn_single_launch_root_loop_v1
KERNEL_SOURCE=src/lumo_flywheel_serving/fr13_gdn_single_launch_root_loop.py
SUPPORT_SOURCE=src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
PATCHER_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
LAUNCHER_SOURCE=scripts/fr13_launch_forked_fa2_tree_server.sh
RUNTIME_MANIFEST_SOURCE=scripts/fr13_runtime_manifest.py
INGRESS_SOURCE=src/lumo_flywheel_serving/inference_proxy.py
VERIFIER=scripts/fr13_gdn_single_launch_live_verdict.py
RESOURCE_AUDIT=results/fr13_fixed32_gdn_single_launch_root_loop_v1_live_ready_20260802
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ENTRYPOINT_PATH=$(realpath "$QUALIFICATION_ENTRYPOINT")

case "$TOPOLOGY" in
  tail6_fixed32)
    LOGICAL_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *) echo "TOPOLOGY must be tail6_fixed32 or hydra27_fixed32" >&2; exit 2 ;;
esac
case "$PROFILE" in
  b1)
    BATCH_SIZE=1
    CONCURRENCY=1
    KV_CACHE_MEMORY_BYTES_VALUE=
    B1_DIAGNOSTIC=1
    B1_GATE=1
    B4_GATE=0
    FR10_METRICS_VALUE=0
    SUBSET=$SUBSET_B1
    SUBSET_SHA256=$SUBSET_B1_SHA256
    ENTRYPOINT_NAME=fr13_run_b1_gdn_single_launch_live_gate.sh
    RUN_CLASSIFICATION=one_real_swe_verified_k64_root1_gdn_root_loop_b1_byte_diagnostic
    ;;
  b4)
    BATCH_SIZE=4
    CONCURRENCY=4
    KV_CACHE_MEMORY_BYTES_VALUE=$B4_KV_CACHE_MEMORY_BYTES
    B1_DIAGNOSTIC=0
    B1_GATE=0
    B4_GATE=1
    FR10_METRICS_VALUE=1
    SUBSET=$SUBSET_B4
    SUBSET_SHA256=$SUBSET_B4_SHA256
    ENTRYPOINT_NAME=fr13_run_b4_gdn_single_launch_live_gate.sh
    RUN_CLASSIFICATION=real_swe_verified_exact4_k64_root1_gdn_root_loop_b4_byte_diagnostic
    ;;
  *) echo "profile must be b1 or b4" >&2; exit 2 ;;
esac
ARM="${TOPOLOGY}_k64_root1_gdn_root_loop_${PROFILE}_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
VERDICT="$RUNROOT_ABS/${PROFILE}_gdn_single_launch_gate_verdict.json"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$ENTRYPOINT_PATH" == "$SCRIPT_DIR/$ENTRYPOINT_NAME" ]] \
  || { echo "qualification entrypoint does not match profile" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "canonical SWE/K64 inputs drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before qualification" >&2; exit 2; }
for probe_value in \
  "${CAPTURE_ONLY:-0}" \
  "${ACCEPT_SPEED_PROBE:-0}" \
  "${PROBE_ONLY:-0}"; do
  [[ "$probe_value" == "0" ]] \
    || { echo "single-launch qualification rejects probe/synthetic traffic" >&2; exit 2; }
done
(cd "$RESOURCE_AUDIT" && sha256sum -c SHA256SUMS >/dev/null) \
  || { echo "single-launch sm_121a resource audit checksum failure" >&2; exit 2; }

RUNNER_SHA256=$(sha256sum "$ENTRYPOINT_PATH" | awk '{print $1}')
CORE_SHA256=$(sha256sum "$CORE_PATH" | awk '{print $1}')
VERIFIER_SHA256=$(sha256sum "$VERIFIER" | awk '{print $1}')
KERNEL_SOURCE_SHA256=$(sha256sum "$KERNEL_SOURCE" | awk '{print $1}')
SUPPORT_SOURCE_SHA256=$(sha256sum "$SUPPORT_SOURCE" | awk '{print $1}')
PATCHER_SOURCE_SHA256=$(sha256sum "$PATCHER_SOURCE" | awk '{print $1}')
LAUNCHER_SOURCE_SHA256=$(sha256sum "$LAUNCHER_SOURCE" | awk '{print $1}')
RUNTIME_MANIFEST_SOURCE_SHA256=$(sha256sum "$RUNTIME_MANIFEST_SOURCE" | awk '{print $1}')
INGRESS_SOURCE_SHA256=$(sha256sum "$INGRESS_SOURCE" | awk '{print $1}')
audit_source_sha256() {
  awk -F '\t' -v path="$1" '$2 == path {print $3}' \
    "$RESOURCE_AUDIT/source_hashes.tsv"
}
[[ "$(audit_source_sha256 "$KERNEL_SOURCE")" == "$KERNEL_SOURCE_SHA256" \
   && "$(audit_source_sha256 "$SUPPORT_SOURCE")" == "$SUPPORT_SOURCE_SHA256" \
   && "$(audit_source_sha256 "$PATCHER_SOURCE")" == "$PATCHER_SOURCE_SHA256" \
   && "$(audit_source_sha256 "$LAUNCHER_SOURCE")" == "$LAUNCHER_SOURCE_SHA256" \
   && "$(audit_source_sha256 "$RUNTIME_MANIFEST_SOURCE")" == "$RUNTIME_MANIFEST_SOURCE_SHA256" \
   && "$(audit_source_sha256 "$INGRESS_SOURCE")" == "$INGRESS_SOURCE_SHA256" \
   && "$(audit_source_sha256 "scripts/$ENTRYPOINT_NAME")" == "$RUNNER_SHA256" \
   && "$(audit_source_sha256 "scripts/fr13_run_gdn_single_launch_live_gate.sh")" == "$CORE_SHA256" \
   && "$(audit_source_sha256 "$VERIFIER")" == "$VERIFIER_SHA256" ]] \
  || { echo "single-launch source differs from the ready audit" >&2; exit 2; }
unset -f audit_source_sha256

export BSIZE=$BATCH_SIZE
export CONC=$CONCURRENCY
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "/workspace/scripts/fr13_dvk_subset_blocks.json" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "fixed32 K64/root1 deployment contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
printf 'classification=%s\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\ncandidate_shadow_only=1\nreference_always_served=1\ncandidate=%s\nmode=%s\nlogical_drafts=%s\nvalid_mask=%s\nphysical_rows_per_request=32\nbatch_size=%s\nconcurrency=%s\ndraft_vocab_k=65536\ndraft_vocab_root=1\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nsource_commit=%s\nkernel_source_sha256=%s\nsupport_source_sha256=%s\nrunner_sha256=%s\ncore_runner_sha256=%s\nverifier_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nprobe_modes=forbidden\nstarted=%s\n' \
  "$RUN_CLASSIFICATION" "$CANDIDATE" "$TOPOLOGY" "$LOGICAL_DRAFTS" "$VALID_MASK" \
  "$BATCH_SIZE" "$CONCURRENCY" "$BLOCK_MAP_SHA256" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$SOURCE_COMMIT" "$KERNEL_SOURCE_SHA256" \
  "$SUPPORT_SOURCE_SHA256" \
  "$RUNNER_SHA256" "$CORE_SHA256" "$VERIFIER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" "$RUNTIME_MANIFEST_SOURCE" \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR="$BATCH_SIZE" \
    SWE_CONCURRENCY="$CONCURRENCY" AGENT_WALL_S= \
    KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES_VALUE" \
    LUMO_SWE_AUTOCOMMIT=0 \
    CAPTURE_ONLY=0 ACCEPT_SPEED_PROBE=0 PROBE_ONLY=0 \
    FR13_FIXED32_B1_DIAGNOSTIC="$B1_DIAGNOSTIC" \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR10_METRICS="$FR10_METRICS_VALUE" \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
    FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
    FR13_SUBTREE_PARALLEL=1 FR13_FIXED32_GDN_PARENT_GROUP=0 \
    FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE=1 \
    FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0 \
    FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_BYTE_AB="$B1_GATE" \
    FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_BYTE_AB="$B4_GATE" \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
    FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$TOPOLOGY" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" "$RUNTIME_MANIFEST_SOURCE" \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during qualification" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during qualification" >&2; exit 14; }
[[ "$(sha256sum "$ENTRYPOINT_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
   && "$(sha256sum "$CORE_PATH" | awk '{print $1}')" == "$CORE_SHA256" \
   && "$(sha256sum "$VERIFIER" | awk '{print $1}')" == "$VERIFIER_SHA256" \
   && "$(sha256sum "$KERNEL_SOURCE" | awk '{print $1}')" == "$KERNEL_SOURCE_SHA256" \
   && "$(sha256sum "$SUPPORT_SOURCE" | awk '{print $1}')" == "$SUPPORT_SOURCE_SHA256" \
   && "$(sha256sum "$PATCHER_SOURCE" | awk '{print $1}')" == "$PATCHER_SOURCE_SHA256" \
   && "$(sha256sum "$LAUNCHER_SOURCE" | awk '{print $1}')" == "$LAUNCHER_SOURCE_SHA256" \
   && "$(sha256sum "$RUNTIME_MANIFEST_SOURCE" | awk '{print $1}')" == "$RUNTIME_MANIFEST_SOURCE_SHA256" \
   && "$(sha256sum "$INGRESS_SOURCE" | awk '{print $1}')" == "$INGRESS_SOURCE_SHA256" ]] \
  || { echo "qualification source changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

"$PYTHON_BIN" "$VERIFIER" \
  --batch-size "$BATCH_SIZE" \
  --mode "$TOPOLOGY" \
  --arm-dir "$ARMDIR" \
  --runner "$ENTRYPOINT_PATH" \
  --runner-sha256 "$RUNNER_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --kernel-source "$KERNEL_SOURCE" \
  --support-source "$SUPPORT_SOURCE" \
  --subset "$SUBSET" \
  --subset-sha256 "$SUBSET_SHA256" \
  --block-map "$BLOCK_MAP" \
  --block-map-sha256 "$BLOCK_MAP_SHA256" \
  --runtime-manifest "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  --external-manifest "$RUNROOT_ABS/external_manifest.at_end.json" \
  --stock-fa2-sha256 "$STOCK_FA2_SHA256" \
  --output "$VERDICT" \
  > "$RUNROOT_ABS/${PROFILE}_gate_verdict_validation.json"
chmod 400 "$VERDICT"
printf 'PASS: %s\n' "$VERDICT"

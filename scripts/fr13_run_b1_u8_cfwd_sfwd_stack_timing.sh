#!/usr/bin/env bash
# Exact-source stock vs U8/packed-CFWD/target/SFWD real B1 timing pair.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_B1_U8_CFWD_SFWD_TIMING:-0}" in
  1) ;;
  0)
    echo "U8/CFWD/SFWD timing is disabled; set FR13_RUN_B1_U8_CFWD_SFWD_TIMING=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_U8_CFWD_SFWD_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the pinned stock FA2 binary}"
: "${DFWD_U8_SO:?set DFWD_U8_SO to the pinned U8 shared object}"
: "${U8_GATE_JSON:?set U8_GATE_JSON to the fresh U8 shadow gate}"
: "${U8_GATE_SHA256:?set its raw SHA-256}"
: "${U8_LIVE_RESULT_JSON:?set U8_LIVE_RESULT_JSON to the shared raw U8 result}"
: "${U8_LIVE_RESULT_SHA256:?set its raw SHA-256}"
: "${SHARED_FINAL_FLUSH_JSON:?set the shared final flush}"
: "${SHARED_FINAL_FLUSH_SHA256:?set its raw SHA-256}"
: "${SHARED_BOUNDARY_SNAPSHOT_JSON:?set the shared boundary snapshot}"
: "${SHARED_BOUNDARY_SNAPSHOT_SHA256:?set its raw SHA-256}"
: "${SHARED_CHAT_TRAFFIC_AUDIT_JSON:?set the shared traffic audit}"
: "${SHARED_CHAT_TRAFFIC_AUDIT_SHA256:?set its raw SHA-256}"
: "${CFWD_LIVE_RESULT_JSON:?set the shared raw CFWD result}"
: "${CFWD_LIVE_RESULT_SHA256:?set its raw SHA-256}"
: "${CFWD_PASS_JSON:?set the CFWD production credential from the shared gate}"
: "${CFWD_PASS_SHA256:?set its raw SHA-256}"
: "${CFWD_U8_COMPOSED_GATE_JSON:?set the fresh shared CFWD/U8 gate}"
: "${CFWD_U8_COMPOSED_GATE_SHA256:?set its raw SHA-256}"
: "${CUTLASS_TARGET_SO:?set the current target shared object}"
: "${CUTLASS_TARGET_PASS:?set the fresh target live PASS}"
: "${CUTLASS_TARGET_PASS_SHA256:?set its raw SHA-256}"
: "${SFWD_CONV_POSTPREP_PASS:?set the fresh SFWD live PASS}"
: "${SFWD_CONV_POSTPREP_PASS_SHA256:?set its raw SHA-256}"
: "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST:?set the fresh SFWD source manifest}"
: "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256:?set its raw SHA-256}"
: "${TARGET_SFWD_COMBINED_SUMMARY:?set the fresh combined target/SFWD summary}"
: "${TARGET_SFWD_COMBINED_SUMMARY_SHA256:?set its raw SHA-256}"
: "${TAW_B1_CREDENTIAL:?set the source-bound Hydra27 B1 credential}"
: "${TAW_B1_CREDENTIAL_SHA256:?set its raw SHA-256}"
: "${TAW_B1_LIVE_BUNDLE:?set the credentialed Hydra27 B1 replay}"
: "${TAW_B1_LIVE_BUNDLE_SHA256:?set its raw SHA-256}"
: "${TAW_REVIEWED_B4_PASS:?set the reviewed Hydra27 exact4 B4 bundle}"
: "${TAW_REVIEWED_B4_PASS_SHA256:?set its raw SHA-256}"
: "${TAW_REVIEWED_B4_VERDICT:?set the reviewed Hydra27 exact4 verdict}"
: "${TAW_REVIEWED_B4_VERDICT_SHA256:?set its raw SHA-256}"
: "${TAW_MERGE_BINDING:?set the Hydra27 B1/B4 merge binding}"
: "${TAW_MERGE_BINDING_SHA256:?set its raw SHA-256}"
: "${TAW_PASS_JSON:?set the merged Hydra27 production bundle}"
: "${TAW_PASS_SHA256:?set its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TASK_SET=${TASK_SET:-exact4}
case "$TASK_SET" in
  exact4)
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
    EXPECTED_TASKS=4
    ;;
  exact16)
    SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
    SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
    EXPECTED_TASKS=16
    ;;
  *)
    echo "TASK_SET must be exactly exact4 or exact16" >&2
    exit 2
    ;;
esac

QUALIFICATION_SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
QUALIFICATION_SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
QUALIFICATION_RUNNER=scripts/fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh
U8_CREDENTIAL=scripts/fr13_dfwd_k64_m1_r64_u8_production_credential.py
U8_SOURCE=csrc/fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu
U8_BUILD=results/fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805/build_attestation.json
PATCH_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
CFWD_SOURCE=scripts/fr13_cfwd_logit_direct_decision_kernel.py
CFWD_RUNTIME_SOURCE=scripts/fr13_device_multidraft_cfwd_packed_v3.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
U8_SO_SHA256=8b27df4f3c6a5a0574261ee984159582a87615c3e6d83f2a267f4fa46a3e421e
U8_SO_BYTES=117904
U8_SOURCE_SHA256=af0044edd84ff58d353a816f6887894d05a62b221e0efa5af933c2c59676b01b
U8_BUILD_SHA256=e7ec95d1fff3b665373ad7b3a14f7e3fad346cf77a5f2f992a90a689e5672c8f
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TARGET_SELECTOR=identity_wide256_fullgrid_b1
TARGET_SHA256=7d762dfa793671d75d1e353bd37d76fc07370cbe387ad1e315e32584d27927d4
TARGET_BYTES=119781296
CFWD_SOURCE_SHA256=a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0
WEIGHT_FLOOR_MS=92.345089436
ONE_SIDED_U95_CAP_MS=106.1968528514
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
QUALIFICATION_RUNNER_SHA256=$(sha256sum "$QUALIFICATION_RUNNER" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
STOCK_ARM="hydra27_fixed32_fullstack_stock_${TASK_SET}_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_fullstack_u8_cfwd_sfwd_${TASK_SET}_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* && ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }

INPUT_BINDINGS=(
  "$U8_GATE_JSON:$U8_GATE_SHA256"
  "$U8_LIVE_RESULT_JSON:$U8_LIVE_RESULT_SHA256"
  "$SHARED_FINAL_FLUSH_JSON:$SHARED_FINAL_FLUSH_SHA256"
  "$SHARED_BOUNDARY_SNAPSHOT_JSON:$SHARED_BOUNDARY_SNAPSHOT_SHA256"
  "$SHARED_CHAT_TRAFFIC_AUDIT_JSON:$SHARED_CHAT_TRAFFIC_AUDIT_SHA256"
  "$CFWD_LIVE_RESULT_JSON:$CFWD_LIVE_RESULT_SHA256"
  "$CFWD_PASS_JSON:$CFWD_PASS_SHA256"
  "$CFWD_U8_COMPOSED_GATE_JSON:$CFWD_U8_COMPOSED_GATE_SHA256"
  "$CUTLASS_TARGET_PASS:$CUTLASS_TARGET_PASS_SHA256"
  "$SFWD_CONV_POSTPREP_PASS:$SFWD_CONV_POSTPREP_PASS_SHA256"
  "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST:$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256"
  "$TARGET_SFWD_COMBINED_SUMMARY:$TARGET_SFWD_COMBINED_SUMMARY_SHA256"
  "$TAW_B1_CREDENTIAL:$TAW_B1_CREDENTIAL_SHA256"
  "$TAW_B1_LIVE_BUNDLE:$TAW_B1_LIVE_BUNDLE_SHA256"
  "$TAW_REVIEWED_B4_PASS:$TAW_REVIEWED_B4_PASS_SHA256"
  "$TAW_REVIEWED_B4_VERDICT:$TAW_REVIEWED_B4_VERDICT_SHA256"
  "$TAW_MERGE_BINDING:$TAW_MERGE_BINDING_SHA256"
  "$TAW_PASS_JSON:$TAW_PASS_SHA256"
)
for binding in "${INPUT_BINDINGS[@]}"; do
  path=${binding%:*}
  digest=${binding##*:}
  [[ "$path" == /* && -f "$path" && ! -L "$path" \
     && "$(stat -c '%h' "$path")" == "1" \
     && "$digest" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] \
    || { echo "credential or evidence identity drifted: $path" >&2; exit 2; }
done
unset binding path digest
for binary in "$STOCK_FA2_SO" "$DFWD_U8_SO" "$CUTLASS_TARGET_SO"; do
  [[ "$binary" == /* && -f "$binary" && ! -L "$binary" \
     && "$(stat -c '%h' "$binary")" == "1" ]] \
    || { echo "candidate binary must be an absolute regular file: $binary" >&2; exit 2; }
done
unset binary
[[ "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" \
   && "$(stat -c '%s' "$DFWD_U8_SO")" == "$U8_SO_BYTES" \
   && "$(sha256sum "$DFWD_U8_SO" | awk '{print $1}')" == "$U8_SO_SHA256" \
   && "$(stat -c '%s' "$CUTLASS_TARGET_SO")" == "$TARGET_BYTES" \
   && "$(sha256sum "$CUTLASS_TARGET_SO" | awk '{print $1}')" == "$TARGET_SHA256" \
   && "$(sha256sum "$U8_SOURCE" | awk '{print $1}')" == "$U8_SOURCE_SHA256" \
   && "$(sha256sum "$U8_BUILD" | awk '{print $1}')" == "$U8_BUILD_SHA256" \
   && "$(sha256sum "$CFWD_SOURCE" | awk '{print $1}')" == "$CFWD_SOURCE_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" \
   && "$(sha256sum "$QUALIFICATION_SUBSET" | awk '{print $1}')" == "$QUALIFICATION_SUBSET_SHA256" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "full-stack binary or committed source identity drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "full-stack timing requires a clean source commit pushed to upstream" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/runtime_inputs" "$RUNROOT_ABS/sidecars"
PREFLIGHT_COMPOSED="$RUNROOT_ABS/cfwd_u8_composed_gate.preflight.json"
"$PYTHON_BIN" scripts/fr13_cfwd_dfwd_u8_composed_gate.py \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --cfwd-credential "$CFWD_PASS_JSON" \
  --cfwd-live-result "$CFWD_LIVE_RESULT_JSON" \
  --dfwd-gate "$U8_GATE_JSON" \
  --dfwd-live-result "$U8_LIVE_RESULT_JSON" \
  --candidate-so "$DFWD_U8_SO" \
  --fa2-so "$STOCK_FA2_SO" \
  --final-flush "$SHARED_FINAL_FLUSH_JSON" \
  --boundary-snapshot "$SHARED_BOUNDARY_SNAPSHOT_JSON" \
  --traffic-audit "$SHARED_CHAT_TRAFFIC_AUDIT_JSON" \
  --out "$PREFLIGHT_COMPOSED" >/dev/null
cmp -s "$PREFLIGHT_COMPOSED" "$CFWD_U8_COMPOSED_GATE_JSON" \
  || { echo "recorded CFWD/U8 composed gate differs from fresh validation" >&2; exit 2; }

PREFLIGHT_U8_CREDENTIAL="$RUNROOT_ABS/u8_production_credential.preflight.json"
"$PYTHON_BIN" "$U8_CREDENTIAL" issue \
  --live-result "$U8_LIVE_RESULT_JSON" \
  --final-flush "$SHARED_FINAL_FLUSH_JSON" \
  --boundary-snapshot "$SHARED_BOUNDARY_SNAPSHOT_JSON" \
  --chat-traffic-audit "$SHARED_CHAT_TRAFFIC_AUDIT_JSON" \
  --repo "$REPO" \
  --candidate-so "$DFWD_U8_SO" \
  --candidate-source "$U8_SOURCE" \
  --build-attestation "$U8_BUILD" \
  --patch-source "$PATCH_SOURCE" \
  --qualification-runner "$QUALIFICATION_RUNNER" \
  --subset "$QUALIFICATION_SUBSET" \
  --vocab-blocks "$BLOCK_MAP" \
  --fa2-so "$STOCK_FA2_SO" \
  --taw-source "$TAW_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --out "$PREFLIGHT_U8_CREDENTIAL" >/dev/null
PREFLIGHT_U8_CREDENTIAL_SHA256=$(sha256sum "$PREFLIGHT_U8_CREDENTIAL" | awk '{print $1}')

"$PYTHON_BIN" scripts/fr13_b1_composed_stack_gate.py validate-eager-credentials \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --combined-summary "$TARGET_SFWD_COMBINED_SUMMARY" \
  --combined-summary-sha256 "$TARGET_SFWD_COMBINED_SUMMARY_SHA256" \
  --target-live "$CUTLASS_TARGET_PASS" \
  --target-live-sha256 "$CUTLASS_TARGET_PASS_SHA256" \
  --sfwd-pass "$SFWD_CONV_POSTPREP_PASS" \
  --sfwd-pass-sha256 "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  --source-manifest "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" >/dev/null
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py validate \
  --live-result "$CUTLASS_TARGET_PASS" \
  --expected-live-sha256 "$CUTLASS_TARGET_PASS_SHA256" \
  --candidate-so "$CUTLASS_TARGET_SO" \
  --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py \
  --expected-source-commit "$SOURCE_COMMIT" \
  --candidate-selector "$TARGET_SELECTOR" \
  --qualification-profile k64_root \
  --diagnostic-task-profile astropy12907 \
  --draft-vocab-blocks "$BLOCK_MAP" >/dev/null
"$PYTHON_BIN" scripts/fr13_sfwd_conv_postprep_gate.py validate-pass \
  --repo "$REPO" \
  --live-pass "$SFWD_CONV_POSTPREP_PASS" \
  --expected-live-pass-sha256 "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  --source-manifest "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST" \
  --expected-source-manifest-sha256 "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT" >/dev/null
"$PYTHON_BIN" scripts/fr13_cfwd_logit_direct_gate.py validate \
  --credential "$CFWD_PASS_JSON" \
  --expected-sha256 "$CFWD_PASS_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --timing-subset "$SUBSET" >/dev/null
"$PYTHON_BIN" scripts/fr13_taw_b1_credential.py validate-production \
  --mode hydra27_fixed32 \
  --source "$TAW_SOURCE" \
  --credential "$TAW_B1_CREDENTIAL" \
  --b1-live-bundle "$TAW_B1_LIVE_BUNDLE" \
  --b4-production-pass "$TAW_REVIEWED_B4_PASS" \
  --b4-gate-verdict "$TAW_REVIEWED_B4_VERDICT" \
  --merge-binding "$TAW_MERGE_BINDING" \
  --production-pass "$TAW_PASS_JSON" >/dev/null

"$PYTHON_BIN" - scripts/fr13_cfwd_logit_direct_gate.py "$CFWD_RUNTIME_SOURCE" <<'PY'
import importlib.util
import sys
from pathlib import Path


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise SystemExit("CFWD integration source contract module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load(sys.argv[1], "fr13_fullstack_cfwd_gate")
device = load(sys.argv[2], "fr13_fullstack_cfwd_device")
contract = device._fr13_cfwd_logit_direct_integration_source_contract()
if (
    contract.get("integration_source_schema") != gate.INTEGRATION_SOURCE_SCHEMA
    or contract.get("integration_source_sha256") != gate.INTEGRATION_SOURCE_SHA256
):
    raise SystemExit("CFWD integration source contract mismatch")
PY

cp -- "$SFWD_CONV_POSTPREP_PASS" "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json"
cp -- "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST" "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json"
chmod 0400 "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json" \
  "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json"
SFWD_PASS_CONTAINER="/workspace/$RUNROOT_REL/runtime_inputs/sfwd_pass.json"
SFWD_MANIFEST_CONTAINER="/workspace/$RUNROOT_REL/runtime_inputs/sfwd_source_manifest.json"
[[ "$(sha256sum "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_PASS_SHA256" \
   && "$(sha256sum "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" ]] \
  || { echo "runtime SFWD credential copy drifted" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1 CONC=1 WALL=0
export FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "25210209416" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "fixed K64/root1 B1 floor contract drifted" >&2; exit 2; }

printf 'classification=real_swe_verified_b1_u8_cfwd_target_sfwd_timing_pair\ntask_set=%s\ntask_count=%s\nbatch_size=1\nconcurrency=1\ntiming_eligible=1\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\nmode=hydra27_fixed32\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\nruntime=FULL_graph_exact_geometry\nsource_commit=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nu8_so_sha256=%s\nu8_gate_sha256=%s\nu8_live_result_sha256=%s\ncfwd_u8_composed_gate_sha256=%s\npreflight_u8_credential_sha256=%s\ncfwd_pass_sha256=%s\ntarget_selector=%s\ntarget_so_sha256=%s\ntarget_pass_sha256=%s\nsfwd_pass_sha256=%s\nsfwd_source_manifest_sha256=%s\ntarget_sfwd_summary_sha256=%s\ntaw_pass_sha256=%s\nstock_arm=%s\ncandidate_arm=%s\nstarted=%s\n' \
  "$TASK_SET" "$EXPECTED_TASKS" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$STOCK_FA2_SHA256" "$U8_SO_SHA256" \
  "$U8_GATE_SHA256" "$U8_LIVE_RESULT_SHA256" \
  "$CFWD_U8_COMPOSED_GATE_SHA256" \
  "$PREFLIGHT_U8_CREDENTIAL_SHA256" "$CFWD_PASS_SHA256" \
  "$TARGET_SELECTOR" "$TARGET_SHA256" "$CUTLASS_TARGET_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
  "$TARGET_SFWD_COMBINED_SUMMARY_SHA256" "$TAW_PASS_SHA256" \
  "$STOCK_ARM" "$CANDIDATE_ARM" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$REPO" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$REPO" --profile fixed32 --sequence "$SEQUENCE" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" \
     && "$(stat -c '%h' "$STOCK_FA2_SO")" == "1" \
     && "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
     && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" \
     && -f "$DFWD_U8_SO" && ! -L "$DFWD_U8_SO" \
     && "$(stat -c '%h' "$DFWD_U8_SO")" == "1" \
     && "$(stat -c '%s' "$DFWD_U8_SO")" == "$U8_SO_BYTES" \
     && "$(sha256sum "$DFWD_U8_SO" | awk '{print $1}')" == "$U8_SO_SHA256" \
     && -f "$CUTLASS_TARGET_SO" && ! -L "$CUTLASS_TARGET_SO" \
     && "$(stat -c '%h' "$CUTLASS_TARGET_SO")" == "1" \
     && "$(stat -c '%s' "$CUTLASS_TARGET_SO")" == "$TARGET_BYTES" \
     && "$(sha256sum "$CUTLASS_TARGET_SO" | awk '{print $1}')" == "$TARGET_SHA256" ]] \
    || { echo "full-stack runner or binary changed during timing" >&2; return 14; }
  for binding in "${INPUT_BINDINGS[@]}"; do
    local path=${binding%:*}
    local digest=${binding##*:}
    [[ -f "$path" && ! -L "$path" && "$(stat -c '%h' "$path")" == "1" \
       && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] \
      || { echo "credential changed during timing: $path" >&2; return 14; }
  done
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifests || { local manifest_rc=$?; (( rc == 0 )) && rc=$manifest_rc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local production=$2
  local device_kernel=/workspace/scripts/fr13_device_multidraft_kernel.py
  local u8_so="" u8_so_sha="" u8_source_sha="" u8_build_sha=""
  local u8_patch_sha="" u8_runner_sha="" u8_subset_sha="" u8_vocab_sha=""
  local u8_fa2_sha="" u8_commit="" u8_instance="" u8_pass="" u8_pass_sha=""
  local u8_flush="" u8_boundary="" u8_traffic=""
  local taw_pass="" cfwd_pass="" cfwd_pass_sha=""
  local target_selector=stock target_so="" target_pass="" target_pass_sha=""
  local sfwd_fusion=0 sfwd_pass="" sfwd_pass_sha="" sfwd_manifest=""
  local sfwd_manifest_sha="" sfwd_commit="" conv_wb_batched=0
  if [[ "$production" == "1" ]]; then
    device_kernel=/workspace/scripts/fr13_device_multidraft_cfwd_packed_v3.py
    u8_so=$DFWD_U8_SO
    u8_so_sha=$U8_SO_SHA256
    u8_source_sha=$U8_SOURCE_SHA256
    u8_build_sha=$U8_BUILD_SHA256
    u8_patch_sha=$PATCH_SOURCE_SHA256
    u8_runner_sha=$QUALIFICATION_RUNNER_SHA256
    u8_subset_sha=$QUALIFICATION_SUBSET_SHA256
    u8_vocab_sha=$BLOCK_MAP_SHA256
    u8_fa2_sha=$STOCK_FA2_SHA256
    u8_commit=$SOURCE_COMMIT
    u8_instance=astropy__astropy-12907
    u8_pass=$U8_LIVE_RESULT_JSON
    u8_pass_sha=$U8_LIVE_RESULT_SHA256
    u8_flush=$SHARED_FINAL_FLUSH_JSON
    u8_boundary=$SHARED_BOUNDARY_SNAPSHOT_JSON
    u8_traffic=$SHARED_CHAT_TRAFFIC_AUDIT_JSON
    taw_pass=$TAW_PASS_JSON
    cfwd_pass=$CFWD_PASS_JSON
    cfwd_pass_sha=$CFWD_PASS_SHA256
    target_selector=$TARGET_SELECTOR
    target_so=$CUTLASS_TARGET_SO
    target_pass=$CUTLASS_TARGET_PASS
    target_pass_sha=$CUTLASS_TARGET_PASS_SHA256
    sfwd_fusion=1
    sfwd_pass=$SFWD_PASS_CONTAINER
    sfwd_pass_sha=$SFWD_CONV_POSTPREP_PASS_SHA256
    sfwd_manifest=$SFWD_MANIFEST_CONTAINER
    sfwd_manifest_sha=$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256
    sfwd_commit=$SOURCE_COMMIT
    conv_wb_batched=1
  fi

  echo "===== $arm: B1 task_set=$TASK_SET full_stack_production=$production ====="
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      LUMO_SWE_AUTOCOMMIT=0 \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
      FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
      FR13_DEVICE_MULTIDRAFT=1 FR13_DEVICE_MULTIDRAFT_KERNEL="$device_kernel" \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_B1_U8_CFWD_SFWD_STACK_TIMING="$production" \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0 \
      FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION="$production" \
      FR13_DRAFT_HEAD_M1_R64_U8_SO="$u8_so" \
      FR13_DRAFT_HEAD_M1_R64_U8_SO_SHA256="$u8_so_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_SHA256="$u8_source_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_BUILD_ATTESTATION_SHA256="$u8_build_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_PATCH_SOURCE_SHA256="$u8_patch_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_RUNNER_SHA256="$u8_runner_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_SUBSET_SHA256="$u8_subset_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_VOCAB_BLOCKS_SHA256="$u8_vocab_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_FA2_SHA256="$u8_fa2_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT="$u8_commit" \
      FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID="$u8_instance" \
      FR13_DRAFT_HEAD_M1_R64_U8_LIVE_PASS_JSON="$u8_pass" \
      FR13_DRAFT_HEAD_M1_R64_U8_LIVE_PASS_SHA256="$u8_pass_sha" \
      FR13_DRAFT_HEAD_M1_R64_U8_LIVE_FINAL_FLUSH_JSON="$u8_flush" \
      FR13_DRAFT_HEAD_M1_R64_U8_LIVE_BOUNDARY_SNAPSHOT_JSON="$u8_boundary" \
      FR13_DRAFT_HEAD_M1_R64_U8_LIVE_CHAT_TRAFFIC_AUDIT_JSON="$u8_traffic" \
      FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_ENGAGEMENT_JSON=/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json \
      FR13_DRAFT_HEAD_FP8=0 FR13_DFWD_K64_TOP3=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$production" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$taw_pass" \
      FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION="$production" \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON="$cfwd_pass" \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256="$cfwd_pass_sha" \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
      FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0 \
      FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE="$target_selector" \
      FR13_FIXED32_CUTLASS_WAVE_SO="$target_so" \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production" \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=k64_root \
      FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE=astropy12907 \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$target_pass" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$target_pass_sha" \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0 \
      FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION="$sfwd_fusion" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0 \
      FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON="$sfwd_pass" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256="$sfwd_pass_sha" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH="$sfwd_manifest" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256="$sfwd_manifest_sha" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT="$sfwd_commit" \
      FR13_CONV_WB_BATCHED="$conv_wb_batched" \
      FR13_TREE_CONV_FUSED=1 FR13_FIXED32_CONV_SOURCE_BATCH=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" hydra27_fixed32 "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi

  local env_path="$RUNROOT_ABS/$arm/container_env.txt"
  [[ -f "$env_path" && ! -L "$env_path" ]] \
    || { echo "$arm lacks container_env.txt" >&2; return 4; }
  for expected in \
      'FR13_FIXED32_MODE=hydra27_fixed32' \
      'FR13_FIXED32_B1_DIAGNOSTIC=0' \
      'FR13_DRAFT_VOCAB_ROOT=1' \
      'FR13_DRAFT_VOCAB_K=65536' \
      'MAX_NUM_SEQS=1' \
      'SWE_CONCURRENCY=1' \
      'ENFORCE_EAGER=0' \
      'CUDAGRAPH_MODE=FULL_AND_PIECEWISE' \
      "FR13_DEVICE_MULTIDRAFT_KERNEL=$device_kernel" \
      'FR13_FA2_QROW16_PRODUCTION=0' \
      'FR13_FA2_QROW32_B1_PRODUCTION_ARM=' \
      'FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0' \
      'FR13_DFWD_K64_TOP3=0' \
      "FR13_B1_U8_CFWD_SFWD_STACK_TIMING=$production" \
      "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=$production" \
      "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=$production" \
      "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=$production" \
      "FR13_FIXED32_CUTLASS_WAVE=$target_selector" \
      "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=$production" \
      "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=$sfwd_fusion"; do
    [[ "$(grep -Fxc "$expected" "$env_path")" -eq 1 ]] \
      || { echo "$arm lacks exact environment pin: $expected" >&2; return 4; }
  done
  unset expected
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" \
    --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$env_path" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
for absent in \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_cfwd_logit_direct.production_engagement.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json"; do
  [[ ! -e "$absent" && ! -L "$absent" ]] \
    || { echo "stock arm emitted production evidence: $absent" >&2; exit 4; }
done
unset absent
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after candidate arm" >&2; exit 2; }

CANDIDATE_DIR="$RUNROOT_ABS/$CANDIDATE_ARM"
U8_PRODUCTION_CREDENTIAL="$CANDIDATE_DIR/logs/fr13_dfwd_k64_m1_r64_u8.production_credential.json"
U8_ENGAGEMENT="$CANDIDATE_DIR/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json"
CFWD_PRODUCTION_PASS="$CANDIDATE_DIR/logs/fr13_cfwd_logit_direct.production_pass.json"
CFWD_ENGAGEMENT="$CANDIDATE_DIR/logs/fr13_cfwd_logit_direct.production_engagement.json"
TAW_PRODUCTION_PASS="$CANDIDATE_DIR/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
TAW_PRODUCTION_ARM="$CANDIDATE_DIR/logs/fr13_fixed32_taw_native_precompute_production.arm"
TARGET_PRODUCTION_SIDECAR="$CANDIDATE_DIR/logs/fr13_fixed32_cutlass_streamk.production_pass.json"
TARGET_BINARY_RECORD="$CANDIDATE_DIR/logs/fr13_fixed32_cutlass_streamk_binary.json"
TARGET_SELECTOR_RECORD="$CANDIDATE_DIR/logs/fr13_fixed32_cutlass_wave.selector"
SFWD_PRODUCTION_PASS="$CANDIDATE_DIR/logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json"
SFWD_PRODUCTION_MANIFEST="$CANDIDATE_DIR/logs/fr13_fixed32_sfwd_conv_postprep.source_manifest.json"
DOCKER_LOG="$CANDIDATE_DIR/docker_after_tasks.log"
for artifact in \
  "$U8_PRODUCTION_CREDENTIAL" "$U8_ENGAGEMENT" \
  "$CFWD_PRODUCTION_PASS" "$CFWD_ENGAGEMENT" \
  "$TAW_PRODUCTION_PASS" "$TAW_PRODUCTION_ARM" \
  "$TARGET_PRODUCTION_SIDECAR" "$TARGET_BINARY_RECORD" \
  "$TARGET_SELECTOR_RECORD" "$SFWD_PRODUCTION_PASS" \
  "$SFWD_PRODUCTION_MANIFEST" "$DOCKER_LOG"; do
  [[ -f "$artifact" && ! -L "$artifact" && "$(stat -c '%h' "$artifact")" == "1" ]] \
    || { echo "candidate production evidence is missing or unsafe: $artifact" >&2; exit 4; }
done
unset artifact
[[ "$(sha256sum "$U8_PRODUCTION_CREDENTIAL" | awk '{print $1}')" == "$PREFLIGHT_U8_CREDENTIAL_SHA256" \
   && "$(sha256sum "$CFWD_PRODUCTION_PASS" | awk '{print $1}')" == "$CFWD_PASS_SHA256" \
   && "$(sha256sum "$TAW_PRODUCTION_PASS" | awk '{print $1}')" == "$TAW_PASS_SHA256" \
   && "$(cat "$TAW_PRODUCTION_ARM")" == "1" \
   && "$(sha256sum "$SFWD_PRODUCTION_PASS" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_PASS_SHA256" \
   && "$(sha256sum "$SFWD_PRODUCTION_MANIFEST" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
   && "$(cat "$TARGET_SELECTOR_RECORD")" == "$TARGET_SELECTOR" ]] \
  || { echo "candidate production credential or selector drifted" >&2; exit 4; }

"$PYTHON_BIN" "$U8_CREDENTIAL" engagement \
  --engagement "$U8_ENGAGEMENT" \
  --expected-credential-sha256 "$PREFLIGHT_U8_CREDENTIAL_SHA256" \
  --expected-source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/u8_engagement_validation.json"
TARGET_PRODUCTION_SIDECAR_SHA256=$(sha256sum "$TARGET_PRODUCTION_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py verify \
  --sidecar "$TARGET_PRODUCTION_SIDECAR" \
  --expected-sidecar-sha256 "$TARGET_PRODUCTION_SIDECAR_SHA256" \
  --candidate-so "$CUTLASS_TARGET_SO" \
  --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py \
  --candidate-selector "$TARGET_SELECTOR" \
  --qualification-profile k64_root \
  --diagnostic-task-profile astropy12907 \
  --draft-vocab-blocks "$BLOCK_MAP" >/dev/null

ENGAGEMENT_VALIDATION="$RUNROOT_ABS/fullstack_engagement_validation.json"
"$PYTHON_BIN" - \
  "$CFWD_ENGAGEMENT" "$TARGET_BINARY_RECORD" "$DOCKER_LOG" \
  "$ENGAGEMENT_VALIDATION" "$SOURCE_COMMIT" "$CFWD_PASS_SHA256" \
  "$TARGET_SELECTOR" "$TARGET_SHA256" <<'PY'
import hashlib
import json
import re
import stat
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
import fr13_cfwd_logit_direct_gate as cfwd


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"runtime artifact is not one regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise SystemExit(f"runtime artifact is not one object: {path}")
    return payload, raw


engagement_path, binary_path, docker_path, out_path = map(Path, sys.argv[1:5])
source_commit, credential_sha, target_selector, target_sha = sys.argv[5:9]
engagement, engagement_raw = load(engagement_path)
binary, binary_raw = load(binary_path)
docker_info = docker_path.lstat()
if docker_path.is_symlink() or not stat.S_ISREG(docker_info.st_mode):
    raise SystemExit("Docker log is not a regular file")
docker_raw = docker_path.read_bytes()
docker_text = docker_raw.decode("utf-8", errors="replace")
expected_engagement_keys = {
    "schema",
    "status",
    "candidate",
    "mode",
    "batch_size",
    "source_commit",
    "candidate_source_sha256",
    "integration_source_schema",
    "integration_source_sha256",
    "production_pass_sha256",
    "served_return",
    "producer_pid",
}
if (
    set(engagement) != expected_engagement_keys
    or engagement.get("schema")
    != "fr13.fixed32.cfwd_logit_direct.production_engagement.v2"
    or engagement.get("status") != "engaged"
    or engagement.get("candidate") != cfwd.CANDIDATE
    or engagement.get("mode") != "hydra27_fixed32"
    or engagement.get("batch_size") != 1
    or engagement.get("source_commit") != source_commit
    or engagement.get("candidate_source_sha256") != cfwd.CANDIDATE_SOURCE_SHA256
    or engagement.get("integration_source_schema") != cfwd.INTEGRATION_SOURCE_SCHEMA
    or engagement.get("integration_source_sha256") != cfwd.INTEGRATION_SOURCE_SHA256
    or engagement.get("production_pass_sha256") != credential_sha
    or engagement.get("served_return") != "logit-direct candidate products"
    or type(engagement.get("producer_pid")) is not int
    or engagement["producer_pid"] < 1
):
    raise SystemExit("CFWD candidate served-return engagement drifted")
if (
    binary.get("schema") != "fr13.fixed32.cutlass_streamk_binary.v2"
    or binary.get("selector") != target_selector
    or binary.get("production_enabled") is not True
    or binary.get("qualification_profile") != "k64_root"
    or not isinstance(binary.get("source"), dict)
    or binary["source"].get("sha256") != target_sha
    or not isinstance(binary.get("destination"), dict)
    or binary["destination"].get("sha256") != target_sha
    or binary.get("installed_mode") != "0555"
):
    raise SystemExit("current target installation evidence drifted")
layers = set(
    re.findall(
        r"\[FR13_SFWD_CONV_POSTPREP\] production engaged "
        r"layer=([^ ]+) B=1 rows=32",
        docker_text,
    )
)
marker = "[FR13_SFWD_CONV_POSTPREP] production engaged layer="
if len(layers) != 48 or docker_text.count(marker) != 48:
    raise SystemExit("SFWD conv/post-prep did not engage exactly 48 layers")
payload = {
    "schema": "fr13.fixed32.b1_u8_cfwd_sfwd.engagement_validation.v1",
    "status": "PASS",
    "source_commit": source_commit,
    "cfwd_engagement_sha256": hashlib.sha256(engagement_raw).hexdigest(),
    "target_binary_record_sha256": hashlib.sha256(binary_raw).hexdigest(),
    "docker_log_sha256": hashlib.sha256(docker_raw).hexdigest(),
    "sfwd_engaged_layers": 48,
    "u8_engagement_validated_separately": True,
}
out_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
PY

finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/u8_engagement_validation.json" \
  "$ENGAGEMENT_VALIDATION" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$TASK_SET" "$EXPECTED_TASKS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$U8_SO_SHA256" "$PREFLIGHT_U8_CREDENTIAL_SHA256" \
  "$CFWD_PASS_SHA256" "$CFWD_U8_COMPOSED_GATE_SHA256" \
  "$TARGET_SELECTOR" "$TARGET_SHA256" "$CUTLASS_TARGET_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" "$TAW_PASS_SHA256" \
  "$WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" <<'PY'
import hashlib
import json
import math
import stat
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
from fr13_b4_timing_math import phase_breakdown


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"timing artifact is not one regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise SystemExit(f"timing artifact is not one object: {path}")
    return payload, raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def positive(payload, key, label):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} lacks numeric {key}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{label} lacks positive finite {key}")
    return value


paths = list(map(Path, sys.argv[1:7]))
subset_path, stock_path, candidate_path, u8_path, stack_path, out_path = paths
task_set, expected_tasks, stock_arm, candidate_arm = sys.argv[7:11]
source_commit, runner_sha, subset_sha = sys.argv[11:14]
stock_fa2_sha, u8_so_sha, u8_credential_sha = sys.argv[14:17]
cfwd_sha, composed_sha, target_selector, target_sha, target_pass_sha = sys.argv[17:22]
sfwd_pass_sha, sfwd_manifest_sha, taw_sha = sys.argv[22:25]
floor_ms, cap_ms = map(float, sys.argv[25:27])
subset, subset_raw = load(subset_path)
stock, stock_raw = load(stock_path)
candidate, candidate_raw = load(candidate_path)
u8_validation, u8_raw = load(u8_path)
stack_validation, stack_raw = load(stack_path)
task_ids = subset.get("instance_ids")
if (
    digest(subset_raw) != subset_sha
    or not isinstance(task_ids, list)
    or len(task_ids) != int(expected_tasks)
    or u8_validation.get("status") != "PASS"
    or u8_validation.get("candidate_served") is not True
    or stack_validation.get("status") != "PASS"
    or stack_validation.get("source_commit") != source_commit
    or stack_validation.get("sfwd_engaged_layers") != 48
):
    raise SystemExit("full-stack timing provenance drifted")


def validate_measure(payload, raw, arm):
    if (
        payload.get("schema") != "fr13.measure.deploy_speed.v1"
        or payload.get("kind") != "speed"
        or payload.get("instrument") != "OFF"
        or payload.get("regime") != "deployment"
        or payload.get("arm") != arm
        or payload.get("batch_size") != 1
        or payload.get("n_tasks") != int(expected_tasks)
        or payload.get("task_instance_ids") != task_ids
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_vocab_k") != 65536
        or payload.get("mandatory_weight_bytes") != 25210209416
        or payload.get("weight_floor_ms") != floor_ms
        or payload.get("floor_ms") != floor_ms
        or payload.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise SystemExit(f"{arm} is not canonical {task_set} K64/root1 B1 timing")
    per_task = payload.get("per_task")
    if (
        not isinstance(per_task, list)
        or len(per_task) != int(expected_tasks)
        or [row.get("instance_id") for row in per_task] != task_ids
        or any(positive(row, "wall_steps", arm + ":task") < 1 for row in per_task)
    ):
        raise SystemExit(f"{arm} lacks complete per-task timing windows")
    phases = phase_breakdown(payload, arm)
    rows_per_step = positive(payload, "rows_per_step", arm)
    if not math.isclose(rows_per_step, 32.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{arm} physical row count drifted")
    return {
        "deploy_speed_sha256": digest(raw),
        "step_wall_ms": positive(payload, "step_wall_ms", arm),
        "measured_tps_fullstep_wall": positive(
            payload, "measured_tps_fullstep_wall", arm
        ),
        "accepted_drafts_per_event": positive(payload, "accept_per_event", arm),
        "committed_tokens_per_event": positive(payload, "committed_per_event", arm),
        "events_per_step": positive(payload, "events_per_step", arm),
        "rows_per_step": rows_per_step,
        "sfwd_gpu_ms_per_step": phases["sfwd_gpu_ms_per_step"],
        "dfwd_gpu_ms_per_step": phases["dfwd_gpu_ms_per_step"],
        "cfwd_gpu_ms_per_step": phases["cfwd_gpu_ms_per_step"],
        "gpu_component_ms_per_step": phases["gpu_component_ms_per_step"],
        "other_wall_ms_per_step": phases["other_wall_ms_per_step"],
    }


s = validate_measure(stock, stock_raw, stock_arm)
c = validate_measure(candidate, candidate_raw, candidate_arm)
summary = {
    "schema": "fr13.fixed32.b1_u8_cfwd_target_sfwd_timing.v1",
    "status": "MEASURED",
    "classification": "real_swe_verified_b1_u8_cfwd_target_sfwd_timing_pair",
    "task_set": task_set,
    "task_count": int(expected_tasks),
    "task_ids": task_ids,
    "batch_size": 1,
    "concurrency": 1,
    "mode": "hydra27_fixed32",
    "physical_rows": 32,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "runtime": "FULL_AND_PIECEWISE",
    "timing_eligible": True,
    "floor_acceptance_eligible": False,
    "production_default_enabled": False,
    "performance_claim": False,
    "source_commit": source_commit,
    "runner_sha256": runner_sha,
    "subset_sha256": subset_sha,
    "component_credentials": {
        "stock_fa2_sha256": stock_fa2_sha,
        "u8_so_sha256": u8_so_sha,
        "u8_production_credential_sha256": u8_credential_sha,
        "u8_engagement_validation_sha256": digest(u8_raw),
        "cfwd_production_credential_sha256": cfwd_sha,
        "cfwd_u8_composed_gate_sha256": composed_sha,
        "fullstack_engagement_validation_sha256": digest(stack_raw),
        "target_selector": target_selector,
        "target_so_sha256": target_sha,
        "target_live_pass_sha256": target_pass_sha,
        "sfwd_live_pass_sha256": sfwd_pass_sha,
        "sfwd_source_manifest_sha256": sfwd_manifest_sha,
        "taw_production_pass_sha256": taw_sha,
    },
    "stock": {"arm": stock_arm, **s},
    "candidate": {"arm": candidate_arm, **c},
    "delta": {
        "step_wall_ms": c["step_wall_ms"] - s["step_wall_ms"],
        "full_wall_tps": (
            c["measured_tps_fullstep_wall"] - s["measured_tps_fullstep_wall"]
        ),
        "accepted_drafts_per_event": (
            c["accepted_drafts_per_event"] - s["accepted_drafts_per_event"]
        ),
    },
    "ratios": {
        "candidate_to_stock_step_wall": c["step_wall_ms"] / s["step_wall_ms"],
        "candidate_to_stock_full_wall_tps": (
            c["measured_tps_fullstep_wall"]
            / s["measured_tps_fullstep_wall"]
        ),
        "candidate_to_weight_floor": c["step_wall_ms"] / floor_ms,
    },
    "one_sided_u95_cap_ms": cap_ms,
    "candidate_gap_to_cap_ms": c["step_wall_ms"] - cap_ms,
}
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

printf 'summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
trap - EXIT
echo "B1 U8/CFWD/target/SFWD timing completed: $RUNROOT_ABS/timing_summary.json"

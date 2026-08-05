#!/usr/bin/env bash
# Canonical one-task SWE-Verified byte gate for fixed32 CFWD logit-direct.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact-safe stock FA2 binary}"
: "${TAW_B1_CREDENTIAL:?set it to the source-bound Hydra27 B1 credential}"
: "${TAW_B1_CREDENTIAL_SHA256:?set its raw SHA-256}"
: "${TAW_B1_LIVE_BUNDLE:?set it to the credentialed Hydra27 B1 replay}"
: "${TAW_B1_LIVE_BUNDLE_SHA256:?set its raw SHA-256}"
: "${TAW_REVIEWED_B4_PASS:?set it to the reviewed Hydra27 exact4 B4 bundle}"
: "${TAW_REVIEWED_B4_PASS_SHA256:?set its raw SHA-256}"
: "${TAW_REVIEWED_B4_VERDICT:?set it to the reviewed Hydra27 exact4 verdict}"
: "${TAW_REVIEWED_B4_VERDICT_SHA256:?set its raw SHA-256}"
: "${TAW_MERGE_BINDING:?set it to the Hydra27 B1/B4 merge binding}"
: "${TAW_MERGE_BINDING_SHA256:?set its raw SHA-256}"
: "${TAW_PASS_JSON:?set it to the merged Hydra27 production bundle}"
: "${TAW_PASS_SHA256:?set TAW_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
TASK_ID=astropy__astropy-12907
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
CANDIDATE_SOURCE=scripts/fr13_cfwd_logit_direct_decision_kernel.py
CANDIDATE_SOURCE_SHA256=c3d5d0f1b210cd545c5ce2dcbc6e50eaa2c7fbb508097d4347db152c428a0192
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
GATE=scripts/fr13_cfwd_logit_direct_gate.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
GATE_SHA256=$(sha256sum "$GATE" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="hydra27_fixed32_cfwd_logit_direct_byte_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in \
  "$FORKED_FA2_SO" "$TAW_B1_CREDENTIAL" "$TAW_B1_LIVE_BUNDLE" \
  "$TAW_REVIEWED_B4_PASS" "$TAW_REVIEWED_B4_VERDICT" \
  "$TAW_MERGE_BINDING" "$TAW_PASS_JSON"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "input must be an absolute regular non-symlink file: $required" >&2; exit 2; }
done
unset required
[[ "$TAW_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_PASS_JSON" | awk '{print $1}')" == "$TAW_PASS_SHA256" \
   && "$TAW_B1_CREDENTIAL_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_B1_CREDENTIAL" | awk '{print $1}')" == "$TAW_B1_CREDENTIAL_SHA256" \
   && "$TAW_B1_LIVE_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_B1_LIVE_BUNDLE" | awk '{print $1}')" == "$TAW_B1_LIVE_BUNDLE_SHA256" \
   && "$TAW_REVIEWED_B4_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_REVIEWED_B4_PASS" | awk '{print $1}')" == "$TAW_REVIEWED_B4_PASS_SHA256" \
   && "$TAW_REVIEWED_B4_VERDICT_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_REVIEWED_B4_VERDICT" | awk '{print $1}')" == "$TAW_REVIEWED_B4_VERDICT_SHA256" \
   && "$TAW_MERGE_BINDING_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_MERGE_BINDING" | awk '{print $1}')" == "$TAW_MERGE_BINDING_SHA256" ]] \
  || { echo "TAW credential or production bundle identity mismatch" >&2; exit 2; }
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" \
   && "$(sha256sum "$CANDIDATE_SOURCE" | awk '{print $1}')" == "$CANDIDATE_SOURCE_SHA256" ]] \
  || { echo "canonical subset, K64 map, or CFWD source identity drifted" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean at a valid source commit" >&2; exit 2; }
"$PYTHON_BIN" - "$GATE" "$TAW_SOURCE" <<'PY'
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


gate = load(sys.argv[1], "fr13_cfwd_gate_contract_preflight")
device = load(sys.argv[2], "fr13_cfwd_device_contract_preflight")
contract = device._fr13_cfwd_logit_direct_integration_source_contract()
if (
    contract.get("integration_source_schema") != gate.INTEGRATION_SOURCE_SCHEMA
    or contract.get("integration_source_sha256") != gate.INTEGRATION_SOURCE_SHA256
):
    raise SystemExit("CFWD integration source contract mismatch")
PY
"$PYTHON_BIN" scripts/fr13_taw_b1_credential.py validate-production \
  --mode hydra27_fixed32 \
  --source "$TAW_SOURCE" \
  --credential "$TAW_B1_CREDENTIAL" \
  --b1-live-bundle "$TAW_B1_LIVE_BUNDLE" \
  --b4-production-pass "$TAW_REVIEWED_B4_PASS" \
  --b4-gate-verdict "$TAW_REVIEWED_B4_VERDICT" \
  --merge-binding "$TAW_MERGE_BINDING" \
  --production-pass "$TAW_PASS_JSON" \
  >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=1 CONC=1 WALL=0
export FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "32666638208" \
   && "$FR13_WEIGHT_FLOOR_MS" == "119.658015414" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "fixed K64/root1 B1 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
printf '%s\n' \
  'classification=real_swe_verified_one_task_cfwd_logit_direct_byte_gate' \
  'acceptance_valid=0' \
  'timing_eligible=0' \
  'floor_acceptance_eligible=0' \
  'reference_always_served=1' \
  'production_enabled=0' \
  'mode=hydra27_fixed32' \
  'batch_size=1' \
  'concurrency=1' \
  'physical_rows=32' \
  'draft_vocab_k=65536' \
  'draft_vocab_root=1' \
  "task_id=$TASK_ID" \
  "source_commit=$SOURCE_COMMIT" \
  "runner_sha256=$RUNNER_SHA256" \
  "gate_sha256=$GATE_SHA256" \
  "subset_sha256=$SUBSET_SHA256" \
  "block_map_sha256=$BLOCK_MAP_SHA256" \
  "stock_fa2_sha256=$STOCK_FA2_SHA256" \
  "taw_b1_credential_sha256=$TAW_B1_CREDENTIAL_SHA256" \
  "taw_b1_live_bundle_sha256=$TAW_B1_LIVE_BUNDLE_SHA256" \
  "taw_reviewed_b4_pass_sha256=$TAW_REVIEWED_B4_PASS_SHA256" \
  "taw_reviewed_b4_verdict_sha256=$TAW_REVIEWED_B4_VERDICT_SHA256" \
  "taw_merge_binding_sha256=$TAW_MERGE_BINDING_SHA256" \
  "taw_pass_sha256=$TAW_PASS_SHA256" \
  "candidate_source_sha256=$CANDIDATE_SOURCE_SHA256" \
  "started=$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during CFWD gate" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during CFWD gate" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$GATE" | awk '{print $1}')" == "$GATE_SHA256" \
     && "$(sha256sum "$CANDIDATE_SOURCE" | awk '{print $1}')" == "$CANDIDATE_SOURCE_SHA256" \
     && "$(sha256sum "$TAW_B1_CREDENTIAL" | awk '{print $1}')" == "$TAW_B1_CREDENTIAL_SHA256" \
     && "$(sha256sum "$TAW_B1_LIVE_BUNDLE" | awk '{print $1}')" == "$TAW_B1_LIVE_BUNDLE_SHA256" \
     && "$(sha256sum "$TAW_REVIEWED_B4_PASS" | awk '{print $1}')" == "$TAW_REVIEWED_B4_PASS_SHA256" \
     && "$(sha256sum "$TAW_REVIEWED_B4_VERDICT" | awk '{print $1}')" == "$TAW_REVIEWED_B4_VERDICT_SHA256" \
     && "$(sha256sum "$TAW_MERGE_BINDING" | awk '{print $1}')" == "$TAW_MERGE_BINDING_SHA256" \
     && "$(sha256sum "$TAW_PASS_JSON" | awk '{print $1}')" == "$TAW_PASS_SHA256" ]] \
    || { echo "CFWD gate source or credential changed during execution" >&2; return 14; }
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

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    LUMO_SWE_AUTOCOMMIT=0 \
    FR13_FIXED32_B1_DIAGNOSTIC=1 FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907 \
    FR10_METRICS=1 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1 \
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
    FR13_DEVICE_MULTIDRAFT=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$TAW_PASS_JSON" \
    FR13_CFWD_LOGIT_DIRECT_BYTE_AB=1 \
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON= \
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256= \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0 \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
(( serve_rc == 0 )) || exit "$serve_rc"

LIVE_RESULT="$ARMDIR/logs/fr13_cfwd_logit_direct.live.json"
FINAL_FLUSH="$ARMDIR/fixed32_final_flush.json"
TRAFFIC_AUDIT="$ARMDIR/fixed32_chat_traffic_audit.json"
FLUSH_GENERATION=$("$PYTHON_BIN" - "$FINAL_FLUSH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
generation = payload.get("ack", {}).get("generation")
if type(generation) is not int or generation < 1:
    raise SystemExit("final flush lacks a valid generation")
print(generation)
PY
)
BOUNDARY="$ARMDIR/logs/fr13_fixed32_boundary_snapshot.${FLUSH_GENERATION}.json"
CREDENTIAL="$ARMDIR/fr13_cfwd_logit_direct.production_credential.json"
for artifact in "$LIVE_RESULT" "$FINAL_FLUSH" "$BOUNDARY" "$TRAFFIC_AUDIT"; do
  [[ -f "$artifact" && ! -L "$artifact" ]] \
    || { echo "CFWD gate artifact is missing or unsafe: $artifact" >&2; exit 4; }
done
unset artifact
"$PYTHON_BIN" "$GATE" issue \
  --live-result "$LIVE_RESULT" \
  --subset "$SUBSET" \
  --final-flush "$FINAL_FLUSH" \
  --boundary-snapshot "$BOUNDARY" \
  --traffic-audit "$TRAFFIC_AUDIT" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --source-commit "$SOURCE_COMMIT" \
  --out "$CREDENTIAL" \
  > "$ARMDIR/cfwd_logit_direct_gate_reduction.json"
CREDENTIAL_SHA256=$(sha256sum "$CREDENTIAL" | awk '{print $1}')
printf 'credential=%s credential_sha256=%s completed=%s\n' \
  "$CREDENTIAL" "$CREDENTIAL_SHA256" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
finalize_manifests

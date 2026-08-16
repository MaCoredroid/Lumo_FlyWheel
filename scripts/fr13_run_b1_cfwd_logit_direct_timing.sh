#!/usr/bin/env bash
# Exact4 real SWE-Verified B1 full-wall timing: native CFWD then logit-direct.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"
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
: "${CFWD_PASS_JSON:?set CFWD_PASS_JSON to the one-task CFWD production credential}"
: "${CFWD_PASS_SHA256:?set CFWD_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
CANDIDATE_SOURCE=scripts/fr13_cfwd_logit_direct_decision_kernel.py
CANDIDATE_SOURCE_SHA256=a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
CFWD_RUNTIME_SOURCE=scripts/fr13_device_multidraft_cfwd_packed_v3.py
GATE=scripts/fr13_cfwd_logit_direct_gate.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
WEIGHT_FLOOR_MS=102.479937172
ONE_SIDED_U95_CAP_MS=117.8519277478
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
GATE_SHA256=$(sha256sum "$GATE" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
STOCK_ARM="hydra27_fixed32_cfwd_native_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_cfwd_logit_direct_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in \
  "$STOCK_FA2_SO" "$TAW_B1_CREDENTIAL" "$TAW_B1_LIVE_BUNDLE" \
  "$TAW_REVIEWED_B4_PASS" "$TAW_REVIEWED_B4_VERDICT" \
  "$TAW_MERGE_BINDING" "$TAW_PASS_JSON" "$CFWD_PASS_JSON"; do
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
[[ "$CFWD_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$CFWD_PASS_JSON" | awk '{print $1}')" == "$CFWD_PASS_SHA256" ]] \
  || { echo "CFWD production credential SHA-256 mismatch" >&2; exit 2; }
[[ "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" \
   && "$(sha256sum "$CANDIDATE_SOURCE" | awk '{print $1}')" == "$CANDIDATE_SOURCE_SHA256" ]] \
  || { echo "canonical exact4 subset, K64 map, or CFWD source identity drifted" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean at a valid source commit" >&2; exit 2; }
"$PYTHON_BIN" - "$GATE" "$CFWD_RUNTIME_SOURCE" <<'PY'
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

# Credential validation is the first candidate-qualifying action. It binds the
# one-task byte PASS to this source commit and separately pins the timing set.
"$PYTHON_BIN" "$GATE" validate \
  --credential "$CFWD_PASS_JSON" \
  --expected-sha256 "$CFWD_PASS_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --timing-subset "$SUBSET" \
  >/dev/null
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
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1 CONC=1 WALL=0
export FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_cfwd_packed_v3.py
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "27977022848" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "fixed K64/root1 B1 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
printf '%s\n' \
  'classification=real_swe_verified_exact4_b1_cfwd_logit_direct_timing_pair' \
  'timing_eligible=1' \
  'floor_acceptance_eligible=0' \
  'production_default_enabled=0' \
  'mode=hydra27_fixed32' \
  'batch_size=1' \
  'concurrency=1' \
  'task_count=4' \
  'physical_rows=32' \
  'draft_vocab_k=65536' \
  'draft_vocab_root=1' \
  'only_arm_delta=FR13_CFWD_LOGIT_DIRECT_PRODUCTION_0_to_1' \
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
  "cfwd_pass_sha256=$CFWD_PASS_SHA256" \
  "candidate_source_sha256=$CANDIDATE_SOURCE_SHA256" \
  "weight_floor_ms=$WEIGHT_FLOOR_MS" \
  "one_sided_u95_cap_ms=$ONE_SIDED_U95_CAP_MS" \
  "stock_arm=$STOCK_ARM" \
  "candidate_arm=$CANDIDATE_ARM" \
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
    || { echo "runtime/source manifest changed during CFWD timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during CFWD timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$GATE" | awk '{print $1}')" == "$GATE_SHA256" \
     && "$(sha256sum "$CANDIDATE_SOURCE" | awk '{print $1}')" == "$CANDIDATE_SOURCE_SHA256" \
     && "$(sha256sum "$TAW_B1_CREDENTIAL" | awk '{print $1}')" == "$TAW_B1_CREDENTIAL_SHA256" \
     && "$(sha256sum "$TAW_B1_LIVE_BUNDLE" | awk '{print $1}')" == "$TAW_B1_LIVE_BUNDLE_SHA256" \
     && "$(sha256sum "$TAW_REVIEWED_B4_PASS" | awk '{print $1}')" == "$TAW_REVIEWED_B4_PASS_SHA256" \
     && "$(sha256sum "$TAW_REVIEWED_B4_VERDICT" | awk '{print $1}')" == "$TAW_REVIEWED_B4_VERDICT_SHA256" \
     && "$(sha256sum "$TAW_MERGE_BINDING" | awk '{print $1}')" == "$TAW_MERGE_BINDING_SHA256" \
     && "$(sha256sum "$TAW_PASS_JSON" | awk '{print $1}')" == "$TAW_PASS_SHA256" \
     && "$(sha256sum "$CFWD_PASS_JSON" | awk '{print $1}')" == "$CFWD_PASS_SHA256" ]] \
    || { echo "CFWD timing source or credential changed during execution" >&2; return 14; }
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
  local pass_json=""
  local pass_sha=""
  if [[ "$production" == "1" ]]; then
    pass_json=$CFWD_PASS_JSON
    pass_sha=$CFWD_PASS_SHA256
  fi
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      LUMO_SWE_AUTOCOMMIT=0 \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1 \
      FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
      FR13_DEVICE_MULTIDRAFT=1 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$TAW_PASS_JSON" \
      FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION="$production" \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON="$pass_json" \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256="$pass_sha" \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= \
      FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
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
      'ENFORCE_EAGER=0' \
      "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=$production"; do
    [[ "$(grep -Fxc "$expected" "$env_path")" -eq 1 ]] \
      || { echo "$arm lacks exact environment pin: $expected" >&2; return 4; }
  done
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
STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_cfwd_logit_direct.production_engagement.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock arm emitted CFWD logit-direct engagement" >&2; exit 4; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_cfwd_logit_direct.production_engagement.json"
[[ -f "$CANDIDATE_ENGAGEMENT" && ! -L "$CANDIDATE_ENGAGEMENT" ]] \
  || { echo "candidate production engagement is missing" >&2; exit 4; }

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$CANDIDATE_ENGAGEMENT" "$CFWD_PASS_JSON" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$STOCK_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" \
  "$RUNNER_SHA256" "$GATE_SHA256" "$SUBSET_SHA256" \
  "$BLOCK_MAP_SHA256" "$STOCK_FA2_SHA256" "$TAW_PASS_SHA256" \
  "$CFWD_PASS_SHA256" "$CANDIDATE_SOURCE_SHA256" \
  "$WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" <<'PY'
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
from fr13_b4_timing_math import phase_breakdown
import fr13_cfwd_logit_direct_gate as cfwd_gate

subset_path, stock_path, candidate_path, engagement_path, credential_path, out_path = map(
    Path, sys.argv[1:7]
)
stock_arm, candidate_arm, source_commit = sys.argv[7:10]
runner_sha, gate_sha, subset_sha, block_map_sha = sys.argv[10:14]
stock_fa2_sha, taw_pass_sha, credential_sha, candidate_source_sha = sys.argv[14:18]
floor_ms, cap_ms = map(float, sys.argv[18:20])


def load(path):
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"timing artifact is not a single-link regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("ascii"))
    if not isinstance(payload, dict):
        raise SystemExit(f"timing artifact is not an object: {path}")
    return payload, raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


subset, subset_raw = load(subset_path)
stock, stock_raw = load(stock_path)
candidate, candidate_raw = load(candidate_path)
engagement, engagement_raw = load(engagement_path)
credential, credential_raw = load(credential_path)
task_ids = subset.get("instance_ids")
if (
    digest(subset_raw) != subset_sha
    or task_ids != [
        "astropy__astropy-12907",
        "astropy__astropy-13033",
        "astropy__astropy-13236",
        "astropy__astropy-13398",
    ]
):
    raise SystemExit("canonical exact4 timing subset drifted")
if digest(credential_raw) != credential_sha:
    raise SystemExit("CFWD production credential drifted during timing")


def positive(payload, key, arm):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{arm} lacks numeric {key}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{arm} lacks positive finite {key}")
    return value


def validate_measure(payload, raw, arm):
    if (
        payload.get("schema") != "fr13.measure.deploy_speed.v1"
        or payload.get("kind") != "speed"
        or payload.get("instrument") != "OFF"
        or payload.get("regime") != "deployment"
        or payload.get("arm") != arm
        or payload.get("batch_size") != 1
        or payload.get("n_tasks") != 4
        or payload.get("task_instance_ids") != task_ids
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_vocab_k") != 65536
        or payload.get("mandatory_weight_bytes") != 27977022848
        or payload.get("weight_floor_ms") != floor_ms
        or payload.get("floor_ms") != floor_ms
        or payload.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise SystemExit(f"{arm} is not canonical exact4 K64/root1 B1 timing")
    per_task = payload.get("per_task")
    if (
        not isinstance(per_task, list)
        or len(per_task) != 4
        or [row.get("instance_id") for row in per_task] != task_ids
        or any(positive(row, "wall_steps", arm + ":task") < 1 for row in per_task)
    ):
        raise SystemExit(f"{arm} lacks four nonempty per-task timing windows")
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
        "committed_tokens_per_event": positive(
            payload, "committed_per_event", arm
        ),
        "events_per_step": positive(payload, "events_per_step", arm),
        "rows_per_step": rows_per_step,
        "floor_ms": positive(payload, "floor_ms", arm),
        "floor_ratio": positive(payload, "floor_ratio", arm),
        "wall_ms_per_event": phases["wall_ms_per_event"],
        "sfwd_gpu_ms_per_event": phases["sfwd_gpu_ms_per_event"],
        "sfwd_gpu_ms_per_step": phases["sfwd_gpu_ms_per_step"],
        "dfwd_gpu_ms_per_step": phases["dfwd_gpu_ms_per_step"],
        "cfwd_gpu_ms_per_step": phases["cfwd_gpu_ms_per_step"],
        "gpu_component_ms_per_step": phases["gpu_component_ms_per_step"],
        "other_wall_ms_per_step": phases["other_wall_ms_per_step"],
    }


expected_engagement_keys = {
    "schema", "status", "candidate", "mode", "batch_size", "source_commit",
    "candidate_source_sha256", "integration_source_schema",
    "integration_source_sha256", "production_pass_sha256", "served_return",
    "producer_pid",
}
if (
    set(engagement) != expected_engagement_keys
    or engagement.get("schema")
    != "fr13.fixed32.cfwd_logit_direct.production_engagement.v2"
    or engagement.get("status") != "engaged"
    or engagement.get("candidate")
    != "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
    or engagement.get("mode") != "hydra27_fixed32"
    or engagement.get("batch_size") != 1
    or engagement.get("source_commit") != source_commit
    or engagement.get("candidate_source_sha256") != candidate_source_sha
    or engagement.get("integration_source_schema")
    != cfwd_gate.INTEGRATION_SOURCE_SCHEMA
    or engagement.get("integration_source_sha256")
    != cfwd_gate.INTEGRATION_SOURCE_SHA256
    or engagement.get("production_pass_sha256") != credential_sha
    or engagement.get("served_return") != "logit-direct candidate products"
    or type(engagement.get("producer_pid")) is not int
    or engagement["producer_pid"] < 1
):
    raise SystemExit("candidate production engagement identity drifted")

s = validate_measure(stock, stock_raw, stock_arm)
c = validate_measure(candidate, candidate_raw, candidate_arm)
summary = {
    "schema": "fr13.fixed32.cfwd_logit_direct.exact4_b1_timing.v1",
    "status": "complete",
    "classification": "real_swe_verified_exact4_b1_cfwd_logit_direct_timing_pair",
    "timing_eligible": True,
    "floor_acceptance_eligible": False,
    "floor_acceptance_reason": "exact4 screen; canonical exact16 U95 remains required",
    "mode": "hydra27_fixed32",
    "batch_size": 1,
    "concurrency": 1,
    "task_count": 4,
    "task_ids": task_ids,
    "physical_rows": 32,
    "draft_vocab_k": 65536,
    "draft_vocab_root": 1,
    "only_arm_delta": "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 to 1",
    "weight_floor_ms": floor_ms,
    "one_sided_u95_cap_ms": cap_ms,
    "stock": {"arm": stock_arm, **s},
    "candidate": {
        "arm": candidate_arm,
        "production_engagement_sha256": digest(engagement_raw),
        **c,
    },
    "delta": {
        "step_wall_ms": c["step_wall_ms"] - s["step_wall_ms"],
        "full_wall_tps": (
            c["measured_tps_fullstep_wall"]
            - s["measured_tps_fullstep_wall"]
        ),
        "accepted_drafts_per_event": (
            c["accepted_drafts_per_event"] - s["accepted_drafts_per_event"]
        ),
        "cfwd_gpu_ms_per_step": (
            c["cfwd_gpu_ms_per_step"] - s["cfwd_gpu_ms_per_step"]
        ),
    },
    "ratios": {
        "candidate_to_stock_step_wall": c["step_wall_ms"] / s["step_wall_ms"],
        "candidate_to_stock_full_wall_tps": (
            c["measured_tps_fullstep_wall"]
            / s["measured_tps_fullstep_wall"]
        ),
        "candidate_to_stock_cfwd_gpu": (
            c["cfwd_gpu_ms_per_step"] / s["cfwd_gpu_ms_per_step"]
        ),
        "candidate_to_weight_floor": c["step_wall_ms"] / floor_ms,
    },
    "candidate_gap_to_cap_ms": c["step_wall_ms"] - cap_ms,
    "source_commit": source_commit,
    "runner_sha256": runner_sha,
    "gate_sha256": gate_sha,
    "subset_sha256": subset_sha,
    "block_map_sha256": block_map_sha,
    "stock_fa2_sha256": stock_fa2_sha,
    "taw_pass_sha256": taw_pass_sha,
    "cfwd_pass_sha256": credential_sha,
    "candidate_source_sha256": candidate_source_sha,
    "integration_source_schema": cfwd_gate.INTEGRATION_SOURCE_SCHEMA,
    "integration_source_sha256": cfwd_gate.INTEGRATION_SOURCE_SHA256,
}
temporary = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
temporary.replace(out_path)
print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
finalize_manifests

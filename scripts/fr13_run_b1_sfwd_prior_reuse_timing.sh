#!/usr/bin/env bash
# Source-gated real SWE-Verified B1 diagnostic: stock then packed x-gather.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the common pinned FA2 shared object}"
: "${PRIOR_REUSE_GATE_JSON:?set PRIOR_REUSE_GATE_JSON to the completed reduced B1 gate}"
: "${PRIOR_REUSE_GATE_SHA256:?set PRIOR_REUSE_GATE_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
TASK_MARKER_SHA256=04fe7f61a0e0bbd48bf28127385c481b85550b291535f3705511494ba24c8463
DRAFT_VOCAB_BLOCKS=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
K64_ROOT_WEIGHT_BYTES=32666638208
K64_ROOT_FLOOR_MS=119.658015414
K64_ROOT_CAP_MS=137.6067177261
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
FA2_SHA256=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
STOCK_ARM="hydra27_fixed32_sfwd_stock_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_sfwd_packed_xgather_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in "$FORKED_FA2_SO" "$PRIOR_REUSE_GATE_JSON"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "input must be an absolute regular non-symlink file: $required" >&2; exit 2; }
done
unset required
[[ "$PRIOR_REUSE_GATE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "PRIOR_REUSE_GATE_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical one-task B1 subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS" && ! -L "$DRAFT_VOCAB_BLOCKS" ]] \
  || { echo "K64 draft-vocabulary block map must be a regular source file" >&2; exit 2; }
[[ "$(sha256sum "$DRAFT_VOCAB_BLOCKS" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "K64 draft-vocabulary block map SHA-256 drift" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$FA2_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "source or FA2 identity is invalid" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

# The reduced gate must match the exact qualified candidate source before Docker/GPU
# query or arm launch. The runtime validates the installed copy again.
"$PYTHON_BIN" scripts/fr13_sfwd_prior_reuse_timing_pass.py validate \
  --gate "$PRIOR_REUSE_GATE_JSON" \
  --expected-gate-sha256 "$PRIOR_REUSE_GATE_SHA256" \
  --candidate-source src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py \
  --candidate-kernel-source \
    src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py \
  >/dev/null

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
unset FR13_NEEDS_ALLOW
unset FR10_ALLOW_LINEAR_FALLBACK
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$K64_ROOT_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$K64_ROOT_FLOOR_MS" ]] \
  || { echo "K64/root1 floor contract drift" >&2; exit 2; }

write_sfwd_prior_reuse_source_manifest() {
  "$PYTHON_BIN" - "$1" "$REPO" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
repo = Path(sys.argv[2]).resolve()
paths = (
    "scripts/fr13_run_b1_sfwd_prior_reuse_timing.sh",
    "scripts/fr13_sfwd_prior_reuse_timing_pass.py",
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py",
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py",
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_timing.py",
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py",
    "src/lumo_flywheel_serving/inference_proxy.py",
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/run_swe_bench_q36_a.py",
    "scripts/fr13_measure.py",
    "scripts/fr13_dvk_subset_blocks.json",
    "config/fr13_fixed32/subset_b1_diagnostic_one.json",
    "results/fr13_fixed32_sfwd_priorreuse_packed_xgather_b1_byte_pass_20260802/gate_summary.json",
    "results/fr13_fixed32_sfwd_priorreuse_packed_xgather_b1_byte_pass_20260802/identity_and_lifecycle.json",
    "results/fr13_fixed32_sfwd_priorreuse_packed_xgather_b1_byte_pass_20260802/record_summary.json",
    "results/fr13_fixed32_sfwd_priorreuse_packed_xgather_b1_byte_pass_20260802/traffic_model.json",
)
files = {}
for relative in paths:
    path = repo / relative
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"SFWD timing source is not regular: {relative}")
    raw = path.read_bytes()
    files[relative] = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
payload = {
    "schema": "fr13.fixed32.sfwd_xgather.timing_source_manifest.v1",
    "files": files,
}
raw = json.dumps(
    payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
) + "\n"
temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
temporary.write_text(raw, encoding="ascii")
temporary.replace(output_path)
PY
}

write_host_zero_census() {
  local checkpoint=$1
  local output_path=$2
  local docker_count
  local compute_count
  local gpu_memory_used_mib
  docker_count=$(docker ps -aq | wc -l)
  compute_count=$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | awk '$1 ~ /^[0-9]+$/ { count += 1 } END { print count + 0 }'
  )
  gpu_memory_used_mib=$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk '$1 ~ /^[0-9]+$/ { total += $1 } END { print total + 0 }'
  )
  [[ "$docker_count" -eq 0 \
     && "$compute_count" -eq 0 \
     && "$gpu_memory_used_mib" -eq 0 ]] || {
    echo "host resource census is nonzero at $checkpoint" >&2
    return 2
  }
  "$PYTHON_BIN" - \
    "$output_path" "$checkpoint" \
    "$docker_count" "$compute_count" "$gpu_memory_used_mib" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "fr13.fixed32.sfwd_xgather.host_zero_census.v1",
    "checkpoint": sys.argv[2],
    "all_zero": True,
    "docker_containers": int(sys.argv[3]),
    "gpu_compute_processes": int(sys.argv[4]),
    "gpu_memory_used_mib": int(sys.argv[5]),
}
raw = json.dumps(
    payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
) + "\n"
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(raw, encoding="ascii")
temporary.replace(path)
PY
}

mkdir -p "$RUNROOT_ABS/sidecars"
write_host_zero_census \
  before_first_arm "$RUNROOT_ABS/host_zero.before_first_arm.json"
write_sfwd_prior_reuse_source_manifest "$RUNROOT_ABS/sfwd_prior_reuse_source_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf '%s\n' \
  'classification=one_real_swe_verified_k64_root_b1_sfwd_packed_xgather_timing_diagnostic' \
  'task_set=one' \
  'task_count=1' \
  'timing_eligible=false' \
  'floor_acceptance_eligible=false' \
  'production_eligible=false' \
  'production_default_enabled=false' \
  'physical_rows_per_request=32' \
  'candidate_conv_launches_per_layer=1' \
  'incumbent_candidate_arm_conv_launches_per_layer=0' \
  'gdn_level_path_programs=1,11' \
  'gdn_physical_launches_per_layer=2' \
  'draft_vocab_root=1' \
  'draft_vocab_k=65536' \
  "draft_vocab_blocks_sha256=$DRAFT_VOCAB_BLOCKS_SHA256" \
  "task_marker_sha256=$TASK_MARKER_SHA256" \
  "source_commit=$SOURCE_COMMIT" \
  "runner_sha256=$RUNNER_SHA256" \
  "subset_sha256=$SUBSET_SHA256" \
  "fa2_sha256=$FA2_SHA256" \
  "prior_reuse_gate_sha256=$PRIOR_REUSE_GATE_SHA256" \
  "sfwd_prior_reuse_source_manifest_sha256=$(sha256sum "$RUNROOT_ABS/sfwd_prior_reuse_source_manifest.at_launch.json" | awk '{print $1}')" \
  "mandatory_weight_bytes=$K64_ROOT_WEIGHT_BYTES" \
  "mandatory_weight_floor_ms=$K64_ROOT_FLOOR_MS" \
  "one_sided_u95_cap_ms=$K64_ROOT_CAP_MS" \
  "stock_arm=$STOCK_ARM" \
  "candidate_arm=$CANDIDATE_ARM" \
  "started=$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  write_sfwd_prior_reuse_source_manifest "$RUNROOT_ABS/sfwd_prior_reuse_source_manifest.at_end.json" \
    || return $?
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 \
    --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" \
    || return $?
  cmp -s \
    "$RUNROOT_ABS/sfwd_prior_reuse_source_manifest.at_launch.json" \
    "$RUNROOT_ABS/sfwd_prior_reuse_source_manifest.at_end.json" \
    || { echo "SFWD timing source manifest changed during timing" >&2; return 14; }
  cmp -s \
    "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s \
    "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "timing runner changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then
      :
    else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local production=$2
  local gate_json=""
  local gate_sha=""
  local real_event_path=""
  if [[ "$production" == "1" ]]; then
    gate_json=$PRIOR_REUSE_GATE_JSON
    gate_sha=$PRIOR_REUSE_GATE_SHA256
    real_event_path=/logs/fr13_fixed32_sfwd_state_fusion.real_event.arm
  fi
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC=1 \
      FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR13_MANDATORY_WEIGHT_BYTES="$K64_ROOT_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$K64_ROOT_FLOOR_MS" \
      ENFORCE_EAGER=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE FR10_METRICS=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0 \
      FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH= \
      FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256= \
      FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_COMMIT= \
      FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_AB=1 \
      FR13_FIXED32_SFWD_PRIOR_REUSE_PRODUCTION="$production" \
      FR13_FIXED32_SFWD_PRIOR_REUSE_GATE_JSON="$gate_json" \
      FR13_FIXED32_SFWD_PRIOR_REUSE_GATE_SHA256="$gate_sha" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON= \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PASS_JSON= FR13_FA2_QROW16_LIVE_PASS_SHA256= \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON= \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_BATCH_GDN_BV8_TIMING=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_JSON= \
      FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_SHA256= \
      FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_JSON= \
      FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_SHA256= \
      FR13_FIXED32_BATCH_GDN_RUNTIME_MANIFEST_JSON= \
      FR13_FIXED32_BATCH_GDN_GATE_RUNNER= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$FORKED_FA2_SO" RUNROOT="$RUNROOT_REL" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" hydra27_fixed32 "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$rc" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$rc"
  fi
  local env_path="$RUNROOT_ABS/$arm/container_env.txt"
  [[ -f "$env_path" && ! -L "$env_path" ]] \
    || { echo "$arm lacks container_env.txt" >&2; return 4; }
  for expected in \
      'FR13_FIXED32_MODE=hydra27_fixed32' \
      'FR13_FIXED32_B1_DIAGNOSTIC=1' \
      'FR13_DRAFT_VOCAB_ROOT=1' \
      'FR13_DRAFT_VOCAB_K=65536' \
      "FR13_DRAFT_VOCAB_BLOCKS=$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      'ENFORCE_EAGER=1' \
      'FR13_FIXED32_SFWD_PRIOR_REUSE_TIMING_AB=1' \
      "FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH=$real_event_path" \
      "FR13_FIXED32_SFWD_PRIOR_REUSE_PRODUCTION=$production"; do
    [[ "$(grep -Fxc "$expected" "$env_path")" -eq 1 ]] \
      || { echo "$arm lacks exact environment pin: $expected" >&2; return 4; }
  done
  "$PYTHON_BIN" - "$RUNROOT_ABS/$arm/docker_after_tasks.log" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
info = os.lstat(path)
if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
    raise SystemExit(f"K64 gather audit input is not a regular file: {path}")
log = path.read_text(encoding="utf-8", errors="replace")
shim_prefix = "[FR13_DRAFT_VOCAB] shim built K=65536 "
root_prefix = "[FR13_DRAFT_VOCAB_ROOT] engaged K=65536 "
disabled_prefix = "[FR13_DRAFT_VOCAB] DISABLED"
shim_lines = [line for line in log.splitlines() if shim_prefix in line]
root_lines = [line for line in log.splitlines() if root_prefix in line]
if len(shim_lines) != 1 or "mode=gather" not in shim_lines[0]:
    raise SystemExit("K64 draft-vocabulary gather shim did not engage exactly once")
if len(root_lines) != 1 or "mode=gather" not in root_lines[0]:
    raise SystemExit("K64 root gather did not engage exactly once")
if disabled_prefix in log:
    raise SystemExit("draft-vocabulary runtime fallback to full vocabulary engaged")
PY
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
STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_sfwd_prior_reuse.timing_engagement.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock arm emitted SFWD candidate engagement" >&2; exit 4; }
write_host_zero_census after_stock_arm "$RUNROOT_ABS/host_zero.after_stock_arm.json"
run_arm "$CANDIDATE_ARM" 1
write_host_zero_census \
  after_candidate_arm "$RUNROOT_ABS/host_zero.after_candidate_arm.json"

CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_sfwd_prior_reuse.timing_engagement.json"
CANDIDATE_REAL_EVENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_sfwd_state_fusion.real_event.arm"
"$PYTHON_BIN" scripts/fr13_sfwd_prior_reuse_timing_pass.py verify-engagement \
  --engagement "$CANDIDATE_ENGAGEMENT" \
  --expected-gate-sha256 "$PRIOR_REUSE_GATE_SHA256" \
  --candidate-source src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py \
  --candidate-kernel-source \
    src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py \
  >/dev/null
[[ -f "$CANDIDATE_REAL_EVENT" && ! -L "$CANDIDATE_REAL_EVENT" \
   && "$(stat -c '%a' "$CANDIDATE_REAL_EVENT")" == "444" \
   && "$(sha256sum "$CANDIDATE_REAL_EVENT" | awk '{print $1}')" == "$TASK_MARKER_SHA256" ]] \
  || { echo "candidate lacks the authenticated real-task engagement marker" >&2; exit 4; }
[[ "$(sha256sum "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_sfwd_prior_reuse.timing_gate.json" | awk '{print $1}')" == "$PRIOR_REUSE_GATE_SHA256" ]] \
  || { echo "candidate installed reduced-gate identity drift" >&2; exit 4; }

finalize_manifests
"$PYTHON_BIN" - \
  "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" \
  "$RUNNER_SHA256" "$SUBSET_SHA256" "$FA2_SHA256" "$PRIOR_REUSE_GATE_SHA256" \
  "$K64_ROOT_WEIGHT_BYTES" "$K64_ROOT_FLOOR_MS" "$K64_ROOT_CAP_MS" \
  "$DRAFT_VOCAB_BLOCKS_SHA256" "$TASK_MARKER_SHA256" <<'PY'
import hashlib
import json
import math
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
stock_arm, candidate_arm = sys.argv[2:4]
source_commit, runner_sha, subset_sha, fa2_sha, gate_sha = sys.argv[4:9]
weight_bytes = int(sys.argv[9])
floor_ms, cap_ms = map(float, sys.argv[10:12])
draft_vocab_blocks_sha256 = sys.argv[12]
task_marker_sha256 = sys.argv[13]


def load(path: Path) -> tuple[dict, bytes]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"non-regular timing artifact: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("ascii"))
    if not isinstance(payload, dict):
        raise SystemExit(f"timing artifact is not an object: {path}")
    return payload, raw


def real_task_marker_sha256(instance_id: object) -> str | None:
    if not isinstance(instance_id, str) or not instance_id:
        return None
    marker = f"swe_verified:{instance_id}\n".encode("ascii", errors="strict")
    return hashlib.sha256(marker).hexdigest()


def gather_log(arm: str) -> dict:
    path = root / arm / "docker_after_tasks.log"
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"non-regular K64 gather audit: {path}")
    raw = path.read_bytes()
    log = raw.decode("utf-8", errors="replace")
    shim_prefix = "[FR13_DRAFT_VOCAB] shim built K=65536 "
    root_prefix = "[FR13_DRAFT_VOCAB_ROOT] engaged K=65536 "
    disabled_prefix = "[FR13_DRAFT_VOCAB] DISABLED"
    shim_lines = [line for line in log.splitlines() if shim_prefix in line]
    root_lines = [line for line in log.splitlines() if root_prefix in line]
    if len(shim_lines) != 1 or "mode=gather" not in shim_lines[0]:
        raise SystemExit(f"{arm} lacks exactly one K64 gather-shim engagement")
    if len(root_lines) != 1 or "mode=gather" not in root_lines[0]:
        raise SystemExit(f"{arm} lacks exactly one K64 root-gather engagement")
    if disabled_prefix in log:
        raise SystemExit(f"{arm} fell back to the full vocabulary")
    return {
        "docker_after_tasks_sha256": hashlib.sha256(raw).hexdigest(),
        "shim_engagements": 1,
        "root_engagements": 1,
        "fallbacks": 0,
    }


def campaign(arm: str) -> dict:
    payload, raw = load(root / arm / "swe_out" / "verified" / "campaign_summary.json")
    if (
        payload.get("instances_total") != 1
        or payload.get("verdict_counts") != {"resolved": 1}
        or payload.get("failure_mode_counts") != {"tests_passed": 1}
        or payload.get("resolved_rate") != 1.0
        or payload.get("fixed32_run_classification")
        != {
            "run_classification": "b1_diagnostic",
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
        }
    ):
        raise SystemExit(f"{arm} did not resolve its one real SWE-Verified task")
    metadata_paths = list(
        (root / arm / "swe_out" / "verified" / "per_task").glob(
            "*/runner_metadata.json"
        )
    )
    if len(metadata_paths) != 1:
        raise SystemExit(f"{arm} lacks exactly one real-task runner metadata file")
    metadata, metadata_raw = load(metadata_paths[0])
    agent = metadata.get("agent")
    codex = metadata.get("codex")
    evaluation = metadata.get("eval")
    if (
        not isinstance(agent, dict)
        or agent.get("timed_out") is not False
        or agent.get("exit_code") != 0
        or not isinstance(codex, dict)
        or codex.get("timed_out") is not False
        or codex.get("exit_code") != 0
        or not isinstance(evaluation, dict)
        or evaluation.get("exit_code") != 0
    ):
        raise SystemExit(f"{arm} real task did not complete cleanly")
    boundary, boundary_raw = load(
        metadata_paths[0].parent / "fixed32_task_boundary.json"
    )
    interval = boundary.get("timing_interval")
    if (
        boundary.get("schema")
        != "fr13-fixed32-eager-timing-task-boundary-v1"
        or real_task_marker_sha256(boundary.get("instance_id"))
        != task_marker_sha256
        or boundary.get("run_classification")
        != "eager_kernel_timing_diagnostic"
        or boundary.get("flush_protocol_used") is not True
        or boundary.get("graph_census_claimed") is not False
        or not isinstance(boundary.get("pre"), dict)
        or not isinstance(boundary.get("post"), dict)
        or not isinstance(boundary.get("pre_runtime_snapshot"), dict)
        or not isinstance(boundary.get("post_runtime_snapshot"), dict)
        or boundary["pre_runtime_snapshot"].get("schema")
        != "fr13-fixed32-eager-timing-boundary-snapshot-v1"
        or boundary["post_runtime_snapshot"].get("schema")
        != "fr13-fixed32-eager-timing-boundary-snapshot-v1"
        or not isinstance(interval, dict)
        or isinstance(interval.get("pure_decode_forward_steps"), bool)
        or not isinstance(interval.get("pure_decode_forward_steps"), int)
        or interval.get("pure_decode_forward_steps", 0) <= 0
        or interval.get("graph_census_events") != 0
        or interval.get("sfwd_steps")
        != interval.get("pure_decode_forward_steps")
        or interval.get("sfwd_drafts")
        != interval.get("pure_decode_forward_steps")
        or interval.get("dfwd_spans")
        != interval.get("pure_decode_forward_steps")
        or interval.get("cfwd_spans")
        != interval.get("pure_decode_forward_steps")
        or interval.get("sfwd_forward_starts")
        != interval.get("pure_decode_forward_steps")
        or interval.get("sfwd_forward_dropped") != 0
        or isinstance(interval.get("wall_steps"), bool)
        or not isinstance(interval.get("wall_steps"), int)
        or interval.get("wall_steps", 0) <= 0
        or interval.get("wall_drafts") != interval.get("wall_steps")
        or not all(
            not isinstance(interval.get(key), bool)
            and isinstance(interval.get(key), (int, float))
            and math.isfinite(float(interval[key]))
            and float(interval[key]) > 0.0
            for key in (
                "sfwd_gpu_seconds",
                "wall_seconds",
                "dfwd_gpu_seconds",
                "cfwd_gpu_seconds",
            )
        )
    ):
        raise SystemExit(f"{arm} lacks a complete authenticated timing bracket")
    return {
        "campaign_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "runner_metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        "task_boundary_sha256": hashlib.sha256(boundary_raw).hexdigest(),
        "instances_total": 1,
        "resolved": 1,
        "tests_passed": 1,
        "agent_timed_out": False,
        "agent_exit_code": 0,
        "eval_exit_code": 0,
    }


def timer(arm: str, suffix: str, schema: str) -> dict:
    paths = sorted((root / "sidecars").glob(f"{arm}{suffix}.json*"))
    paths = [
        path for path in paths
        if path.is_file() and not path.is_symlink() and ".samples." not in path.name
    ]
    if not paths:
        raise SystemExit(f"missing {schema} timer for {arm}")
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    payload, raw = load(path)
    if payload.get("schema") != schema:
        raise SystemExit(f"timer schema mismatch: {path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema": schema,
    }


def measure(arm: str) -> tuple[dict, bytes]:
    payload, raw = load(root / arm / "deploy_speed_fullwall.json")
    required = {
        "schema": "fr13.measure.deploy_speed.v1",
        "regime": "deployment",
        "instrument": "OFF",
        "batch_size": 1,
        "n_tasks": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "floor_is_full_step_hardware_floor": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise SystemExit(f"{arm} measure {key} mismatch")
    task_instance_ids = payload.get("task_instance_ids")
    if (
        not isinstance(task_instance_ids, list)
        or len(task_instance_ids) != 1
        or real_task_marker_sha256(task_instance_ids[0]) != task_marker_sha256
    ):
        raise SystemExit(f"{arm} measure task identity mismatch")
    for key in (
        "step_wall_ms", "measured_tps_fullstep_wall", "accept_per_event",
        "s_per_fwd_gpu", "drafter_gpu_ms_per_step", "committer_gpu_ms_per_step",
        "derived_tps_fullstep_gpu", "committed_per_event", "events_per_step",
        "wall_steps_measured", "floor_ms", "weight_floor_ms",
    ):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) <= 0:
            raise SystemExit(f"{arm} measure lacks positive {key}")
    residual = payload.get("overhead_other_ms_per_event")
    if isinstance(residual, bool) or not isinstance(residual, (int, float)) \
            or not math.isfinite(float(residual)):
        raise SystemExit(f"{arm} measure lacks finite overhead_other_ms_per_event")
    if not math.isclose(
        float(payload["events_per_step"]), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SystemExit(f"{arm} B1 measure must have exactly one event per step")
    if payload.get("mandatory_weight_bytes") != weight_bytes:
        raise SystemExit(f"{arm} mandatory weight-byte ledger mismatch")
    if not math.isclose(
        float(payload["weight_floor_ms"]), floor_ms, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(f"{arm} K64/root1 weight floor mismatch")
    if not math.isclose(
        float(payload["floor_ms"]), floor_ms, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(f"{arm} B1 floor is not the K64/root1 weight bound")
    return payload, raw


def phase_breakdown(payload: dict) -> dict:
    events_per_step = float(payload["events_per_step"])
    sfwd_ms = float(payload["s_per_fwd_gpu"]) * 1000.0 * events_per_step
    drafter_ms = float(payload["drafter_gpu_ms_per_step"])
    committer_ms = float(payload["committer_gpu_ms_per_step"])
    other_ms = float(payload["overhead_other_ms_per_event"]) * events_per_step
    component_sum_ms = sfwd_ms + drafter_ms + committer_ms + other_ms
    wall_ms = float(payload["step_wall_ms"])
    if not math.isclose(component_sum_ms, wall_ms, rel_tol=1e-9, abs_tol=1e-6):
        raise SystemExit("full-step phase breakdown does not close to measured wall")
    return {
        "sfwd_verify_gpu_ms": sfwd_ms,
        "dfwd_drafter_gpu_ms": drafter_ms,
        "cfwd_committer_gpu_ms": committer_ms,
        "other_wall_ms": other_ms,
        "component_sum_ms": component_sum_ms,
        "measured_wall_ms": wall_ms,
        "events_per_step": events_per_step,
    }


def zero_census(name: str, checkpoint: str) -> dict:
    payload, raw = load(root / name)
    required = {
        "schema": "fr13.fixed32.sfwd_xgather.host_zero_census.v1",
        "checkpoint": checkpoint,
        "all_zero": True,
        "docker_containers": 0,
        "gpu_compute_processes": 0,
        "gpu_memory_used_mib": 0,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise SystemExit(f"host zero census mismatch at {checkpoint}")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "all_zero": True,
        "docker_containers": 0,
        "gpu_compute_processes": 0,
        "gpu_memory_used_mib": 0,
    }


stock, stock_raw = measure(stock_arm)
candidate, candidate_raw = measure(candidate_arm)
engagement, engagement_raw = load(
    root / candidate_arm / "logs" /
    "fr13_fixed32_sfwd_prior_reuse.timing_engagement.json"
)
source_manifest, source_manifest_raw = load(
    root / "sfwd_prior_reuse_source_manifest.at_launch.json"
)
source_manifest_end, source_manifest_end_raw = load(
    root / "sfwd_prior_reuse_source_manifest.at_end.json"
)
runtime_manifest, runtime_manifest_raw = load(
    root / "runtime_manifest.at_launch.json"
)
runtime_manifest_end, runtime_manifest_end_raw = load(
    root / "runtime_manifest.at_end.json"
)
external_manifest, external_manifest_raw = load(
    root / "external_manifest.at_launch.json"
)
external_manifest_end, external_manifest_end_raw = load(
    root / "external_manifest.at_end.json"
)
if (
    source_manifest.get("schema")
    != "fr13.fixed32.sfwd_xgather.timing_source_manifest.v1"
    or len(source_manifest.get("files", {})) != 18
    or source_manifest != source_manifest_end
    or source_manifest_raw != source_manifest_end_raw
    or runtime_manifest != runtime_manifest_end
    or runtime_manifest_raw != runtime_manifest_end_raw
    or external_manifest != external_manifest_end
    or external_manifest_raw != external_manifest_end_raw
):
    raise SystemExit("SFWD packed x-gather manifest identity is incomplete")
host_zero_census = {
    "before_first_arm": zero_census(
        "host_zero.before_first_arm.json", "before_first_arm"
    ),
    "after_stock_arm": zero_census(
        "host_zero.after_stock_arm.json", "after_stock_arm"
    ),
    "after_candidate_arm": zero_census(
        "host_zero.after_candidate_arm.json", "after_candidate_arm"
    ),
}
summary = {
    "schema": "fr13.fixed32.sfwd_xgather.b1_timing_pair.v1",
    "status": "complete_diagnostic",
    "run_classification": (
        "one_real_swe_verified_k64_root_b1_sfwd_packed_xgather_timing_diagnostic"
    ),
    "task_marker_sha256": task_marker_sha256,
    "task_count": 1,
    "batch_size": 1,
    "physical_rows_per_request": 32,
    "draft_vocab_k": 65536,
    "draft_vocab_root": 1,
    "draft_vocab_blocks_sha256": draft_vocab_blocks_sha256,
    "source_commit": source_commit,
    "runner_sha256": runner_sha,
    "subset_sha256": subset_sha,
    "fa2_sha256": fa2_sha,
    "reduced_gate_sha256": gate_sha,
    "sfwd_prior_reuse_source_manifest_sha256": hashlib.sha256(
        source_manifest_raw
    ).hexdigest(),
    "source_manifest_launch_end_equal": True,
    "runtime_manifest_sha256": hashlib.sha256(runtime_manifest_raw).hexdigest(),
    "runtime_manifest_launch_end_equal": True,
    "external_manifest_sha256": hashlib.sha256(external_manifest_raw).hexdigest(),
    "external_manifest_launch_end_equal": True,
    "host_zero_census": host_zero_census,
    "mandatory_weight_bytes": weight_bytes,
    "mandatory_weight_floor_ms": floor_ms,
    "floor_scope": "optimistic_mandatory_weight_read_lower_bound",
    "floor_is_full_step_hardware_floor": False,
    "one_sided_u95_cap_ms": cap_ms,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "production_eligible": False,
    "acceptance_blocker": "one task is diagnostic only; exact4 or exact16 required",
    "stock": {
        "arm": stock_arm,
        "candidate_production_enabled": False,
        "candidate_engagement_present": False,
        "deploy_speed_sha256": hashlib.sha256(stock_raw).hexdigest(),
        "step_wall_ms": stock["step_wall_ms"],
        "fullstep_wall_tps": stock["measured_tps_fullstep_wall"],
        "accept_per_event": stock["accept_per_event"],
        "committed_per_event": stock["committed_per_event"],
        "derived_fullstep_gpu_tps": stock["derived_tps_fullstep_gpu"],
        "phase_breakdown_ms_per_step": phase_breakdown(stock),
        "k64_gather": gather_log(stock_arm),
        "real_task_outcome": campaign(stock_arm),
        "sfwd": timer(stock_arm, "", "fr13.sfwd_gpu_timer.v2"),
        "dfwd": timer(stock_arm, "_dfwd", "fr13.span_gpu_timer.v1"),
        "cfwd": timer(stock_arm, "_cfwd", "fr13.span_gpu_timer.v1"),
    },
    "candidate": {
        "arm": candidate_arm,
        "candidate": engagement.get("candidate"),
        "candidate_kernel": engagement.get("candidate_kernel"),
        "conv_num_warps": engagement.get("conv_num_warps"),
        "deploy_speed_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "engagement_sha256": hashlib.sha256(engagement_raw).hexdigest(),
        "real_event_marker_sha256": hashlib.sha256(
            (root / candidate_arm / "logs" /
             "fr13_fixed32_sfwd_state_fusion.real_event.arm").read_bytes()
        ).hexdigest(),
        "candidate_served": engagement.get("candidate_served"),
        "sole_conv_source_producer": engagement.get("sole_conv_source_producer"),
        "fallback_permitted": engagement.get("fallback_permitted"),
        "step_wall_ms": candidate["step_wall_ms"],
        "fullstep_wall_tps": candidate["measured_tps_fullstep_wall"],
        "accept_per_event": candidate["accept_per_event"],
        "committed_per_event": candidate["committed_per_event"],
        "derived_fullstep_gpu_tps": candidate["derived_tps_fullstep_gpu"],
        "phase_breakdown_ms_per_step": phase_breakdown(candidate),
        "k64_gather": gather_log(candidate_arm),
        "real_task_outcome": campaign(candidate_arm),
        "sfwd": timer(candidate_arm, "", "fr13.sfwd_gpu_timer.v2"),
        "dfwd": timer(candidate_arm, "_dfwd", "fr13.span_gpu_timer.v1"),
        "cfwd": timer(candidate_arm, "_cfwd", "fr13.span_gpu_timer.v1"),
    },
    "candidate_over_stock_step_wall_ratio": (
        candidate["step_wall_ms"] / stock["step_wall_ms"]
    ),
    "candidate_over_optimistic_floor_ratio": candidate["step_wall_ms"] / floor_ms,
}
(root / "timing_summary.pending.json").write_text(
    json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

mv -- \
  "$RUNROOT_ABS/timing_summary.pending.json" \
  "$RUNROOT_ABS/timing_summary.json"
trap - EXIT
cat "$RUNROOT_ABS/timing_summary.json"
printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

#!/usr/bin/env bash
# Source-gated real SWE-Verified B1 timing diagnostic: stock then SFWD fusion.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the common pinned FA2 shared object}"
: "${SFWD_PASS_JSON:?set SFWD_PASS_JSON to the completed authenticated B1 live PASS}"
: "${SFWD_PASS_SHA256:?set SFWD_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
TASK_ID=astropy__astropy-12907
FULL_VOCAB_WEIGHT_BYTES=42025179008
FULL_VOCAB_FLOOR_MS=153.9383846446886
FULL_VOCAB_CAP_MS=177.0291423413919
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
FA2_SHA256=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
STOCK_ARM="hydra27_fixed32_sfwd_stock_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_sfwd_fusion_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in "$FORKED_FA2_SO" "$SFWD_PASS_JSON"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "input must be an absolute regular non-symlink file: $required" >&2; exit 2; }
done
unset required
[[ "$SFWD_PASS_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "SFWD_PASS_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical one-task B1 subset SHA-256 drift" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$FA2_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "source or FA2 identity is invalid" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

# The future live PASS must match this exact kernel source before any Docker/GPU
# query or arm launch. The runtime validates the installed copy again.
"$PYTHON_BIN" scripts/fr13_sfwd_state_fusion_pass.py validate \
  --live-result "$SFWD_PASS_JSON" \
  --expected-live-sha256 "$SFWD_PASS_SHA256" \
  --kernel-source src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0'
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$FULL_VOCAB_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$FULL_VOCAB_FLOOR_MS" ]] \
  || { echo "full-vocabulary floor contract drift" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf '%s\n' \
  'classification=one_real_swe_verified_full_vocab_b1_sfwd_state_fusion_timing_diagnostic' \
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
  "task_id=$TASK_ID" \
  "source_commit=$SOURCE_COMMIT" \
  "runner_sha256=$RUNNER_SHA256" \
  "subset_sha256=$SUBSET_SHA256" \
  "fa2_sha256=$FA2_SHA256" \
  "sfwd_pass_sha256=$SFWD_PASS_SHA256" \
  "mandatory_weight_bytes=$FULL_VOCAB_WEIGHT_BYTES" \
  "mandatory_weight_floor_ms=$FULL_VOCAB_FLOOR_MS" \
  "one_sided_u95_cap_ms=$FULL_VOCAB_CAP_MS" \
  "stock_arm=$STOCK_ARM" \
  "candidate_arm=$CANDIDATE_ARM" \
  "started=$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

validate_eager_lifecycle() {
  local arm=$1
  "$PYTHON_BIN" - "$RUNROOT_ABS/$arm" "$TASK_ID" <<'PY'
import json
import stat
import sys
from pathlib import Path

armdir = Path(sys.argv[1])
task_id = sys.argv[2]
classification = "eager_kernel_timing_diagnostic"


def load(path: Path) -> dict:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"non-regular eager lifecycle artifact: {path}")
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise SystemExit(f"eager lifecycle artifact is not an object: {path}")
    return payload


terminal = load(armdir / "fixed32_final_flush_skipped.json")
for key, expected in {
    "schema": "fr13-fixed32-eager-kernel-terminal-v1",
    "run_classification": classification,
    "acceptance_valid": False,
    "flush_protocol_used": False,
}.items():
    if terminal.get(key) != expected:
        raise SystemExit(f"{armdir.name} terminal {key} mismatch")

traffic = load(armdir / "fixed32_chat_traffic_audit_skipped.json")
for key, expected in {
    "schema": "fr13-fixed32-eager-kernel-traffic-audit-skip-v1",
    "run_classification": classification,
    "acceptance_valid": False,
    "authenticated_engine_ledger_snapshotted": True,
    "graph_census_audit_used": False,
}.items():
    if traffic.get(key) != expected:
        raise SystemExit(f"{armdir.name} traffic audit {key} mismatch")

boundaries = list(armdir.glob("swe_out/*/per_task/*/fixed32_task_boundary.json"))
if len(boundaries) != 1:
    raise SystemExit(
        f"{armdir.name} expected one eager task boundary, found {len(boundaries)}"
    )
boundary = load(boundaries[0])
for key, expected in {
    "schema": "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1",
    "instance_id": task_id,
    "run_classification": classification,
    "acceptance_valid": False,
    "flush_protocol_used": False,
}.items():
    if boundary.get(key) != expected:
        raise SystemExit(f"{armdir.name} task boundary {key} mismatch")
for key in ("pre_metrics", "post_metrics"):
    if not isinstance(boundary.get(key), dict):
        raise SystemExit(f"{armdir.name} task boundary lacks {key}")
PY
}

run_arm() {
  local arm=$1
  local production=$2
  local pass_json=""
  local pass_sha=""
  if [[ "$production" == "1" ]]; then
    pass_json=$SFWD_PASS_JSON
    pass_sha=$SFWD_PASS_SHA256
  fi
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC=1 \
      FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0 \
      FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0' \
      FR13_MANDATORY_WEIGHT_BYTES="$FULL_VOCAB_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$FULL_VOCAB_FLOOR_MS" \
      ENFORCE_EAGER=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE FR10_METRICS=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_TIMING=1 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION="$production" \
      FR13_FIXED32_SFWD_STATE_FUSION_LIVE_PASS_JSON="$pass_json" \
      FR13_FIXED32_SFWD_STATE_FUSION_LIVE_PASS_SHA256="$pass_sha" \
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
      'FR13_DRAFT_VOCAB_ROOT=0' \
      'FR13_DRAFT_VOCAB_K=0' \
      'ENFORCE_EAGER=1' \
      'FR13_FIXED32_SFWD_STATE_FUSION_TIMING=1' \
      "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=$production"; do
    [[ "$(grep -Fxc "$expected" "$env_path")" -eq 1 ]] \
      || { echo "$arm lacks exact environment pin: $expected" >&2; return 4; }
  done
  validate_eager_lifecycle "$arm"
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
STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_sfwd_state_fusion.production_engagement.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock arm emitted SFWD candidate engagement" >&2; exit 4; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_sfwd_state_fusion.production_engagement.json"
"$PYTHON_BIN" scripts/fr13_sfwd_state_fusion_pass.py verify-engagement \
  --engagement "$CANDIDATE_ENGAGEMENT" \
  --expected-live-sha256 "$SFWD_PASS_SHA256" \
  --kernel-source src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  >/dev/null
[[ "$(sha256sum "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_sfwd_state_fusion.production_pass.json" | awk '{print $1}')" == "$SFWD_PASS_SHA256" ]] \
  || { echo "candidate installed PASS identity drift" >&2; exit 4; }

"$PYTHON_BIN" - \
  "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" \
  "$RUNNER_SHA256" "$SUBSET_SHA256" "$FA2_SHA256" "$SFWD_PASS_SHA256" \
  "$FULL_VOCAB_FLOOR_MS" "$FULL_VOCAB_CAP_MS" <<'PY'
import hashlib
import json
import math
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
stock_arm, candidate_arm = sys.argv[2:4]
source_commit, runner_sha, subset_sha, fa2_sha, live_sha = sys.argv[4:9]
floor_ms, cap_ms = map(float, sys.argv[9:11])


def load(path: Path) -> tuple[dict, bytes]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"non-regular timing artifact: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("ascii"))
    if not isinstance(payload, dict):
        raise SystemExit(f"timing artifact is not an object: {path}")
    return payload, raw


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
        "draft_vocab_k": 0,
        "draft_vocab_root": 0,
        "floor_is_full_step_hardware_floor": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise SystemExit(f"{arm} measure {key} mismatch")
    if payload.get("task_instance_ids") != ["astropy__astropy-12907"]:
        raise SystemExit(f"{arm} measure task identity mismatch")
    for key in (
        "step_wall_ms", "measured_tps_fullstep_wall", "accept_per_event",
        "s_per_fwd_gpu", "drafter_gpu_ms_per_step", "committer_gpu_ms_per_step",
    ):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) <= 0:
            raise SystemExit(f"{arm} measure lacks positive {key}")
    return payload, raw


stock, stock_raw = measure(stock_arm)
candidate, candidate_raw = measure(candidate_arm)
engagement, engagement_raw = load(
    root / candidate_arm / "logs" /
    "fr13_fixed32_sfwd_state_fusion.production_engagement.json"
)
summary = {
    "schema": "fr13.fixed32.sfwd_state_fusion.b1_timing_pair.v1",
    "status": "complete_diagnostic",
    "run_classification": (
        "one_real_swe_verified_full_vocab_b1_sfwd_state_fusion_timing_diagnostic"
    ),
    "task_ids": ["astropy__astropy-12907"],
    "task_count": 1,
    "batch_size": 1,
    "physical_rows_per_request": 32,
    "source_commit": source_commit,
    "runner_sha256": runner_sha,
    "subset_sha256": subset_sha,
    "fa2_sha256": fa2_sha,
    "live_pass_sha256": live_sha,
    "mandatory_weight_floor_ms": floor_ms,
    "one_sided_u95_cap_ms": cap_ms,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "production_eligible": False,
    "acceptance_blocker": "one task is diagnostic only; exact4 or exact16 required",
    "stock": {
        "arm": stock_arm,
        "deploy_speed_sha256": hashlib.sha256(stock_raw).hexdigest(),
        "step_wall_ms": stock["step_wall_ms"],
        "fullstep_wall_tps": stock["measured_tps_fullstep_wall"],
        "accept_per_event": stock["accept_per_event"],
        "sfwd": timer(stock_arm, "", "fr13.sfwd_gpu_timer.v2"),
        "dfwd": timer(stock_arm, "_dfwd", "fr13.span_gpu_timer.v1"),
        "cfwd": timer(stock_arm, "_cfwd", "fr13.span_gpu_timer.v1"),
    },
    "candidate": {
        "arm": candidate_arm,
        "deploy_speed_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "engagement_sha256": hashlib.sha256(engagement_raw).hexdigest(),
        "candidate_served": engagement.get("candidate_served"),
        "step_wall_ms": candidate["step_wall_ms"],
        "fullstep_wall_tps": candidate["measured_tps_fullstep_wall"],
        "accept_per_event": candidate["accept_per_event"],
        "sfwd": timer(candidate_arm, "", "fr13.sfwd_gpu_timer.v2"),
        "dfwd": timer(candidate_arm, "_dfwd", "fr13.span_gpu_timer.v1"),
        "cfwd": timer(candidate_arm, "_cfwd", "fr13.span_gpu_timer.v1"),
    },
    "candidate_over_stock_step_wall_ratio": (
        candidate["step_wall_ms"] / stock["step_wall_ms"]
    ),
    "candidate_over_floor_ratio": candidate["step_wall_ms"] / floor_ms,
}
(root / "timing_summary.json").write_text(
    json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during timing" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during timing" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "timing runner changed during execution" >&2; exit 14; }
printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

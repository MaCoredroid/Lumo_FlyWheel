#!/usr/bin/env bash
# One authenticated real SWE-Verified Hydra27 B1 byte gate for qrow32 split2.
# The incumbent Qrow16 FULL graph is served; this script emits no timing data.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_QROW32_SPLIT2_LIVE_GATE:-0}" in
  1) ;;
  0)
    echo "qrow32 split2 live gate is disabled; set FR13_RUN_QROW32_SPLIT2_LIVE_GATE=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_QROW32_SPLIT2_LIVE_GATE must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW32_B1_FA2_SO:?set QROW32_B1_FA2_SO to the pinned combined binary}"
: "${QROW32_B1_FA2_SOURCE:?set QROW32_B1_FA2_SOURCE to the pinned FA2 source closure}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=hydra27_fixed32
TASK_ID=astropy__astropy-12907
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
CANDIDATE_SHA256=5eec90f317cf6126cd57ab7f77b392ae6a1430d28210dcb31756abe788ef3467
CANDIDATE_BYTES=300140712
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=c10888e721335ff99f93dabdfea7d8a524fbd7e21e8aee3f425f50af06bf5d84
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
PATCH_SOURCE_SHA256=$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="hydra27_fixed32_fa2_qrow32_split2_k64_b1_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in "$QROW32_B1_FA2_SO" "$QROW32_B1_FA2_SOURCE"; do
  [[ "$required" == /* && -e "$required" && ! -L "$required" ]] \
    || { echo "pinned input must be absolute, present, and not a symlink: $required" >&2; exit 2; }
done
unset required
[[ -f "$QROW32_B1_FA2_SO" \
   && "$(stat -c '%s' "$QROW32_B1_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$QROW32_B1_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "QROW32_B1_FA2_SO is not the pinned qrow32 split2 binary" >&2; exit 2; }
[[ -d "$QROW32_B1_FA2_SOURCE" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "source closure, B1 task, or K64 block map drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py validate-source \
  --source-root "$QROW32_B1_FA2_SOURCE" >/dev/null
PYTHONPATH="$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" - "$QROW32_B1_FA2_SO" "$SOURCE_COMMIT" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_fixed32_contract as contract
from scripts import fr13_qrow32_b1_pass_sidecar as qrow

qrow.validate_candidate(Path(sys.argv[1]), qrow.CANDIDATE_SHA256)
qrow.validate_patch_source(
    Path("scripts/fr13_patch_fa2_tree_bias.py"),
    expected_source_commit=sys.argv[2],
)
if (
    contract.QROW32_B1_SPLIT2_FA2_SHA256 != qrow.CANDIDATE_SHA256
    or contract.QROW32_B1_SPLIT2_FA2_SIZE != qrow.CANDIDATE_SIZE
):
    raise SystemExit("runtime and pass-sidecar binary pins differ")
PY
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the B1 gate" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "$BLOCK_MAP_CONTAINER" \
   && -z "$FR13_NEEDS_ALLOW" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "K64 ROOT=1 B1 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=authenticated_one_real_swe_verified_qrow32_split2_byte_gate\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_enabled=0\nqrow16_reference_served=1\ncandidate_returned=0\nmode=%s\ntask_count=1\ntask_id=%s\nsubset_sha256=%s\nbatch_size=1\nconcurrency=1\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\ndraft_vocab_blocks_sha256=%s\ncandidate_so_sha256=%s\ncandidate_so_bytes=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\nsource=%s\npatch_source_sha256=%s\nrunner_sha256=%s\nstarted=%s\n' \
  "$FIXED32_MODE" "$TASK_ID" "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" \
  "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$FA2_HEAD" \
  "$SOURCE_CLOSURE_SHA256" "$SOURCE_COMMIT" "$PATCH_SOURCE_SHA256" \
  "$RUNNER_SHA256" "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    LUMO_SWE_AUTOCOMMIT=0 FR13_FIXED32_B1_DIAGNOSTIC=1 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" FR13_NEEDS_ALLOW= \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_SFWD_GPU_TIMER=0 FR13_DFWD_GPU_TIMER=0 FR13_CFWD_GPU_TIMER=0 \
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
    FR13_FA2_QROW32_B1_LIVE_AB_ARM=split2 \
    FR13_FA2_QROW32_B1_LIVE_AB_INSTANCE_ID="$TASK_ID" \
    FR13_FA2_QROW32_B1_LIVE_AB_JSON=/logs/fr13_fa2_qrow32_b1_split2_live_paged_ab.json \
    FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
    FR13_FA2_QROW32_B1_SO_SHA256="$CANDIDATE_SHA256" \
    FR13_FA2_QROW32_B1_SO_SIZE="$CANDIDATE_BYTES" \
    FR13_FA2_QROW32_B1_FA2_HEAD="$FA2_HEAD" \
    FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
    FR13_FA2_QROW32_B1_SOURCE_COMMIT="$SOURCE_COMMIT" \
    FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$QROW32_B1_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$FIXED32_MODE" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during qrow32 gate" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during qrow32 gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "qrow32 gate runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

CONTAINER_ENV="$ARMDIR/container_env.txt"
for expected in \
  'FR13_FIXED32_MODE=hydra27_fixed32' \
  'FR13_FIXED32_B1_DIAGNOSTIC=1' \
  'ENFORCE_EAGER=0' \
  'CUDAGRAPH_MODE=FULL_AND_PIECEWISE' \
  'FR13_DRAFT_VOCAB_ROOT=1' \
  'FR13_DRAFT_VOCAB_K=65536' \
  'MAX_NUM_SEQS=1' \
  'SWE_CONCURRENCY=1' \
  'FR13_FA2_QROW16_LIVE_PAGED_AB=0' \
  'FR13_FA2_QROW16_PRODUCTION=0' \
  'FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0' \
  'FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0' \
  'FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0' \
  'FR13_FA2_QROW32_B1_LIVE_AB_ARM=split2' \
  "FR13_FA2_QROW32_B1_SO_SHA256=$CANDIDATE_SHA256" \
  "FR13_FA2_QROW32_B1_SO_SIZE=$CANDIDATE_BYTES" \
  "FR13_FA2_QROW32_B1_FA2_HEAD=$FA2_HEAD" \
  "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256=$SOURCE_CLOSURE_SHA256" \
  "FR13_FA2_QROW32_B1_SOURCE_COMMIT=$SOURCE_COMMIT" \
  "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256=$PATCH_SOURCE_SHA256"; do
  [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks exact qrow32 split2 gate pin: $expected" >&2; exit 4; }
done
unset expected

LIVE_RESULT="$ARMDIR/logs/fr13_fa2_qrow32_b1_split2_live_paged_ab.json"
DIAGNOSTIC="$ARMDIR/fixed32_b1_diagnostic.json"
HEALTH="$ARMDIR/health.json"
TRAFFIC_AUDIT="$ARMDIR/fixed32_chat_traffic_audit.json"
for artifact in "$LIVE_RESULT" "$DIAGNOSTIC" "$HEALTH" "$TRAFFIC_AUDIT"; do
  [[ -f "$artifact" && ! -L "$artifact" ]] \
    || { echo "qrow32 split2 gate artifact is missing or unsafe: $artifact" >&2; exit 4; }
done
unset artifact
"$PYTHON_BIN" - \
  "$LIVE_RESULT" "$DIAGNOSTIC" "$HEALTH" "$TRAFFIC_AUDIT" \
  "$QROW32_B1_FA2_SO" "$SOURCE_COMMIT" "$PATCH_SOURCE_SHA256" \
  "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" \
  "$ARMDIR/qrow32_split2_live_verification.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from scripts import fr13_qrow32_b1_pass_sidecar as qrow

(
    live_path, diagnostic_path, health_path, traffic_path, candidate_path,
    source_commit, patch_sha, subset_sha, block_sha, output_path,
) = sys.argv[1:]
live, live_raw = qrow.load_json(Path(live_path))
summary = qrow.validate_live_result(
    live,
    candidate_sha256=qrow.CANDIDATE_SHA256,
    arm=qrow.ARM,
    source_commit=source_commit,
    patch_source_sha256=patch_sha,
)
qrow.validate_candidate(Path(candidate_path), qrow.CANDIDATE_SHA256)
diagnostic, diagnostic_raw = qrow.load_json(Path(diagnostic_path))
if diagnostic != {
    "schema": "fr13-fixed32-b1-diagnostic-v1",
    "run_classification": "b1_diagnostic",
    "gate_eligible": False,
    "floor_acceptance_eligible": False,
    "max_num_seqs": 1,
    "swe_concurrency": 1,
    "subset_path": str(Path("config/fr13_fixed32/subset_b1_diagnostic_one.json").resolve()),
    "subset_sha256": subset_sha,
    "task_ids": [qrow.EXACT4_TASK_IDS[0]],
}:
    raise SystemExit("real SWE-Verified B1 diagnostic binding drifted")
health, health_raw = qrow.load_json(Path(health_path))
tasks = health.get("tasks")
if (
    health.get("swe_orchestrator_rc") != 0
    or not isinstance(tasks, list)
    or len(tasks) != 1
    or tasks[0].get("instance_id") != qrow.EXACT4_TASK_IDS[0]
    or tasks[0].get("codex_timed_out") is not False
    or tasks[0].get("verdict") != "resolved"
):
    raise SystemExit("real SWE-Verified task health drifted")
traffic, traffic_raw = qrow.load_json(Path(traffic_path))
checks = traffic.get("checks")
if (
    traffic.get("schema")
    not in {
        "fr13-fixed32-chat-task-provenance-audit-v2",
        "fr13-fixed32-chat-task-provenance-audit-v3",
    }
    or traffic.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
    or traffic.get("mode") != "hydra27_fixed32"
    or traffic.get("subset", {}).get("sha256") != subset_sha
    or traffic.get("subset", {}).get("task_ids") != [qrow.EXACT4_TASK_IDS[0]]
    or traffic.get("subset", {}).get("task_count") != 1
    or not isinstance(checks, dict)
    or not checks
    or any(value is not True for value in checks.values())
    or traffic.get("exact_proxy_engine_attempt_parity") is not True
    or set(traffic.get("tasks", {})) != {qrow.EXACT4_TASK_IDS[0]}
):
    raise SystemExit("authenticated real-task traffic audit drifted")
payload = {
    "schema": "fr13.fixed32.fa2_qrow32_split2_k64_b1_live_verification.v1",
    "status": "PASS",
    "suite": "SWE-Verified",
    "task_ids": [summary["instance_id"]],
    "subset_sha256": subset_sha,
    "block_map_sha256": block_sha,
    "batch_size": 1,
    "concurrency": 1,
    "physical_rows": 32,
    "topology": "hydra27_fixed32",
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "candidate_so_sha256": qrow.CANDIDATE_SHA256,
    "candidate_so_size": qrow.CANDIDATE_SIZE,
    "fa2_head": qrow.FA2_HEAD,
    "fa2_source_closure_sha256": qrow.SOURCE_CLOSURE_SHA256,
    "source_commit": source_commit,
    "patch_source_sha256": patch_sha,
    "live_result_sha256": hashlib.sha256(live_raw).hexdigest(),
    "diagnostic_binding_sha256": hashlib.sha256(diagnostic_raw).hexdigest(),
    "health_sha256": hashlib.sha256(health_raw).hexdigest(),
    "traffic_audit_sha256": hashlib.sha256(traffic_raw).hexdigest(),
    "layers_sha256": summary["layers_sha256"],
    "served_return": "qrow16 captured graph output unchanged",
    "performance_measurement": False,
}
Path(output_path).write_text(
    json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
PY

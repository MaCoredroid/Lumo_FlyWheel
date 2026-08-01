#!/usr/bin/env bash
# Real SWE-Verified exact4 shadow byte gate for the full-vocab B1-B4 draft head.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the canonical FA2 shared object}"
: "${FR13_DRAFT_HEAD_M1_SO:?set FR13_DRAFT_HEAD_M1_SO to the B1-B4 SO}"
: "${FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION:?set the pinned build attestation}"
: "${FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT:?pin the exact source commit}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
CANDIDATE_SOURCE=csrc/fr13_bf16_gemvx_b1_b4.cu
PATCHER=scripts/fr13_phase4_patch_vllm_tree_gdn_b1_b4.py
CANONICAL_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
CANONICAL_FA2_SIZE=299183936
B4_KV_CACHE_MEMORY_BYTES=42949672960
RUNROOT_ABS=$(realpath -m "$RUNROOT")

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT" == output/* && "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must be repo-relative below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drifted" >&2; exit 2; }
for path in \
  "$FORKED_FA2_SO" \
  "$FR13_DRAFT_HEAD_M1_SO" \
  "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION"; do
  [[ -f "$path" && ! -L "$path" && "$path" == /* ]] \
    || { echo "binary must be an absolute regular non-symlink: $path" >&2; exit 2; }
done
[[ "$(stat -c %s "$FORKED_FA2_SO")" == "$CANONICAL_FA2_SIZE" \
   && "$(sha256sum "$FORKED_FA2_SO" | cut -d' ' -f1)" \
      == "$CANONICAL_FA2_SHA256" ]] \
  || { echo "B1-B4 gate requires the canonical FA2 binary identity" >&2; exit 2; }

ARM="hydra27_fixed32_${TAG}"
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)
CANDIDATE_SOURCE_SHA256=$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)
PATCHER_SHA256=$(sha256sum "$PATCHER" | cut -d' ' -f1)
CANDIDATE_SO_SHA256=$(sha256sum "$FR13_DRAFT_HEAD_M1_SO" | cut -d' ' -f1)
CANDIDATE_SO_SIZE=$(stat -c %s "$FR13_DRAFT_HEAD_M1_SO")
BUILD_ATTESTATION_SHA256=$(
  sha256sum "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" | cut -d' ' -f1
)
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && "$SOURCE_COMMIT" == "$FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT" \
   && "$CANDIDATE_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$PATCHER_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$BUILD_ATTESTATION_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$CANDIDATE_SO_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$CANDIDATE_SO_SIZE" -gt 0 ]]
[[ -z "$(git status --porcelain=v1)" ]] \
  || { echo "B1-B4 gate requires a clean checkout" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "B1-B4 gate requires no existing Docker containers" >&2; exit 2; }

"$PYTHON_BIN" - \
  "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" \
  "$CANDIDATE_SOURCE_SHA256" "$CANDIDATE_SO_SHA256" \
  "$CANDIDATE_SO_SIZE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
expected = {
    "schema": "fr13.fixed32.bf16_gemvx_b1_b4_build.v1",
    "status": "BUILT_UNQUALIFIED",
    "performance_measurement": False,
    "byte_equality_claim": False,
    "production_default_enabled": False,
    "torch_version": "2.10.0+cu130",
    "cuda_release": "13.0",
    "cuda_arch": "12.1a",
}
if any(payload.get(key) != value for key, value in expected.items()):
    raise SystemExit("B1-B4 build attestation identity drifted")
if payload.get("source", {}).get("sha256") != sys.argv[2]:
    raise SystemExit("B1-B4 build source binding drifted")
binary = payload.get("binary", {})
if binary.get("sha256") != sys.argv[3] or binary.get("bytes") != int(sys.argv[4]):
    raise SystemExit("B1-B4 build binary binding drifted")
contract = payload.get("kernel_contract", {})
if (
    contract.get("supported_batch_sizes") != [1, 2, 3, 4]
    or contract.get("candidate_launches_per_head") != 1
    or contract.get("logical_weight_element_loads_per_head") != 1271398400
):
    raise SystemExit("B1-B4 build kernel contract drifted")
PY

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_DRAFT_VOCAB_BLOCKS=
export FR13_MANDATORY_WEIGHT_BYTES=42025179008
export FR13_WEIGHT_FLOOR_MS=153.938384645
export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"
export FR13_FLOOR_ORDER=TH

source scripts/fr13_canonical_env.sh
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_DRAFT_VOCAB_BLOCKS=
export FR13_MANDATORY_WEIGHT_BYTES=42025179008
export FR13_WEIGHT_FLOOR_MS=153.938384645
export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_DRAFT_VOCAB_BLOCKS=

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b1_b4_shadow_byte_gate\nperformance_measurement=0\nfloor_acceptance_eligible=0\nproduction_eligible=0\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\ncandidate_source_sha256=%s\npatcher_sha256=%s\nbuild_attestation_sha256=%s\ncandidate_so_sha256=%s\nfa2_sha256=%s\nstarted=%s\n' \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$CANDIDATE_SOURCE_SHA256" "$PATCHER_SHA256" \
  "$BUILD_ATTESTATION_SHA256" "$CANDIDATE_SO_SHA256" \
  "$CANONICAL_FA2_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"

if OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
  KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
  FR13_FIXED32_B1_DIAGNOSTIC=0 \
  FR10_METRICS=1 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
  FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
  FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
  FR13_DRAFT_HEAD_M32_TIMING_ARM=0 \
  FR13_DRAFT_HEAD_M1_LIVE_AB=1 FR13_DRAFT_HEAD_M1_MAX_BATCH=4 \
  FR13_DRAFT_HEAD_M1_PRODUCTION=0 FR13_DRAFT_HEAD_M1_TIMING_ARM=0 \
  FR13_DRAFT_HEAD_M1_INSTANCE_ID= \
  FR13_DRAFT_HEAD_M1_LIVE_JSON=/logs/fr13_draft_head_b1_b4.live.json \
  FR13_DRAFT_HEAD_M1_SO="$FR13_DRAFT_HEAD_M1_SO" \
  FR13_DRAFT_HEAD_M1_SO_SHA256="$CANDIDATE_SO_SHA256" \
  FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION="$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" \
  FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION_SHA256="$BUILD_ATTESTATION_SHA256" \
  FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
  FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
  FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
  FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
  FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
  FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
  FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
  FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
  FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
  FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
  FR13_FIXED32_ATTRIBUTION_ONLY=0 \
  FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0" \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  FR13_FA2_QROW16_SO_SHA256="$CANONICAL_FA2_SHA256" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh \
    "$ARM" hydra27_fixed32 "$SUBSET" \
    > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during B1-B4 gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)" == "$RUNNER_SHA256" \
   && "$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)" == "$CANDIDATE_SOURCE_SHA256" \
   && "$(sha256sum "$PATCHER" | cut -d' ' -f1)" == "$PATCHER_SHA256" \
   && "$(sha256sum "$FR13_DRAFT_HEAD_M1_SO" | cut -d' ' -f1)" == "$CANDIDATE_SO_SHA256" \
   && "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
   && -z "$(git status --porcelain=v1)" ]] \
  || { echo "B1-B4 gate identity changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

"$PYTHON_BIN" - \
  "$RUNROOT_ABS/$ARM" "$CANDIDATE_SOURCE_SHA256" "$PATCHER_SHA256" \
  "$BUILD_ATTESTATION_SHA256" "$CANDIDATE_SO_SHA256" \
  "$SUBSET_SHA256" "$RUNNER_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

arm = Path(sys.argv[1])
source_sha, patcher_sha, build_sha, so_sha = sys.argv[2:6]
subset_sha, runner_sha = sys.argv[6:8]
live_path = arm / "logs" / "fr13_draft_head_b1_b4.live.json"
live_raw = live_path.read_bytes()
live = json.loads(live_raw.decode("ascii"))
expected_tasks = {
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
}
health = json.loads((arm / "health.json").read_text(encoding="utf-8"))
tasks = health.get("tasks")
if (
    not isinstance(tasks, list)
    or len(tasks) != 4
    or {task.get("instance_id") for task in tasks} != expected_tasks
    or health.get("swe_orchestrator_rc") != 0
):
    raise SystemExit("B1-B4 gate did not complete canonical exact4")
if (
    live.get("schema") != "fr13.fixed32.draft_head_full_b1_b4_live_ab.v1"
    or live.get("status") != "PASS"
    or live.get("suite") != "SWE-Verified"
    or live.get("concurrency") != 4
    or live.get("batch_size") != 4
    or live.get("candidate_source_sha256") != source_sha
    or live.get("patcher_sha256") != patcher_sha
    or live.get("build_attestation_sha256") != build_sha
    or live.get("binary", {}).get("sha256") != so_sha
    or live.get("raw_bf16_mismatches") != 0
    or live.get("performance_measurement") is not False
    or live.get("acceptance_eligible") is not False
    or live.get("production_default_enabled") is not False
    or live.get("served_return")
    != "stock reference BF16 logits computed first and unchanged"
):
    raise SystemExit("B1-B4 live shadow evidence failed")
completed = live.get("completed_events")
per_head = live.get("per_head")
if (
    not isinstance(completed, int)
    or completed < 4
    or not isinstance(per_head, list)
    or len(per_head) != 5
    or any(row.get("full_logit_comparisons") != completed for row in per_head)
    or live.get("full_logit_comparisons") != completed * 5
):
    raise SystemExit("B1-B4 full-logit comparison census failed")
verdict = {
    "schema": "fr13.fixed32.draft_head_b1_b4.exact4_shadow_gate.v1",
    "status": "PASS",
    "suite": "SWE-Verified",
    "task_ids": sorted(expected_tasks),
    "subset_sha256": subset_sha,
    "candidate_source_sha256": source_sha,
    "patcher_sha256": patcher_sha,
    "build_attestation_sha256": build_sha,
    "candidate_so_sha256": so_sha,
    "runner_sha256": runner_sha,
    "live_result_sha256": hashlib.sha256(live_raw).hexdigest(),
    "completed_request_rows": completed,
    "full_logit_comparisons": completed * 5,
    "raw_bf16_mismatches": 0,
    "reference_always_served": True,
    "performance_measurement": False,
    "floor_acceptance_eligible": False,
    "production_default_enabled": False,
}
out = arm.parent / "b1_b4_exact4_shadow_gate_verdict.json"
out.write_text(
    json.dumps(verdict, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
print(json.dumps(verdict, sort_keys=True))
PY

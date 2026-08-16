#!/usr/bin/env bash
# One real SWE-Verified K64 ROOT=1 B1 paged byte gate for qrow16 division-free.
# The stock FULL graph is always served; this script emits no timing samples.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_QROW16_DIVFREE_LIVE_GATE:-0}" in
  1) ;;
  0)
    echo "qrow16 division-free live gate is disabled; set FR13_RUN_QROW16_DIVFREE_LIVE_GATE=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_QROW16_DIVFREE_LIVE_GATE must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned division-free qrow16 SO}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=hydra27_fixed32
TASK_ID=astropy__astropy-12907
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
CANDIDATE_SHA256=106e54d1c82ec7ce7576cbb44bb4aa2342b2985bb58e97aeeca5503275bee3e2
CANDIDATE_BYTES=299491544
MANDATORY_WEIGHT_BYTES=25210209416
MANDATORY_WEIGHT_FLOOR_MS=92.345089436
ONE_SIDED_U95_CAP_MS=106.1968528514
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="${FIXED32_MODE}_fa2_qrow16_divfree_k64_b1_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the pinned division-free qrow16 binary" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "canonical B1 task or K64 block map drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the B1 gate" >&2; exit 2; }

"$PYTHON_BIN" - "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" <<'PY'
import sys

sys.path.insert(0, "scripts")
import fr13_fixed32_contract as contract

if (
    contract.QROW16_DIVFREE_FA2_SHA256 != sys.argv[1]
    or contract.QROW16_DIVFREE_FA2_SIZE != int(sys.argv[2])
):
    raise SystemExit("division-free qrow16 contract pin drifted")
PY

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
printf 'classification=one_real_swe_verified_k64_root1_b1_qrow16_divfree_paged_byte_gate\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_enabled=0\nreference_always_served=1\ncandidate_returned=0\nmode=%s\ntask_count=1\ntask_id=%s\nsubset_sha256=%s\nbatch_size=1\nconcurrency=1\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\ncandidate_so_sha256=%s\ncandidate_so_bytes=%s\nsource=%s\nrunner_sha256=%s\nstarted=%s\n' \
  "$FIXED32_MODE" "$TASK_ID" "$SUBSET_SHA256" "$BLOCK_MAP_CONTAINER" \
  "$BLOCK_MAP_SHA256" "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

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
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=1 \
    FR13_FA2_QROW16_LIVE_PAGED_AB_INSTANCE_ID="$TASK_ID" \
    FR13_FA2_QROW16_LIVE_PAGED_AB_JSON=/logs/fr13_fa2_qrow16_divfree_live_paged_ab.json \
    FR13_FA2_QROW16_SO_SHA256="$CANDIDATE_SHA256" \
    FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
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
  || { echo "runtime/source manifest changed during qrow16 gate" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during qrow16 gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "qrow16 gate runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

CONTAINER_ENV="$ARMDIR/container_env.txt"
for expected in \
  'FR13_FIXED32_B1_DIAGNOSTIC=1' \
  'FR13_DRAFT_VOCAB_ROOT=1' \
  'FR13_DRAFT_VOCAB_K=65536' \
  "FR13_DRAFT_VOCAB_BLOCKS=$BLOCK_MAP_CONTAINER" \
  'MAX_NUM_SEQS=1' \
  'SWE_CONCURRENCY=1' \
  'FR13_FA2_QROW16_LIVE_PAGED_AB=1' \
  "FR13_FA2_QROW16_SO_SHA256=$CANDIDATE_SHA256"; do
  [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks exact qrow16 B1 gate pin: $expected" >&2; exit 4; }
done
unset expected

LIVE_RESULT="$ARMDIR/logs/fr13_fa2_qrow16_divfree_live_paged_ab.json"
DIAGNOSTIC="$ARMDIR/fixed32_b1_diagnostic.json"
"$PYTHON_BIN" - \
  "$LIVE_RESULT" "$DIAGNOSTIC" "$FORKED_FA2_SO" "$SOURCE_COMMIT" \
  "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" "$ARMDIR/qrow16_divfree_live_verification.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import fr13_fixed32_contract as contract
import fr13_qrow16_pass_sidecar as qrow

live_path, diagnostic_path, candidate_path, source_commit = map(Path, sys.argv[1:5])
subset_sha256, block_map_sha256, output_path = sys.argv[5], sys.argv[6], Path(sys.argv[7])
live, live_raw = qrow.load_json(live_path)
summary = qrow.validate_live_result(
    live,
    candidate_sha256=contract.QROW16_DIVFREE_FA2_SHA256,
)
if live.get("draft_vocab_root") != 1 or live.get("draft_vocab_k") != 65536:
    raise SystemExit("qrow16 live result is not bound to K64 ROOT=1")
diagnostic, diagnostic_raw = qrow.load_json(diagnostic_path)
if diagnostic != {
    "schema": "fr13-fixed32-b1-diagnostic-v1",
    "run_classification": "b1_diagnostic",
    "gate_eligible": False,
    "floor_acceptance_eligible": False,
    "max_num_seqs": 1,
    "swe_concurrency": 1,
    "subset_path": str(Path("config/fr13_fixed32/subset_b1_diagnostic_one.json").resolve()),
    "subset_sha256": subset_sha256,
    "task_ids": ["astropy__astropy-12907"],
}:
    raise SystemExit("real SWE-Verified B1 diagnostic binding drifted")
if (
    candidate_path.stat().st_size != contract.QROW16_DIVFREE_FA2_SIZE
    or qrow.sha256_file(candidate_path) != contract.QROW16_DIVFREE_FA2_SHA256
):
    raise SystemExit("qrow16 division-free candidate identity drifted")
payload = {
    "schema": "fr13.fixed32.fa2_qrow16_divfree_k64_b1_live_verification.v1",
    "status": "PASS",
    "suite": "SWE-Verified",
    "task_ids": [summary["instance_id"]],
    "subset_sha256": subset_sha256,
    "block_map_sha256": block_map_sha256,
    "batch_size": 1,
    "concurrency": 1,
    "physical_rows": 32,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "candidate_so_sha256": contract.QROW16_DIVFREE_FA2_SHA256,
    "source_commit": str(source_commit),
    "live_result_sha256": hashlib.sha256(live_raw).hexdigest(),
    "diagnostic_binding_sha256": hashlib.sha256(diagnostic_raw).hexdigest(),
    "output_sha256": summary["output_sha256"],
    "lse_sha256": summary["lse_sha256"],
    "served_return": "stock captured graph output unchanged",
    "performance_measurement": False,
}
output_path.write_text(
    json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
PY

#!/usr/bin/env bash
# Canonical exact4 K64/root1 B1 full-step timing pair for FP8 quant regcache.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 binary}"
: "${FP8_QUANT_SO:?set FP8_QUANT_SO to the pinned regcache runtime binary}"
: "${FP8_QUANT_SO_SHA256:?set FP8_QUANT_SO_SHA256 to its raw SHA-256}"
: "${FP8_QUANT_PASS:?set FP8_QUANT_PASS to the real-B1 production PASS}"
: "${FP8_QUANT_PASS_SHA256:?set FP8_QUANT_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
STOCK_ARM="hydra27_fixed32_k64_fp8_quant_stock_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_k64_fp8_quant_regcache_${TAG}"
PATCH_SOURCE=scripts/fr13_patch_fp8_quant_fixed32.py
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
MANDATORY_WEIGHT_BYTES=25210209416
WEIGHT_FLOOR_MS=92.345089436
ONE_SIDED_U95_CAP_MS=106.1968528514

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
for required in "$FORKED_FA2_SO" "$FP8_QUANT_SO" "$FP8_QUANT_PASS"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "required input is not an absolute regular file: $required" >&2; exit 2; }
done
unset required
[[ "$FP8_QUANT_SO_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$FP8_QUANT_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$FP8_QUANT_SO" | awk '{print $1}')" == "$FP8_QUANT_SO_SHA256" \
   && "$(sha256sum "$FP8_QUANT_PASS" | awk '{print $1}')" == "$FP8_QUANT_PASS_SHA256" ]] \
  || { echo "FP8 quant binary or PASS identity drifted" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "canonical exact4 subset or K64 block map drifted" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_fp8_quant_regcache_runtime.py verify-binary \
  "$FP8_QUANT_SO" --expected-sha256 "$FP8_QUANT_SO_SHA256" >/dev/null
"$PYTHON_BIN" scripts/fr13_fp8_quant_regcache_pass.py verify \
  --sidecar "$FP8_QUANT_PASS" \
  --expected-sidecar-sha256 "$FP8_QUANT_PASS_SHA256" \
  --candidate-so "$FP8_QUANT_SO" \
  --expected-candidate-sha256 "$FP8_QUANT_SO_SHA256" \
  --patch-source "$PATCH_SOURCE" >/dev/null

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=real_swe_verified_exact4_k64_root_b1_fp8_quant_timing_pair\ntask_count=4\nbatch_size=1\nconcurrency=1\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=FR13_FIXED32_B1_FP8_QUANT_REGCACHE_0_to_1\nmandatory_weight_bytes=%s\nweight_floor_ms=%s\none_sided_u95_cap_ms=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nblock_map_sha256=%s\ncandidate_sha256=%s\nproduction_pass_sha256=%s\nstarted=%s\n' \
  "$MANDATORY_WEIGHT_BYTES" "$WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" \
  "$STOCK_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" "$FP8_QUANT_SO_SHA256" \
  "$FP8_QUANT_PASS_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$FP8_QUANT_SO" | awk '{print $1}')" == "$FP8_QUANT_SO_SHA256" \
     && "$(sha256sum "$FP8_QUANT_PASS" | awk '{print $1}')" == "$FP8_QUANT_PASS_SHA256" ]] \
    || { echo "timing input changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifests || { local mrc=$?; (( rc == 0 )) && rc=$mrc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local selector=$2
  local pass_path=
  local pass_sha=
  if [[ "$selector" == "1" ]]; then
    pass_path=$FP8_QUANT_PASS
    pass_sha=$FP8_QUANT_PASS_SHA256
  fi
  echo "===== $arm: real exact4 B1 FP8 quant selector=$selector ====="
  (
    export BSIZE=1 CONC=1 WALL=0
    export FR13_DRAFT_VOCAB_ROOT=1
    export FR13_DRAFT_VOCAB_K=65536
    export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
    export FR13_NEEDS_ALLOW=
    export FR13_FLOOR_ORDER=HT
    source scripts/fr13_canonical_env.sh
    run_variant() { :; }
    source "$SEQUENCE"
    unset -f run_variant
    [[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
       && "$FR13_WEIGHT_FLOOR_MS" == "$WEIGHT_FLOOR_MS" ]] \
      || { echo "K64/root1 B1 floor contract drifted" >&2; exit 2; }

    env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      LUMO_SWE_AUTOCOMMIT=0 \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
      FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" FR13_NEEDS_ALLOW= \
      FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$WEIGHT_FLOOR_MS" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_FIXED32_B1_FP8_QUANT_REGCACHE="$selector" \
      FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO="$FP8_QUANT_SO" \
      FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO_SHA256="$FP8_QUANT_SO_SHA256" \
      FR13_FIXED32_B1_FP8_QUANT_REGCACHE_PASS_JSON="$pass_path" \
      FR13_FIXED32_B1_FP8_QUANT_REGCACHE_PASS_SHA256="$pass_sha" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_DRAFT_HEAD_FP8=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$FORKED_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" hydra27_fixed32 "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1
    "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
      --arm "$arm" \
      --out-root "$RUNROOT_ABS/$arm/swe_out" \
      --expected-tok-per-draft 31 \
      --batch-size 1 \
      --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  )
  printf 'arm=%s selector=%s serve_rc=0 ended=%s\n' \
    "$arm" "$selector" "$(date -u +%FT%TZ)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after candidate arm" >&2; exit 2; }

finalize_manifests
"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_b1_fp8_quant_regcache.binary.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_b1_fp8_quant_regcache.binary.json" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$STOCK_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" "$FP8_QUANT_SO_SHA256" \
  "$FP8_QUANT_PASS_SHA256" "$WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

subset_path, stock_path, candidate_path, stock_binary_path, candidate_binary_path, out_path = map(
    Path, sys.argv[1:7]
)
stock_arm, candidate_arm, source, runner_sha = sys.argv[7:11]
subset_sha, block_map_sha, binary_sha, pass_sha = sys.argv[11:15]
floor_ms, cap_ms = map(float, sys.argv[15:17])


def load(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("ascii")), raw


subset, _ = load(subset_path)
stock, stock_raw = load(stock_path)
candidate, candidate_raw = load(candidate_path)
stock_binary, stock_binary_raw = load(stock_binary_path)
candidate_binary, candidate_binary_raw = load(candidate_binary_path)
task_ids = sorted(subset["instance_ids"])


def validate_measure(payload, arm):
    if (
        payload.get("schema") != "fr13.measure.deploy_speed.v1"
        or payload.get("kind") != "speed"
        or payload.get("instrument") != "OFF"
        or payload.get("regime") != "deployment"
        or payload.get("arm") != arm
        or payload.get("batch_size") != 1
        or payload.get("n_tasks") != 4
        or sorted(payload.get("task_instance_ids", [])) != task_ids
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_vocab_k") != 65536
        or payload.get("mandatory_weight_bytes") != 25210209416
        or payload.get("weight_floor_ms") != floor_ms
    ):
        raise SystemExit(f"{arm} is not a canonical exact4 K64/root1 B1 measure")
    fields = (
        "step_wall_ms",
        "measured_tps_fullstep_wall",
        "accept_per_event",
        "committed_per_event",
        "s_per_fwd_gpu",
        "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step",
    )
    for field in fields:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f"{arm} lacks finite {field}")
    return {field: payload[field] for field in fields}


stock_metrics = validate_measure(stock, stock_arm)
candidate_metrics = validate_measure(candidate, candidate_arm)
for payload, selector, production, sidecar in (
    (stock_binary, "0", False, None),
    (candidate_binary, "1", True, pass_sha),
):
    if (
        payload.get("schema") != "fr13.fixed32.b1_fp8_quant_regcache.binary.v1"
        or payload.get("status") != "INSTALLED"
        or payload.get("selector") != selector
        or payload.get("production_enabled") is not production
        or payload.get("candidate_sha256") != binary_sha
        or payload.get("production_sidecar_sha256") != sidecar
    ):
        raise SystemExit("FP8 quant runtime attestation drifted during timing")

summary = {
    "schema": "fr13.fixed32.b1_fp8_quant_regcache.exact4_timing.v1",
    "status": "COMPLETE",
    "run_classification": "real_swe_verified_exact4_k64_root_b1_fp8_quant_timing_pair",
    "task_ids": task_ids,
    "task_count": 4,
    "batch_size": 1,
    "concurrency": 1,
    "physical_rows": 32,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "timing_eligible": True,
    "formal_floor_acceptance_eligible": False,
    "formal_floor_acceptance_reason": "exact4 screen; canonical exact16 U95 remains required",
    "only_arm_delta": "FR13_FIXED32_B1_FP8_QUANT_REGCACHE=0 to 1",
    "weight_floor_ms": floor_ms,
    "one_sided_u95_cap_ms": cap_ms,
    "stock": {"arm": stock_arm, **stock_metrics},
    "candidate": {"arm": candidate_arm, **candidate_metrics},
    "candidate_over_stock_step_wall": candidate_metrics["step_wall_ms"] / stock_metrics["step_wall_ms"],
    "candidate_over_floor": candidate_metrics["step_wall_ms"] / floor_ms,
    "candidate_gap_to_cap_ms": candidate_metrics["step_wall_ms"] - cap_ms,
    "candidate_tps_over_stock": candidate_metrics["measured_tps_fullstep_wall"] / stock_metrics["measured_tps_fullstep_wall"],
    "source_commit": source,
    "runner_sha256": runner_sha,
    "subset_sha256": subset_sha,
    "block_map_sha256": block_map_sha,
    "candidate_sha256": binary_sha,
    "production_pass_sha256": pass_sha,
    "stock_measure_sha256": hashlib.sha256(stock_raw).hexdigest(),
    "candidate_measure_sha256": hashlib.sha256(candidate_raw).hexdigest(),
    "stock_binary_attestation_sha256": hashlib.sha256(stock_binary_raw).hexdigest(),
    "candidate_binary_attestation_sha256": hashlib.sha256(candidate_binary_raw).hexdigest(),
}
out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="ascii")
print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

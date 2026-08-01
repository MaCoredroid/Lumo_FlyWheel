#!/usr/bin/env bash
# Exact4 real SWE-Verified B4 full-wall timing: stock CUTLASS vs persistent M128.
# This paired screen is not the formal statistical hardware-floor acceptance gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"
: "${CUTLASS_B4_SO:?set CUTLASS_B4_SO to the pinned persistent-M128 binary}"
: "${CUTLASS_B4_PASS_JSON:?set CUTLASS_B4_PASS_JSON to an authenticated exact4 byte-gate PASS}"
: "${CUTLASS_B4_PASS_SHA256:?set CUTLASS_B4_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
CANDIDATE_SHA256=6988f6a994c29e9196b6addc039e1d63bf08c32f268f9be3d2f14c5d863be1de
CANDIDATE_BYTES=112698512
PATCH_SOURCE=scripts/fr13_patch_cutlass_fixed32_wave.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
B4_KV_CACHE_MEMORY_BYTES=42949672960
TIMING_KIND=hydra27_fixed32
STOCK_ARM="hydra27_fixed32_cutlass_stock_b4_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_cutlass_persistent_m128_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in "$STOCK_FA2_SO" "$CUTLASS_B4_SO" "$CUTLASS_B4_PASS_JSON"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "timing input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
[[ "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(stat -c '%s' "$CUTLASS_B4_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$CUTLASS_B4_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "CUTLASS_B4_SO is not the pinned persistent-M128 candidate" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ "$CUTLASS_B4_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$CUTLASS_B4_PASS_JSON" | awk '{print $1}')" == "$CUTLASS_B4_PASS_SHA256" ]] \
  || { echo "CUTLASS B4 live PASS identity mismatch" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py validate \
  --live-result "$CUTLASS_B4_PASS_JSON" \
  --expected-live-sha256 "$CUTLASS_B4_PASS_SHA256" \
  --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --candidate-selector persistent_b4_m128 >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0'
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "42025179008" \
   && "$FR13_WEIGHT_FLOOR_MS" == "153.9383846446886" ]] \
  || { echo "canonical B4 full-vocabulary floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=real_swe_verified_exact4_b4_timing_candidate\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=CUTLASS_stock_to_persistent_b4_m128\nbatch_size=4\nconcurrency=4\nfixed_rows=128\ndraft_vocab_root=0\ndraft_vocab_k=0\nfr13_needs_allow=FR13_DRAFT_VOCAB_K=0\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=177.0291423413919\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstock_fa2_bytes=%s\ncandidate_sha256=%s\ncandidate_bytes=%s\nlive_pass_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" "$$" \
  "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" \
  "$RUNNER_SHA256" "$SUBSET_SHA256" "$STOCK_FA2_SHA256" "$STOCK_FA2_BYTES" \
  "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$CUTLASS_B4_PASS_SHA256" \
  "$B4_KV_CACHE_MEMORY_BYTES" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "B4 CUTLASS timing runner changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then :; else
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
  local selector=stock
  local candidate_so=""
  local pass_json=""
  local pass_sha=""
  if [[ "$production" == "1" ]]; then
    selector=persistent_b4_m128
    candidate_so=$CUTLASS_B4_SO
    pass_json=$CUTLASS_B4_PASS_JSON
    pass_sha=$CUTLASS_B4_PASS_SHA256
  fi
  echo "===== $arm: exact4 B4 CUTLASS production=$production ====="
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0 \
      FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0' \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_CUTLASS_WAVE="$selector" \
      FR13_FIXED32_CUTLASS_WAVE_SO="$candidate_so" \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$pass_json" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$pass_sha" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$TIMING_KIND" "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi
  local container_env="$RUNROOT_ABS/$arm/container_env.txt"
  [[ -f "$container_env" && ! -L "$container_env" ]] \
    || { echo "$arm lacks a regular container environment artifact" >&2; return 4; }
  [[ "$(grep -Fxc 'FR13_FIXED32_MODE=hydra27_fixed32' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_DRAFT_VOCAB_ROOT=0' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_DRAFT_VOCAB_K=0' "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the canonical B4 full-vocabulary contract" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 --batch-size 4 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

STOCK_ATTESTATION="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_cutlass_streamk_binary.json"
CANDIDATE_ATTESTATION="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_streamk_binary.json"
CANDIDATE_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_streamk.production_pass.json"
CANDIDATE_SELECTOR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_wave.selector"
[[ ! -e "$STOCK_ATTESTATION" && ! -L "$STOCK_ATTESTATION" ]] \
  || { echo "stock arm emitted a CUTLASS candidate attestation" >&2; exit 4; }
[[ -f "$CANDIDATE_ATTESTATION" && ! -L "$CANDIDATE_ATTESTATION" \
   && -f "$CANDIDATE_SIDECAR" && ! -L "$CANDIDATE_SIDECAR" \
   && -f "$CANDIDATE_SELECTOR" && ! -L "$CANDIDATE_SELECTOR" \
   && "$(<"$CANDIDATE_SELECTOR")" == "persistent_b4_m128" ]] \
  || { echo "candidate arm lacks persistent-M128 identity artifacts" >&2; exit 4; }
CANDIDATE_SIDECAR_SHA256=$(sha256sum "$CANDIDATE_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py verify \
  --sidecar "$CANDIDATE_SIDECAR" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
  --candidate-selector persistent_b4_m128 >/dev/null
"$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py attestation \
  --attestation "$CANDIDATE_ATTESTATION" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  > "$RUNROOT_ABS/$CANDIDATE_ARM/cutlass_b4_production_binding.json"

finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/cutlass_b4_production_binding.json" \
  "$RUNROOT_ABS/timing_summary.json" "$CANDIDATE_SIDECAR_SHA256" \
  "$CUTLASS_B4_PASS_SHA256" "$CANDIDATE_SHA256" "$STOCK_FA2_SHA256" <<'PY'
import json
import math
import sys
from pathlib import Path

subset_path, stock_path, candidate_path, binding_path, out_path = map(Path, sys.argv[1:6])
sidecar_sha256, live_sha256, candidate_sha256, fa2_sha256 = sys.argv[6:10]
task_ids = sorted(json.loads(subset_path.read_text(encoding="ascii"))["instance_ids"])
stock = json.loads(stock_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
binding = json.loads(binding_path.read_text(encoding="ascii"))

def positive(record, key):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{key} is missing from full-wall timing evidence")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{key} is not finite and positive")
    return value

def validate(record, label):
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("batch_size") != 4
        or record.get("n_tasks") != 4
        or sorted(record.get("task_instance_ids", [])) != task_ids
        or record.get("draft_vocab_root") != 0
        or record.get("draft_vocab_k") != 0
        or record.get("mandatory_weight_bytes") != 42_025_179_008
        or not math.isclose(
            float(record.get("weight_floor_ms", math.nan)),
            153.9383846446886,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or record.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise SystemExit(f"{label} deploy-speed provenance is not exact4 B4")
    for key in (
        "measured_tps_fullstep_wall", "step_wall_ms", "accept_per_event",
        "committed_per_event", "wall_steps_measured", "events_per_step",
        "s_per_fwd_gpu", "drafter_gpu_ms_per_step", "committer_gpu_ms_per_step",
        "weight_floor_ms", "floor_ms", "floor_ratio",
    ):
        positive(record, key)

validate(stock, "stock")
validate(candidate, "candidate")
if (
    binding.get("schema") != "fr13.fixed32.cutlass_b4.production_binding.v1"
    or binding.get("status") != "BOUND"
    or binding.get("selector") != "persistent_b4_m128"
    or binding.get("candidate_sha256") != candidate_sha256
    or binding.get("production_sidecar_sha256") != sidecar_sha256
    or binding.get("live_result_sha256") != live_sha256
    or binding.get("qualified_fixed_rows") != 128
    or binding.get("qualified_topology") != "hydra27_fixed32"
    or binding.get("qualified_comparison_call_limit") != 320
):
    raise SystemExit("candidate lacks persistent-M128 production binding")
stock_wall = positive(stock, "step_wall_ms")
candidate_wall = positive(candidate, "step_wall_ms")
stock_tps = positive(stock, "measured_tps_fullstep_wall")
candidate_tps = positive(candidate, "measured_tps_fullstep_wall")
stock_floor = positive(stock, "floor_ms")
candidate_floor = positive(candidate, "floor_ms")
if not math.isclose(stock_floor, candidate_floor, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit("stock and candidate floor values differ")
summary = {
    "schema": "fr13.fixed32.cutlass_persistent_b4_m128.full_wall_timing_pair.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_b4_timing",
    "task_count": 4,
    "batch_size": 4,
    "concurrency": 4,
    "arm": "hydra27_fixed32",
    "task_ids": task_ids,
    "decision_metric": "measured_tps_fullstep_wall",
    "stock_reference": {
        "selector": "stock",
        "fa2_sha256": fa2_sha256,
        "step_wall_ms": stock_wall,
        "measured_tps_fullstep_wall": stock_tps,
        "accepted_drafts_per_event": float(stock["accept_per_event"]),
        "step_wall_to_optimistic_floor_ratio": float(stock["floor_ratio"]),
    },
    "candidate": {
        "selector": "persistent_b4_m128",
        "candidate_sha256": candidate_sha256,
        "live_result_sha256": live_sha256,
        "production_sidecar_sha256": sidecar_sha256,
        "step_wall_ms": candidate_wall,
        "measured_tps_fullstep_wall": candidate_tps,
        "accepted_drafts_per_event": float(candidate["accept_per_event"]),
        "step_wall_to_optimistic_floor_ratio": float(candidate["floor_ratio"]),
    },
    "optimistic_floor_ms": stock_floor,
    "optimistic_floor_is_full_step_hardware_floor": False,
    "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
    "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
    "formal_floor_acceptance_eligible": False,
    "formal_floor_acceptance_reason": (
        "paired exact4 timing candidate only; run the canonical statistical "
        "Tail/Hydra floor gate after a positive screen"
    ),
    "production_default_enabled": False,
}
temporary = out_path.with_name(out_path.name + ".tmp")
temporary.write_text(
    json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
    encoding="ascii",
)
temporary.replace(out_path)
print(json.dumps(summary, sort_keys=True))
PY

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

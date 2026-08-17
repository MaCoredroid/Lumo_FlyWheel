#!/usr/bin/env bash
# FR14 — the hydra27 B1 lever pair: gqa_pair OFF vs gqa_pair ON, K0 production shape.
#
# WHY A PAIR AND NOT A SINGLE ARM. The FA2 qrow32 B1 selector is legal only in
# hydra27_fixed32, so the levered arm is a HYDRA27 arm. Every FR14 headline so
# far is tail6, which makes tail6 the wrong control: a tail6-vs-hydra27
# comparison would confound the lever with the tree topology (23 active nodes vs
# 27, different mask). Both arms here are hydra27, same K0 identity, same floor,
# same boundary, same tasks -- the lever is the only variable.
#
# ARM ORDER: stock first, then levered. Disclosed because order is not neutral --
# the second arm runs against a warmer page cache and a longer-lived host. Stock
# first is the conservative choice: it gives the INCUMBENT the colder start, so
# any lever win this pair reports is not manufactured by ordering. The
# per-arm memory preflight drops caches between arms to shrink what remains.
#
# WHY NOT THE CAMPAIGN DRIVER. The driver only activates fixed32 mode for the
# canonical sequence file, which runs tail6 AND hydra27 -- two campaigns would
# have spent ~3 GPU-h on tail6 arms this pair does not need. This drives
# fr13_bigdenom_swe_serve_variant.sh directly (the gate runner's idiom) and then
# reduces with the driver's OWN deploy-speed invocation, so the numbers are
# produced exactly as every banked arm's were.
#
# CREDENTIAL. The levered arm needs a production sidecar issued from a gate
# earned AT THE SERVING COMMIT. fr13_qrow32_b1_pass_sidecar.py says so in its own
# words: "By PATH, never by runroot: the credential must be issued against a gate
# re-run at the production plumbing commit." The launcher enforces it as
# FR13_FA2_QROW32_B1_SOURCE_COMMIT == $(git rev-parse HEAD), which compares
# against HEAD REGARDLESS OF WHAT CHANGED -- a results-only commit invalidates a
# credential just as surely as a code one. This script therefore refuses to run
# unless HEAD still equals the gate's source commit.
#
# Usage:
#   GATE_RUNROOT=output/fr14_gqa_k0_gate_<ts> \
#   TAG=leverpair bash scripts/fr14_run_b1_hydra27_lever_pair.sh
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${GATE_RUNROOT:?set GATE_RUNROOT to the K0 gate runroot that earned the credential}"
TAG=${TAG:-leverpair}
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "TAG unsafe" >&2; exit 2; }

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FA2_DIR=${FA2_DIR:-/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810}
CANDIDATE_SO="$FA2_DIR/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so"
CANDIDATE_SHA256=3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae
CANDIDATE_BYTES=299815552
SOURCE_CLOSURE_SHA256=172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
MANDATORY_WEIGHT_BYTES=25430574256
MANDATORY_WEIGHT_FLOOR_MS=93.15228665201465
QUALIFICATION_PROFILE=full_vocab
EXPECT_TOK_PER_DRAFT=31
GATE_ARM_GLOB="$GATE_RUNROOT/hydra27_fixed32_fa2_qrow32_gqa_pair_k0_b1_gate_"*

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_hydra27_lever_pair_$TS
RUNROOT_ABS=$(realpath -m "$RUNROOT")
[[ ! -e "$RUNROOT_ABS" ]] || { echo "RUNROOT must be new" >&2; exit 2; }
mkdir -p "$RUNROOT_ABS"

# ---- the freeze: HEAD must still be the gate's source commit -----------------
SOURCE_COMMIT=$(git rev-parse HEAD)
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
GATE_ARM_DIR=$(ls -d $GATE_ARM_GLOB 2>/dev/null | head -1)
[[ -d "$GATE_ARM_DIR" ]] || { echo "gate arm dir not found under $GATE_RUNROOT" >&2; exit 2; }
GATE_JSON=$(ls "$GATE_ARM_DIR"/qrow32_gqa_pair_live_verification.json 2>/dev/null | head -1)
LIVE_RESULT="$GATE_ARM_DIR/logs/fr13_fa2_qrow32_b1_gqa_pair_live_paged_ab.json"
for f in "$GATE_JSON" "$LIVE_RESULT" "$CANDIDATE_SO"; do
  [[ -f "$f" && ! -L "$f" ]] || { echo "gate input missing/unsafe: $f" >&2; exit 2; }
done
GATE_SOURCE_COMMIT=$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['source_commit'])" "$GATE_JSON")
[[ "$GATE_SOURCE_COMMIT" == "$SOURCE_COMMIT" ]] || {
  echo "FREEZE VIOLATED: the gate was earned at $GATE_SOURCE_COMMIT but HEAD is $SOURCE_COMMIT." >&2
  echo "The launcher compares the credential's SOURCE_COMMIT against HEAD regardless of what" >&2
  echo "changed, so ANY commit since the gate -- results-only included -- invalidates it." >&2
  echo "Re-run the K0 gate at this HEAD before running the levered arm." >&2
  exit 2
}
GATE_SHA256=$(sha256sum "$GATE_JSON" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | awk '{print $1}')

# ---- issue the production credential from that gate --------------------------
SIDECAR="$RUNROOT_ABS/fr14_qrow32_b1_gqa_pair_k0_production_pass.json"
"$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py issue-gqa-pair \
  --gate "$GATE_JSON" --expected-gate-sha256 "$GATE_SHA256" \
  --live-result "$LIVE_RESULT" \
  --candidate-so "$CANDIDATE_SO" --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm gqa_pair --patch-source scripts/fr13_patch_fa2_tree_bias.py \
  --expected-source-commit "$SOURCE_COMMIT" --out "$SIDECAR" \
  || { echo "credential issuance FAILED" >&2; exit 2; }
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
echo "[pair] credential issued: $SIDECAR ($SIDECAR_SHA256)"

run_arm() {  # label lever(0|1)
  local label=$1 lever=$2
  local arm="hydra27_fixed32_${label}_${TAG}"
  echo "===== ARM $arm (gqa_pair=$lever) $(date -u +%H:%M:%SZ) ====="
  sync; sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
  awk '/^MemFree:/{printf "[pair] MemFree=%.2fGiB\n",$2/1048576}' /proc/meminfo
  awk '/^MemFree:/{exit ($2/1048576 < 82.3)}' /proc/meminfo \
    || { echo "[$arm] FAIL: unified-memory preflight" >&2; return 2; }
  [[ -z "$(docker ps -q)" ]] || { echo "[$arm] FAIL: docker not empty" >&2; return 2; }

  local lever_env=()
  if [[ "$lever" == "1" ]]; then
    lever_env=(
      FR13_FA2_QROW32_B1_PRODUCTION_ARM=gqa_pair
      FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR="$SIDECAR"
      FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR_SHA256="$SIDECAR_SHA256"
      FR13_FA2_QROW32_B1_SO_SHA256="$CANDIDATE_SHA256"
      FR13_FA2_QROW32_B1_SO_SIZE="$CANDIDATE_BYTES"
      FR13_FA2_QROW32_B1_FA2_HEAD="$FA2_HEAD"
      FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256"
      FR13_FA2_QROW32_B1_SOURCE_COMMIT="$SOURCE_COMMIT"
      FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256"
      FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256="$SUBSET_SHA256"
      FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE="$QUALIFICATION_PROFILE"
      FORKED_FA2_SO="$CANDIDATE_SO"
    )
  fi

  env RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=5400 \
    FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0 \
    FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0" \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    FR13_DEVICE_MULTIDRAFT=1 \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}.json \
    FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json \
    FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json \
    "${lever_env[@]}" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$arm" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$arm.runlog" 2>&1
  local rc=$?
  echo "[$arm] serve rc=$rc $(date -u +%H:%M:%SZ)"

  # The campaign driver's OWN reduction, so the numbers are produced exactly as
  # every banked arm's were.
  local census="$RUNROOT_ABS/$arm/logs/fr13_fixed32_work_census.jsonl"
  local census_args=()
  [[ -f "$census" && ! -L "$census" ]] && census_args=(--work-census "$census")
  local np
  np=$(find "$RUNROOT_ABS/$arm/swe_out" -name vllm_metrics_post.txt 2>/dev/null | wc -l)
  if (( np >= 1 )); then
    "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
      --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
      --expected-tok-per-draft "$EXPECT_TOK_PER_DRAFT" --batch-size 1 \
      "${census_args[@]}" \
      --out "$RUNROOT_ABS/$arm/deploy_speed_${TAG}.json" 2>&1 | tail -12 \
      || echo "[$arm] deploy-speed reduce FAILED"
  else
    echo "[$arm] NO post-brackets — deploy-speed VACUOUS"
  fi
  return $rc
}

mkdir -p output/fr13_sfwd_sidecar
run_arm stock 0; STOCK_RC=$?
run_arm levered 1; LEVER_RC=$?
printf 'stock_rc=%s levered_rc=%s ended=%s\n' \
  "$STOCK_RC" "$LEVER_RC" "$(date -u +%FT%TZ)" | tee "$RUNROOT_ABS/pair_meta.txt"
echo "[pair] done -> $RUNROOT_ABS"

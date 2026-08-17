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
# BUT THE SEQUENCE STILL HAS TO BE SOURCED. Bypassing the driver does NOT mean
# bypassing fr13_fixed32_floor_timers_seq.sh: that file is the only place
# FR13_FIXED32_TAW_WALK_CAP and ~50 other fixed32 route/committer/census pins
# are exported, and the launcher reads them from the PROCESS environment as
# "${NAME:-}", not from any env-prefix array. The first launch of this pair
# skipped it and both arms died in 6 s with the opaque "fixed32 integer route
# pin is malformed" -- an unset walk cap arriving as "" and failing int("")
# deep inside the launcher. So publish the sequence with run_variant stubbed
# (the gate runner's idiom), which yields the driver's exact environment
# without the driver's arm schedule.
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
# both (default) = the full pair. levered = salvage a levered arm against an
# already-banked clean stock arm named by PAIR_STOCK_REF.
PAIR_ARMS=${PAIR_ARMS:-both}
PAIR_STOCK_REF=${PAIR_STOCK_REF:-}
case "$PAIR_ARMS" in both|levered) ;; *) echo "PAIR_ARMS must be both|levered" >&2; exit 2 ;; esac

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FA2_DIR=${FA2_DIR:-/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810}
CANDIDATE_SO="$FA2_DIR/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so"
CANDIDATE_SHA256=3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae
CANDIDATE_BYTES=299815552
SOURCE_CLOSURE_SHA256=172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
# The launcher demands the exact4 identity verbatim for any B1 production arm.
EXACT4_TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398
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

# ---- PREFLIGHT ONLY: prove the gate can still mint a credential --------------
# The launcher mints the SERVED credential itself and refuses a caller-supplied
# one, so this sidecar is NEVER passed to the run. It is a cheap dry run of the
# exact issuance the launcher will perform ~8 min into the boot, and it has
# already caught three defects (a NameError in the validator, a missing
# qualification_profile, a K64-shaped required_runtime) at zero GPU cost. Keep it
# for that, and do not mistake its output for the credential.
SIDECAR="$RUNROOT_ABS/fr14_qrow32_b1_gqa_pair_k0_production_pass.preflight.json"
"$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py issue-gqa-pair \
  --gate "$GATE_JSON" --expected-gate-sha256 "$GATE_SHA256" \
  --live-result "$LIVE_RESULT" \
  --candidate-so "$CANDIDATE_SO" --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm gqa_pair --patch-source scripts/fr13_patch_fa2_tree_bias.py \
  --expected-source-commit "$SOURCE_COMMIT" --out "$SIDECAR" \
  || { echo "credential issuance PREFLIGHT FAILED" >&2; exit 2; }
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
echo "[pair] credential preflight OK (launcher will mint the served one): $SIDECAR_SHA256"

# ---- publish the fixed32 route pins ------------------------------------------
# Identity FIRST, then the sequence: the sequence reads the draft-vocab config
# as "${FR13_DRAFT_VOCAB_K:-65536}:$FR13_DRAFT_VOCAB_ROOT" and derives the floor
# from it, so exporting K0 up front makes it publish the K0 floor itself. The
# block map is deliberately NOT set here -- fr13_canonical_env.sh supplies the
# canonical default and the launcher demands exactly that value, so production
# carries it and so must this pair. Assert the whole identity survived sourcing.
export BSIZE=1 CONC=1 WALL=5400
# THE PER-TASK WALL MUST BE A DECLARED BUDGET, NOT THE LEGACY AGENT WALL.
# WALL/AGENT_WALL_S is the legacy per-attempt wall, and run_swe_bench_q36_a.py
# treats its timeout as PROVENANCE-FATAL on purpose: "those mean the harness cut
# a run it did not intend to". The first stock arm proved the consequence --
# astropy__astropy-13398 ran past 5400 s, raised Fixed32BoundaryError("agent
# terminal state is incomplete: exit_code=-1 timed_out=True"), aborted the
# orchestrator, left both ingress campaigns unfinalisable (HTTP 409), and so
# voided an otherwise complete 2h25m arm including its three good tasks.
# FR13_CAMPAIGN_TASK_BUDGET_S is the instrument the codebase built for exactly
# this: a declared budget whose terminal is ACCOUNTED (verdict "capped") instead
# of provenance-fatal. Its own docs say "ON for timing and gate arms, where wall
# determinism is the point and the resolve verdict is not" -- which is this pair.
# 5400 is the recommended value and equals WALL, so the effective timeout is the
# budget and campaign_budget_was_the_binding_limit corroborates as true.
# This weakens NO check: the fatal path stays fatal for undeclared cuts.
export FR13_CAMPAIGN_TASK_BUDGET_S=5400
export FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
export FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE="$QUALIFICATION_PROFILE"
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "0" \
   && "$FR13_DRAFT_VOCAB_K" == "0" \
   && "$FR13_NEEDS_ALLOW" == "FR13_DRAFT_VOCAB_K=0" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "/workspace/scripts/fr13_dvk_subset_blocks.json" \
   && "$FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE" == "full_vocab" \
   && "$FR13_FIXED32_TAW_WALK_CAP" == "12" \
   && "$FR13_CAMPAIGN_TASK_BUDGET_S" == "5400" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "K0 hydra27 pair floor contract drifted" >&2; exit 2; }
echo "[pair] route pins published: walk_cap=$FR13_FIXED32_TAW_WALK_CAP" \
     "floor=${FR13_MANDATORY_WEIGHT_BYTES}B/${FR13_WEIGHT_FLOOR_MS}ms K=0 ROOT=0"

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
    # THE CREDENTIAL IS PRESENTED AS ITS GATE, NOT AS A MINTED SIDECAR.
    # FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR{,_SHA256} are LAUNCHER-PRIVATE:
    # the launcher mints the sidecar itself, inside the run, from the sealed gate
    # plus the live A/B result the gate binds by digest, and it REFUSES outright
    # ("FR13 qrow32 production sidecar credentials are launcher-private") if a
    # caller sets them. Passing our own pre-minted sidecar is exactly what killed
    # the first levered arm in 5 s. The caller's job is to hand over the gate BY
    # PATH and let the launcher re-derive the whole chain at issue time.
    lever_env=(
      FR13_FA2_QROW32_B1_PRODUCTION_ARM=gqa_pair
      FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON="$GATE_JSON"
      FR13_FA2_QROW32_B1_GQA_PAIR_GATE_SHA256="$GATE_SHA256"
      FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON="$LIVE_RESULT"
      FR13_FA2_QROW32_B1_EXACT4_TASK_IDS="$EXACT4_TASK_IDS"
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
LEVER_RC=""
STOCK_RC=0
if [[ "$PAIR_ARMS" == "both" ]]; then
  run_arm stock 0; STOCK_RC=$?
else
  # levered-only salvage. Legitimate ONLY when a clean stock arm already exists
  # to pair against, and it must be disclosed with the commit delta between the
  # two arms, because they will not have been served at the same HEAD.
  echo "[pair] PAIR_ARMS=levered -- stock arm NOT run; pairing against $PAIR_STOCK_REF"
  [[ -n "$PAIR_STOCK_REF" ]] \
    || { echo "PAIR_ARMS=levered requires PAIR_STOCK_REF (the banked stock arm dir)" >&2; exit 2; }
  [[ -f "$PAIR_STOCK_REF/deploy_speed_${TAG}.json" ]] \
    || { echo "PAIR_STOCK_REF carries no deploy_speed_${TAG}.json: $PAIR_STOCK_REF" >&2; exit 2; }
fi

# A failed arm 1 ends the pair. Running arm 2 anyway cannot produce a pair, and
# the first attempt showed it actively obscures the diagnosis: arm 1 failed its
# terminal, the serve variant DELIBERATELY preserved its container (fail-closed
# evidence retention after the engine-ledger snapshot could not be materialised),
# and arm 2 then died on "unified-memory preflight" -- pointing at memory when
# the actual cause was three steps upstream. Name the real cause here.
if (( STOCK_RC != 0 )); then
  echo "[pair] STOCK ARM FAILED rc=$STOCK_RC -- the pair is void; levered arm NOT run." >&2
  leaked=$(docker ps -q)
  if [[ -n "$leaked" ]]; then
    echo "[pair] a serving container is still up: $(docker ps --format '{{.Names}}')" >&2
    echo "[pair] this is DELIBERATE preservation, not a leak -- the arm's engine-ledger" >&2
    echo "[pair] snapshot failed, so the container is held for manual evidence recovery." >&2
    echo "[pair] Recover the ledger from its /logs bind mount BEFORE removing it." >&2
  fi
else
  run_arm levered 1; LEVER_RC=$?
  (( LEVER_RC == 0 )) || echo "[pair] LEVERED ARM FAILED rc=$LEVER_RC" >&2
fi

printf 'stock_rc=%s levered_rc=%s ended=%s\n' \
  "$STOCK_RC" "${LEVER_RC:-not_run}" "$(date -u +%FT%TZ)" | tee "$RUNROOT_ABS/pair_meta.txt"
echo "[pair] done -> $RUNROOT_ABS"
# EXIT STATUS IS THE PAIR'S VERDICT. Without this the script's status was that of
# its last command -- the `printf | tee` above -- so the first attempt reported
# PAIR_EXIT=0 while carrying stock_rc=13 and levered_rc=2. A wrapper that chains
# on success would have walked straight past a void pair.
if (( STOCK_RC != 0 )); then exit "$STOCK_RC"; fi
if [[ -n "$LEVER_RC" ]] && (( LEVER_RC != 0 )); then exit "$LEVER_RC"; fi
exit 0

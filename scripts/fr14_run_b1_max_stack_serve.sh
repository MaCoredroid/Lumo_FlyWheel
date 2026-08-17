#!/usr/bin/env bash
# FR14 MAX-STACK serve and its METRICS=1 stock control (Mark's composed number).
#
# TWO ARMS, ONE VARIABLE:
#   maxstack  gqa_pair (promoted, armed from the run-local credential pointer)
#             + FR13_HOST_TAIL_PREP_BAKE + FR13_HOST_TAIL_DEFER
#   (control  NOT REQUIRED -- see the arm_env note below)
#
# WHY THE CONTROL IS NOT THE BANKED 215.899 ANCHOR. single_launch's production
# predicate requires FR10_METRICS=1, but the promoted config -- and the banked
# anchor -- run FR10_METRICS=0. Arming single_launch therefore moves a second
# variable. FR13 already solved this exact conflict with TIMING_CONTROL: a flag
# that moves a PLAIN fixed32 arm into the metrics=1 class while selecting no
# kernel, with symmetric counter cost. Its guard refuses to sit beside any GDN
# selector -- "a control arm serves the incumbent and nothing else" -- which is
# what keeps it honest as metrics-classification-only.
#
# CONTROL B, not A. TIMING_CONTROL's refusal list covers GDN selectors and does
# NOT mention the FA2 B1 selector, so a control arm CAN still carry gqa_pair.
# This runner deliberately does not: it NAMES the B1 production arm EMPTY, which
# the launcher obeys verbatim as a deliberate opt-out, giving true stock at
# metrics=1 -- the metrics-twin of the banked anchor. The other question (isolate
# the new levers with gqa_pair held constant) is recoverable by arithmetic
# against the banked gqa_pair verdict; it does not need its own serve.
#
# VERDICT INSTRUMENT: step_wall_ms + s_per_fwd_gpu. Acceptance is REPORTED but is
# not the verdict, per the ruling that a byte-exact lever cannot move acceptance
# and that trajectory noise at n=4 swamps a ~2% kernel effect.
#
# PER-LEVER ATTRIBUTION IS FORFEIT BY DESIGN (Mark's one-go ruling): this arm
# claims the STACK, not any lever's individual value. Say so wherever it is
# banked.
#
# Usage:
#   ARM_KIND=maxstack|control \
#   FR13_FIXED32_GDN_SINGLE_LAUNCH_PASS_JSON=<credential>   # maxstack only
#   bash scripts/fr14_run_b1_max_stack_serve.sh
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

ARM_KIND=${ARM_KIND:-maxstack}
case "$ARM_KIND" in maxstack) ;; *) echo "ARM_KIND must be maxstack" >&2; exit 2 ;; esac
TAG=${TAG:-maxstack}
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "TAG unsafe" >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FA2_DIR=${FA2_DIR:-/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810}
CANDIDATE_SO="$FA2_DIR/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so"
SUBSET=config/fr13_fixed32/subset_b4_four.json
MANDATORY_WEIGHT_BYTES=25430574256
MANDATORY_WEIGHT_FLOOR_MS=93.15228665201465
EXPECT_TOK_PER_DRAFT=31

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_${ARM_KIND}_$TS
RUNROOT_ABS=$(realpath -m "$RUNROOT")
[[ ! -e "$RUNROOT_ABS" ]] || { echo "RUNROOT must be new" >&2; exit 2; }
mkdir -p "$RUNROOT_ABS"
ARM="hydra27_fixed32_${ARM_KIND}_${TAG}"

SOURCE_COMMIT=$(git rev-parse HEAD)
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ -z "$(docker ps -aq)" ]] || { echo "docker must be empty" >&2; exit 2; }
awk '/^MemFree:/{exit ($2/1048576 < 82.3)}' /proc/meminfo \
  || { echo "unified-memory preflight failed" >&2; exit 2; }

# ---- route pins + K0 identity + the DECLARED task budget --------------------
# 9000 s for the marathon: 13398 is stochastic (1679 s to >5400 s observed), so
# 9000 usually lets it finish naturally and is still an ACCOUNTED cap if not.
# The legacy per-attempt wall stays equal to it, so the budget is the binding
# limit and campaign_budget_was_the_binding_limit corroborates true.
export BSIZE=1 CONC=1 WALL=9000
export FR13_CAMPAIGN_TASK_BUDGET_S=9000
export FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
export FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=full_vocab
export FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE=full_vocab
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_K" == "0" && "$FR13_DRAFT_VOCAB_ROOT" == "0" \
   && "$FR13_FIXED32_TAW_WALK_CAP" == "12" \
   && "$FR13_CAMPAIGN_TASK_BUDGET_S" == "9000" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "K0 max-stack floor contract drifted" >&2; exit 2; }

# THE COMPOSED ARM, REDUCED. single_launch is deferred: its K0 arming is a
# ~16-site port across six files, and K64 lives in the credential CONTENT
# (draft_vocab_k is both baked in and verified), not merely in env checks.
#
# Dropping it SIMPLIFIES the verdict rather than weakening it. single_launch is
# the only reason FR10_METRICS=1 was needed -- plain fixed32 refuses metrics=1
# outright ("fixed32 requires FR10_METRICS=0, got 1") -- so without it this arm
# runs at metrics=0, the banked 215.899 ms anchor's OWN setting, and pairs
# against that anchor directly. No Control B serve is required.
#
# gqa_pair is UNNAMED on purpose: the promoted default must arm it from the
# run-local credential pointer, which is the path production takes.
arm_env=(
  FR10_METRICS=0
  FR13_HOST_TAIL_PREP_BAKE=1
  FR13_HOST_TAIL_DEFER=1
)
printf 'arm_kind=%s\nsource_commit=%s\nbudget_s=9000\nverdict_instrument=step_wall_ms+s_per_fwd_gpu\nper_lever_attribution=forfeit_by_design\nstarted=%s\n' \
  "$ARM_KIND" "$SOURCE_COMMIT" "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/arm_meta.txt"

echo "===== $ARM ($ARM_KIND) $(date -u +%H:%M:%SZ) ====="
env RUNROOT="$RUNROOT_ABS" \
  OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=9000 \
  FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
  FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}.json \
  FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_dfwd.json \
  FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_cfwd.json \
  FORKED_FA2_SO="$CANDIDATE_SO" \
  "${arm_env[@]}" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" hydra27_fixed32 "$SUBSET" \
  > "$RUNROOT_ABS/$ARM.runlog" 2>&1
rc=$?
echo "[$ARM] serve rc=$rc $(date -u +%H:%M:%SZ)"

census="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_work_census.jsonl"
census_args=()
[[ -f "$census" && ! -L "$census" ]] && census_args=(--work-census "$census")
np=$(find "$RUNROOT_ABS/$ARM/swe_out" -name vllm_metrics_post.txt 2>/dev/null | wc -l)
if (( np >= 1 )); then
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$ARM" --out-root "$RUNROOT_ABS/$ARM/swe_out" \
    --expected-tok-per-draft "$EXPECT_TOK_PER_DRAFT" --batch-size 1 \
    "${census_args[@]}" \
    --out "$RUNROOT_ABS/$ARM/deploy_speed_${TAG}.json" 2>&1 | tail -12 \
    || echo "[$ARM] deploy-speed reduce FAILED"
else
  echo "[$ARM] NO post-brackets — deploy-speed VACUOUS"
fi
printf 'serve_rc=%s\nended=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/arm_meta.txt"
echo "[$ARM] done -> $RUNROOT_ABS"
exit "$rc"

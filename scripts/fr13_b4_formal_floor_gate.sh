#!/usr/bin/env bash
# FR13 B4 FORMAL STATISTICAL FLOOR GATE -- Tail23 + Hydra27, exact4, N repeats.
#
# Every B4 timing pair banked before this is a SCREEN: one draw of four agent
# trajectories reported as a point estimate.  The campaign log's standing gap is
# "formal Tail/Hydra statistical floor gate -- all pairs so far are screens".
#
# WHAT THIS RUNS
#   N independent PASSES.  One pass = scripts/fr13_b4_campaign_driver.sh over
#   scripts/fr13_fixed32_floor_timers_seq.sh, which boots BOTH topologies back to
#   back (tail6_fixed32 then hydra27_fixed32, or the reverse) at BSIZE=CONC=4 on
#   the byte-pinned exact4 subset, and then runs the canonical
#   scripts/fr13_floor_gate.py over the pass.
#
#   Nothing here re-implements measurement.  The arm vehicle, the route preamble,
#   the bracket reduction, the work census and the statistics are all the existing
#   tested code; this script only supplies repetition, balance and provenance.
#
# WHY REPEATS
#   fr13_floor_gate.py's B4 uncertainty model is a moving-block bootstrap WITHIN a
#   run.  That is right for step wall (CV ~2%) and wrong for aggregate TPS, which
#   is events_per_step * per_request_step_tps -- and events_per_step is set by
#   agent-trajectory co-residency, which a within-run resample cannot see.
#   Measured across repeats of the same config: aggregate TPS CV ~29%, per-request
#   step TPS CV ~9%.  Only between-pass repetition bounds the first.
#
# WHY THE ORDER ALTERNATES
#   The second arm of a pass inherits a warmer page cache and a differently-aged
#   host.  FR13_FLOOR_ORDER alternates TH/HT across passes so arm position is
#   balanced and cannot alias into the topology contrast.
#
# NO DEFAULT IS FLIPPED.  FR13_MAMBA_SPEC_BLOCKS_CDIV keeps its branch default
# (0 = narrowing OFF); the flip awaits Mark.  FR13_B4_TASK_REFILL stays 0 because
# the driver states refill output is NOT exact4-citable without a contract update.
# The reducer records the observed stack state per pass and refuses to call a run
# citable if any of them moved.
#
# USAGE
#   PASSES=4 bash scripts/fr13_b4_formal_floor_gate.sh
# Long runs must be launched DETACHED (a 120s tool timeout SIGTERMs the process
# group), e.g.
#   setsid nohup env PASSES=4 bash scripts/fr13_b4_formal_floor_gate.sh \
#     > /home/mark/shared/b4gate.log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

PASSES=${PASSES:-4}
# MAKEUP MODE: append passes to an EXISTING gate root instead of starting a new
# campaign. Used when infrastructure (not the stack) destroys a pass -- e.g. the
# 2026-08-11 ON campaign, where a transient alienware->GB10 Tailscale blip at the
# preflight moment killed pass_03's Tail23 arm before it served a single request.
# The replacement pass MUST reuse the dead pass's TH/HT slot, so the order is
# forced rather than derived from index parity.
MAKEUP_GATE_ROOT=${MAKEUP_GATE_ROOT:-}
PASS_INDEX_START=${PASS_INDEX_START:-0}
FLOOR_ORDER_OVERRIDE=${FLOOR_ORDER_OVERRIDE:-}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh

# The gate admits exactly 4 or 16 included passes -- that is what keeps the
# between-pass interval on the repo's pinned one-sided t criticals (df 3 or 15)
# instead of inventing a new constant.
# A FRESH campaign must be 4 or 16 so the between-pass interval lands on the
# repo's pinned one-sided t criticals. A MAKEUP run is exempt: it tops an
# existing campaign back up, and the reducer -- not this loop -- enforces the
# final 4-or-16 admissibility per topology.
if [[ -z "$MAKEUP_GATE_ROOT" ]]; then
  case "$PASSES" in
    4|16) ;;
    *) echo "FAIL: PASSES must be 4 or 16 (pinned t criticals); got $PASSES" >&2; exit 2 ;;
  esac
fi
case "$FLOOR_ORDER_OVERRIDE" in
  ""|TH|HT) ;;
  *) echo "FAIL: FLOOR_ORDER_OVERRIDE must be TH or HT" >&2; exit 2 ;;
esac

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SOURCE_COMMIT=$(git rev-parse HEAD)
if [[ -n "$MAKEUP_GATE_ROOT" ]]; then
  GATE_ROOT=$(realpath -m "$MAKEUP_GATE_ROOT")
  [[ -d "$GATE_ROOT" ]] \
    || { echo "FAIL: makeup gate root does not exist: $GATE_ROOT" >&2; exit 2; }
else
  GATE_ROOT="$REPO/output/fr13_b4_formal_floor_gate_${STAMP}"
fi

# ---------------------------------------------------------------- preflight --
[[ -f "$SUBSET" && ! -L "$SUBSET" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "FAIL: exact4 subset is not the canonical byte-pinned set" >&2; exit 2; }
[[ -f "$SEQUENCE_FILE" && ! -L "$SEQUENCE_FILE" ]] \
  || { echo "FAIL: missing $SEQUENCE_FILE" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "FAIL: no $PYTHON_BIN" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "FAIL: tracked worktree must be clean" >&2; exit 2; }
if [[ -z "$MAKEUP_GATE_ROOT" ]]; then
  [[ ! -e "$GATE_ROOT" && ! -L "$GATE_ROOT" ]] \
    || { echo "FAIL: gate root must be new: $GATE_ROOT" >&2; exit 2; }
fi

# GPU coordination: never contend with another campaign.
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "FAIL: Docker is not clean -- another campaign may be serving" >&2; exit 2; }
if pgrep -af '[f]r13_bigdenom_swe_serve_variant|[f]r13_b4_campaign_driver' >/dev/null 2>&1; then
  echo "FAIL: an fr13 serve/driver process is already running" >&2
  exit 2
fi

mkdir -p "$GATE_ROOT"
{
  printf 'gate_root=%s\n' "$GATE_ROOT"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'passes=%s\n' "$PASSES"
  printf 'subset=%s\n' "$SUBSET"
  printf 'subset_sha256=%s\n' "$SUBSET_SHA256"
  printf 'sequence=%s\n' "$SEQUENCE_FILE"
  printf 'started=%s\n' "$(date -u +%FT%TZ)"
} > "$GATE_ROOT/gate_meta.txt"

echo "===== B4 FORMAL FLOOR GATE $STAMP ($PASSES passes x 2 topologies) ====="

# ------------------------------------------------------------------- passes --
completed=0
for (( offset=0; offset<PASSES; offset++ )); do
  index=$(( PASS_INDEX_START + offset ))
  pass_dir="$GATE_ROOT/pass_$(printf '%02d' "$index")"
  # Balance first-arm/second-arm position across passes.
  if (( index % 2 == 0 )); then order=TH; else order=HT; fi
  [[ -n "$FLOOR_ORDER_OVERRIDE" ]] && order=$FLOOR_ORDER_OVERRIDE
  [[ ! -e "$pass_dir" ]] \
    || { echo "FAIL: pass dir already exists: $pass_dir" >&2; exit 2; }
  mkdir -p "$pass_dir"
  printf '%s\n' "$order" > "$pass_dir/floor_order.txt"

  echo "----- pass $index (FR13_FLOOR_ORDER=$order) -> $pass_dir -----"
  printf 'pass=%s order=%s started=%s\n' "$index" "$order" "$(date -u +%FT%TZ)" \
    >> "$GATE_ROOT/gate_meta.txt"

  # Both arms boot inside the driver; it reduces deploy-speed after each and then
  # runs the canonical floor gate over the pass runroot.
  if env \
      BSIZE=4 \
      CONC=4 \
      WALL=0 \
      TAG=gate${index} \
      RUNROOT="$pass_dir" \
      SUBSET="$SUBSET" \
      SEQUENCE_FILE="$SEQUENCE_FILE" \
      FR13_FLOOR_ORDER="$order" \
      FR13_B4_TASK_REFILL=0 \
      FR13_DRAFT_VOCAB_ROOT=1 \
      FR13_DRAFT_VOCAB_K=65536 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      bash scripts/fr13_b4_campaign_driver.sh \
      > "$pass_dir/driver.runlog" 2>&1; then
    completed=$(( completed + 1 ))
    printf 'pass=%s rc=0 ended=%s\n' "$index" "$(date -u +%FT%TZ)" \
      >> "$GATE_ROOT/gate_meta.txt"
  else
    rc=$?
    # A failed pass is RECORDED, not repaired and not retried in place: the
    # reducer excludes it with a reason and reports how many survived.
    echo "FAIL: pass $index driver rc=$rc (recorded; pass will be excluded)" >&2
    printf 'pass=%s rc=%s ended=%s\n' "$index" "$rc" "$(date -u +%FT%TZ)" \
      >> "$GATE_ROOT/gate_meta.txt"
  fi

  # Evidence-first teardown: containers must be gone before the next pass boots.
  if [[ "$(docker ps -aq | wc -l)" -ne 0 ]]; then
    docker ps -a > "$pass_dir/docker_ps_after_pass.txt" 2>&1 || true
    for cid in $(docker ps -aq); do
      docker logs "$cid" > "$pass_dir/docker_logs_${cid}.txt" 2>&1 || true
    done
    echo "FAIL: containers survived pass $index; evidence captured, stopping" >&2
    break
  fi
done

printf 'passes_completed=%s ended=%s\n' "$completed" "$(date -u +%FT%TZ)" \
  >> "$GATE_ROOT/gate_meta.txt"

# ------------------------------------------------------------------ reduce ---
echo "===== reducing $completed/$PASSES passes ====="
# --finalize is MANDATORY here: the campaign driver writes an ungated
# deploy_speed_${TAG}.json, so without it every arm is excluded for a missing
# deploy_speed_fullwall.json and the verdict is vacuous.
"$PYTHON_BIN" scripts/fr13_b4_floor_gate_reduce.py \
  --repo "$REPO" \
  --gate-root "$GATE_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --finalize \
  --min-passes 4
rc=$?
echo "===== B4 FORMAL FLOOR GATE DONE rc=$rc ====="
echo "verdict: $GATE_ROOT/fr13_b4_formal_floor_gate.json"
exit "$rc"

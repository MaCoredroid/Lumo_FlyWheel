#!/usr/bin/env bash
# FR13 B4 POOL16 REFILL TIMING GATE -- Tail23 + Hydra27, 16-task pool, 4 slots.
#
# WHAT THIS MEASURES
#   The SCHEDULE FRONT.  exact4 is the degenerate pool where pool == slots: the
#   wave decays 4->3->2->1 with nothing behind it and the batch is full width for
#   only ~36% of the arm.  A 16-task pool behind the same 4 slots backfills, so
#   events/step -- and therefore aggregate TPS -- should rise at flat per-request
#   service speed.  This is where mamba narrowing's retired throughput claim is
#   supposed to reappear as real throughput: narrowing's citable value is CAPACITY
#   (KV peak 18% vs 74%), and capacity only pays if something keeps the batch full.
#
# WHAT FR13_B4_TASK_REFILL ACTUALLY IS
#   NOT a schedule lever.  Its own docstring (run_swe_bench_q36_a.py:3253) records
#   that ThreadPoolExecutor.map already backfills a worker the instant its job
#   returns, so admission TIMING is identical with the flag off.  What the flag
#   adds is the admission LEDGER, completion-order collection with a circuit
#   breaker, and a hard peak_depth <= slots invariant.  It is pinned to 1 here
#   because the ledger is the only artifact that witnesses the occupancy this
#   class claims -- read the pin as "instrumented", not "accelerated".  The
#   schedule change under test is the POOL SIZE.
#
#   That is also why there is no refill-OFF comparator arm: it would spend ~21
#   GPU-hours measuring a structural null AND would produce no ledger, so it could
#   not be validated as a pool run at all.  The comparator is exact4, and exact4 is
#   already sealed and citable four passes deep on both topologies.
#
# THE CONTRACT UPDATE
#   scripts/fr13_b4_campaign_driver.sh:38-41 says refill output "is NOT exact4-
#   citable without a contract update".  That update is the run class
#   `pool16_refill_timing` in scripts/fr13_b4_floor_gate_reduce.py.  It does not
#   make refill exact4-citable -- it defines a SEPARATE class with its own binding,
#   its own required bracket topology (staggered envelope, not nested), its own
#   primary statistic (aggregate, with per-request kept as a mandatory
#   non-regression companion), and an explicit list of what it does not claim:
#   exact4 comparability, cap verdicts, and the exact16 agent-quality band (Mark's
#   2026-08-10 ruling that exact16 is QUALITY CONTROL is untouched by this class).
#
# WHY REPEATS, AND WHY EXACTLY 4 OR 16
#   Same reason as the formal floor gate: aggregate TPS is
#   events_per_step * per_request_step_tps and events_per_step is set by agent
#   trajectory co-residency, which a within-run bootstrap cannot see.  Only
#   between-pass repetition bounds it, and only N in {4, 16} lands on the repo's
#   pinned one-sided t criticals.  A 2-pass campaign is a SCREEN: the reducer
#   returns NOT_EVALUATED_INSUFFICIENT_PASSES and citable=false.  No df=1 critical
#   is invented to rescue a short run.
#
# WHY THE ORDER ALTERNATES
#   The second arm of a pass inherits a warmer page cache and a differently-aged
#   host.  FR13_FLOOR_ORDER alternates TH/HT across passes so arm position is
#   balanced and cannot alias into the topology contrast.
#
# NO DEFAULT IS FLIPPED.  FR13_MAMBA_SPEC_BLOCKS_CDIV keeps its shipped default
# (1 = narrowing ON since the 2026-08-10 promotion); the reducer resolves it from
# scripts/fr13_canonical_env.sh at the campaign's own commit and refuses a campaign
# whose arms did not all run the same state.
#
# COST.  ~2.4-3.0 h per arm at 16 tasks, ~5-6 h per pass, ~21-24 GPU-h for four.
# Every pass is self-contained evidence and the reducer is offline and idempotent,
# so the campaign can be read at any point and stopped early -- at the cost of
# citability only.
#
# USAGE
#   PASSES=4 bash scripts/fr13_b4_pool16_refill_gate.sh
# Long runs must be launched DETACHED (a 120s tool timeout SIGTERMs the process
# group), e.g.
#   setsid nohup env PASSES=4 bash scripts/fr13_b4_pool16_refill_gate.sh \
#     > /home/mark/shared/b4pool16.log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

PASSES=${PASSES:-4}
# MAKEUP MODE: append passes to an EXISTING gate root instead of starting a new
# campaign.  Used when infrastructure -- not the stack -- destroys a pass.  The
# replacement pass MUST reuse the dead pass's TH/HT slot, so the order is forced
# rather than derived from index parity.
MAKEUP_GATE_ROOT=${MAKEUP_GATE_ROOT:-}
PASS_INDEX_START=${PASS_INDEX_START:-0}
FLOOR_ORDER_OVERRIDE=${FLOOR_ORDER_OVERRIDE:-}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
RUN_CLASS=pool16_refill_timing
SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh
# The sealed exact4 ON gate.  OPTIONAL and DESCRIPTIVE: the reducer emits the
# contrast with its confounds enumerated and refuses it outright unless both
# campaigns measured the same stack state.  Its absence never blocks the gate.
EXACT4_REFERENCE=${EXACT4_REFERENCE:-/home/mark/lumoFlyWheel-main-integration-20260802/output/fr13_b4_formal_floor_gate_20260811T041931Z/fr13_b4_formal_floor_gate.json}

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
  GATE_ROOT="$REPO/output/fr13_b4_pool16_refill_gate_${STAMP}"
fi

# ---------------------------------------------------------------- preflight --
[[ -f "$SUBSET" && ! -L "$SUBSET" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "FAIL: 16-task subset is not the canonical byte-pinned set" >&2; exit 2; }
[[ -f "$SEQUENCE_FILE" && ! -L "$SEQUENCE_FILE" ]] \
  || { echo "FAIL: missing $SEQUENCE_FILE" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "FAIL: no $PYTHON_BIN" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "FAIL: tracked worktree must be clean" >&2; exit 2; }
if [[ -z "$MAKEUP_GATE_ROOT" ]]; then
  [[ ! -e "$GATE_ROOT" && ! -L "$GATE_ROOT" ]] \
    || { echo "FAIL: gate root must be new: $GATE_ROOT" >&2; exit 2; }
fi
# Every precondition this gate imposes must be one the run can actually satisfy --
# five separate campaign fossils were runners bound to an artifact nothing wrote.
# So the reducer and its class are resolved BEFORE any GPU time is spent.
"$PYTHON_BIN" - "$RUN_CLASS" <<'PY' || exit 2
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from fr13_b4_floor_gate_reduce import resolve_run_class

spec = resolve_run_class(sys.argv[1])
assert spec["task_count"] == 16, spec["task_count"]
assert spec["required_bracket_topology"] == "staggered"
assert spec["requires_pool_ledger"] is True
assert spec["contract_pinned_stack"] == {"FR13_B4_TASK_REFILL": "1"}
print(f"run class {sys.argv[1]} resolves: {spec['classification']}")
PY
if [[ -f "$EXACT4_REFERENCE" ]]; then
  echo "exact4 reference: $EXACT4_REFERENCE"
else
  echo "WARN: no exact4 reference at $EXACT4_REFERENCE; the descriptive contrast" >&2
  echo "      will be omitted. This does NOT affect citability." >&2
  EXACT4_REFERENCE=""
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
  printf 'run_class=%s\n' "$RUN_CLASS"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'passes=%s\n' "$PASSES"
  printf 'subset=%s\n' "$SUBSET"
  printf 'subset_sha256=%s\n' "$SUBSET_SHA256"
  printf 'sequence=%s\n' "$SEQUENCE_FILE"
  printf 'task_refill=1\n'
  printf 'task_pool=16\n'
  printf 'slots=4\n'
  printf 'exact4_reference=%s\n' "$EXACT4_REFERENCE"
  printf 'started=%s\n' "$(date -u +%FT%TZ)"
} > "$GATE_ROOT/gate_meta.txt"

echo "===== B4 POOL16 REFILL GATE $STAMP ($PASSES passes x 2 topologies) ====="

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
  # runs the canonical floor gate over the pass runroot.  That in-pass gate is
  # what validates the admission ledger live (fr13_floor_gate.py:7151); the
  # reducer reads the same bytes again independently.
  if env \
      BSIZE=4 \
      CONC=4 \
      WALL=0 \
      TAG=pool${index} \
      RUNROOT="$pass_dir" \
      SUBSET="$SUBSET" \
      SEQUENCE_FILE="$SEQUENCE_FILE" \
      FR13_FLOOR_ORDER="$order" \
      FR13_B4_TASK_REFILL=1 \
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
# deploy_speed_fullwall.json and the verdict is vacuous.  It matters twice as much
# for a pool arm: fr13_measure.py REFUSES a staggered reduction without the work
# census, and --finalize is what supplies it.
reduce_args=(
  --repo "$REPO"
  --gate-root "$GATE_ROOT"
  --source-commit "$SOURCE_COMMIT"
  --run-class "$RUN_CLASS"
  --finalize
  --min-passes 4
  # Its own filename: a pool16 verdict is NOT a formal floor gate and must never be
  # picked up by anything globbing for one.
  --out "$GATE_ROOT/fr13_b4_pool16_refill_gate.json"
)
[[ -n "$EXACT4_REFERENCE" ]] && reduce_args+=(--exact4-reference "$EXACT4_REFERENCE")
"$PYTHON_BIN" scripts/fr13_b4_floor_gate_reduce.py "${reduce_args[@]}"
rc=$?
echo "===== B4 POOL16 REFILL GATE DONE rc=$rc ====="
echo "verdict: $GATE_ROOT/fr13_b4_pool16_refill_gate.json"
exit "$rc"

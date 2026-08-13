#!/usr/bin/env bash
# FR13 B4 HYDRA27-ONLY SEALING CAMPAIGN -- N paired passes at the width-4
# operating point, stock dispatch vs the padded GQA-pair candidate.
#
# MARK'S RULING 2026-08-13: "seal hydra27 only".
#
# WHAT THIS CONVERTS, AND FROM WHAT
#   The width-4 screen (2026-08-13, output/fr13_b4_width4_timing_padded_*)
#   returned REVERSES_THE_EXACT4_NULL at n=1 and said so in its own words: "At
#   n=1 this is a SCREEN result: it is grounds to fund a four-pass paired
#   campaign, not a promotion."  This is that campaign.  It takes the -29.5 ms
#   width-4 screen plus the padded width-3 extension and turns them into a
#   citable claim, or it fails to and the lever closes honestly.
#
# WHAT ONE PASS IS
#   One pass = ONE invocation of scripts/fr13_run_b4_gqa_width4_timing.sh, which
#   serves BOTH arms -- stock dispatch and the padded GQA-pair candidate -- into
#   this campaign's own pass_NN directory at pool16 behind 4 slots on Hydra27,
#   then reduces that pair.  Nothing here re-implements measurement, attestation
#   or statistics; this script supplies repetition, ARM BALANCE and provenance,
#   exactly as fr13_b4_formal_floor_gate.sh does for the topology contrast.
#
# WHY THE ARM ORDER ALTERNATES -- the whole reason this wrapper exists
#   The second arm of a pass inherits a warmer page cache and a differently-aged
#   host.  In the formal floor gate the two arms of a pass are TOPOLOGIES, so arm
#   position is a nuisance variable and alternating it merely tidies up.  Here
#   the two arms are STOCK and CANDIDATE: arm position aliases DIRECTLY into the
#   contrast being sealed.  Serving stock first in every pass would hand the
#   candidate the warmer host in every pass -- a systematic bias in the
#   candidate's favour that repetition does NOT remove, it only makes more
#   precise.  So ARM_ORDER alternates SC/CS on pass parity and the reducer
#   refuses a campaign whose SC and CS counts are not balanced.
#
# WHY EXACTLY 4 OR 16 PASSES
#   Same reason as the two existing multi-pass gates: only N in {4, 16} lands on
#   the repo's pinned one-sided t criticals (df 3 or 15).  A short campaign is a
#   SCREEN and the reducer returns NOT_EVALUATED_INSUFFICIENT_PASSES rather than
#   inventing a df=1 constant to rescue it.
#
# WHAT IT DOES NOT CLAIM
#   Hydra27 only -- Tail23 is NOT sealed by this campaign and its width-4 MDE
#   (6.42 ms) is not used here.  This is a timing class, not the formal
#   statistical hardware-floor acceptance gate, and not the exact16 agent-quality
#   band (Mark's 2026-08-10 ruling that exact16 is QUALITY CONTROL is untouched).
#
# COST.  ~1.4 h stock arm + ~1.6 h candidate arm ~= 3 h per pass, ~12 h for four.
# Every pass is self-contained evidence and the reducer is offline and
# idempotent, so the campaign can be read at any point and stopped early.
#
# USAGE (must be launched DETACHED -- a 120 s tool timeout SIGTERMs the group)
#   setsid nohup env PASSES=4 \
#     QROW32_GQA_PAIR_FA2_SO=... QROW32_GQA_PAIR_FA2_SOURCE=... \
#     QROW32_GQA_PAIR_DUAL_GATE_JSON=... QROW32_GQA_PAIR_DUAL_GATE_SHA256=... \
#     bash scripts/fr13_b4_hydra27_sealing_campaign.sh \
#     > /home/mark/shared/hydra27_seal.log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

PASSES=${PASSES:-4}
# MAKEUP MODE: append passes to an EXISTING campaign root instead of starting a
# new one. Used when INFRASTRUCTURE (not the stack) destroys a pass. The
# replacement pass MUST reuse the dead pass's SC/CS slot, so the order is forced
# rather than derived from index parity -- otherwise a makeup pass silently
# unbalances the arm-position design this campaign is built on.
MAKEUP_CAMPAIGN_ROOT=${MAKEUP_CAMPAIGN_ROOT:-}
PASS_INDEX_START=${PASS_INDEX_START:-0}
ARM_ORDER_OVERRIDE=${ARM_ORDER_OVERRIDE:-}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=hydra27_fixed32
TIMING_RUNNER=scripts/fr13_run_b4_gqa_width4_timing.sh
REDUCER=scripts/fr13_b4_hydra27_sealing_reduce.py

: "${QROW32_GQA_PAIR_FA2_SO:?set QROW32_GQA_PAIR_FA2_SO to the pinned candidate binary}"
: "${QROW32_GQA_PAIR_FA2_SOURCE:?set QROW32_GQA_PAIR_FA2_SOURCE to the regenerated FA2 source}"
: "${QROW32_GQA_PAIR_DUAL_GATE_JSON:?set it to the dual raw-byte gate PASS produced at HEAD}"
: "${QROW32_GQA_PAIR_DUAL_GATE_SHA256:?set it to that PASS artifact SHA-256}"

if [[ -z "$MAKEUP_CAMPAIGN_ROOT" ]]; then
  case "$PASSES" in
    4|16) ;;
    *) echo "FAIL: PASSES must be 4 or 16 (pinned t criticals); got $PASSES" >&2; exit 2 ;;
  esac
fi
case "$ARM_ORDER_OVERRIDE" in
  ""|SC|CS) ;;
  *) echo "FAIL: ARM_ORDER_OVERRIDE must be SC or CS" >&2; exit 2 ;;
esac

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SOURCE_COMMIT=$(git rev-parse HEAD)
if [[ -n "$MAKEUP_CAMPAIGN_ROOT" ]]; then
  CAMPAIGN_ROOT=$(realpath -m "$MAKEUP_CAMPAIGN_ROOT")
  [[ -d "$CAMPAIGN_ROOT" ]] \
    || { echo "FAIL: makeup campaign root does not exist: $CAMPAIGN_ROOT" >&2; exit 2; }
else
  CAMPAIGN_ROOT="$REPO/output/fr13_b4_hydra27_sealing_campaign_${STAMP}"
fi

# ---------------------------------------------------------------- preflight --
[[ -f "$TIMING_RUNNER" && ! -L "$TIMING_RUNNER" ]] \
  || { echo "FAIL: missing $TIMING_RUNNER" >&2; exit 2; }
[[ -f "$REDUCER" && ! -L "$REDUCER" ]] \
  || { echo "FAIL: missing $REDUCER" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "FAIL: no $PYTHON_BIN" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "FAIL: cannot resolve HEAD" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "FAIL: tracked worktree must be clean" >&2; exit 2; }
# The credential is void off this commit, so a campaign that would move HEAD
# under itself is refused before it spends a single GPU-hour.
[[ "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "FAIL: source commit must be pushed before the campaign" >&2; exit 2; }
if [[ -z "$MAKEUP_CAMPAIGN_ROOT" ]]; then
  [[ ! -e "$CAMPAIGN_ROOT" && ! -L "$CAMPAIGN_ROOT" ]] \
    || { echo "FAIL: campaign root must be new: $CAMPAIGN_ROOT" >&2; exit 2; }
fi
# EVERY PRECONDITION MUST BE ONE THE RUN CAN SATISFY: resolve the reducer before
# any GPU time is spent, not hours later at reduce time.
"$PYTHON_BIN" "$REDUCER" --self-check \
  || { echo "FAIL: sealing reducer failed its own self-check" >&2; exit 2; }

# GPU coordination: never contend with another campaign.
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "FAIL: Docker is not clean -- another campaign may be serving" >&2; exit 2; }
if pgrep -af '[f]r13_bigdenom_swe_serve_variant|[f]r13_b4_campaign_driver' >/dev/null 2>&1; then
  echo "FAIL: an fr13 serve/driver process is already running" >&2
  exit 2
fi

mkdir -p "$CAMPAIGN_ROOT"
{
  printf 'campaign_root=%s\n' "$CAMPAIGN_ROOT"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'passes=%s\n' "$PASSES"
  printf 'topology=%s\n' "$FIXED32_MODE"
  printf 'logical_topology=Hydra27\n'
  printf 'timing_runner=%s\n' "$TIMING_RUNNER"
  printf 'timing_runner_sha256=%s\n' "$(sha256sum "$TIMING_RUNNER" | awk '{print $1}')"
  printf 'reducer_sha256=%s\n' "$(sha256sum "$REDUCER" | awk '{print $1}')"
  printf 'dual_gate_sha256=%s\n' "$QROW32_GQA_PAIR_DUAL_GATE_SHA256"
  printf 'started=%s\n' "$(date -u +%FT%TZ)"
} > "$CAMPAIGN_ROOT/campaign_meta.txt"

echo "===== B4 HYDRA27 SEALING CAMPAIGN $STAMP ($PASSES paired passes) ====="

# ------------------------------------------------------------------- passes --
completed=0
for (( offset=0; offset<PASSES; offset++ )); do
  index=$(( PASS_INDEX_START + offset ))
  pass_dir="$CAMPAIGN_ROOT/pass_$(printf '%02d' "$index")"
  run_root="$CAMPAIGN_ROOT/run_$(printf '%02d' "$index")"
  # Balance stock-first/candidate-first position across passes.
  if (( index % 2 == 0 )); then arm_order=SC; else arm_order=CS; fi
  [[ -n "$ARM_ORDER_OVERRIDE" ]] && arm_order=$ARM_ORDER_OVERRIDE
  [[ ! -e "$pass_dir" && ! -e "$run_root" ]] \
    || { echo "FAIL: pass or run dir already exists for index $index" >&2; exit 2; }

  echo "----- pass $index (ARM_ORDER=$arm_order) -> $pass_dir -----"
  printf 'pass=%s arm_order=%s started=%s\n' \
    "$index" "$arm_order" "$(date -u +%FT%TZ)" >> "$CAMPAIGN_ROOT/campaign_meta.txt"

  # The timing runner owns RUNROOT (its own manifests, attestations and pair
  # reduction) and writes its ARMS into PASS_ROOT/pass_NN, so the sealed window
  # reducer sees the whole campaign as pass_00..pass_NN of one root.
  if env \
      FR13_RUN_B4_GQA_WIDTH4_TIMING=1 \
      RUNROOT="$run_root" \
      TAG="seal${index}" \
      PASS_ROOT="$CAMPAIGN_ROOT" \
      PASS_INDEX="$index" \
      ARM_ORDER="$arm_order" \
      QROW32_GQA_PAIR_FIXED32_MODE="$FIXED32_MODE" \
      QROW32_GQA_PAIR_FA2_SO="$QROW32_GQA_PAIR_FA2_SO" \
      QROW32_GQA_PAIR_FA2_SOURCE="$QROW32_GQA_PAIR_FA2_SOURCE" \
      QROW32_GQA_PAIR_DUAL_GATE_JSON="$QROW32_GQA_PAIR_DUAL_GATE_JSON" \
      QROW32_GQA_PAIR_DUAL_GATE_SHA256="$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
      PYTHON_BIN="$PYTHON_BIN" \
      bash "$TIMING_RUNNER" \
      > "$CAMPAIGN_ROOT/pass_$(printf '%02d' "$index").runlog" 2>&1; then
    completed=$(( completed + 1 ))
    printf 'pass=%s rc=0 ended=%s\n' "$index" "$(date -u +%FT%TZ)" \
      >> "$CAMPAIGN_ROOT/campaign_meta.txt"
  else
    rc=$?
    # A failed pass is RECORDED, not repaired and not retried in place: the
    # reducer excludes it with a reason and reports how many survived. Retrying
    # in place would silently replace a pass whose failure may BE the result.
    echo "FAIL: pass $index rc=$rc (recorded; pass will be excluded)" >&2
    printf 'pass=%s rc=%s ended=%s\n' "$index" "$rc" "$(date -u +%FT%TZ)" \
      >> "$CAMPAIGN_ROOT/campaign_meta.txt"
  fi

  # Evidence-first teardown: containers must be gone before the next pass boots,
  # and if they are not, the evidence is captured BEFORE anything is removed.
  if [[ "$(docker ps -aq | wc -l)" -ne 0 ]]; then
    mkdir -p "$pass_dir"
    docker ps -a > "$pass_dir/docker_ps_after_pass.txt" 2>&1 || true
    for cid in $(docker ps -aq); do
      docker logs "$cid" > "$pass_dir/docker_logs_${cid}.txt" 2>&1 || true
    done
    echo "FAIL: containers survived pass $index; evidence captured, stopping" >&2
    break
  fi
done

printf 'passes_completed=%s ended=%s\n' "$completed" "$(date -u +%FT%TZ)" \
  >> "$CAMPAIGN_ROOT/campaign_meta.txt"

# ------------------------------------------------------------------ reduce ---
echo "===== reducing $completed/$PASSES passes ====="
"$PYTHON_BIN" "$REDUCER" \
  --campaign-root "$CAMPAIGN_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --dual-gate-sha256 "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  --min-passes 4 \
  --out "$CAMPAIGN_ROOT/fr13_b4_hydra27_sealing_campaign.json"
rc=$?
echo "===== B4 HYDRA27 SEALING CAMPAIGN DONE rc=$rc ====="
echo "verdict: $CAMPAIGN_ROOT/fr13_b4_hydra27_sealing_campaign.json"
exit "$rc"

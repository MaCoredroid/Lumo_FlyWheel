#!/usr/bin/env bash
# FR13 B4 EXACT16 AGENT-QUALITY-CONTROL GATE -- ONE uncapped 16-task pass served
# through the SEALED B4 GQA-pair PADDED candidate at production widths {3,4}.
#
# MARK'S RULING (2026-08-12, restated 2026-08-13): exact16 is AGENT QUALITY
# CONTROL at batched-optimization milestones.  It proves the speed levers did not
# degrade task solving.  It is NOT a timing run and this script therefore emits
# NO timing statistic, NO deploy-speed reduction and NO acceptance claim.  The
# per-task /metrics brackets the serve variant writes anyway are left unreduced
# on purpose: an unreduced bracket cannot be mistaken for a timing verdict.
#
# WHY ONE ARM AND NOT A PAIR
#   The paired stock/candidate contrast is the SEALING campaign's job and it is
#   already done (output/fr13_b4_hydra27_sealing_campaign_*).  The QC question is
#   narrower and one-sided: does the shipping candidate stack still land inside
#   the historical agent-behaviour band?  A second stock arm would double the GPU
#   cost, answer a timing question this class does not ask, and invite exactly
#   the timing read the ruling forbids.  The comparator is the BANKED historical
#   profile, supplied to the reducer as reference arms.
#
# THE CAP IS OFF, STRUCTURALLY
#   Mark requires QC to observe UNCAPPED agent behaviour.  At this commit the
#   per-task campaign budget cap does not exist -- it lives on
#   codex/fr13-b1-flip-and-13398-cap-20260813 (6cbfcab9d) and is not an ancestor
#   of HEAD.  AGENT_WALL_S is additionally passed EMPTY, so no --agent-wall-s
#   reaches run_swe_bench_q36_a.py and its default (0 = unlimited) stands.  Both
#   facts are asserted below rather than assumed, and recorded in launcher_meta.
#
# WHAT IS SERVED
#   Byte-for-byte the sealing campaign's CANDIDATE arm: pool16 behind 4 slots,
#   B=4, concurrency 4, Hydra27, refill ledger on, root-64K draft vocabulary,
#   mamba spec-block narrowing at its shipped default (ON since 749f83af6), and
#   FR13_FA2_QROW32_B4_PRODUCTION_ARM=gqa_pair -- the padded candidate serving
#   real traffic at BOTH byte-gated widths 3 and 4.
#
# WHAT THIS DOES NOT DECIDE
#   Flipping gqa_pair-padded to the DEFAULT B4 serving arm is NOT ruled by Mark.
#   This gate validates exactly that candidate configuration so his decision has
#   agent-quality evidence under it.  It confers no default.
#
# USAGE (must be launched DETACHED -- a 120 s tool timeout SIGTERMs the group)
#   setsid nohup env \
#     QROW32_GQA_PAIR_FA2_SO=... QROW32_GQA_PAIR_FA2_SOURCE=... \
#     QROW32_GQA_PAIR_DUAL_GATE_JSON=... QROW32_GQA_PAIR_DUAL_GATE_SHA256=... \
#     bash scripts/fr13_b4_exact16_qc_gate.sh \
#     > /home/mark/shared/exact16_qc.log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TAG=${TAG:-qc0}
FIXED32_MODE=${QROW32_GQA_PAIR_FIXED32_MODE:-hydra27_fixed32}
RUN_CLASS=exact16_qc
GATE=scripts/fr13_fa2_qrow32_gqa_pair_gate.py
SIDECAR=scripts/fr13_qrow32_b4_pass_sidecar.py
PATCH_SOURCE=scripts/fr13_patch_fa2_tree_bias.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
REDUCER=scripts/fr13_b4_exact16_qc_reduce.py

# EVIDENCE_SETS[16] -- byte-pinned identically to fr13_floor_gate.EVIDENCE_SETS.
SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398,astropy__astropy-13453,astropy__astropy-13579,astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
TASK_COUNT=16
SLOTS=4
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
CANDIDATE_SHA256=af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb
CANDIDATE_BYTES=299813360
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81
SELECTOR_SENTINEL=131092
B4_KV_CACHE_MEMORY_BYTES=49392123904
DRAFT_VOCAB_ROOT=1
DRAFT_VOCAB_K=65536
# The commit that made mamba spec-block narrowing the shipped fixed32 B4 default.
# One of the three levers this QC pass is validating; asserted to be an ancestor.
MAMBA_NARROWING_COMMIT=749f83af6
# The per-task campaign budget cap. QC must see UNCAPPED behaviour, so this
# commit must NOT be reachable from HEAD.
TASK_BUDGET_CAP_COMMIT=6cbfcab9d
LAUNCH_CLASSIFICATION=real_swe_verified_pool16_b4_fa2_qrow32_gqa_pair_width34_exact16_qc

: "${QROW32_GQA_PAIR_FA2_SO:?set QROW32_GQA_PAIR_FA2_SO to the pinned candidate binary}"
: "${QROW32_GQA_PAIR_FA2_SOURCE:?set QROW32_GQA_PAIR_FA2_SOURCE to the regenerated FA2 source}"
: "${QROW32_GQA_PAIR_DUAL_GATE_JSON:?set it to the dual raw-byte gate PASS produced at HEAD}"
: "${QROW32_GQA_PAIR_DUAL_GATE_SHA256:?set it to that PASS artifact SHA-256}"

case "$FIXED32_MODE" in
  hydra27_fixed32) LOGICAL_TOPOLOGY=Hydra27; ACTIVE_DRAFTS=27; VALID_MASK=0x7abdffff ;;
  tail6_fixed32)   LOGICAL_TOPOLOGY=Tail23;  ACTIVE_DRAFTS=23; VALID_MASK=0x7a9ce7ff ;;
  *) echo "FAIL: QROW32_GQA_PAIR_FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2; exit 2 ;;
esac

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNROOT_ABS="$REPO/output/fr13_b4_exact16_qc_${STAMP}"
ARM="${FIXED32_MODE}_exact16_qc_gqa_pair_b4_${TAG}"
ARM_ROOT="$RUNROOT_ABS/arm_root"
RUNNER_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------- preflight --
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "FAIL: TAG contains unsafe characters" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "FAIL: no $PYTHON_BIN" >&2; exit 2; }
for f in "$GATE" "$SIDECAR" "$PATCH_SOURCE" "$SEQUENCE" "$REDUCER" "$SUBSET"; do
  [[ -f "$f" && ! -L "$f" ]] || { echo "FAIL: missing $f" >&2; exit 2; }
done
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "FAIL: 16-task subset is not the canonical byte-pinned set" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "FAIL: pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
for input in "$QROW32_GQA_PAIR_FA2_SO" "$QROW32_GQA_PAIR_DUAL_GATE_JSON"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "FAIL: input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
[[ "$QROW32_GQA_PAIR_FA2_SOURCE" == /* && -d "$QROW32_GQA_PAIR_FA2_SOURCE" \
   && ! -L "$QROW32_GQA_PAIR_FA2_SOURCE" ]] \
  || { echo "FAIL: FA2 source must be an absolute non-symlink directory" >&2; exit 2; }
[[ "$(stat -c '%s' "$QROW32_GQA_PAIR_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$QROW32_GQA_PAIR_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "FAIL: QROW32_GQA_PAIR_FA2_SO is not the pinned dual-gate candidate" >&2; exit 2; }
[[ "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$QROW32_GQA_PAIR_DUAL_GATE_JSON" | awk '{print $1}')" == "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" ]] \
  || { echo "FAIL: dual raw-byte gate PASS identity mismatch" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "FAIL: cannot resolve HEAD" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "FAIL: tracked worktree must be clean" >&2; exit 2; }
[[ "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "FAIL: source commit must be pushed before the QC pass" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "FAIL: runroot must be new: $RUNROOT_ABS" >&2; exit 2; }

# THE MILESTONE BATCH, ASSERTED RATHER THAN ASSUMED.
git merge-base --is-ancestor "$MAMBA_NARROWING_COMMIT" HEAD \
  || { echo "FAIL: mamba narrowing default $MAMBA_NARROWING_COMMIT is not in this tree" >&2; exit 2; }
if git cat-file -e "${TASK_BUDGET_CAP_COMMIT}^{commit}" 2>/dev/null \
   && git merge-base --is-ancestor "$TASK_BUDGET_CAP_COMMIT" HEAD 2>/dev/null; then
  echo "FAIL: the per-task budget cap $TASK_BUDGET_CAP_COMMIT is reachable from HEAD;" >&2
  echo "      exact16 QC must observe UNCAPPED agent behaviour" >&2
  exit 2
fi
# The cap's own env symbol, checked where it would have to be implemented. This
# script mentions the symbol only inside this guard, so scripts/ is scanned with
# this file excluded rather than by a bare recursive grep that would match itself.
CAP_ENV_SYMBOL='FR13_CAMPAIGN_TASK_BUDGET_S'
if grep -rln --exclude="$(basename "$RUNNER_PATH")" "$CAP_ENV_SYMBOL" scripts/ 2>/dev/null | grep -q .; then
  echo "FAIL: task-budget-cap machinery is present in scripts/; QC must run uncapped" >&2
  exit 2
fi

# EVERY PRECONDITION MUST BE ONE THE RUN CAN SATISFY: the reducer is resolved
# BEFORE any GPU time is spent, not three hours later at reduce time.
"$PYTHON_BIN" "$REDUCER" --self-check \
  || { echo "FAIL: exact16 QC reducer failed its own self-check" >&2; exit 2; }

# GPU coordination: never contend with another campaign.
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "FAIL: Docker is not clean -- another campaign may be serving" >&2; exit 2; }
if pgrep -af '[f]r13_bigdenom_swe_serve_variant|[f]r13_b4_campaign_driver' >/dev/null 2>&1; then
  echo "FAIL: an fr13 serve/driver process is already running" >&2
  exit 2
fi

mkdir -p "$RUNROOT_ABS" "$ARM_ROOT"
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
GATE_SHA256=$(sha256sum "$GATE" | awk '{print $1}')
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
REDUCER_SHA256=$(sha256sum "$REDUCER" | awk '{print $1}')

"$PYTHON_BIN" "$GATE" validate-candidate \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
  > "$RUNROOT_ABS/candidate_identity.at_launch.json" \
  || { echo "FAIL: candidate identity validation failed" >&2; exit 2; }
"$PYTHON_BIN" "$SIDECAR" validate \
  --dual-gate "$QROW32_GQA_PAIR_DUAL_GATE_JSON" \
  --expected-dual-gate-sha256 "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm gqa_pair \
  --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/dual_gate_binding.at_launch.json" \
  || { echo "FAIL: dual-gate binding validation failed" >&2; exit 2; }
DUAL_GATE_BINDING_SHA256=$(
  sha256sum "$RUNROOT_ABS/dual_gate_binding.at_launch.json" | awk '{print $1}'
)

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
# Narrowing must be at its shipped default -- this pass is validating that
# default, so a stale override in the environment would invalidate the QC read.
[[ "${FR13_MAMBA_SPEC_BLOCKS_CDIV:-}" == "1" ]] \
  || { echo "FAIL: mamba narrowing is not at its shipped default (=1)" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=%s\nrun_class=%s\ntiming_eligible=0\nformal_floor_acceptance_eligible=0\nacceptance_valid=0\nqc_only=1\ndoes_not_claim=timing,acceptance,exact4_comparability,tier_b_qualification\ncandidate_arm_selector=gqa_pair\nproduction_widths=3,4\nselector_sentinel=%s\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=4\nconcurrency=4\nslots=%s\ntask_pool=%s\ntask_refill=1\nagent_wall=none\ntask_budget_cap=absent_at_this_commit\nmamba_spec_blocks_cdiv=%s\nfixed_rows=128\ntask_ids=%s\nsubset=%s\nsubset_sha256=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nlauncher_pid=%s\nrunroot=%s\narm_root=%s\narm=%s\nsource_commit=%s\ncandidate_so=%s\ncandidate_so_sha256=%s\ncandidate_so_size=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\ndual_gate_json=%s\ndual_gate_sha256=%s\ndual_gate_binding_sha256=%s\nrunner_sha256=%s\ngate_sha256=%s\nsidecar_sha256=%s\npatch_source_sha256=%s\nreducer_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$LAUNCH_CLASSIFICATION" "$RUN_CLASS" "$SELECTOR_SENTINEL" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$SLOTS" "$TASK_COUNT" "$FR13_MAMBA_SPEC_BLOCKS_CDIV" \
  "$TASK_IDS" "$SUBSET" "$SUBSET_SHA256" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$$" "$RUNROOT_ABS" "$ARM_ROOT" "$ARM" "$SOURCE_COMMIT" \
  "$QROW32_GQA_PAIR_FA2_SO" "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$FA2_HEAD" \
  "$SOURCE_CLOSURE_SHA256" "$QROW32_GQA_PAIR_DUAL_GATE_JSON" \
  "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" "$DUAL_GATE_BINDING_SHA256" \
  "$RUNNER_SHA256" "$GATE_SHA256" "$SIDECAR_SHA256" "$PATCH_SOURCE_SHA256" \
  "$REDUCER_SHA256" "$B4_KV_CACHE_MEMORY_BYTES" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

echo "===== B4 EXACT16 QC GATE $STAMP -- $ARM (production_arm=gqa_pair, widths 3,4) ====="
echo "runroot: $RUNROOT_ABS"

# ---------------------------------------------------------------- the serve --
# Byte-for-byte the sealing campaign's CANDIDATE arm env, with the timing-arm
# retag dropped: this pass serves the production arm only and measures nothing.
# AGENT_WALL_S is EMPTY on purpose -- QC observes uncapped agent behaviour.
serve_rc=0
if env \
    RUNROOT="$ARM_ROOT" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
    FR13_B4_TASK_REFILL=1 \
    KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
    FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
    FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
    FR13_NEEDS_ALLOW= \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${ARM}.json" \
    FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${ARM}_dfwd.json" \
    FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${ARM}_cfwd.json" \
    FR13_FA2_QROW32_B4_TIMING_ARM=gqa_pair \
    FR13_FA2_QROW32_B4_PRODUCTION_ARM=gqa_pair \
    FR13_FA2_QROW32_B4_DUAL_GATE_JSON="$QROW32_GQA_PAIR_DUAL_GATE_JSON" \
    FR13_FA2_QROW32_B4_DUAL_GATE_SHA256="$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
    FR13_FA2_QROW32_B4_EXACT4_TASK_IDS="$TASK_IDS" \
    FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256="$SUBSET_SHA256" \
    FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
    FR13_FA2_QROW32_SO_SHA256="$CANDIDATE_SHA256" \
    FR13_FA2_QROW32_SO_SIZE="$CANDIDATE_BYTES" \
    FR13_FA2_QROW32_FA2_HEAD="$FA2_HEAD" \
    FR13_FA2_QROW32_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
    FR13_FA2_QROW32_SOURCE_COMMIT="$SOURCE_COMMIT" \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB_ARM= \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$QROW32_GQA_PAIR_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$FIXED32_MODE" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  :
else
  serve_rc=$?
  echo "FAIL: serve variant rc=$serve_rc (evidence retained; see $RUNROOT_ABS/$ARM.runlog)" >&2
fi
printf 'arm=%s serve_rc=%s ended=%s\n' "$ARM" "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

# -------------------------------------------------- evidence-first teardown --
if [[ "$(docker ps -aq | wc -l)" -ne 0 ]]; then
  docker ps -a > "$RUNROOT_ABS/docker_ps_after_arm.txt" 2>&1 || true
  for cid in $(docker ps -aq); do
    docker logs "$cid" > "$RUNROOT_ABS/docker_logs_${cid}.txt" 2>&1 || true
  done
  echo "FAIL: containers survived the QC arm; evidence captured" >&2
fi

# ------------------------------------------------------- served-what checks --
ARM_DIR="$ARM_ROOT/$ARM"
CONTAINER_ENV="$ARM_DIR/container_env.txt"
selector_ok=1
if [[ -f "$CONTAINER_ENV" && ! -L "$CONTAINER_ENV" ]]; then
  [[ "$(grep -Fxc "FR13_FIXED32_MODE=$FIXED32_MODE" "$CONTAINER_ENV")" -eq 1 \
     && "$(grep -Fxc "FR13_B4_TASK_REFILL=1" "$CONTAINER_ENV")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_PRODUCTION_ARM=gqa_pair" "$CONTAINER_ENV")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_SO_SHA256=$CANDIDATE_SHA256" "$CONTAINER_ENV")" -eq 1 \
     && "$(grep -Fxc "FR13_MAMBA_SPEC_BLOCKS_CDIV=1" "$CONTAINER_ENV")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_LIVE_PAGED_AB=0" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "FAIL: the QC arm did not serve the declared padded production selector" >&2; selector_ok=0; }
else
  echo "FAIL: the QC arm lacks a regular container environment artifact" >&2
  selector_ok=0
fi
for required in \
    "$ARM_DIR/logs/fr13_fa2_qrow32_b4_production_engagement.json" \
    "$ARM_DIR/logs/fr13_fa2_qrow32_b4_production_pass.json" \
    "$ARM_DIR/swe_out/verified/fr13_task_refill_ledger.jsonl" \
    "$ARM_DIR/swe_out/verified/fr13_task_refill_summary.json" \
    "$ARM_DIR/health.json"; do
  [[ -f "$required" && ! -L "$required" ]] \
    || { echo "FAIL: the QC arm lacks $required" >&2; selector_ok=0; }
done

# The launch-time manifests must still describe the tree that just served.
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "FAIL: runtime/source manifest changed during the QC pass" >&2; selector_ok=0; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "FAIL: external manifest changed during the QC pass" >&2; selector_ok=0; }
"$PYTHON_BIN" "$GATE" validate-candidate \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
  > "$RUNROOT_ABS/candidate_identity.at_end.json"
cmp -s "$RUNROOT_ABS/candidate_identity.at_launch.json" "$RUNROOT_ABS/candidate_identity.at_end.json" \
  || { echo "FAIL: candidate binary/source identity changed during the QC pass" >&2; selector_ok=0; }
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "FAIL: frozen source changed during the QC pass" >&2; selector_ok=0; }
printf 'selector_ok=%s ended=%s\n' "$selector_ok" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

echo "===== B4 EXACT16 QC GATE SERVE DONE serve_rc=$serve_rc selector_ok=$selector_ok ====="
echo "arm dir: $ARM_DIR"
echo "reduce with: $PYTHON_BIN $REDUCER --repo $REPO --qc-runroot $RUNROOT_ABS ..."
if (( serve_rc != 0 || selector_ok == 0 )); then
  exit 3
fi
exit 0

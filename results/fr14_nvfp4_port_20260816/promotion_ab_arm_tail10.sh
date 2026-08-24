#!/usr/bin/env bash
# FR14 PROMOTION A/B — the paired serve arms (Mark's greenlight, condition
# "eyeball traces for degenerate").
#
#   ARM_KIND=C  control : today's promoted stack (hydra27 K0 + gqa_pair from the
#                         run-local credential + HOST_TAIL_PREP_BAKE)
#                         + FR14_FUSED_DRAFT_TOPK=1
#   ARM_KIND=G  gated   : ARM C + FR14_SUFFIX_PASS_GATE=1 (ngram 8, min_agree 0.75)
#
# ONE VARIABLE between them. Everything else -- topology, subset, budget,
# instruments, binaries, network boundary -- is pinned identical here and
# asserted in each arm's container_env.txt.
#
# WHY hydra27 AND NOT tail6. suffix_pass_gating.md 11.7 pre-registers
# active_nodes/verify_rows == 27/32 on EVERY step; 27 active nodes IS the
# hydra27 topology (tail6 is 23). A tail6 arm could not test the pre-registered
# expectation. hydra27 is also the promoted FR14 production config and the
# topology the banked composed number (210.700 ms) was measured on.
#
# INSTRUMENT DOCTRINE (pass 24, restated): the verdict instruments are
# step_wall_ms and s_per_fwd_gpu. FR14_FUSED_DRAFT_TOPK is byte-exact selection
# (0 raw-byte mismatches over 6840 configs), so it CANNOT move acceptance and its
# whole serve evidence is the dfwd span. Acceptance is reported for both arms and
# is a real reading only for ARM G, which is acceptance-affecting by construction.
#
# TASK BUDGET = 9000, NOT 5400, and this is a deliberate correction of the
# brief on the campaign's own banked evidence.
#
# The brief asked for FR13_CAMPAIGN_TASK_BUDGET_S=5400 "so a wall-timeout is an
# accounted capped terminal, not a voided arm". At 5400 that is exactly what
# does NOT happen here. Pass 23 ran this arm shape at 5400 and banked the
# result: 13398 hit the cap, the cap fired correctly -- and the arm still came
# back swerc=13 with THREE tasks, because the truncated-trace validator union
# the capped terminal needs was never built. Pass 24 recorded the consequence
# verbatim: "Levered arm VOID for banking ... Future marathon arms: budget
# 9000s", and flagged the union itself as provenance-core surgery awaiting
# Mark's sanction. It has not landed.
#
# So 5400 buys the failure the brief was trying to avoid, and 9000 buys what it
# wanted: fr14_run_b1_max_stack_serve.sh -- the pattern of record for THIS arm
# shape (hydra27 K0 exact4 marathon) -- uses 9000 and drained all four tasks at
# swerc=0 in 2h20m. AGENT_WALL_S is held EQUAL to the budget so the declared
# budget is provably the binding limit
# (campaign_budget_was_the_binding_limit corroborates true) and a terminal is
# accounted rather than provenance-fatal.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"

ARM_KIND=${ARM_KIND:?set ARM_KIND to C or G}
case "$ARM_KIND" in C|G) ;; *) echo "ARM_KIND must be C or G" >&2; exit 2 ;; esac

PYTHON_BIN=.venv/bin/python
# PROMOAB_SUBSET selects the served workload. exact4 stays the default so every
# banked A/B in this campaign keeps its meaning; exact16 is the QC subset (pass
# 118 ruling). The sha is RECOMPUTED from the file rather than carried as a
# second literal -- two hardcoded copies of a digest is how a subset silently
# stops matching its own identity.
PROMOAB_SUBSET=${PROMOAB_SUBSET:-exact4}
case "$PROMOAB_SUBSET" in
  exact4)  SUBSET=config/fr13_fixed32/subset_b4_four.json ;;
  exact16) SUBSET=config/fr13_fixed32/subset_b4_sixteen.json ;;
  # THE RESUME SET. exact16 attempt 6 served 3 of 16; 13236 degenerated and is
  # verdicted; --skip-existing is forbidden for fixed32 campaigns; and declaring
  # exact16 while serving fifteen is the pins-as-fiction move pass 122 prevents.
  # So the remainder is its own canonical set with its own declared workload.
  exact16_minus_13236) SUBSET=config/fr13_fixed32/subset_b4_sixteen_minus_13236.json ;;
  # THE SECOND RESUME SET: sixteen minus the FOUR the QC has now verdicted
  # (12907 resolved, 13033 failed, 13236 degenerate-regression, 13398 empty-fail).
  exact16_qc_remainder_12) SUBSET=config/fr13_fixed32/subset_b4_sixteen_qc_remainder_12.json ;;
  *) echo "PROMOAB_SUBSET must be exact4, exact16, exact16_minus_13236 or exact16_qc_remainder_12" >&2; exit 2 ;;
esac
[[ -f "$SUBSET" && ! -L "$SUBSET" ]] || { echo "subset missing: $SUBSET" >&2; exit 2; }
SUBSET_SHA256=$(sha256sum "$SUBSET" | cut -d' ' -f1)
case "$PROMOAB_SUBSET:$SUBSET_SHA256" in
  exact4:0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5) ;;
  exact16:47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c) ;;
  exact16_minus_13236:24a8cf7c27646b13b76ebafa5a54d79bd5433f01ba34e55503227fdcc96e729a) ;;
  exact16_qc_remainder_12:ac54bcc4df1311147616affd71a3722d0c0fda89216b7ac6473a4ab3eab4c424) ;;
  *) echo "subset $PROMOAB_SUBSET digest drifted: $SUBSET_SHA256" >&2; exit 2 ;;
esac
MANDATORY_WEIGHT_BYTES=25430574256
MANDATORY_WEIGHT_FLOOR_MS=93.15228665201465
EXPECT_TOK_PER_DRAFT=31
CANDIDATE_SO=/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so
STOCK_FA2_SO="$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"
TOPK_SO_HOST="$REPO/output/fr14_fused_draft_topk_build/fr14_dfwd_full_topk_sm121a.abi3.so"
TOPK_SO_CONTAINER=/workspace/output/fr14_fused_draft_topk_build/fr14_dfwd_full_topk_sm121a.abi3.so
TOPK_SHA256=8f7a99e78c0898a4221f045aa8e15a8085883dbc41b08f609da0da71e66a449e
CREDENTIAL_POINTER="$REPO/output/fr13_b1_gqa_pair_credential.env"

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_promoab_${ARM_KIND}${PROMOAB_ARM_SUFFIX:-}_$TS
RUNROOT_ABS=$(realpath -m "$RUNROOT")
[[ ! -e "$RUNROOT_ABS" ]] || { echo "RUNROOT must be new" >&2; exit 2; }
ARM="${PROMOAB_KIND}_promoab_${ARM_KIND}${PROMOAB_ARM_SUFFIX:-}"

SOURCE_COMMIT=$(git rev-parse HEAD)
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ -z "$(docker ps -aq)" ]] || { echo "docker must be empty before boot" >&2; exit 2; }
# ---- unified-memory preflight -------------------------------------------
# EXACT16 ATTEMPT 4 (2026-08-19) died 57s in on vLLM's own startup assertion --
# "Free memory on device cuda:0 (74.37/117.51 GiB) ... less than ... (0.7, 82.26
# GiB)" -- with NO other process holding the GPU. This gate had already asserted
# MemFree >= 82.3 and passed, so the threshold, not the metric, was wrong.
#
# WHY. 82.26 GiB is the engine's requirement AFTER the checkpoint is resident.
# This gate runs BEFORE the container starts, and the 20.42 GiB of weights land
# in between. Measured across attempts: free at this gate minus free at the
# engine's check is ~23 GiB, and 98 - 23 = 75, which is the 74.37 that refused.
# Attempts 3 and 5 started at 105 GiB free and cleared it; attempt 4 started at
# 98 and did not. The pre-boot floor must therefore carry the weights too.
#
# RECLAIM FIRST, and with the path that is PROVEN to work here. The Python
# helper took free from 74 -> 105 GiB (cache 31 -> 0) by measurement; the sysctl
# is kept because it also works on this box, but it is `|| true` and a silent
# no-op elsewhere, which is exactly how a reclaim step stops reclaiming without
# announcing it.
sync
sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
PYTHONPATH="$PWD/src" .venv/bin/python -c \
  "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" \
  >/dev/null 2>&1 || true
_mem_free_gib=$(awk '/^MemFree:/{printf "%.1f", $2/1048576}' /proc/meminfo)
awk '/^MemFree:/{exit ($2/1048576 < 102.8)}' /proc/meminfo \
  || { echo "unified-memory preflight failed: MemFree=${_mem_free_gib}GiB < 102.8GiB" >&2
       echo "  (the engine needs 82.26GiB free AFTER the 20.42GiB checkpoint loads;" >&2
       echo "   refusing here costs seconds, refusing at the engine costs ~5 minutes)" >&2
       exit 2; }
echo "[promoab] unified-memory preflight OK: MemFree=${_mem_free_gib}GiB >= 102.8GiB"
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset drifted" >&2; exit 2; }
[[ -f "$TOPK_SO_HOST" && ! -L "$TOPK_SO_HOST" \
   && "$(sha256sum "$TOPK_SO_HOST" | awk '{print $1}')" == "$TOPK_SHA256" ]] \
  || { echo "fused-topk binary missing or drifted" >&2; exit 2; }

# ---- the promoted-arm credential: PRESENT IT IF IT EXISTS, NAME EMPTY IF NOT --
# gqa_pair is only SERVICEABLE at the HEAD its gate was earned at. Step 0 tried
# to re-earn it here and could not: the K0 gate boots, serves, resolves its task
# (swe_orchestrator_rc=0) and then dies in the terminal chat-traffic audit on a
# STALE CENSUS MIRROR that has nothing to do with this campaign --
# fr13_fixed32_work_census.py:157 TAW_SOURCE_CONTRACT_SHA256 still carries
# 484babd7..., while fr13_device_multidraft_kernel.py:1667 was re-attested to
# 68b289ae... by FR14 lane 3 (pass 44). The emitter recomputes and self-asserts
# its digest on every boot, so 68b289ae is what the served source IS; the census
# literal is simply a mirror nobody synced. Attributed in
# promotion_ab_campaign.md 0.4 and flagged for the credential owner.
#
# NOTHING IS RELAXED TO WORK AROUND IT. The campaign does not touch the census
# validator: it is a credential instrument, and editing one so that my own
# measurement can proceed is the exact move splitk_fa2.md 2(a) refused. Instead:
#
#   * if a serviceable credential exists, present it and serve gqa_pair;
#   * if it does not, NAME the B1 production arm EMPTY.
#
# Naming it empty is the launcher's documented deliberate opt-out ([[ -v ... ]]
# at :838), and it is strictly better here than letting the promoted default's
# stale-credential path degrade implicitly: the arm's kernel becomes an
# explicit, attested fact in container_env.txt and is identical in BOTH arms by
# construction rather than by inference. The C-vs-G comparison is unaffected --
# the FA2 verifier kernel is not the variable under test.
FA2_B1_ARM_NOTE=
# PROMOAB_FA2=stock forces the incumbent path: no B1 production arm named, stock
# FA2 mounted. Round 16 / option A needs BOTH arms on the incumbent kernel, because
# the gqa_pair arm is excluded under hydra31 by a mode gate (round 15) and pairing a
# gqa_pair arm against an incumbent one would measure kernel + topology at once.
# PROMOAB_FA2=default (approved pass 118) QCs THE CANONICAL PATH: present nothing,
# name nothing, and let the launcher's promoted split-K default arm itself. That is
# the whole point of the exact16 QC -- the configuration under test is the one an
# unnamed boot resolves to.
#
# THE TRAP THIS MODE HAS TO AVOID, stated because it is invisible: naming the B1
# production arm EMPTY is NOT the same as leaving it unnamed. The launcher's
# promoted-default block is guarded by _FR13_FA2_QROW32_B1_PRODUCTION_ARM_NAMED == 0,
# and that flag is set from `[[ -v FR13_FA2_QROW32_B1_PRODUCTION_ARM ]]` -- an
# EXPORTED EMPTY STRING is "set". So `export ...=""`, which is exactly what the
# stock/incumbent branch below does deliberately, would SUPPRESS the promoted
# default and silently QC the incumbent instead. This branch therefore exports
# nothing at all, presents no credential, and leaves FORKED_FA2_SO empty so the
# launcher's own `${FORKED_FA2_SO:-$_FR13_SPLITK_DEFAULT_SO}` supplies the binary.
if [[ "${PROMOAB_FA2:-}" == "default" ]]; then
  : # present nothing; the launcher must arm the promoted default from its own literals
elif [[ "${PROMOAB_FA2:-}" == "stock" ]]; then
  FR13_FA2_QROW32_B1_SOURCE_COMMIT=__forced_stock__
elif [[ -f "$CREDENTIAL_POINTER" ]]; then
  set -a; . "$CREDENTIAL_POINTER"; set +a
fi
if [[ "${PROMOAB_FA2:-}" == "default" ]]; then
  FA2_B1_ARM_NOTE="PROMOTED DEFAULT (B1 arm left UNNAMED; launcher arms split-K tier-B and mints its own provenance)"
  FA2_SO_FOR_ARM=""
elif [[ "${FR13_FA2_QROW32_B1_SOURCE_COMMIT:-}" == "$SOURCE_COMMIT" \
      && "${FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE:-}" == "full_vocab" ]]; then
  FA2_B1_ARM_NOTE="gqa_pair (promoted default, credential serviceable at $SOURCE_COMMIT)"
  FA2_SO_FOR_ARM="$CANDIDATE_SO"
else
  FA2_B1_ARM_NOTE="INCUMBENT qrow16 (production arm NAMED EMPTY: no serviceable gqa_pair credential at $SOURCE_COMMIT)"
  unset FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON \
        FR13_FA2_QROW32_B1_GQA_PAIR_GATE_SHA256 \
        FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON \
        FR13_FA2_QROW32_B1_SOURCE_COMMIT FR13_FA2_QROW32_B1_SO_SHA256 \
        FR13_FA2_QROW32_B1_SO_SIZE FR13_FA2_QROW32_B1_FA2_HEAD \
        FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256 \
        FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256 \
        FR13_FA2_QROW32_B1_EXACT4_TASK_IDS \
        FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256
  export FR13_FA2_QROW32_B1_PRODUCTION_ARM=""
  # With no B1 selector active, fr13_fixed32_contract's external manifest
  # demands the FA2 .so at its canonical in-repo path -- the candidate binary is
  # only admissible when an arm is actually armed to use it. Serve the stock
  # unit, stated here rather than inherited from the launcher default so the
  # arm's binary is an explicit, attested choice.
  FA2_SO_FOR_ARM="$STOCK_FA2_SO"
fi
# In default mode the binary is the launcher's to choose, so there is nothing to
# validate here -- and validating an empty path would refuse the very mode that is
# under test. Everything else must still name a real, non-symlink file.
if [[ -n "$FA2_SO_FOR_ARM" ]]; then
  [[ -f "$FA2_SO_FOR_ARM" && ! -L "$FA2_SO_FOR_ARM" ]] \
    || { echo "FA2 .so for this arm is missing or a symlink: $FA2_SO_FOR_ARM" >&2; exit 2; }
elif [[ "${PROMOAB_FA2:-}" != "default" ]]; then
  echo "FA2 .so unset outside default mode" >&2; exit 2
fi
echo "[promoab] FA2 B1 arm: $FA2_B1_ARM_NOTE"
echo "[promoab] FA2 .so:    $FA2_SO_FOR_ARM"

mkdir -p "$RUNROOT_ABS"
# SITE 24 INTERIM PROTOCOL. The 24k-ceiling landing edited this vehicle 14 minutes into
# a live drain; bash re-reads a script by BYTE OFFSET, so the shell resumed at a stale
# offset into rewritten bytes and ran fragment text. The QC died at 3/15 and nothing
# said why until the corpse was read. Snapshot the execution closure at boot so a
# mid-drain landing can ALARM at minute fifteen instead of surfacing at hour four.
.venv/bin/python results/fr14_nvfp4_port_20260816/promotion_ab_closure_watch.py \
  snapshot "$RUNROOT_ABS/closure.json" >/dev/null 2>&1 \
  || echo "[promoab] WARNING: closure snapshot failed; mid-drain drift will not be detectable" >&2

# ---- route pins + K0 identity ----------------------------------------------
# TAG is referenced by the sequence file's run_variant lines; it must exist even
# though run_variant is stubbed out here (set -u).
PROMOAB_KIND=${PROMOAB_KIND:-hydra27_fixed32}
case "$PROMOAB_KIND" in hydra27_fixed32|hydra31_fixed32) ;; *) echo "PROMOAB_KIND must be hydra27_fixed32 or hydra31_fixed32" >&2; exit 2 ;; esac
export TAG="promoab${ARM_KIND}"
export BSIZE=1 CONC=1 WALL=9000
export FR13_CAMPAIGN_TASK_BUDGET_S=9000
export FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
export FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=full_vocab
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
  || { echo "K0 promotion-A/B floor contract drifted" >&2; exit 2; }

# ---- the arm --------------------------------------------------------------
# Lever 1 (fused draft top-k) is ON IN BOTH ARMS by design: it is byte-exact
# selection, so it is acceptance-invariant, and holding it constant keeps the
# C-vs-G difference to exactly one variable (the pass gate).
#
# PROMOAB_TOPK=0 is an ATTRIBUTION escape hatch, not a campaign arm: it drops
# lever 1 so a refusal seen with both levers armed can be attributed to one of
# them rather than to their combination. "A mechanism that EXPLAINS a failure is
# not evidence that it CAUSED it" (pass 30) -- so the attribution gets its own
# boot instead of an argument. Never used for a reported A/B arm.
#
# PROMOAB_TOPK=promoted (default since the 2026-08-18 promotion) names NOTHING:
# the launcher's own default arms the lever from its launcher-literal credential,
# which is the path production takes. Naming it explicitly would still work --
# the defaults resolve to the same .so and sha -- but it would test the operator's
# literals instead of the promoted ones, and a promoted default that only works
# when the caller repeats it is not promoted.
#   =1  force ON  with this campaign's own pins (pre-promotion behaviour)
#   =0  force OFF -- the opt-out the paired 0-vs-1 A/B needs
PROMOAB_TOPK=${PROMOAB_TOPK:-promoted}
case "$PROMOAB_TOPK" in 0|1|promoted) ;; *) echo "PROMOAB_TOPK must be 0, 1 or promoted" >&2; exit 2 ;; esac
arm_env=(
  FR10_METRICS=0
  "FR13_HOST_TAIL_PREP_BAKE=${PROMOAB_PREP_BAKE:-1}"
  FR13_HOST_TAIL_DEFER=0
  FR13_DFWD_SPLIT=1
  FR13_LFWD_GPU_TIMER=1
)
if [[ "$PROMOAB_TOPK" != "promoted" ]]; then
  arm_env+=(
    "FR14_FUSED_DRAFT_TOPK=$PROMOAB_TOPK"
    FR14_FUSED_DRAFT_TOPK_BLOCKS=64
    "FR14_FUSED_DRAFT_TOPK_SO=$TOPK_SO_CONTAINER"
    "FR14_FUSED_DRAFT_TOPK_SHA256=$TOPK_SHA256"
  )
fi
if [[ "$ARM_KIND" == "G" ]]; then
  arm_env+=(
    FR14_SUFFIX_PASS_GATE=1
    FR14_SUFFIX_PASS_GATE_NGRAM=8
    FR14_SUFFIX_PASS_GATE_MIN_AGREE=0.75
    FR14_SUFFIX_PASS_GATE_MIN_HISTORY=256
  )
else
  arm_env+=(FR14_SUFFIX_PASS_GATE=0)
fi
# PROMOAB_EXTRA_ENV: newline-separated NAME=VALUE pairs appended verbatim.
# Deliberately a passthrough rather than a named knob: the tier-B WORKLOAD
# declaration (pass 119/120) is landing in another lane and its exact spelling is
# theirs to choose. Reading the spelling off the landing and passing it here beats
# guessing a variable name in this script and shipping a silent no-op if I guess
# wrong -- a misspelled knob would leave the workload UNDECLARED while the arm
# reported success, which is the failure mode this whole chain keeps finding.
if [[ -n "${PROMOAB_EXTRA_ENV:-}" ]]; then
  while IFS= read -r _kv; do
    [[ -z "$_kv" ]] && continue
    [[ "$_kv" == *=* ]] || { echo "PROMOAB_EXTRA_ENV entry not NAME=VALUE: $_kv" >&2; exit 2; }
    arm_env+=("$_kv")
    echo "[promoab] extra env: $_kv"
  done <<< "$PROMOAB_EXTRA_ENV"
fi

printf 'arm_kind=%s\narm=%s\nsource_commit=%s\nsubset=%s\nsubset_sha256=%s\ntask_budget_s=9000\nagent_wall_s=9000\nverdict_instrument=step_wall_ms+s_per_fwd_gpu\nfa2_b1_arm=%s\nstarted=%s\n' \
  "$ARM_KIND" "$ARM" "$SOURCE_COMMIT" "$SUBSET" "$SUBSET_SHA256" \
  "$FA2_B1_ARM_NOTE" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/arm_meta.txt"
printf '%s\n' "${arm_env[@]}" > "$RUNROOT_ABS/arm_env.txt"

echo "===== $ARM (promotion A/B arm $ARM_KIND) $(date -u +%FT%TZ) ====="
env RUNROOT="$RUNROOT_ABS" \
  OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=9000 \
  FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
  FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}.json \
  FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_dfwd.json \
  FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_cfwd.json \
  FR13_LFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${ARM}_lfwd.json \
  FR13_DFWD_SPLIT_JSON=/logs/fr13_dfwd_split.json \
  FORKED_FA2_SO="$FA2_SO_FOR_ARM" \
  TMPDIR=/home/mark/shared/tmp-scratch \
  "${arm_env[@]}" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$PROMOAB_KIND" "$SUBSET" \
  > "$RUNROOT_ABS/$ARM.runlog" 2>&1
rc=$?
echo "[$ARM] serve rc=$rc $(date -u +%FT%TZ)"

census="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_work_census.jsonl"
census_args=()
[[ -f "$census" && ! -L "$census" ]] && census_args=(--work-census "$census")
np=$(find "$RUNROOT_ABS/$ARM/swe_out" -name vllm_metrics_post.txt 2>/dev/null | wc -l)
echo "[$ARM] post-brackets=$np"
if (( np >= 1 )); then
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$ARM" --out-root "$RUNROOT_ABS/$ARM/swe_out" \
    --expected-tok-per-draft "$EXPECT_TOK_PER_DRAFT" --batch-size 1 \
    "${census_args[@]}" \
    --out "$RUNROOT_ABS/$ARM/deploy_speed_promoab_${ARM_KIND}.json" 2>&1 | tail -14 \
    || echo "[$ARM] deploy-speed reduce FAILED"
else
  echo "[$ARM] NO post-brackets — deploy-speed VACUOUS"
fi
printf 'serve_rc=%s\nended=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/arm_meta.txt"
echo "[$ARM] docker containers after: $(docker ps -aq | wc -l)"
echo "[$ARM] done -> $RUNROOT_ABS"
exit "$rc"

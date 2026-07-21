# COMMITTER BATCHED validation, BATCHED-FIRST (deployed path; default baseline 52-60ms already measured cb1).
# FR13_COMMITTER_NATIVE_BATCHED=1 routes the LIVE committer to _fr13_native_committer_all_layers_batched
# (hoisted layout + batched gather; BYTE-IDENTICAL committed state). Validate arm1 (batched) EARLY:
#   - needle '[FR13_COMMITTER_NATIVE_BATCHED ENGAGED]', 0 fatal, COHERENT generation (no garble),
#   - accept ~4-5 (NOT collapsed to ~0 => committer correct), committer_gpu DROPS from ~60ms.
# arm2 (default) second for a same-campaign matched committer comparison if arm1 validates.
# Both COMMITTER_NATIVE=1, APC_SNAP_FIX=0, cache-off, PARENT_GATHER=1, --async-scheduling, COMMIT_FULL timer.
# Launch: RUNROOT=output/fr13_combatch2 TAG=cb2 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_committer_batched_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh combatch2
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=0
export FR13_APC_SNAP_FIX=0
export FR13_COMMITTER_NATIVE=1
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm1: BATCHED (validate first) ----
export FR13_COMMITTER_NATIVE_BATCHED=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_combatch2/cf_batched_${TAG}.json
run_variant tail6_batched_${TAG}  tail6  21  1
# ---- arm2: DEFAULT (matched comparison) ----
export FR13_COMMITTER_NATIVE_BATCHED=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_combatch2/cf_default_${TAG}.json
run_variant tail6_default_${TAG}  tail6  21  1

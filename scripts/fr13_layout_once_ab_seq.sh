# LAYOUT-ONCE committer-dispatch A/B (user 2026-07-21: optimize the 60ms native committer replay
# instead of pb; PARK PB). The deployed native committer validates each of 48 layers' staging flags
# via 2 blocking .item() D2H syncs/layer (~96 stalls/step). FR13_COMMITTER_LAYOUT_ONCE=1 batches all
# layers' flags into ONE .tolist() -> kills ~94 syncs. Replay untouched => committed state BIT-IDENTICAL.
# Measures whether the 60.52ms COMMIT_FULL (commdiag) is dispatch-stall-bound (drops => keep optimizing:
# batch gathers + graph-capture) or fused_sigmoid compute-bound (no drop => fold is the only way).
#   arm1 = LAYOUT_ONCE=0 : baseline (per-layer .item() syncs) -- expect ~60ms COMMIT_FULL.
#   arm2 = LAYOUT_ONCE=1 : batched flag validation -- COMMIT_FULL delta = the sync-stall tax.
# Both non-pb tail6, COMMITTER_NATIVE=1 (deployed native committer), cache-OFF (committer cache-indep +
# fast boot), PARENT_GATHER=1, --async-scheduling, FR13_COMMIT_FULL_GPU_TIMER=1. Timer reads early (few k
# drafts) => small subset (subset_collapse3, 3 tasks) suffices. B4. Watch needle '[FR13_COMMITTER_LAYOUT_ONCE ENGAGED]'.
# Launch: RUNROOT=output/fr13_layoutonce TAG=lo1 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_layout_once_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh layoutonce
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=0
export FR13_COMMITTER_NATIVE=1
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm1: baseline (per-layer .item() syncs) ----
export FR13_COMMITTER_LAYOUT_ONCE=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_layoutonce/cf_base_${TAG}.json
run_variant tail6_base_${TAG}  tail6  21  1
# ---- arm2: layout-once (batched flag validation) ----
export FR13_COMMITTER_LAYOUT_ONCE=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_layoutonce/cf_lo_${TAG}.json
run_variant tail6_lo_${TAG}  tail6  21  1

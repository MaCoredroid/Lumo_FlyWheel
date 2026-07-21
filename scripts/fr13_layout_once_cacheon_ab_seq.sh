# LAYOUT-ONCE committer-dispatch A/B, CACHE-ON, arm2-first (user 2026-07-21: "run arm2 then arm1, both
# tail6 cache on, same other config as previous run"). Measures whether the native committer's ~60ms
# COMMIT_FULL is host-dispatch-stall-bound (drops with layout-once) or fused_sigmoid compute-bound (flat).
# FR13_COMMITTER_LAYOUT_ONCE=1 batches the 48-layer staging-flag validation into ONE .tolist() (kills ~94
# per-layer .item() D2H syncs); replay untouched => committed state BIT-IDENTICAL.
#   arm2 FIRST = LAYOUT_ONCE=1 (batched validation)  -- watch needle '[FR13_COMMITTER_LAYOUT_ONCE ENGAGED]'.
#   arm1 SECOND = LAYOUT_ONCE=0 (per-layer .item() syncs, ~60ms baseline).
#   COMMIT_FULL(arm1) - COMMIT_FULL(arm2) = the sync-stall tax.
# CACHE-ON config MATCHED to the previous cache-on run pd1 (ENABLE_APC=1 + MAMBA_BLOCK_SIZE=1024 +
# APC_BLOCK_SIZE=1024 + SSM_CACHE_DTYPE=float32). Both non-pb tail6, COMMITTER_NATIVE=1, PARENT_GATHER=1,
# --async-scheduling, FR13_COMMIT_FULL_GPU_TIMER=1. B4, subset_collapse3 (same 3 tasks as pd1).
# Launch: RUNROOT=output/fr13_layoutonce_co TAG=loco1 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_layout_once_cacheon_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh layoutonce_co
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
# CACHE ON (matched to pd1 reference):
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_NATIVE=1
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm2 FIRST: layout-once (batched flag validation) ----
export FR13_COMMITTER_LAYOUT_ONCE=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_layoutonce_co/cf_lo_${TAG}.json
run_variant tail6_lo_${TAG}  tail6  21  1
# ---- arm1 SECOND: baseline (per-layer .item() syncs) ----
export FR13_COMMITTER_LAYOUT_ONCE=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_layoutonce_co/cf_base_${TAG}.json
run_variant tail6_base_${TAG}  tail6  21  1

# REROUTE-THROUGH-NATIVE A/B, CACHE-ON (user 2026-07-21 "pls run all arm cache on"). Same as the rr1
# reroute A/B but with APC ENABLED = deployment-representative: agentic turns reuse the conversation-prefix
# KV (attn + GDN recurrent state), so prefill collapses to the per-turn delta -> higher aggregate TPS +
# real TTFT. Accept is cache-INDEPENDENT so it must match the cache-off rr1 read; the win of cache-ON is
# the deployment speed/prefill picture. Tree+cache lossless is proven (project_fr13_treecache_SOLVED).
#   arm1 = tail6_pb  : pb CHAIN FOLD (custom, lossy) -- expect deep-task accept collapse (cache-on).
#   arm2 = tail6     : non-pb NATIVE SEED (the fix) -- expect ~5 deep == native-lossless (cache-on).
# Cache-ON config MATCHED to the historic reference pbab1 (5.03/5.31 deep): ENABLE_APC=1 +
# MAMBA_BLOCK_SIZE=1024 + APC_BLOCK_SIZE=1024 + SSM_CACHE_DTYPE=float32. PARENT_GATHER=1, --async-scheduling,
# FR13_COMMIT_FULL_GPU_TIMER. B4, subset_b4_sixteen (same 16 tasks as rr1 -> clean cache-on-vs-off compare).
# Launch: RUNROOT=output/fr13_reroute_co TAG=rrco1 SUBSET=subset_b4_sixteen.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_reroute_native_cacheon_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh reroute_co
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
# CACHE ON (deployment, matched to pbab1 reference):
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm1: pb CHAIN FOLD (lossy custom-seed baseline), cache-ON ----
export FR13_PB_BASE_COL_INVARIANT=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_reroute_co/cf_pbfold_${TAG}.json
run_variant tail6pb_fold_${TAG}  tail6_pb  29  1
# ---- arm2: non-pb NATIVE SEED (the fix), cache-ON ----
unset FR13_PB_BASE_COL_INVARIANT
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_reroute_co/cf_native_${TAG}.json
run_variant tail6_native_${TAG}  tail6  21  1

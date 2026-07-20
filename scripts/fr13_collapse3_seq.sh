# COLLAPSE-3 ATTRIBUTION (user directive 2026-07-20: recover the 3 collapse tasks).
# allon5 (tail6_pb cache-ON, END-of-16-run) tanks 14539/14598/14995 to 3.6-3.9;
# rg1 (plain tail6 cache-OFF) HOLDS them at 4.9-5.7. Run BOTH arms FRESH on just
# these 3 tasks (cache-OFF is position-independent; cache-ON fresh vs allon5
# end-of-run separates cache-effect from cache-accumulation):
#   arm1 tail6_pb CACHE-OFF: ~5   => not intrinsically hard for pb; collapse is cache.
#                            ~3.6 => PIGGYBACK collapses them regardless of cache => pb bug.
#   arm2 tail6_pb CACHE-ON:  ~3.6 => cache-ON collapses them position-independently (cache effect).
#                            ~5   => cache-ON alone is fine FRESH => allon5's collapse = cache
#                                    ACCUMULATION over the 13 prior tasks (end-of-run state).
# Committer timers folded in (bonus). Fast (3 tasks each, ~1h/arm).
# Launch: RUNROOT=output/fr13_collapse3 TAG=c3 SUBSET=subset_collapse3.json
#   WALL=0 BSIZE=4 CONC=4 HEALTH_TIMEOUT_S=3600
#   SEQUENCE_FILE=scripts/fr13_collapse3_seq.sh
#   bash scripts/fr13_campaign_tmux.sh collapse3
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_MULTIDRAFT_GPU_TIMER=1
# arm 1: CACHE-OFF (primary discriminator: pb-intrinsic vs cache)
export FR13_ENABLE_APC=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_collapse3/cf_nocache_c3.json
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_collapse3/md_nocache_c3.json
run_variant tail6pb_nocache_${TAG}  tail6_pb  29  1
# arm 2: CACHE-ON fresh (cache-effect vs cache-accumulation)
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_collapse3/cf_cache_c3.json
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_collapse3/md_cache_c3.json
run_variant tail6pb_cache_${TAG}  tail6_pb  29  1

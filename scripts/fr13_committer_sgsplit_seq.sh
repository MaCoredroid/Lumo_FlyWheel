# COMMITTER SG-SPLIT PROFILE (user 2026-07-21: "keep the replay, but it shouldn't take that long").
# The native committer replay (~88ms/commit, LOSSLESS bit-exact) is a 48-layer Python loop, each doing
# 4 torch.cat gathers + 1 fused_sigmoid. Hypothesis: HOST-bound (gathers+dispatch), fused_sigmoid tiny.
# This splits it: FR13_COMMITTER_SG_TIMER accumulates ONLY the fused_sigmoid GPU time; fr13_committer_gpu
# span (~88ms) - sg_gpu = the host gathers + dispatch gaps (= what batch/graph can kill, losslessly).
#   sg_gpu << committer_gpu  => HOST-bound => build batched-gather + graph-capture (big lossless win).
#   sg_gpu ~= committer_gpu  => fused_sigmoid-bound => harder (kernel/HBM).
# Single arm: tail6 non-pb COMMITTER_NATIVE=1, cache-off, PARENT_GATHER=1, --async-scheduling.
# Launch: RUNROOT=output/fr13_sgsplit TAG=sg1 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_committer_sgsplit_seq.sh
#   bash scripts/fr13_campaign_tmux.sh sgsplit
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=0
export FR13_COMMITTER_NATIVE=1
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMITTER_SG_TIMER=1
export FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sgsplit/sg_${TAG}.json
run_variant tail6_sgsplit_${TAG}  tail6  21  1

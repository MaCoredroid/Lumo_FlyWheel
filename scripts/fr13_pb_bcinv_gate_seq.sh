# FR13_PB_BASE_COL_INVARIANT GATE (2026-07-20): validate the pb base-first attn-col
# permute FIX on the 3 deep collapse tasks. tail6_pb + FR13_PB_BASE_COL_INVARIANT=1,
# cache-OFF (matches collapse3 arm1 baseline 3.5-3.6 for a DIRECT accept comparison).
# MECHANISM: chain at attn cols 0-7 shifts the base subtree off non-pb's canonical
# columns -> FA2 verify reduction tile-misaligned -> tie-tips head verify -> Arctic
# amplifies to -1.5. FIX permutes the base subtree (nodes 8..N-1) onto phys cols
# 0..N-9 (== non-pb) so the base FA2 reduction is byte-canonical; dead chain+pos-0
# parked at the tail. Col-0-safe (chain rows attend only paged context).
# GATE READS:
#   BOOT: engagement needle "FR13_SLOT_REORDER ENGAGED (tree_attn bias): tree_n=30
#         pi=[8, 9, ...]" AND "(runner): ... pi=[8, 9, ...]" -- BOTH must show the
#         base-first pi (starts with 8), else the permute did not engage.
#   NO fatal ('EngineCore encountered a fatal error'); NO garble (coherent patches,
#         resolve doesn't crash); col-0 lossless (piggyback stays correct).
#   ACCEPT (decode-bracketed, deep tasks 14539/14598/14995):
#     ~5   => FIX WORKS (base verify canonical -> head recovers -> deep tail lifts). WIN.
#     ~3.6 => permute engaged but accept flat => mechanism wrong OR pi mismatch. Reassess.
#     crash/garble => permute bug (bias/slot/remap inconsistency). Debug from the needle.
# Launch: RUNROOT=output/fr13_bcinv TAG=bc1 SUBSET=subset_collapse3.json
#   WALL=0 BSIZE=4 CONC=4 HEALTH_TIMEOUT_S=3600
#   SEQUENCE_FILE=scripts/fr13_pb_bcinv_gate_seq.sh
#   bash scripts/fr13_campaign_tmux.sh bcinv
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
# THE FIX (default OFF; =1 engages the pb base-first attn-col permute):
export FR13_PB_BASE_COL_INVARIANT=1
# FR13_PARENT_GATHER=1: validated byte-identical O(N) GDN kernel (task #36). The O(N^2)
# default compiles a huge Triton IR that spikes HOST RAM below the gpu_oom_guard 9GB floor
# -> docker-kill (exit 137) during the ~10min compile. The O(N) IR compiles fast + light.
export FR13_PARENT_GATHER=1
# cache-OFF: match the collapse3 arm1 baseline (3.5) so the delta is the FIX alone.
export FR13_ENABLE_APC=0
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_bcinv/cf_bcinv_bc1.json
run_variant tail6pb_bcinv_${TAG}  tail6_pb  29  1

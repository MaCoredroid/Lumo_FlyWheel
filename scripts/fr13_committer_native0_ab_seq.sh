# COMMITTER_NATIVE=0 ISOLATION (user "its a config drift?" 2026-07-21): the DEFINITIVE test.
# Deployed COMMITTER_NATIVE=1 reroutes non-pb's col-0 to the NATIVE committer while pb's chain fold
# stays CUSTOM -> the pb-vs-nonpb accept gap is a committer-arithmetic asymmetry. Set BOTH arms to
# COMMITTER_NATIVE=0 (custom committer everywhere) = the LIVE version of the never-run V0(d) gate
# (chain-fold == custom replay). PREDICTION:
#   pb-custom ~= nonpb-custom (both ~4.1)  => CARRIER = the native asymmetry; pb LOSSLESS vs custom;
#                                             deployed gap = chain-fold-not-native (fix = SCAN_ALIGN).
#   pb-custom << nonpb-custom               => a REAL pb-vs-custom divergence (chain fold not lossless).
# No SCAN_ALIGN recompile (avoids the guard-kill). base-col + PARENT_GATHER=1. Deep tasks. cache-OFF.
# Launch: RUNROOT=output/fr13_cn0 TAG=cn0 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_committer_native0_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh cn0
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_COMMITTER_NATIVE=0
export FR13_ENABLE_APC=0
# arm1: pb (custom chain fold) at COMMITTER_NATIVE=0
export FR13_PB_BASE_COL_INVARIANT=1
run_variant tail6pb_cn0_${TAG}  tail6_pb  29  1
# arm2: non-pb (custom replay) at COMMITTER_NATIVE=0
unset FR13_PB_BASE_COL_INVARIANT
run_variant tail6_nonpb_cn0_${TAG}  tail6  21  1

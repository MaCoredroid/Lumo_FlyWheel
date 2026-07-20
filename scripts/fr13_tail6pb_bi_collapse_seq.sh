# FIX-ATTEMPT (user: fix + rerun): residual -1.1 likely = 29-col fused forward
# autotune numerics diverging from 21-col (within-floor ULP tipping temp-0.6
# ties). BATCH_INVARIANT=1 + FR13_BI_TREE_ATTN=1 (BI allowlist for TREE_ATTN,
# required per launcher:52) pins the tree attention deterministic -> stops the
# autotune fork. tail6_pb cache-OFF + NEW mask, 3 collapse tasks. ~5 => fix;
# ~3.9 => inherent 29-vs-21 cost -> build localization gate.
export BATCH_INVARIANT=1
export FR13_BI_TREE_ATTN=1
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_ENABLE_APC=0
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
run_variant tail6pb_bi_${TAG}  tail6_pb  29  1

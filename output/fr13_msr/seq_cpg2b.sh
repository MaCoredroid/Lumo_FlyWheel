# REGATE 2b (regate_queue.sh): FR13_CONV_PREGATHER alone, WITH the row-id
# token fix (composite (req_ids, col0 page-ids); stage refuses without the
# col0 publish). Legs: loop-watch (same-counter vs bv1 events 32/71/197/114),
# accept-inflation vs bv1x comb 4.718 (same subset), garble eyeball,
# [FR13_CPG_ROWID_TOKEN] miss needle ~0, AND [FR13_CPG_SERVE] served>0
# (vacuity audit — a served=0 arm proves nothing). One lever only.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_CONV_PREGATHER=1
run_variant cpg2b_rowid  tail6  21  1

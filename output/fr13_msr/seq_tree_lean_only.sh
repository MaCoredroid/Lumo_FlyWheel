# B-sweep LEAN tree arm body (stack recomposition by measured merit,
# 2026-07-25): keep the isolation-gate WINNERS only — parent_gather (35.22),
# pregather (34.16), flags_inkernel (33.35), subtree (+4.7% B=1, byte-sealed).
# UNBAKED vs the sealed stack: NODEBANK (28.05, bank tax), SPEC_BLOCKS_CAP
# (29.62; cache-hit lever — WATCH marginal hit-rate for the squeeze),
# WB_BATCHED (31.14, dual-arm double-write tax). HC stays off (shadowed by
# subtree). Arm name via BSWEEP_ARM.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1
export FR13_HC_INTERNAL=0
export FR13_CONV_WB_BATCHED=0
export FR13_CONV_NODEBANK=0
export FR13_SPEC_BLOCKS_CAP=0
export FR13_SUBTREE_PARALLEL=1
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
run_variant "${BSWEEP_ARM:?}"  tail6  21  1

# B-sweep tree arm body: the byte-sealed stack (same six levers as seq_g3),
# arm name via BSWEEP_ARM. Used by run_bsweep.sh at BSIZE 1/8.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1
export FR13_HC_INTERNAL=0
export FR13_CONV_WB_BATCHED=1
export FR13_CONV_NODEBANK=1
export FR13_SPEC_BLOCKS_CAP=12
export FR13_SUBTREE_PARALLEL=1
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
run_variant "${BSWEEP_ARM:?}"  tail6  21  1

# BAR CONFIRM (queue step 3): FULL candidate stack, 16-task, CLEAN (zero
# instruments — the speed-of-record per the two-kinds rule). Stack = the six
# queue-validated candidates: parent_gather + pregather(row-id token) +
# flags_inkernel + wb_batched + nodebank + spec_cap(12). Bar = native5@0.70
# 50.99 measured_tps_fullstep_wall. Legs: loop-watch (same-counter), accept
# band, garble eyeball, serve/derivation needles, "X pass Y fail Z finished".
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
run_variant bar17_stack  tail6  21  1

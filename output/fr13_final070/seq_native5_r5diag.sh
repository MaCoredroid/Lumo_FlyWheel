# r5: BLOCKING + GUARD together — true faulting site + guard-state witness.
# r4 (guard, non-blocking) crashed w/ rescued=0 + sample-assert silent =>
# corruption is NOT in the two scatter slot lists => suspect length/offset
# desync poisoning multiple consumers (embedding r3; rearrange_mixed_qkv r4).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export CUDA_LAUNCH_BLOCKING=1
export FR13_INPUTPREP_GUARD=1
run_native  native5_f70_r5diag  5  5  1

# B-sweep native control arm body: stock native MTP-5 serve (the E5 bar
# vehicle), arm name via BSWEEP_ARM. Used by run_bsweep.sh at BSIZE 1/4/8.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
run_native "${BSWEEP_ARM:?}"  5  5  1

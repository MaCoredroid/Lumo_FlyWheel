# GPU_UTIL raise probe (cache-hit lever 1). Env matches the ovl16 gate exactly
# (overlap armed, timers on); the ONLY intended delta vs ovl16 is GPU_UTIL,
# passed by launch_probe.sh. 4-task slice; speed numbers DIAGNOSTIC ONLY.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMIT_OVERLAP=1
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant "utilprobe_tail6_${FR13_PROBE_TAG:-up70}"  tail6  21  1

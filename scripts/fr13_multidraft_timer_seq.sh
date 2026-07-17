# MULTIDRAFT-COMMITTER decomposition (the REAL temp-0.6 committer, per user's catch that temp0.6 uses
# rejection sampling not greedy LCP): tail6_mt = tail6 + FR13_MULTIDRAFT_GPU_TIMER=1 -> sidecar
# output/fr13_sfwd_sidecar/tail6_mt_md.json (per-step GPU-time of fr13_device_multidraft_commit).
# Settles what the 94ms committer_gpu span IS: multidraft_ms ~=90 => the rejection committer IS the cost
# (a real optimization target I missed); multidraft_ms ~=few => the 94ms is result-DtoH + verify-wait
# (pipeline, needs async). Short run: read sidecar after ~50 decode steps, then can kill. No drift (timer OFF default).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6_mt_${TAG}  tail6_mt  21  1

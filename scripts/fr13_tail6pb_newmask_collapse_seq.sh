# GENERALIZED-MASK TEST: tail6_pb (cache-OFF async) on the 3 collapse tasks WITH
# the new patcher (generalized S1(b) row-0 ghost for all pb trees + fail-loud
# shape/flag guard). Compare accept to old tail6_pb cache-OFF (14539=3.647,
# 14598=3.348, 14995=3.506). If the row-0 ghost was part of the chain leak,
# accept moves up; if unchanged, the -1.7 deep-tail loss is elsewhere (overflow
# deep-draft quality / trajectory amplification).
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_ENABLE_APC=0
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
run_variant tail6pb_nm_${TAG}  tail6_pb  29  1

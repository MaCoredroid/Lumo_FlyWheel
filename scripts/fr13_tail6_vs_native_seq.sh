# The unmeasured deliverable: does tail6 (accept 5.237) actually beat NATIVE MTP-5 throughput?
# The design doc only compared tail6 to the depth-5 TREE, never to native. nativeE5 = native MTP-5 (no tree).
# tail6 = the accept>5 deploy config. Compare derived_tps_gpu + derived_tps_fullstep_gpu at B=4, same subset.
export GPU_UTIL=0.72
run_native nativeE5 5 5 1
run_variant tail6 tail6 21 1

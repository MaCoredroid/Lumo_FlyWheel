# FR13 BURN-REDUNDANCY A/B on TAIL6 (depth-11, the R1 speed-pair kind, ~5 accept anchor).
# Same experiment as the cat9f seq, but on tail6 so the accept lines up with the tail6 trend and the
# committer burn is measured on the deployed speed baseline. Vary ONLY the burn (no config drift).
# GATE: burn-OFF resolve + accept + garble MATCH burn-ON (within-floor, temp 0.6, live SWE, B>1 via CONC=4);
#       speed = burn-OFF s_per_fwd / TPS better (committer minus the ~memory-bound burn). burn-OFF first (fail-fast).

# arm 1: BURN OFF (commit/init stay ON; toggle bypasses the tri-flag assert)
export FR13_BURN_REDUNDANCY_TEST=1
run_variant burnoff_${TAG}  tail6  21  1

# arm 2: BURN ON (deployed baseline)
export FR13_BURN_REDUNDANCY_TEST=0
run_variant burnon_${TAG}   tail6  21  1

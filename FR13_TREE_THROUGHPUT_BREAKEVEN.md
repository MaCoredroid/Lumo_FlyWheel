# Task #42 — why native throughput (5.49) > tree (4.89) despite tree accept (4.32) > native (3.42)

## The break-even model (ties the committer floor to throughput)
Throughput of committed output = accept_per_step / step_time. This session PROVED the committer's
72ms per-layer GDN replay is FUNDAMENTAL (copy-not-replay infeasible = 13.7GB per-node state export;
the verify deliberately does not export per-node states -- fr10_gdn_tree_kernel.py:925,2084). That 72ms
is on the CRITICAL PATH: the next forward cannot start until the accepted-path state is committed.

  native step ~= forward(98) + committer(7)               ~= 105 ms,  accept 3.42
  tree   step ~= forward(98) + committer(7 + 72 replay)   ~= 177 ms,  accept 4.32-5.15

  tree beats native  iff  accept_tree/177 > 3.42/105
                     iff  accept_tree > 3.42 * 177/105 ~= 5.76   (MODEL estimate; step times rough)

## Consequence
- Prewarm accept 5.15 is BELOW the ~5.76 break-even => native still wins on throughput. This is the
  quantitative answer to task #42: the tree's accept advantage is real but does NOT clear the bar set
  by the (fundamental) 72ms replay overhead that native lacks.
- async-scheduling (task #40) overlaps drafter/host time (per_req 4.89->5.029) but CANNOT hide the
  replay -- it is a data-dependency on the critical path, not host overhead.
- => The ONLY path to "tree beats native" is accept ABOVE break-even (~5.76). That is exactly the
  accept-beyond-5 design (FR13_ACCEPT_BEYOND5_DESIGN.md: suffix tail past d5 + complement branches,
  monotone-lossless, target 6+). It is not a nice-to-have; it is the deciding lever.

## Confirming experiment (SCOPED, needs clean arm wiring -- deferred, no session-tail config drift)
Live B4-16 (subset_b4_sixteen), 3 arms, byte/accept-identical when levers off:
  A) native  = kind `nativemtp5`            (pre-baked; break-even reference)
  B) tree    = kind `tail6` + --async-scheduling + FR13_PREWARM_TRIE=<corpus>   (accept ~5.15)
  C) tree+   = B + accept-beyond-5 (suffix tail + complement)                    (target accept >5.76)
Wiring TODO before launch (each is a known vacuous-flag trap -- verify -e passthrough into container):
  - FR13_PREWARM_TRIE not currently threaded through fr13_b4_campaign_driver.sh / serve-variant.
  - --async-scheduling must reach the vLLM launch (task #40 did this one-off; not a baked kind).
  - accept-beyond-5 is DESIGNED (task #34) not yet a baked, gated arm.
Gate: derived_tps_gpu (committed/s_per_fwd_gpu), matched prefill_frac, same seed, temp 0.6.
HYPOTHESIS to confirm/refute: B < native, C >= native. If C still < native, the tree architecture
cannot beat native MTP-5 on this workload and the deliverable is accept-parity + lossless branched cache,
NOT a throughput win. MEASURE before claiming either way.

# FR13 Gate Blind-Spot: scalar accept is blind to per-token argmax flips

Date: 2026-06-13. User question: why didn't the previous superset/accept gate catch the
verify-forward greedy-loss (tree verify flips greedy argmax ~4.3% at deep committed spine
rows)? Answer: the gate measured a SCALAR, which is mathematically blind to a small-rate
per-token argmax flip. This is the SAME blind-spot CLASS the lossy-superset audit
(wf_8100e9a6, f9cf125e) named for the old multi-spine `superset_violations=0` gate.

## The four reasons the superset/accept gate passed while the defect was live
1. **Scalar-average blindness.** accept/event ~= 3.18; a 4.3% per-position argmax flip moves
   the AVERAGE by a fraction of a token — inside the cross-boot accept band (~0.2-0.3 wide).
   "cat9 3.1789 > native 3.1613" PASSED with the defect present. NO per-token argmax check.
2. **Short-context measurement.** The crossing was at 64-tok pinned probes; the flips
   accumulate with depth/context (gap grows 1.02x@64tok -> 1.05x@11k; deep-row flips need
   longer contexts + structural boundaries). 64 tok had few enough deep-boundary positions
   that the tree still barely edged native (+0.018).
3. **Same class as the multi-spine count-only gate** (f9cf125e): scalar/count vs an INTERNAL
   reference, never per-token output-equality vs an EXTERNAL one. The verify-forward gap is
   the dual hole: a per-token argmax MISS a scalar accept average cannot resolve.
4. **Coherent served stream.** Flipped tokens are near-equivalent at structural boundaries
   (` code`<->` files`, `Let`<->code-fence), so the within-floor served-stream check passed
   it via first-fork bracketing. Only per-token argmax-vs-clean could see it.

## Standing gate requirement (the missing instrument)
accept/event — and ANY scalar accept metric — is NECESSARY but NOT SUFFICIENT for either
losslessness OR superset; it is blind to small-rate per-token argmax flips. The BINDING
instrument is **per-token argmax-vs-clean-reference** (the teacher-forced/in-process probe:
fr13_gold_margin_probe.py + FR13_COMMIT_ARGMAX_GATE), which MUST be a permanent part of every
superset/lossless gate, not an afterthought. This is the dual of the lossy-superset audit's
GAP-2 (two-sided over-acceptance): GAP-2 catches accept ABOVE native unaccounted; this catches
a per-token argmax MISS that keeps the scalar accept within band. Add BOTH to the corruption
gate + the superset gate. See [[reference_lossless_specdecode_gate_methodology]],
the lossy-superset audit (research/fr13_workflows/lossy_superset_gate_audit_wf_8100e9a6.raw.json),
FR13_B1_SWE_GOLD_BIND.md.

# FR13 wide256 data-parallel real-B1 rejection

Status: **candidate rejected; no timing or hardware-floor claim**.

The diagnostic ran the pinned real SWE-Verified task
`astropy__astropy-12907` at B1, fixed physical row count 32, and full draft
vocabulary (`root=0`, `K=0`). The task resolved cleanly. Every diagnostic call
served the stock result while comparing every BF16 output byte against the
`wide256_dataparallel` candidate.

## Kernel result

- Candidate binary: `5b921ab7b428f2c5cfeefc0daed0314ff903d73bb0d4f8a790b17234c9d60890`
- Candidate geometry: `256x32x128`
- Scheduling: one CTA owns the full K range; no Stream-K workspace reduction
- Comparisons: `256`
- Unequal comparisons: `256`
- Compared bytes: `249,888,768`
- Differing bytes: `10,504` (`42.0347 ppm`)
- First mismatch offset even in every record: yes

Removing the explicit K split was insufficient. Changing the tile geometry
still changed the per-output MMA/epilogue association relative to the stock
kernel, producing sparse BF16 low-byte drift. The candidate is therefore
ineligible for production and timing.

## Campaign validity

The task evaluation itself resolved with harness exit code 0. The shared
filesystem then filled during ingress-campaign finalization. That prevented
the formal live-gate reducer from running and left the run exit code at 15.
This artifact does not upgrade that incomplete campaign into a formal gate
result. The raw, authenticated real-task comparisons are nevertheless
decisive negative evidence because every candidate output differed.

The run also exposed a gate-bound issue: 256 target-projection calls exhaust
the old comparator cap before the first required MTP projection. The next
candidate must use a minimally larger bound and bind the compiled binary and
reducer to the same limit.

The applicable full-vocabulary floor remains `153.9383846446886 ms/step`;
the one-sided U95 acceptance cap is `177.0291423413919 ms/step`. No latency,
TPS, acceptance-rate, B4, or hardware-floor result is claimed here.

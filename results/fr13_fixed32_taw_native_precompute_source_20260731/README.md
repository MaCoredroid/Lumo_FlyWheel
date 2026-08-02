# Fixed32 TAW native precompute byte gate

## Status

This is a default-off, reference-returning diagnostic candidate. It is not a
deployment result and has not run on a GPU.

Set `FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1` only for the real-task byte gate.
The route computes the existing per-level native PyTorch reference and a
fixed-row batched native `torch.softmax` candidate on the same logits and
uniforms. It compares the raw FP32 bits of every causally consumed self/target
probability row and all five final integer TAW products. Mismatch counts remain
on device during capture. The next uncaptured root fails loud on a nonzero
counter. The function always returns the reference products.

The default route is unchanged when the variable is unset or `0`. Any other
value fails closed.

## Fixed work

The source rows are the union required by both fixed modes, not a mode-sized
set:

- self leaf rows: 13 draft-local rows
  `[1,2,4,5,9,10,12,14,15,19,20,21,30]`;
- target first-child rows: 17 draft-local rows
  `[0,3,6,7,8,11,12,13,16,18,21,23,25,27,28,29,30]`.

Both Tail6 and Hydra27 therefore issue the same candidate work: one native
softmax over the 13 self rows and one over the 17 target rows. Dynamic cache
gathers preserve the fixed 12-level causal walk and every existing
normalization, overlap, source, acceptance, q-mix, residual, and inverse-CDF
operation. Values mapped for dead/internal rows are irrelevant and are not
counted by the probability gate; every row that can affect a product is gated.

## Evidence and bound

The real SWE-Verified B1 attribution artifact at
`results/fr13_fixed32_b1_nsys_20260731T013952Z_curated/nsys_attribution.json`
has SHA256
`685c410a0ba09d00c8b244bfa06809530337cea383b88fa92a5da1013eadf2d0`.
It is pre-final attribution evidence, not acceptance evidence. Across 881 CFWD
ranges it reports:

- CFWD projected GPU envelope: 22.755077 ms/event;
- native CUDA softmax: 26 launches and 2.304176 ms/event;
- the indexed FP32 row-gather kernel: 30 launches and 0.141708 ms/event.

The audited TAW source accounts for 24 of the 26 native softmax calls. A
deployed candidate would reduce TAW softmax calls from 24 to 2, but no latency
saving may be inferred before profiling. The diagnostic byte gate intentionally
runs both walks and reports its doubled work separately.

For vocab 248,320 and FP32 rows, minimum softmax read-plus-write traffic is:

- current 24 rows: 47,677,440 bytes/request;
- candidate 30 rows: 59,596,800 bytes/request, 11,919,360 bytes more;
- candidate probability cache: 29,798,400 bytes/request, or 119,193,600 bytes
  at B4;
- candidate source plus dynamic probability gathers: 107,274,240
  bytes/request versus 47,677,440 currently.

The candidate trades 22 softmax launches for 25% more softmax row work and
59,596,800 extra gather bytes/request. It is viable only if launch and small-row
occupancy savings exceed that traffic. B1 and B4 GPU measurement are both
required.

## Gates run

CPU-only tests exercised Tail6 and Hydra27 at B1 and B4 over four finite-random
seeds per configuration. All causally consumed probability bytes and all final
product bytes matched. Source-contract and work-census tests also passed: 12
targeted tests total.

CPU equality does not prove CUDA row-batch reduction identity. The next step is
one real SWE-Verified B1 task with this reference-returning route, requiring
zero probability and product mismatches. Only after that passes may a separate
candidate-returning source change be considered and profiled on the standing
real 4-task/B4 and 16-task sets.

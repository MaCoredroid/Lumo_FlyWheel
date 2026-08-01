# Fixed32 CFWD all-parent commit fusion

Status: **source candidate only; default off; live-event equality and timing pending**.

## Scope and attribution

This branch starts at compliant full-vocabulary source `c3720f2ba`. It does not
change vocab size, dtype, model weights, RNG draws, tree topology, accepted-path
semantics, or the fixed 32-row contract.

The real SWE-Verified Nsight artifact at
`results/fr13_fixed32_b1_nsys_20260731T013952Z_curated/nsys_attribution.json`
attributes 1,377,003 GPU operations across 881 CFWD ranges: exactly 1,563 GPU
operations/event and a projected 22.755077 ms/event envelope. The later real
exact4 Hydra27 timer measured 20.677391 ms/event. Those older runs used the
now-invalid 64K draft-vocab floor, so they are attribution only, not formal
floor evidence.

The incumbent fixed32 TAW performs a 12-level causal walk. It launches 24
full-vocabulary softmax/CDF paths and 12 integer commit kernels per event. The
base already contains a separate default-off 48-to-1 GDN layer-batch candidate;
this change does not duplicate or claim that candidate's 4.082147 ms/event
kernel group.

## Candidate

The historical native-precompute diagnostic/production selector now identifies
`fixed32_all_parent_commit_v2` and is source-bound to
`fr13-fixed32-taw-all-parent-v4`.

The candidate exploits the fixed topology without changing the causal rule:

- Batch the fixed union of 13 self rows and 17 target-parent rows with native
  PyTorch FP32 softmax and the existing normalization, q-mix, residual, and
  inverse-CDF expressions.
- Bind each row to the exact uniform slot its parent would consume in the
  sequential walk. The same pre-drawn per-request uniforms remain authoritative.
- Publish the already-decided path with one integer-only Triton program per
  request. The kernel contains no floating-point sampling implementation.
- Keep Tail6 and Hydra27 physical candidate work identical: 13 self rows, 17
  target rows, two softmax calls, and one exact commit launch for B1 through B4.

The diagnostic route runs the incumbent and candidate on the same logits and
uniforms, compares every causally consumed FP32 probability row as raw bits,
compares all five integer products, and returns the incumbent products. A new
live PASS is mandatory; the prior v1 PASS cannot activate this v2 source.

## Static evidence

CPU randomized equality covered both fixed modes at B1 and B4, 16 seeds per
combination, plus zero-overlap and duplicate-sibling edge cases. Every output
token, output length, accepted path row, accepted length, and final row matched
byte-for-byte. The local CPU-focused result was 30 passed with the CUDA module
skipped. Independent review then compiled the Triton kernel in the serving GPU
image and passed the four Tail6/Hydra27 B1/B4 CUDA cases, including ordinary and
source/accept/residual threshold-boundary products. That CUDA run was a focused
correctness check, not a throughput measurement or real-task qualification.
Ruff, Python compile, and `git diff --check` passed.

## Expected value and decision gate

This replaces 12 sequential TAW decision groups with one all-parent group,
reduces softmax launches from 24 to 2, and reduces exact integer commit launches
from 12 to 1. It evaluates 30 fixed probability rows instead of 24, so it trades
25% more full-vocabulary row work for substantially fewer device launch
boundaries. Based on the Nsight operator groups, a defensible pre-GPU expectation
is 2-6 ms/event off CFWD, not the entire 20.677 ms phase. Reject the candidate if
the real B1 diagnostic gain is below 1 ms/event or if B4 regresses.

Qualification must use full-vocabulary `K=0`, `root=0` real SWE-Verified exact4:

1. Reference-returning diagnostic at B1 and B4: zero FP32 probability-bit
   mismatches and zero product mismatches for every used occupancy.
2. Source-bound production pair on the same four tasks: reference versus v2,
   identical fixed32 work/topology/dtypes and unchanged acceptance semantics.
3. Nsight: 24-to-2 TAW softmax launches and 12-to-1 exact commit launches, with
   no fallback, cache miss, graph death, or dynamic-work drift.
4. Report full-wall TPS and CFWD ms/event. This candidate is additive progress,
   not by itself a claim that the 177.029142341 ms full-vocab cap is met.

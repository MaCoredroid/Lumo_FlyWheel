# FR13 fixed32 GDN parent-group corrected source candidate

Status: **source candidate only, default OFF, no GPU measurement**.

## Corrected execution contract

The logical fixed32 schedule remains `[1, 11]`, with logical critical path 12.
The grouped level-1 kernel has five programs whose total node counts are
`[9, 12, 2, 2, 2]`. Because each program executes its member paths serially,
the physical level maxima are `[5, 12]` and the physical critical path is 17.

The observer now records logical and physical work separately:

- B1 grouped: per-request and event z grids `[1, 5]`, two launches/layer.
- B4 grouped and batch-folded: per-request `[1, 5]`, event `[4, 20]`, two
  launches/layer.
- B4 incumbent reference: `[1, 11]` repeated four times, eight
  launches/layer.

Physical parent-state program units remain 12 to 6 per request/layer. The
previous modeled parent-read savings are unchanged, but they do not imply a
latency reduction because the grouped kernel has a longer serial path and may
increase registers or local-memory traffic.

## Qualification wiring

B2-B4 byte gates now force the incumbent eleven-path implementation for the
reference arm. A cached parent-group descriptor cannot select the candidate in
both arms.

B1 has a default-off, eager-only authenticated gate. Before the authenticated
real SWE-Verified event it serves only the incumbent. On the real event it runs
incumbent and grouped arms from the same state, compares raw bytes for output,
full export scratch, K/V/A/B rings, flags, and counter, restores all reference
state, and serves the incumbent result. A mismatch fails closed. The gate does
not authorize production.

Parent grouping and the parent-gather selfcheck are explicitly incompatible,
preventing that higher-priority dispatch from bypassing the candidate while
the observer reports grouped work.

## Verification boundary

Static and CPU tests passed. No GPU command, Triton compile, byte-equivalence
campaign, resource inspection, CUDA graph replay, SWE-Verified timing, or
full-wall measurement was run. The candidate is not acceptance eligible and
has no production authorization.


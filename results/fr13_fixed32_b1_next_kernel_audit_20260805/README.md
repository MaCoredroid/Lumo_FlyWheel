# Fixed32 B1 next-kernel audit

Status: **evidence audit only; no new CUDA candidate; no performance claim**.

This audit reconciles the latest valid Hydra27 B1 wall measurement, the
canonical real SWE-Verified Nsight attribution, current `main`, and the active
Qrow32, R32/R64 draft-head, and BM8 branches. It deliberately does not create
a timing probe or mutate a kernel without evidence that the mutation can
preserve the exact-output contract.

## Authoritative measured point

The latest valid Hydra27 exact-four arm is still the older Qrow16 production
run at source commit `04c98057fa9d9e658282cf0b32422716660e1210`:

| Metric | ms/step |
| --- | ---: |
| Full wall | 232.779790071 |
| SFWD GPU | 159.619263244 |
| DFWD GPU | 36.813368134 |
| CFWD GPU | 20.677390557 |
| Host/unattributed residual | 15.669768137 |
| Mandatory-weight floor | 119.658015414 |
| Acceptance cap | 137.606717726 |
| Gap to cap | 95.173072345 |

The real-task Nsight artifact is attribution-only and was captured at an older
source commit. It cannot be compared to the unprofiled wall point as a
regression, but it is the only bound per-kernel ranking available.

## Coverage reconciliation

| Measured group | Nsight ms/step | Current source coverage |
| --- | ---: | --- |
| Target FP8 CUTLASS | 112.312954 | Full-tile/full-grid candidate is on `main`; real qualification and timing pending. |
| Five DFWD BF16 draft heads | 26.227316 | R32 and R64 exact-order candidates are staged; real five-site byte qualification pending. |
| Tree attention FA2 | 24.708601 | Qrow32 interleaved-KV v3 is on `main`; Gate A evidence pending. |
| Conv select/copy/writeback | 15.014090 | SFWD conv/post-prep fusion is on `main`; current-stack real qualification pending. |
| Tree GDN | 14.019520 | GQA-group3 candidate is on `main`; current-stack real qualification pending. |
| Verifier full-vocab head | 12.152307 | No reduced-vocab or alternate-math candidate is admissible for verifier semantics. |
| MTP FP8 CUTLASS | 8.514285 | Exact MTP direct scheduler candidate is on `main`; current-stack timing pending. |
| Unified attention | 6.967564 | BM8 is staged on its composed branch; current-stack timing pending. |
| CFWD GDN replay | 4.082147 | The 48-layer batch kernel and exact-byte gate are already on `main`. |

The table avoids double-counting the verifier head as draft-head work. The
Nsight trace has five DFWD BF16 head calls per event and a separate one-call
postprocess verifier projection.

## Next unaddressed kernel

The separate verifier projection is the highest measured distinct group not
covered by an active source candidate. Its full BF16 weight tensor is
2,542,796,800 bytes. At the campaign bandwidth premise of 273 GB/s, weight
read alone has a 9.314273993 ms lower bound. The measured kernel is
12.152306947 ms/event, leaving only 2.838032954 ms between the observation and
that ideal lower bound, or 1.304697x the lower bound.

This is a lower-bound comparison, not proof that the kernel is memory-bound.
The canonical artifact explicitly records compute-versus-memory NCU as
unmeasured. A custom reduction, reduced verifier vocabulary, FP8 verifier
weights, or alternate GEMM schedule has no current raw-byte evidence and can
change verifier logits or reduction order. Staging one now would be an
unsupported probe, not a defensible kernel optimization.

## Decision

No new CUDA source candidate is added by this branch. The next authoritative
step is to finish the existing real-task qualification ladder and obtain a
current-stack exact-four measurement plus real-workload attribution. Only then
can the remaining verifier-head excess, or a newly exposed group, be ranked
without relying on stale phase shares.

The current-main CFWD layer-batch source contract was rechecked with 26 focused
CPU/static tests; all passed.

No GPU command, Docker runtime, synthetic workload, timing run, or acceptance
campaign was used for this audit.

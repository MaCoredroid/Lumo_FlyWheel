# FR13 fixed32 SFWD state-fusion source candidate

This artifact records a source-only, default-off kernel candidate. No GPU was
used, no synthetic/probe timing was run, and this is not B1 or B4 acceptance
evidence.

## What changed

The incumbent already exports GDN replay rings and freshness flags inside the
two exact fixed32 path launches. Those launches retain the physical `[1, 11]`
schedule, with request batch folded into the program grid as `[B, 11*B]` for
B1-B4.

The new candidate attacks the remaining per-layer conv path. One B-folded
Triton launch directly reads the current col0 BF16 conv state, reproduces the
four BF16 tap products and ordered FP32 adds, applies the same ex2-compatible
SiLU, writes BF16 conv output, and materializes the persistent
`prior + x + zero` source consumed by the post-accept direct col0 committer.
This removes redundant state snapshots and source round trips at the source
level without changing dtype, topology, or mandatory model-weight traffic.

## Safety boundary

There is no production selector. The launch entrypoint refuses CUDA graph
capture and only the authenticated real-event byte-gate wiring can call it.
That wiring executes the candidate first, then executes and serves the complete
incumbent path. It compares `conv_out` and `commit_source_stage` as raw bytes,
records mismatches, and emits a 48-layer pass that remains explicitly
`production_eligible=false`.

## Verification

Focused deterministic tests cover B1-B4 geometry, the exact fixed32 conv-window
descriptor, CPU reference equivalence for direct indexing/source materializing,
strict byte comparison, default-off real-event gating, and generated patch
syntax. Existing fixed32 GDN schedule tests confirm the legacy path kernel AST
is unchanged.

The broader relevant run completed 472 tests with 22 skips; its 11 failures are
unchanged on the base commit (six stale stateless-tree reference expectations
and five stale fused-conv wiring expectations). The patcher's standalone
`--self-test` also fails identically on the base at the older conv-pregather
runtime fixture. These pre-existing harness failures were not modified.

## Next qualification

Run the byte hook only through authenticated SWE-Verified ingress, then add a
persistent graph-safe scratch and repeat on the final FULL B4 graph. Production
selection still requires byte-clean B1/B4 gates and matched exact4
full-vocabulary full-wall timing. The historical 15.014 ms conv-state-motion
attribution is stale and is motivation only, not a savings estimate.

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

No production path is enabled. The explicit production selector fails closed
until future authenticated B1 and exact4 B4 byte prerequisites are bound, and
even that binding does not permit candidate serving. The launch entrypoint
refuses CUDA graph capture and only the authenticated exact4 real-event byte
gate can call it. That wiring executes the candidate first, then executes and
serves the complete incumbent path. It compares `conv_out` and
`commit_source_stage` as raw bytes, records mismatches, and emits a 48-layer
byte-only result that remains explicitly `production_eligible=false`.

## Verification

Focused deterministic tests cover B1-B4 kernel geometry, the exact fixed32
conv-window descriptor, CPU reference equivalence for direct indexing/source
materializing, strict byte comparison, authenticated exact4 marker handling,
default-off reference serving, generated patch syntax, and the eager boot-warm
lifecycle required before the fail-closed conv-pregather consumer. The composed
SFWD, ingress, PAD-slot, preseed, process-attestation, and floor-propagation run
completed 155 tests without failures. The broader fixed32 regression run
completed 746 tests with 7 skips and no failures.

## Next qualification

Run the prepared byte hook only through authenticated canonical SWE-Verified
exact4 ingress and require B4/concurrency 4, 32 physical rows per request, full
vocabulary root0/K0, both byte surfaces, and all 48 layers. The B1 credential
must be rerun and pass with the boot-warm lifecycle repair before it can be
bound to any B4 qualification. Production remains default-off and unavailable;
this preflight makes no acceptance, timing, or savings claim. The historical
15.014 ms conv-state-motion attribution is stale and is motivation only.

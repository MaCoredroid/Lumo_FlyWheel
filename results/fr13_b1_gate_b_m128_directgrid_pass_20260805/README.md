# B1 Gate B M128 direct-grid pass

Status: **PASS** for the authenticated one-task byte-correctness gate.

Gate B ran from exact pushed source commit
`2c349c938b0ea85f43c1a9f85dea316aa179ee13` with fixed Hydra27,
physical32, K64/root1, B1, and the real SWE-Verified task
`astropy__astropy-12907`. The task resolved and the launcher exited zero.

The corrected M128 direct-grid target candidate completed 320 comparisons
over all five fixed32 projection shapes. Every compared byte matched stock,
with zero mismatching comparisons. Qrow16 engaged in eager mode on all 16
attention layers. The SFWD conv/post-prep source-only gate completed 336
comparisons across 48 layers with zero mismatches and zero errors. Launch and
end runtime, external-input, and source manifests were byte-identical.

This is a correctness qualification, not a timing run. The candidate remained
shadowed and returned the stock/reference result, so this package makes no
timing, TPS, acceptance, speedup, production, or hardware-floor claim. The
latest valid B1 performance result therefore remains unchanged.

The reduced artifact contains final gate records, the Qrow16 engagement
record, an aggregate target-comparator summary, the resolved-task summary,
source identity, checksums, and this explanation. It excludes prompts,
responses, patches, traces, raw logs, environment dumps, process identities,
binaries, tensors, and the raw byte-comparator streams.

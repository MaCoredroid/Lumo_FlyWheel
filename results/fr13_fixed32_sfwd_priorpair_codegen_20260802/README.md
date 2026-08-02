# Fixed32 SFWD prior-pair static codegen

This reduced artifact records the CPU-only SM121a codegen result for
`fixed32_sfwd_priorpair_quadweight_xgather_rowgroup32_c64_w16_v1`.

The kernel keeps the fixed 32-row, C64, W16 launch shape. It pairs the first two
contiguous prior-state BF16 loads as one 32-bit word and retains the existing
four-weight 64-bit load. Relative to the packed-xgather source before prior
pairing, static SASS has 19 rather than 33 `LDG` instructions, 391 rather than
408 non-control instructions, and the same 55-register allocation. The emitted
kernel has no stack, local-memory, `LDL`, `STL`, or call instructions.

Two builds from empty compiler caches reproduced byte-identical B1 and B4
outputs. B1 and B4 emit the same cubin; only launch cardinality and compile-key
identity differ. This is a static readiness result, not a latency, throughput,
losslessness, or hardware-floor verdict. A new real SWE-Verified byte gate is
required before this candidate may enter matched timing.

# Fixed32 B1 full-grid CUTLASS byte PASS

This reduced artifact records the authenticated one-task SWE-Verified B1
qualification of `identity_onen_n5120_fullgrid_b1` at K64/root1 and physical
row count 32.

The diagnostic executed the stock and candidate kernels on 320 production
projection calls, covering all five admitted `(N,K)` shapes. All 320 outputs
were byte-identical, with zero mismatching comparisons and zero differing
bytes. The diagnostic always served stock; production remained disabled.

The real task reached a terminal `failed` verdict and the complete serving
lifecycle exited zero. Task quality is not used as kernel correctness evidence;
the gate is based on authenticated task-path execution and byte equality. The
run produced no timing, TPS, acceptance, or hardware-floor evidence.

The byte PASS credential immediately released the standing exact-four B1
stock-versus-candidate full-step campaign on the same frozen source head. Its
result is not included here.

This package excludes prompts, responses, task patches, comparator JSONL,
logs, environments, secrets, and process or container identifiers.

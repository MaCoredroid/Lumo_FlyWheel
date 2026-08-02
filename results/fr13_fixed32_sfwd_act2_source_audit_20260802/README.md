# Fixed32 SFWD paired-activation source audit

This reduced artifact binds the source-only audit for
`fixed32_sfwd_channel_serial_r32_b1c128w2_bxc256w4_u32x2_frontier5_loadonce_act2_v4`
to source commit `9b54beec6cb4e196cdfde7d9daa1e58af94da64e`.

The candidate retains the exact load-once frontier-5 node order and groups the
32 nodes into 16 adjacent two-node activation windows. Each first FP32
accumulator remains live while the second node performs the same four
BF16-rounded products and ordered FP32 additions. Both SiLU expressions then
appear before either output or source-stage store.

`source_audit.json` records the source invariants and file identities.
`verification.md` records the checks that passed. `SHA256SUMS` authenticates
the reduced artifact files.

This is not a performance result. No Triton or CUDA compile, GPU execution,
real-task request, timing measurement, or hardware-floor acceptance run was
performed for this candidate while the independent B4 gate owned the device.

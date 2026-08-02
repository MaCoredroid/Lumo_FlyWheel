# Fixed32 DFWD K64 M1 full-warp R32 pair2 build

Status: pinned host build and static codegen audit pass; default off and not
runtime-integrated. No GPU, synthetic probe, workload, or real task was run.

The candidate assigns one 32-lane warp to each output row in a 1,024-thread
CTA. Each lane loads an aligned BF16 pair, converts both elements to FP32, and
performs two ordered FP32 FMAs. Relative to the scalar warp32/R32 parent, total
weight bytes and FMAs remain unchanged, load width grows from 16 to 32 bits,
and loop iterations fall from 160 to 80. The compiled loop has 16 instructions
per iteration instead of 11, reducing estimated dynamic loop instructions from
1,760 to 1,280 per lane (27.3%). This is a codegen observation, not timing.

The pair partition changes draft logit rounding and can change draft argmax.
That remains lossless only under the fixed32 deterministic proposal contract:
`draft_probs=None`, argmax proposal selection, and one-hot multi-draft rejection
sampling. All seven reference tests pass. Runtime binding still must prove that
the candidate output is used only for proposal token selection.

Static codegen uses 20 registers/thread and zero stack, local, or shared memory.
It contains no CTA barrier, local load/store, atomic, or call. The immutable
114,304-byte binary has SHA256
`60cad2aa39f2d36769070d8947294759e9af6158c02b9c9feaa8c43860236d72`
and a GLIBC ceiling of 2.32.

Source commit: `840bd0fd5806fc114dd921e3e9a9298c44f7c354` on
`agent/fixed32-dfwd-k64-m1-warp32-r32-pair2-20260802`. Full-step speed,
acceptance, task quality, and hardware-floor impact remain unmeasured.

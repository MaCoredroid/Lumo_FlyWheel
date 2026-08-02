# Fixed32 DFWD K64 M1 full-warp R32 pair2bits build

Status: pinned host build and static codegen audit pass; default off and not
runtime-integrated. No GPU, synthetic probe, workload, or real task was run.

The candidate assigns one 32-lane warp to each output row in a 1,024-thread
CTA. Each lane loads one aligned 32-bit word containing two BF16 values, shifts
or masks each BF16 bit pattern into the high half of an FP32 word, then performs
two ordered FP32 FMAs. An exhaustive CPU reference test verifies this bit
expansion for every non-NaN BF16 pattern.

Relative to scalar warp32/R32, total weight bytes and FMAs remain unchanged,
while load width grows from 16 to 32 bits and loop iterations fall from 160 to
80. The compiled loop has 14 instructions per iteration instead of 11,
reducing estimated dynamic loop instructions from 1,760 to 1,120 per lane
(36.4%). It also removes two instructions per loop relative to the built-in
pair conversion candidate. These are codegen observations, not timing.

The pair partition changes draft logit rounding and can change draft argmax.
That remains lossless only under the fixed32 deterministic proposal contract:
`draft_probs=None`, argmax proposal selection, and one-hot multi-draft rejection
sampling. All seven reference tests pass. Runtime binding still must prove that
the candidate output is used only for proposal token selection.

Static codegen uses 20 registers/thread and zero stack, local, or shared memory.
It contains no CTA barrier, local load/store, atomic, or call. The immutable
114,328-byte binary has SHA256
`559391fa1de3a6231327ca54f5a5ea7171ac1defd9b1ba0ce714104d25642d88`
and a GLIBC ceiling of 2.32.

Source commit: `7f3f2598fc4adc60f5f49fef387257e25382e6a6` on
`agent/fixed32-dfwd-k64-m1-warp32-r32-pair2bits-20260802`. Full-step speed,
acceptance, task quality, and hardware-floor impact remain unmeasured.

# Fixed32 DFWD K64 M1 full-warp R32 pair4bits build

Status: pinned host build and static codegen audit pass; default off and not
runtime-integrated. No GPU, synthetic probe, workload, or real task was run.

The candidate assigns one 32-lane warp to each output row in a 1,024-thread
CTA. Each lane loads one aligned 64-bit word containing four BF16 values,
expands the four bit patterns into exact FP32 values, and performs four ordered
FP32 FMAs. The exhaustive BF16 bit-expansion reference test remains in the
focused source suite.

Relative to scalar warp32/R32, total weight bytes and FMAs remain unchanged,
while load width grows from 16 to 64 bits and loop iterations fall from 160 to
40. The compiled loop has 20 instructions per iteration instead of 11,
reducing estimated dynamic loop instructions from 1,760 to 800 per lane
(54.5%). It also reduces this estimate by 28.6% relative to pair2bits. These
are codegen observations, not timing.

The quad partition changes draft logit rounding and can change draft argmax.
That remains lossless only under the fixed32 deterministic proposal contract:
`draft_probs=None`, argmax proposal selection, and one-hot multi-draft rejection
sampling. All seven reference tests pass. Runtime binding still must prove that
the candidate output is used only for proposal token selection.

Static codegen uses 24 registers/thread and zero stack, local, or shared memory.
It contains no CTA barrier, local load/store, atomic, or call. The immutable
114,328-byte binary has SHA256
`63b33b12f6034f17ed2a1406b085e8e9ec56ce3f213bc489bc7f7c3cd9d0a2e1`
and a GLIBC ceiling of 2.32.

Source commit: `ecddc53b89633f6eaf4439b84d0e2c32d27a4628` on
`agent/fixed32-dfwd-k64-m1-warp32-r32-pair4bits-20260802`. Full-step speed,
acceptance, task quality, and hardware-floor impact remain unmeasured.

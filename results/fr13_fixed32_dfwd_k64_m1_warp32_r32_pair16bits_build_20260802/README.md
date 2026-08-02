# Fixed32 DFWD K64 M1 full-warp R32 pair16bits build

Status: pinned host build and static codegen audit pass; default off and not
runtime-integrated. No GPU, synthetic probe, workload, or real task was run.

The candidate assigns one 32-lane warp to each output row in a 1,024-thread
CTA. Each lane issues four 128-bit loads for 16 input/weight BF16 values,
expands the bit patterns into exact FP32 values, and performs 16 ordered FP32
FMAs. The exhaustive BF16 bit-expansion reference test remains in the focused
source suite.

Relative to scalar warp32/R32, total weight bytes and FMAs remain unchanged,
while loop iterations fall from 160 to 10. The compiled loop has 58
instructions per iteration instead of 11, reducing estimated dynamic loop
instructions from 1,760 to 580 per lane (67.0%). Relative to pair8bits this is
only a 9.4% instruction reduction, while registers rise from 28 to 36 and both
candidates execute 40 dynamic 128-bit loads per lane. This closes the offline
load-width curve at sharply diminishing returns; it is not a timing result.

The 16-element partition changes draft logit rounding and can change draft
argmax. That remains lossless only under the fixed32 deterministic proposal
contract: `draft_probs=None`, argmax proposal selection, and one-hot
multi-draft rejection sampling. All seven reference tests pass. Runtime binding
still must prove that the candidate output is used only for proposal selection.

Static codegen uses 36 registers/thread and zero stack, local, or shared memory.
It contains no CTA barrier, local load/store, atomic, or call. The immutable
114,328-byte binary has SHA256
`e3c65cd4a855de2dc5edc91f81bc2b7b08f16efd5652623c9ae98d20a90f19fa`
and a GLIBC ceiling of 2.32.

Source commit: `957bb1c307bdc1fc64ad2359f000dca542bc0fce` on
`agent/fixed32-dfwd-k64-m1-warp32-r32-pair16bits-20260802`. Full-step speed,
acceptance, task quality, and hardware-floor impact remain unmeasured.

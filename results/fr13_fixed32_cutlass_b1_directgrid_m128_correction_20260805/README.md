# Fixed32 B1 direct-grid M128 correctness correction

Status: **corrected SM121a shared object built and host-verified; real
SWE-Verified GPU byte gate and timing pending**.

## Why the prior arm was rejected

The authenticated B1 diagnostic on `astropy__astropy-12907` reached a real
`m32 n16384 k5120` projection and found 22 differing bytes in the 1,048,576
byte output. That result rejects the prior `256x32x128` cooperative
collective for byte-exact qualification. It produced no valid timing or
hardware-floor evidence.

There is no M, N, or K tail in that record: the byte count is exactly
`32 * 16384 * 2`, N is divisible by 256 and 128, and K is divisible by 128.
The sparse mismatch therefore does not support a gross tile-coverage or
output-layout failure. The direct scheduler was not active in that failed
runtime binary, so the scheduler substitution is not implicated by that
record.

## Corrected kernel

Commit `33fb5acb833b81b2c82b90d76b5107a7718beefb` restores the exact
`128x32x128`, StageCount2, ping-pong collective used by the correctness-backed
B1 one-N full-grid arm. It retains only the direct bounded linear scheduler.
The legacy selector name still contains `wide256`, but its corrected tile M
is 128.

The corrected FP16 and BF16 direct kernels each compile to 752 SASS
instructions, 44 branch-class operations, 168 registers, 8 bytes of stack,
1,024 bytes of static shared memory, three `LDL`, one `STL`, and no `CALL`.
The generic one-N full-grid neighbor is 968 instructions and 45 branch-class
operations with no stack or local load/store instructions. This is static
codegen evidence only.

## Validation and next gate

The pinned binary identity verifier, CPU import with CUDA hidden, 173 focused
tests, Python byte compilation, and `git diff --check` passed. No GPU runtime
was used for this build artifact.

The next authoritative step is the authenticated real SWE-Verified B1
physical32/K64/root1 raw-byte gate over all five admitted projection shapes.
It must produce 320 comparisons, zero mismatching comparisons, and zero
differing bytes while serving stock output. Performance tuning and acceptance
must then use the standing exact 4-task set, followed by the exact 16-task set;
one-task diagnostics and synthetic probes are not acceptance evidence.


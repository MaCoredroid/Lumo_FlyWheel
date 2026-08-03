# Fixed32 B4 N5120 one-tile scheduler host audit

This reduced artifact records a host-only CUDA 13.0 SM121a compile, link,
import, resource, SASS, and exact-geometry audit of the fixed32 K64 B4 hybrid
CUTLASS route. The two exact `N=5120` projections already use one cooperative
`128x128x128` output tile per CTA. This candidate replaces CUTLASS's general
persistent device scheduler for those projections with an exact `(40, 1, 1)`
X-axis scheduler. The other three B4 projection shapes retain the admitted
two-M ping-pong kernel.

## Result

The candidate is retained for real SWE-Verified exact4 byte qualification.
Both FP16 and BF16 kernels compile at 168 registers with zero stack, local
memory, `LDL`, `STL`, or `CALL`. SASS falls from 1,352 to 1,008 instructions
per dtype, a reduction of 344 instructions or 25.44%. Branches fall from 40 to
33. The arithmetic and data-movement census is unchanged at 128 QMMA, 128
FFMA, 48 LDSM, and 16 STSM instructions.

The incumbent cooperative kernel was emitted alongside the candidate and is
byte-identical to the corresponding kernel in the prior hybrid build for both
dtypes. The candidate therefore changes only the explicitly selected N5120
specialization; the default and retained hybrid routes keep their prior
codegen.

The linked extension imports successfully with `CUDA_VISIBLE_DEVICES` empty.
No GPU kernel, Docker service, synthetic workload, SWE-Verified task, timing
run, or hardware-floor measurement was used. The object, cubin, linked binary,
generated dispatch, raw resource dump, raw disassembly, build logs, task data,
and credentials are not published.

## Qualification boundary

This artifact is not an acceptance result. The next valid ladder is the
canonical real SWE-Verified exact4 K64/root byte gate for Tail23 and Hydra27,
followed by the exact4 full-wall timing pair only if both byte gates pass. The
hardware-floor distance is unchanged until that real-task timing exists.

# Fixed32 FA2 B1 qrow32 no-split SM121a checkpoint

Status: **host codegen passes; real-task byte and timing qualification are
pending**.

The exact B1 qrow32 candidate replaces 48 one-warp qrow16 CTAs with 24
two-warp CTAs. Both variants therefore expose 48 query warps per layer, but
the qrow32 CTA shares one ordered K/V scan across its two query warps. The
candidate remains hidden and default-off behind a distinct exact-geometry
sentinel.

Fresh paired CUDA 13.0 SM121a builds from FA2 commit `2921022186` report:

| kernel | CTAs | warps/CTA | registers | stack/local/spills | static SASS |
| --- | ---: | ---: | ---: | --- | ---: |
| qrow16 static strides | 48 | 1 | 212 | 0/0/0 | 5,000 |
| qrow32 B1 no-split | 24 | 2 | 246 | 0/0/0 | 3,984 |

The qrow32 kernel has eight registers of headroom below its explicit 254 cap.
Its static instruction text is 20.32% smaller. Ordered attention work is
preserved at 512 HMMAs, 132 FFMAs, 264 FMULs, 288 `LDSM`, and 38 `STG`
instructions. `LDGSTS` sites fall from 336 to 176, a 47.62% reduction.

Multiplying static sites by the launch's total 48 warp instances gives a
clearly labeled event proxy, not dynamic instruction counts: `LDGSTS` falls
from 16,128 to 8,448 and total sites from 240,000 to 191,232. CTA-fixed work
is not represented by that proxy; qrow32 additionally halves CTA count. The
24-CTA grid can underfill a 48-SM GPU, so split-K=2 is being evaluated as the
next candidate using FA2's existing combine implementation.

This checkpoint used the host compiler and disassembler only. It contains no
GPU timing, synthetic probe, real task, task data, raw compiler output, raw
SASS, object, cubin, prompt, response, credentials, or environment dump.
Acceptance requires lossless output qualification and full-step measurement
on the standing real SWE-Verified 4-task set (or the established 16-task set).

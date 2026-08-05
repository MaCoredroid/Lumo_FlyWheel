# Fixed32 DFWD K64 M1 shared R64-U8 fallback

Status: static SM121a codegen pass, default off, runtime unwired. No GPU,
Docker, real SWE-Verified task, correctness gate, timing, or acceptance claim
was performed.

The kernel preserves the fast U8 per-lane load/FMA order and replaces the
width-16 shuffle reduction with the incumbent library's shared-memory
`8/4/2/1` association. It keeps the 1,024-CTA, 64-row-per-CTA geometry. This is
a separate comparator/fallback; it does not replace the quality-changing fast
shuffle candidate.

CUDA 13.0.88 device-only compilation for `sm_121a` produced 30 registers per
thread, zero stack/spills/local memory, one barrier resource, and no shuffle
instructions. Static SASS contains the expected four CTA barriers, eight
shared loads, four shared stores, four FP32 adds, and the eight-FMA unrolled
steady body. The launch requests 4,352 bytes of dynamic shared memory.

Required next evidence is a linked build followed by a real B1 five-depth
comparison against the incumbent. Only then can it be called exact. If exact,
measure it as the fallback point beside the fast shuffle acceptance/TPS point;
do not substitute this static artifact for a real-task result.

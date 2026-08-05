# Fixed32 DFWD K64 M1 exact-order R64 static build

Status: **default off, runtime unwired, static SM121a codegen pass; no GPU
execution, byte qualification, real task, timing, or production admission**.

This bounded candidate changes only the B1 K64 M1 drafter-head CTA ownership
from the R32 parent's 32 rows in 512 threads to 64 rows in 1,024 threads. It
therefore launches 1,024 CTAs instead of 2,048. Each output row retains the
same 16-lane K partition, 320 dependent scalar FP32 `__fmaf_rn` operations per
lane, width-16 `8+4+2+1` `__fadd_rn` shuffle tree, alpha-one/beta-positive-zero
epilogue, and BF16 rounding. Focused tests compare the complete per-row
arithmetic source body directly with the R32 parent.

The candidate was compiled offline with the immutable deployed build image
`sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`:
Torch `2.11.0+cu130`, CUDA 13.0, target `sm_121a`, network disabled, and no GPU
device exposed. Extension registration passed without executing the kernel.

The linked binary has SHA-256
`95f3a63200af4af622ca2f788f29c5c422fa425aca6c5e58953190dbd296e009`
and is 113,680 bytes. The cubin reports 18 registers/thread, zero stack, local,
or shared bytes, and a 1,024-thread launch bound. SASS has four width-16
shuffles, four FP32 adds, two static FP32 FMAs, and no CTA `BAR`, local
load/store, call, or atomic instruction. `BSSY`/`BSYNC` reconvergence matches
the R32 control-flow shape and is not a CTA synchronization barrier.

The CUDA 13.0 build image does not package `nvdisasm`, so the extracted CUDA
13.0 cubin was decoded offline by local CUDA 13.1 `nvdisasm`; the pinned CUDA
13.0 `cuobjdump` produced the ELF and resource metadata. Both audit containers
had network disabled and no GPU exposure. Raw outputs and the fail-closed
machine verdict are included here.

Source checkpoint: `fc744b40009c88f873a53edb2c7e37c2bf81154f`.
Static checker checkpoint: `54bf895d56b1505f4a9f8835c60be560326dfb23`.

No performance result follows from halving CTA count. Before any runtime or
timing use, R64 requires an authenticated real SWE-Verified B1 full-logit byte
gate at the root and all four MTP sites, serving the incumbent reference, then
matched real-task timing. This artifact provides no selector or runtime path.

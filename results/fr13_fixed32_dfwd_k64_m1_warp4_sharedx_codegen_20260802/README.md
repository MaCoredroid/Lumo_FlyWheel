# Fixed32 DFWD K64 M1 warp4 shared-x codegen

This artifact records host-only SM121a code generation for a default-off BF16
K64 M1 draft-head candidate. Eight warps per CTA each compute four output rows,
while the CTA stages the 5,120-element hidden vector once in shared memory.

Compared with the preceding one-row-per-warp pair8 kernel, the candidate keeps
the same 2,048 CTAs and weight loads, reduces logical input loads per CTA from
20,480 to 640, and reduces total warps per call from 65,536 to 16,384. Each
output row retains the same lane assignment, K order, eight-FMA order, and
width-32 shuffle reduction order.

The exact cubin uses 38 registers, no stack/local storage, one vector shared
load, one vector shared store, and one barrier. Ptxas reports 11,264 bytes of
shared memory: the nominal 10,240-byte input stage plus the SM121a 1,024-byte
reported baseline overhead.

This is not a loadable Torch extension and no GPU kernel was executed. It makes
no live correctness, task, acceptance, throughput, B4, production, or
hardware-floor claim. A pinned-image extension build and real SWE-Verified gate
remain required.

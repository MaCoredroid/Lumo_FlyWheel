# Fixed32 DFWD K64 M1 warp4 global-x codegen

This artifact records the direct-global-input control for the warp4 K64 M1
draft-head curve. Eight warps per CTA each compute four output rows and reuse
one globally loaded hidden octet across those rows. The candidate keeps the
same per-row FMA and shuffle order as pair8bits.

Relative to one-row-per-warp pair8, logical input loads and total warps per call
both fall 4x while weight loads remain unchanged. Relative to warp4 shared-x,
this control removes the shared stage and barrier but performs eight times as
many logical input loads per CTA (5,120 versus 640).

The exact SM121a cubin uses 40 registers with no shared memory, barriers,
stack, local traffic, spills, or calls. This is host-only codegen, not a
loadable Torch extension. No GPU, task, acceptance, timing, B4, production, or
hardware-floor claim is made.

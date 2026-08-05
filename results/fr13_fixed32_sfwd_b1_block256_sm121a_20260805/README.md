# Fixed32 B1 SFWD BLOCK_C=256 candidate

Status: **OFFLINE CODEGEN PASS; REAL B1 BYTE GATE REQUIRED**.

The fixed32 B1 conv/post-prep launcher now uses the already compiled B4 tile
geometry: `BLOCK_C=256`, four warps, and eight gate rows per program. The
kernel arithmetic source is unchanged. Offline SM121a codegen reports 56
registers, no stack/local/shared spills, and no calls. Its cubin and SASS are
byte-identical to the prior audited B4 build of the same kernel geometry.

For one physical32 B1 request, the grid falls from 88 to 44 programs per
layer: channel programs fall from 80 to 40 and gate programs from 8 to 4.
Across 48 layers that is 4,224 to 2,112 programs. Total scheduled warps remain
176 per layer because each program grows from two to four warps. Requested
gate bytes fall from 20,736 to 19,584 per layer because invariant gate loads
are issued by four programs instead of eight.

This is source and offline codegen evidence only. It is not measured GPU
timing, DRAM/HBM traffic, runtime byte equality, TPS, or hardware-floor
evidence. The next required step is the real SWE-Verified B1 eager Qrow16
target/SFWD byte gate on the exact merged source, followed by full-step timing.

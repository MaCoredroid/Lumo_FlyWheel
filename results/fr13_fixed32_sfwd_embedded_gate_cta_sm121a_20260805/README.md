# Fixed32 SFWD embedded gate CTAs

Status: **OFFLINE SM121a CODEGEN PASS; REAL B1/B4 BYTE AND TIMING GATES REQUIRED**.

The candidate at `086da781207322601fc4876f9f6d69292a4a71a1` removes the
four standalone gate CTAs from each fixed32 request. The unchanged 8-row gate
tiles run after the channel stores in channel CTAs 0 through 3. The launch grid
therefore contains only the 40 `BLOCK_C=256` channel CTAs per request.

For B1, the grid falls from 44 to 40 CTAs per layer and from 2,112 to 1,920
CTAs across 48 layers. For the whole B4 batch, it falls from 8,448 to 7,680.
The number of launched warps falls by 768 for B1 and 3,072 for B4. Kernel
launches remain 48 per event, and the gate computation groups, arithmetic,
addresses, and requested gate bytes are unchanged.

Two independent cold-cache offline builds targeting SM121a produced
byte-identical summaries and binaries. Both B1 and B4 compile to 56 registers
with zero stack, local, shared, spill, or call use. Relative to the standalone
gate baseline, encoded SASS falls from 3,040 to 3,024 instructions and static
SASS from 2,889 to 2,875; LDG and STG counts remain 85 and 336.

CPU tests enumerate every physical32 channel and row-by-head gate output for
B1 and B4 and prove exact-once coverage. This is static and CPU evidence only.
It is not device byte equality, measured GPU timing, DRAM/HBM traffic, TPS, or
hardware-floor acceptance evidence. The next gate is real SWE-Verified B1 and
B4 byte equality on the exact merged source, followed by exact4 and exact16
full-step timing.

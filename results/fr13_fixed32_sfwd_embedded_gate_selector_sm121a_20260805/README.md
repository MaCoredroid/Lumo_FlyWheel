# Fixed32 SFWD embedded-gate selector

Status: **OFFLINE SM121a CODEGEN PASS; REAL B1/B4 BYTE GATES REQUIRED**.

Source commit `7e99008327eb1b0609793277a10c282c3d85b7d8` restores the
44-CTA standalone-gate schedule as the default and exposes the 40-CTA
embedded-gate schedule only through
`FR13_FIXED32_SFWD_EMBED_GATE_CTA=1`. The opt-in route is shadow-only and is
restricted to exact Hydra27 physical32 K64/root1 B1 or B4 byte gates.

Both selector values were compiled from the same generated kernel source for
B1 and B4. Each specialization uses 56 registers with zero stack, local,
shared, spill, or call use. The default specialization emits the same SASS
instruction counts as the prior standalone schedule: 3,040 encoded and 2,889
static instructions. The embedded specialization emits 3,024 encoded and
2,875 static instructions. Both emit 85 LDG and 336 STG instructions.

The 40-CTA schedule removes four standalone CTAs per request and layer. Across
48 layers, that is 192 fewer CTAs and 768 fewer launched warps for B1, or 768
fewer CTAs and 3,072 fewer launched warps for the whole B4 batch. Gate
computation groups, source addresses, requested gate bytes, and kernel launch
count are unchanged.

Two independent cold-cache offline builds produced byte-identical summaries
and binaries. This is compiler and static-work evidence only. It is not device
byte equality, measured GPU timing, DRAM/HBM traffic, TPS, or hardware-floor
acceptance evidence. The next required evidence is real SWE-Verified B1 and
exact4 B4 byte equality on the bound source commit, followed by exact4 and
exact16 full-step timing.

# Fixed32 SFWD paired-weight x-gather codegen

Status: **OFFLINE_CODEGEN_PASS_REAL_BYTE_GATE_REQUIRED**.

Source commit `d6a7aec63` keeps the current tap in the gather layout and loads
the exact contiguous BF16 convolution weights as two 32-bit pairs. The
arithmetic remains ordered as taps 0, 1, 2, then 3; the bitcasts do not change
the BF16 operands.

Fresh-cache SM121a builds produced identical cubin, PTX, and SASS hashes for
B1 and B4 and across two independent rebuilds. Relative to the real-B1
byte-qualified packed x-gather source at `7c9fda4bc`, the C64/W16 candidate
reduces static SASS instructions from 408 to 395, LDG from 33 to 28, LDS from
8 to 6, STS from 10 to 6, and BAR from 5 to 3. Registers remain 55, launch
shared memory remains 4,096 bytes, and stack, local memory, spills, and calls
remain zero.

This is static codegen evidence only. The paired-weight source has not run a
GPU kernel or real task and is ineligible for timing or floor claims until a
fresh reference-served real SWE-Verified byte gate passes.

The package excludes raw binaries, PTX, SASS, IR, compiler logs, task/model
content, requests, responses, patches, environment values, process/container
identifiers, and secrets.

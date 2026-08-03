# Fixed32 B4 FA2 qrow32 register recovery

Status: **host codegen pass; register rejection resolved; real byte gates
pending**.

## Kernel result

The previously rejected carried-page qrow32 kernel compiled at the SM121a
architectural ceiling of 255 registers. This change gives the exact
BM32/BN64/two-warp specialization a private non-templated entry point with a
254-register source ceiling. A fresh detached FA2 checkout was patched from
code commit `9fc17c8b5c785d29f59ca9b6df39ffe840a0c600` and compiled with CUDA
cache disabled.

The paired default-ptxas result is:

- registers: 255 to 252;
- stack, local memory, spill loads, spill stores, SASS `LDL`, `STL`, and
  `CALL`: zero in both arms;
- SASS slots through the terminal branch: 4,082 to 3,984, down 98 (2.401%);
- `LDG`: 68 to 68 and `LEA`: 202 to 195;
- mandatory work unchanged: 512 BF16 HMMAs, 132 FFMAs, 264 FMULs, 176
  `LDGSTS`, 288 `LDSM`, and 38 global stores;
- static shared memory remains 1,024 bytes and launch dynamic shared memory
  remains 80 KiB.

The paired incumbent rebuild reproduced the historical carried-page SASS
byte-for-byte. The fresh candidate rebuild reproduced the earlier candidate
SASS byte-for-byte. A register-usage-level-3 sensitivity build also remained
spill-free at 246 registers and 3,972 slots, but it is not runtime-ranked here.

## Default-off evidence

This remains a gate-only candidate. The source generator reports the qrow32
live selector off, exposes no qrow32 production selector, leaves the stock
translation unit unchanged, and routes the private kernel only through the
exact B4 diagnostic batch-stride sentinel and fail-closed geometry checks.
The admission gate now rejects more than 254 registers or any SASS local
memory/call instruction.

## Scope and next gate

This was a host-only CUDA compile and disassembly audit. No GPU workload,
synthetic probe, real SWE-Verified task, byte comparison, timing sample, or
hardware-floor measurement was used. The kernel is not production-eligible or
timing-eligible until independent canonical real SWE-Verified exact4 B4 raw
byte gates pass for both Tail23 and Hydra27.

This directory contains only aggregate metadata and hashes. It excludes
objects, cubins, raw compiler logs, raw SASS, task content, prompts, responses,
credentials, and process or environment dumps.

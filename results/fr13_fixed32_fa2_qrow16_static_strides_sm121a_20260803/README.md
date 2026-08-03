# Fixed32 FA2 qrow16 static K/V strides

Status: **default-off SM121a codegen passes; the real B1 K64/root1 byte gate
and full-step timing remain pending**.

## Change

The private B1 qrow16 `BlockM=16`, `BlockN=64` translation unit now has an
opt-in trait for the canonical contiguous paged-K/V layout:

- page stride: `1024 * 4 * 256`
- row stride: `4 * 256`
- head stride: `256`

The hidden dispatch checks all six K/V stride values exactly before entering
the specialized kernel. Stock FA2 traits retain zero-valued compile-time
strides and therefore use their original runtime K/V strides. A stock HD256
BF16 split-forward explicit-instantiation TU was compiled successfully from
the same patched header as an independent fallback control.

The specialization changes address formation only. Query-row mapping, the
complete ordered K loop, QK/PV accumulation order, masking, tree bias,
softmax, output stores, launch geometry, and `num_splits=0` remain unchanged.

## Paired SM121a codegen

Both objects were generated from pristine FA2 commit
`29210221863736a08f71a866459e368ad1ac4a95` with CUDA 13.0.88, target
`sm_121a`, ptxas register-usage level 3, and the existing 216-register source
cap. CUDA visibility was explicitly empty.

| Metric | RU3/R216 incumbent | Static-stride candidate | Delta |
| --- | ---: | ---: | ---: |
| Registers/thread | 216 | 212 | -4 |
| Stack/local bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| Spill stores/loads | 0 / 0 | 0 / 0 | 0 / 0 |
| Static shared bytes | 1,024 | 1,024 | 0 |
| Launch dynamic shared bytes | 73,728 | 73,728 | 0 |
| SASS text slots | 5,064 | 5,000 | -64 (-1.264%) |
| Object bytes | 162,000 | 160,512 | -1,488 |

The arithmetic and data-movement opcode census is identical: 512 HMMAs, 132
FFMAs, 264 FMULs, 336 LDGSTS, 288 LDSM, 75 LDG, and 38 STG instructions.
Static address-instruction families fall as follows:

| Family | Incumbent | Candidate | Delta |
| --- | ---: | ---: | ---: |
| LEA + ULEA | 301 | 294 | -7 |
| IMAD + UIMAD | 337 | 310 | -27 |
| IADD | 67 | 63 | -4 |
| IADD3 + UIADD3 | 264 | 250 | -14 |
| LOP3 + PLOP3 + ULOP3 | 73 | 76 | +3 |

The final candidate SASS SHA-256 is
`1ed3d6bfc235e0577f290ebf4ed86b6501a8ca9c7d65b24b137e20ec276c22da`.
The independently regenerated final disassembly is byte-identical to the
prior candidate disassembly.

## Residency

The launch consumes `73,728 + 1,024 = 74,752` shared bytes per CTA. On compute
capability 12.x, NVIDIA documents 131,072 shared bytes per SM, 101,376 shared
bytes per block, 65,536 registers per SM, 48 resident warps per SM, and 32
resident blocks per SM. Shared memory therefore limits both variants to one
CTA per SM. Register limits are nine CTAs for both
(`floor(65536 / (registers * 32))`). The B1 grid remains 48 one-warp CTAs, so
the register reduction does not change static occupancy.

Reference: [NVIDIA Blackwell tuning guide, occupancy](https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-tuning-guide/index.html#occupancy).

## Rejected BlockN128 alternative

A bounded `BlockN=128` attempt was rejected before integration. Its best clean
RU3 variants used 254 or 248 registers, but every variant required 139,264
dynamic plus 1,024 static shared bytes, or 140,288 bytes per CTA. That exceeds
both the 101,376-byte block limit and the 131,072-byte SM capacity, so it has
zero legal residency. No BlockN128 code is retained.

## Qualification boundary

This is host compiler/disassembler evidence only. No GPU kernel, synthetic
probe, service, task, request, timing, or acceptance campaign was run. The
candidate is not timing-eligible or production-authorized until it passes the
standing real SWE-Verified B1 K64/root1 byte gate. Only then may full-step TPS
and component timing be compared on the standing real task set.

The checked-in package excludes objects, cubins, PTX, SASS, raw compiler logs,
task/model/request/response/patch data, credentials, environment dumps,
process IDs, and container IDs.

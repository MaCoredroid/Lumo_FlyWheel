# Fixed32 B1 FP8 activation-quantization register cache

Status: **SM121a codegen gate passed; default off; real-task byte and timing
qualification pending**.

## Kernel change

The deployed Qwen3.5 target model has 128 block-FP8 activation-quantization
calls with input `[32,5120]` per target forward: one attention/GDN input
projection and one MLP gate/up projection across each of 64 layers. Each call
contains 1,280 independent groups of 128 BF16 values.

The stock CUDA kernel assigns one group to a 16-lane half warp. Each lane loads
eight BF16 values, writes them to shared memory, participates in the same
half-warp maximum reduction, waits at a 256-thread barrier, reloads its eight
values, and emits FP8 values plus one FP32 scale. The candidate retains those
eight values in registers across the unchanged reduction and conversion.

For every admitted call this removes:

- 4,096 bytes of dynamic shared-memory allocation per CTA;
- 4,096 shared-store and 4,096 shared-load bytes per CTA;
- one CTA-wide barrier per CTA;
- 80 CTAs, so 655,360 shared-memory bytes and 80 barriers per call.

Across the 128 known target-model calls, that is 83,886,080 shared-memory bytes
and 10,240 CTA barriers removed per target forward. Global input reads, FP8
output writes, scale writes, launch count, group maxima, scales, FP8 casts,
CUTLASS GEMMs, weight bytes, and accumulation order are unchanged.

The selector `FR13_FIXED32_B1_FP8_QUANT_REGCACHE=1` is default off. Admission
requires BF16 input `[32,5120]`, FP8 E4M3 output `[32,5120]`, group size 128,
non-UE8M0 scaling, column-major FP32 scales `[32,40]` with stride `[1,32]`,
and compatible pointer alignment. Any drift uses stock.

## Paired SM121a result

Baseline and candidate were compiled from independent detached copies of vLLM
`fe9c3d6c5f66c873d196800384ed6880687b9e52` with the same CUDA 13.0 command,
`arch=compute_121a,code=sm_121a`, and distinct initially empty CUDA cache
directories. No GPU runtime was used.

| Kernel | Registers | Stack/local | Dynamic shared launch | BAR | LDS | STS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock BF16, column-major, non-UE8M0 | 48 | 0 / 0 | 4,096 B | 1 | 12 | 40 |
| Register-cache candidate | 26 | 0 / 0 | 0 B | 0 | 0 | 0 |

`cuobjdump` reports a 1,024-byte shared metadata record for both functions on
SM121a. The operational distinction is explicit: stock launches with 4,096
dynamic shared bytes and contains shared load/store plus barrier opcodes; the
candidate launches with zero dynamic shared bytes and contains none of those
opcodes. The untouched stock function has identical normalized SASS in the
baseline and candidate objects.

The dumped function images contain 2,120 stock versus 408 candidate encoded
instructions, and 2,109 versus 398 after excluding NOP. These are static code
image counts including compiler-emitted conversion helpers, not a dynamic
instruction or performance measurement.

## Qualification boundary

- Focused patch tests: 7 passed.
- Python byte compilation and `git diff --check`: passed.
- Paired fresh-source/fresh-cache SM121a compile: passed.
- Candidate: 26 registers, zero stack/local spill, zero `BAR`/`LDS`/`STS`.
- No GPU kernel, synthetic probe, Docker run, SWE-Verified task, raw-byte
  comparison, B1 timing sample, or hardware-floor acceptance run was performed.
- Expected impact is SFWD only. DFWD and CFWD are unchanged.

The next gate is an authenticated real SWE-Verified K64/root1 B1 raw-byte A/B
on fixed32 Tail23/Hydra27. Full-step timing is permitted only after byte equality
passes.

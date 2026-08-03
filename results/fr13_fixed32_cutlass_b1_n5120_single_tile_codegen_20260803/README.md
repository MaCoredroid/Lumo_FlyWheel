# Fixed32 B1 N5120 single-tile scheduler codegen

Status: **SM121a codegen gate passed; default off; real-task raw-byte and
timing qualification pending**.

## Kernel change

The two admitted B1 projections with physical `N=5120` contain exactly 40
scheduler-M tiles after swap-AB and launch a `(1,40,1)` grid. Every CTA owns one
complete output tile. `Fr13B1N5120SingleTileScheduler100` encodes that invariant
directly: the initial tile is `{blockIdx.y,0,0}`, every tile is terminal, and
next work is invalid. This removes runtime problem-tile state, grid-stride
advance, and next-tile bounds work from those two kernels.

The candidate retains the qualified `128x32x128` tile, cluster `(1,1,1)`,
cooperative mainloop, StageCount2, identity epilogue, swap-AB layout, and full-K
accumulation order. The three wider admitted projections continue to use the
existing one-N ping-pong scheduler. Unset, unknown, and out-of-contract shapes
remain stock.

## Paired SM121a result

Baseline and candidate were compiled as complete vLLM translation units from
independent detached copies of pinned vLLM, with distinct initially empty CUDA
cache directories and the same CUDA 13.0 SM121a command.

| Dtype | Kernel | Encoded | Non-NOP | Branches | Registers | Stack/local |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | existing one-N | 568 | 540 | 29 | 168 | 0 / 0 |
| FP16 | N5120 single-tile | 520 | 493 | 26 | 168 | 0 / 0 |
| BF16 | existing one-N | 568 | 540 | 29 | 168 | 0 / 0 |
| BF16 | N5120 single-tile | 520 | 493 | 26 | 168 | 0 / 0 |

The candidate removes 48 encoded instructions (8.45%) and 47 operational
instructions per dtype. Both candidate kernels have zero `LDL`, `STL`, and
`CALL`. Their resource records remain 168 registers, zero stack, zero local
memory, and 1,024 bytes static shared memory.

The existing FP16/BF16 one-N cooperative and ping-pong SASS functions are
byte-for-byte identical between the paired objects. Thus the codegen delta is
isolated to the new N5120 specialization.

## Qualification boundary

- Focused patch tests: 38 passed.
- Python byte compilation and `git diff --check`: passed.
- Paired fresh-source/fresh-cache SM121a compile and static resource/SASS
  audit: passed.
- No GPU kernel, Docker container, synthetic probe, SWE-Verified task, raw-byte
  comparison, timing sample, or hardware-floor acceptance run was performed.
- The diagnostic selector is stock-serving. Production remains default off.

The next gate is an authenticated real SWE-Verified K64/root B1 byte A/B run,
followed by standing-rule real-task full-step timing only if bytes pass.


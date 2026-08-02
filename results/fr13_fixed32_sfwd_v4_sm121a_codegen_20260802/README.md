# Fixed32 SFWD v4 SM121a codegen

Status: **offline SM121a codegen passes; no production source change is
justified by static codegen**.

The merged v4 load-once/two-activation-window kernel at source revision
`3295f4d38045486244b8cea1b1f647edc5617cc0` was compiled against its retained
v3 load-once incumbent at
`ac8d848b63278a9c956ebbb31b9b7836372816f1`. Both revisions use the production
selector geometry:

- 32 physical rows per request, 10,240 channels, width 4, state length 34
- padded x row stride 16,384 and compact conv-state row stride 348,160
- B1: `BLOCK_C=128`, two warps, 80 CTAs/request
- B4: `BLOCK_C=256`, four warps, 40 CTAs/request
- BF16 inputs/products, ordered FP32 accumulation, no bias
- CUDA target `sm_121a`

CUDA visibility was explicitly empty. No GPU kernel, service, task, request,
timing run, or acceptance run was launched. The fixed32 fail-closed selector
and K64 drafter route were not modified.

## Result

The B1 and B4 specializations have the same per-CTA static resources and
instruction counts. B4 changes only launch geometry and thread count.

| Metric | v3 incumbent | v4 candidate | Delta |
|---|---:|---:|---:|
| Registers/thread | 48 | 48 | 0 |
| Stack / local bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Static / encoded SASS | 1,834 / 1,848 | 1,828 / 1,840 | -6 / -8 |
| LDG / STG | 75 / 136 | 75 / 136 | 0 / 0 |
| FADD / FMUL / MUFU | 312 / 384 / 128 | 312 / 384 / 128 | 0 / 0 / 0 |
| FSETP / F2FP | 128 / 32 | 128 / 32 | 0 / 0 |
| MOV / PRMT | 39 / 261 | 31 / 260 | -8 / -1 |
| Cubin bytes | 109,520 | 115,936 | +6,416 |

The paired source schedule preserves 32 unique current-row loads, zero x
reloads, 32 activations, 128 ordered product assignments, and 68 stores. Its
deliberate source-level change is 16 saved first-of-pair accumulators and a
peak of two live accumulator values instead of one. Backend allocation remains
48 registers/thread with no spills.

Primary and fresh-cache rebuild cubins, PTX, SASS, resource reports, and ELF
reports were byte-identical. The verifier independently re-disassembled every
cubin, checked `.target sm_121a`, checked B1/B4 thread counts, recounted static
instructions, and enforced the register, stack, local-memory, spill, and call
gates.

## Decision

Keep v4. Static codegen shows no defect: it preserves register allocation and
memory operations while slightly reducing static and encoded SASS. A source
rewrite would be speculative and is therefore out of scope. These results do
not establish latency, throughput, full-step TPS, or hardware-floor progress;
those require the standing real SWE-Verified byte and timing gates.

The checked-in package contains only reduced summaries and reproduction code.
It excludes cubin, PTX, SASS, compiler caches, raw logs, task/model/request/
response/patch content, credentials, environment dumps, process IDs, and
container IDs.

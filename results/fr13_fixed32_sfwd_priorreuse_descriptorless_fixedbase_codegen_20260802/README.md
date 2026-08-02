# Descriptorless SFWD fixed-base codegen

Status: **offline SM121a codegen improves the descriptorless candidate while
retaining its register and global-memory counts; live correctness and timing
remain unqualified**.

This package audits source commit
`83b8eb0f697eb1e5c98470aa214e6b31317d9e8d` at B1 and B4 with 32
physical rows per request, row group 32, `C=10240`, `BLOCK_C=64`, width 4,
state length 34, eight warps, and three stages. CUDA visibility was explicitly
empty. No GPU kernel, Docker service, SWE task, request, timing run, or
acceptance run was launched.

## Change

The candidate combines the pushed descriptorless fixed-topology kernel with
the source-bound padded `x` stride of 16,384, dense `[4,1]` weight strides,
and explicit fixed-layout address terms. `x` and `out` use per-request base
pointers. Source stage uses a scalar per-request element offset; making all
three batch bases pointers raised reported registers to 44 and was rejected.

The selected formulation keeps:

- all 96 non-final source descriptor loads absent
- exact ordered BF16 products and FP32 adds
- reported/allocated registers at 40/40 per thread
- LDG/STG at 37/20 per CTA
- int64 conv-state bank-row addressing with runtime state strides

The host contract requires `x` strides `[16384,1]`, dense output/source-stage
strides `[10240,1]`, and dense weight strides `[4,1]`, with exact B1-B4 shapes.
A future launcher must call `fixed32_specialized_layout_contract` before
integrating this kernel.

The layout premise is hash-bound to the prior reduced real-B1 artifact at
commit `b38bb5b3af1ea2f1e9479e299a7d06272d1d5cce`. It is layout evidence
only and does not qualify correctness for this source.

## Result

| Metric | Descriptorless parent | Fixed base | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Reported / allocated registers/thread | 40 / 40 | 40 / 40 | 0 / 0 |
| Allocated registers/CTA | 10,240 | 10,240 | 0 |
| Register-budget CTAs / warps per SM | 6 / 48 | 6 / 48 | 0 / 0 |
| Static / encoded SASS | 885 / 896 | 718 / 736 | -167 / -160 |
| Warp-weighted static / encoded SASS | 7,080 / 7,168 | 5,744 / 5,888 | -1,336 / -1,280 |
| IMAD / UIMAD | 262 / 7 | 152 / 11 | -110 / +4 |
| IADD / IADD3 | 57 / 97 | 36 / 64 | -21 / -33 |
| LDG / STG / LDS / STS | 37 / 20 / 0 / 0 | 37 / 20 / 0 / 0 | 0 / 0 / 0 / 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 60,720 | 51,184 | -9,536 |

Static SASS falls 18.87%, encoded SASS falls 17.86%, and IMAD falls 41.98%.
Combined IMAD plus UIMAD falls by 106 instructions, or 39.41%. These are
static per-CTA counts, not measured issue rates or latency.

Two separate fresh-cache builds from the exact source commit reproduced B1
and B4. Cubin, PTX, SASS, resource reports, and non-launch metrics are
byte-identical across B1/B4 and between builds. The verifier independently
re-disassembled the cubins, recounted SASS and address-operation classes,
checked target/thread metadata, and enforced no register, LDG, STG, spill,
stack, local-memory, shared-memory, or call regression.

The descriptorless, int32-address, and state-fusion suites passed 28 tests.
They cover all fixed topology edges and source rows, exact ordered convolution
math, B1/B4 offset bounds, every specialized layout predicate, batch-base
equivalence, absent descriptor arguments, and int64 conv-state addressing.

## Scope

This module remains offline-only with no serving launcher or selector. A
source-bound real B1 reference-served byte gate is required before timing, and
exact4 B4 remains required for acceptance. The package excludes cubin, PTX,
SASS, IR, model/task content, requests, responses, patches, environment dumps,
credentials, process IDs, container IDs, and raw logs.

## Reproduction

Run from the repository root with two new cache/output roots:

```bash
ART=results/fr13_fixed32_sfwd_priorreuse_descriptorless_fixedbase_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
REV=83b8eb0f697eb1e5c98470aa214e6b31317d9e8d

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/tmp/fr13_descbase_primary_cache \
  "$PY" "$ART/offline_codegen_audit.py" --repo . --revision "$REV" \
  --canonical-path src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py \
  --output /tmp/fr13_descbase_primary --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/tmp/fr13_descbase_rebuild_cache \
  "$PY" "$ART/offline_codegen_audit.py" --repo . --revision "$REV" \
  --canonical-path src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py \
  --output /tmp/fr13_descbase_rebuild --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary /tmp/fr13_descbase_primary --rebuild /tmp/fr13_descbase_rebuild
```

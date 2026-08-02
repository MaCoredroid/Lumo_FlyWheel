# SFWD prior-reuse fixed-stride codegen

Status: **offline SM121a codegen improves the int32-address candidate;
current-source live correctness and runtime performance remain unqualified**.

This package audits source commit
`6ab0a4882e7c34d6e41ca88727094609932acc9e` at the fixed B1/B4
specialization: 32 physical rows per request, row group 32, 10,240 channels,
`BLOCK_C=64`, width 4, state length 34, eight warps, and three stages. CUDA
visibility was explicitly empty. No GPU kernel, Docker service, SWE task,
request, timing run, or acceptance run was launched.

## Change

The kernel keeps the int32 source descriptor, C64 schedule, prior-vector reuse,
and exact ordered BF16-product/FP32-add convolution. It removes runtime `x`
and weight strides only after binding their fixed layouts, and gives `x`,
`out`, and source stage explicit per-request base pointers. Conv-state bank-row
addressing remains int64 with runtime strides.

The host contract requires:

- `x`: shape `[B*32,10240]`, padded strides `[16384,1]`
- `out`: shape `[B*32,10240]`, dense strides `[10240,1]`
- source stage: shape `[B*36,10240]`, dense strides `[10240,1]`
- weights: shape `[10240,4]`, dense strides `[4,1]`
- int32 source descriptor: shape `[96]`, stride `[1]`

These predicates are implemented in `fixed32_specialized_layout_contract` and
must be called by a future launcher before this kernel is integrated.

The fixed padded `x` stride is bound to prior reduced real-B1 evidence at
artifact commit `b38bb5b3af1ea2f1e9479e299a7d06272d1d5cce`. That artifact
observed `x=[32,10240]` with strides `[16384,1]`, dense output/source-stage
strides `[10240,1]`, and dense weight strides `[4,1]`. This is layout evidence
only, not correctness evidence for the current source.

## Result

| Metric | Int32 address baseline | Fixed stride | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 0 / 0 |
| Reported registers/thread | 56 | 42 | -14 |
| Allocated registers/thread | 56 | 48 | -8 |
| Allocated registers/CTA | 14,336 | 12,288 | -2,048 |
| Static / encoded SASS | 912 / 928 | 778 / 792 | -134 / -136 |
| Warp-weighted static / encoded SASS | 7,296 / 7,424 | 6,224 / 6,336 | -1,072 / -1,088 |
| LDG / STG / LDS / STS | 64 / 20 / 0 / 0 | 64 / 20 / 0 / 0 | 0 / 0 / 0 / 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 64,040 | 56,968 | -7,072 |

Static and encoded SASS fall 14.69% and 14.66%; cubin size falls 11.04%.
Under the project SM121 65,536-register ledger, register-limited residency
moves from four to five CTAs per SM, or 32 to 40 warps. This is a register-only
proxy, not measured occupancy or latency.

Two separate fresh-cache builds from the exact source commit compiled B1 and
B4. Their cubin, PTX, SASS, resource report, and non-launch metrics are
byte-identical, and both complete output trees reproduce across the two builds.
The independent verifier re-disassembled the cubins, recounted instruction
classes, checked target/thread metadata, enforced no stack/local/spills/calls,
and enforced strict improvements over the int32-address baseline without LDG
or STG regression.

The focused state-fusion/descriptor/layout suite passed 22 tests. It covers
the source descriptor, exact ordered convolution math, B1/B4 signed-int32
offset bounds, every specialized layout predicate, batch-base offset
equivalence, and retention of int64 conv-state bank addressing.

## Rejections

The logical-dense `x` stride `[10240,1]` prototype is invalid and rejected: it
conflicts with the source-bound real-B1 layout `[16384,1]`. A constant
state-index stride prototype was also rejected because static SASS rose by one
instruction without a register, memory-operation, or encoded-SASS gain.
Additional state/descriptor/weight base aliases produced identical SASS and a
32-byte larger cubin debug payload, so they were not retained.

## Scope

This module is offline-only and has no launcher or production selector. A
source-bound real B1 reference-served byte gate is required before timing, and
exact4 B4 remains required for acceptance. This package contains only source,
helper hashes, and derived summaries. Cubin, PTX, SASS, IR, model/task content,
requests, responses, patches, environment dumps, credentials, process IDs,
container IDs, and raw logs are excluded.

## Reproduction

Run both commands from the repository root with two new cache/output roots:

```bash
ART=results/fr13_fixed32_sfwd_priorreuse_fixed_stride_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
REV=6ab0a4882e7c34d6e41ca88727094609932acc9e

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/tmp/fr13_stride_primary_cache \
  "$PY" "$ART/offline_codegen_audit.py" --repo . --revision "$REV" \
  --canonical-path src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_i32_descriptor.py \
  --output /tmp/fr13_stride_primary --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/tmp/fr13_stride_rebuild_cache \
  "$PY" "$ART/offline_codegen_audit.py" --repo . --revision "$REV" \
  --canonical-path src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_i32_descriptor.py \
  --output /tmp/fr13_stride_rebuild --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary /tmp/fr13_stride_primary --rebuild /tmp/fr13_stride_rebuild
```

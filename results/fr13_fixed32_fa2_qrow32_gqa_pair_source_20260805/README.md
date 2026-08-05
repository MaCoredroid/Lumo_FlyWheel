# Fixed32 B4 FA2 qrow32 GQA-pair source candidate

Status: **static source pass; SM121a codegen and real B4 byte gates pending**.

This artifact records a default-off source candidate for physical32 B4 tree
attention. It is not a performance result and does not authorize a timing
arm. No GPU, Docker container, CUDA cache, synthetic timing probe, or real
task was touched while preparing it.

## Audit finding

The currently staged B4 qrow32 kernel uses BM32/BN64, two warps, and one CTA
per `(batch, query_head)`: `6 x 4 x 4 = 96` CTAs per layer. It has no split-K
or combine launch. Its latest tracked host result is already clean at 252
registers with zero stack, local memory, spills, `LDL`, `STL`, or `CALL`, but
it has no canonical real SWE-Verified exact4 B4 byte pass or timing result.

Splitting the K dimension is therefore not the grounded next lever. The
remaining structural duplication is GQA: each KV head serves six query heads,
and six independent CTAs stage the same K/V tiles for each `(batch, kv_head)`.
The source candidate maps two adjacent query heads to one standard BM64,
four-warp CTA:

- grid: `3 head pairs x B4 x 4 KV heads = 48` CTAs per layer;
- logical M rows `0..31`: first query head in the pair;
- logical M rows `32..63`: second query head in the pair;
- K/V scans per `(batch, kv_head)`: six to three;
- total query rows, attention arithmetic, launched threads, and launched
  warps per layer: unchanged relative to qrow32;
- split-K launches: zero; combine launches: zero.

This repurposes the second 32-row half that stock B4 BM64 masks as outside the
per-sequence query extent. It does not add a custom attention algorithm.

## Exactness design

The query and output tensors use a hierarchical M layout
`((query_row_32, head_in_pair_2), head_dim_256)`. LSE uses
`(query_row_32, head_in_pair_2)` with head stride `total_q`. The layout keeps
runtime row strides, including the observed fused-QKV query row stride of
8192, and only requires the canonical head stride of 256.

Each head keeps its own two-warp row assignment and ordered QK, softmax, and
PV accumulation. Tree-bias rows map with `logical_row % 32`. The gate excludes
dropout, ALiBi, local/causal windows, appended KV, split-K, cache-batch remaps,
and padded LSE layouts, so every remaining head-dependent address is either
the shared KV head or explicitly mapped through the pair layout.

CPU tests exhaust all 96 `(batch, query_head)` outputs, compare Q/O/LSE scalar
and grouped addresses, verify warp-local row order, and check the private
48-CTA launch. A fresh archive of pinned FA2 commit
`29210221863736a08f71a866459e368ad1ac4a95` was patched twice: the first pass
changed only the expected source set and the second pass reported every file
unchanged.

## Admission boundary

The source is intentionally build-only and has no live or production
selector. Before any timing claim, it still requires:

1. Fresh CUDA 13 SM121a compile, target-symbol resource audit, and ABI/ELF
   parity. The 96 KiB/four-warp kernel must have zero stack/local/spills and no
   SASS `LDL`, `STL`, or `CALL`.
2. Same-EngineCore retained-operand raw-byte comparison on the canonical real
   SWE-Verified exact4 B4 set for both Tail23 and Hydra27, covering all 16
   tree-attention layers and BF16 output plus FP32 LSE bytes.
3. Only after both byte gates pass, a clean exact4 B4 timing pair with full
   step wall TPS and the existing breakdown contract.

The deterministic work reduction in `work_model.tsv` is not a measured DRAM
reduction or speedup. L2 reuse, register allocation, shared-memory residency,
and scheduling can only be resolved by the pending compile and real B4 gates.

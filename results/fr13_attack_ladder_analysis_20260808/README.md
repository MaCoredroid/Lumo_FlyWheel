# FR13 fixed32 B1 — kernel attack ladder from the post-Qrow Nsight capture

Offline, read-only analysis over banked evidence. **No GPU was touched. No
serving code was changed.** This is a design document, not a performance
result and not an acceptance run.

- Source capture: `output/fr13_fixed32_b1_nsys_20260808T212056Z/tail6_fixed32_b1_nsys_f32_20260808T212056Z/logs/fr13_fixed32_b1_real_swe.nsys-rep`
  (218,692,330 B, sha256 `241d4541f5c4767a649fe49968a4af2991346156bf073a71fae6752980f45c48`)
- Curated reduction: `results/fr13_fixed32_b1_nsys_attribution_20260808T212056Z/`
- Queried through the co-located `fr13_fixed32_b1_real_swe.sqlite` export
  (`/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys export`), 4,324,314 GPU
  kernel rows, 4,105,181 CUDA runtime rows, 1146 complete
  `fr13.fixed32.step` NVTX instances.
- Hardware: GB10, 48 SMs, 273 GB/s LPDDR5X (unified CPU/GPU).
- Constraint honoured throughout: **exact-math**. Target-GEMM output bytes must
  be byte-identical, so only scheduler / launch-policy / persistence changes on
  stock CUTLASS arithmetic are admissible. Tile geometry and split-K are dead
  (proven byte drift), and every lever below is labelled legal, blocked, or dead.

`attribution_only=true`, `acceptance_valid=false`. The capture is a profiling
window. Absolute host-side numbers carry profiler inflation (see *Caveats*).

---

## 0. The headline

**The target GEMM has no legal scheduler lever.** It is 48.4% of the step
envelope, and every mechanism the campaign has been holding in reserve —
persistent/megakernel scheduling, CUDA-graph node fusion, stream ordering, L2
policy — is worth a combined **0.009 ms/step** against it, because:

- SFWD is **already one CUDA graph** (`graphId=812`, 1 replay/step, 1894 kernel
  nodes). All 293,549 target-GEMM instances in SFWD are graph nodes.
- The **total** inter-node gap across all 1894 SFWD nodes is **0.404 ms/step**
  (0.26% of the phase, 213 ns/node). The 256 GEMM instances account for
  **9.17 µs/step** of it (35.8 ns mean gap).
- There is **no wave-quantization penalty**. The 40-CTA shapes, which leave 8 of
  48 SMs idle for the whole kernel, achieve **214.7–220.6 GB/s** — equal to or
  better than the 272-CTA shapes at 216.2 GB/s. The kernel is limited by the
  memory controller, not by SM fill, so under-filling the machine costs nothing.
- At the 5th percentile the GEMM already runs at **85.8% of the 273 GB/s
  roofline**. That is the practical ceiling for a streaming LPDDR5X read.

The recoverable time is elsewhere and is mostly **host-side**: 15.81 ms/step of
GPU idle inside the step envelope, of which **10.08 ms sits in a single
post-DFWD tail** where the GPU does nothing while Python finishes the step.

Reality check: the campaign floor is `119.658 ms/step` with an acceptance cap of
`137.607 ms/step`. The full legal ladder below totals a **19.4 ms/step ceiling**.
Applying all of it lands near 213 ms/step — still ~1.55x the cap. **This ladder
cannot close the acceptance gap.** Only arithmetic changes could, and those are
barred by the exact-math rule. That conclusion is the deliverable.

---

## 1. The attack ladder

Modelled ms/step against the 237.248 ms/step envelope.

| # | Lever | Modelled ms/step | % env | Verdict | Risk / dependency |
|---:|---|---:|---:|---|---|
| 1 | Move the post-DFWD host tail off the step critical path | **10.08** ceiling, 5–7 realistic | 4.2% | **LEGAL** | Threading/ordering only; zero GPU arithmetic. Needs the 2.3 `cudaStreamSynchronize`/step off the submit path. |
| 2 | GDN two-level scan → fused single launch | **3.45** | 1.45% | **LEGAL** | Codegen change; byte gate + `FR13_SUBTREE_PARALLEL_SELFCHECK` already exist. Requires per-node ready flags, *not* a grid barrier. |
| 3 | CUDA-graph capture of CFWD's 1111 eager ops | **2.99** | 1.26% | **LEGAL** | Low. CFWD already has `graphId=809` (78 nodes); extend coverage. |
| 4 | CFWD→DFWD boundary bubble | **1.22** | 0.52% | **LEGAL** | Medium. Likely a sampled-token dependency; needs the handoff kept on-device. |
| 5 | Complete DFWD graph capture (105 eager ops) | **1.16** | 0.49% | **LEGAL** | Low. `graphId=815` already covers 176 of 299 ops. |
| 6 | SFWD intra-graph gap | 0.52 | 0.22% | **AT FLOOR** | Already 213 ns/node — the graph-replay floor. |
| 7 | Target GEMM persistent / megakernel / graph-node fusion | **0.009** | 0.004% | **DEAD** | 256 GEMM gaps = 9.17 µs/step. |
| 8 | Target GEMM wave / tile quantization | **≈0** | 0% | **DEAD** | 40-CTA shapes match or beat 272-CTA shapes on GB/s. |
| 9 | Target GEMM L2 policy | **0** | 0% | **DEAD** | 24.83 GB streamed/step, zero intra-step reuse. |
| 10 | Target GEMM stream ordering / concurrency | **0** | 0% | **DEAD** | Single stream; strict layer dependency chain. |
| 11 | FA2 persistent across layers | **0.0016** | 0% | **DEAD** | 16 gaps × 102 ns. |
| 12 | FA2 KV L2-persistence window (`cudaAccessPolicyWindow`) | unquantified, ≤17.0 | ≤7.2% | **LEGAL, speculative** | Launch policy only. Needs an L2-metrics run; this trace has no metric sampling. |
| — | Target GEMM bandwidth shortfall | 15.73 (+4.26 jitter) | 8.4% | **NOT ADDRESSABLE** | 85.8% of roofline at p5; 82.4% at mean. |
| — | FA2 tile geometry (kBlockM 16→32, GQA-pair) | ≤17.0 | ≤7.2% | **BLOCKED** | Exact-math. B1 byte gates pending. |

**Legal, evidence-backed total: 19.4 ms/step ceiling (8.2%); 10–14 ms/step
realistic (4–6%). None of it comes from the target GEMM.**

---

## 2. Target GEMM (Q1) — 114.812 ms/step, 256.2 inst/step, 48.4% of envelope

### 2.1 Launch geometry and occupancy

`void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<bfloat16_t,
128, 1, 128, tuple<C<128>,C<32>,C<128>>, tuple<C<1>,C<1>,C<1>>,
EpilogueScheduleAuto, KernelTmaWarpSpecializedBlockwiseCooperativeSm120,
true>>::GemmKernel>`

| property | value |
|---|---|
| block | 384, 1, 1 (3 warpgroups — cooperative) |
| registers/thread | 168 |
| dynamic shared memory | 93,184 B (91 KB) |
| cluster | (1, 1, 1) |
| grid (decode) | (1, N_tiles, 1) — CTA count **= gridY** |
| **CTAs resident per SM** | **1** (91 KB of ≤100 KB SM shared memory) |
| launchType | REGULAR |
| CUDA graph | **`graphId=812`, all 293,549 SFWD instances are graph nodes** |
| stream | 7 (the only stream in the trace) |

### 2.2 Per-instance distribution — six distinct shapes

The 256.0 instances/step resolve into exactly six populations. `gridY=40` is
trimodal; splitting it on duration gives 16 / 48 / 64 per step, which pins the
model as **64 layers = 48 GDN + 16 attention** (corroborated independently by
`results/fr13_fixed32_sfwd_conv_postprep_fusion_20260803` — "48 layers").

| projection | CTAs | inst/step | p5 µs | p50 µs | mean µs | ms/step | N × K | bytes/inst | mean GB/s |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| MLP gate_up | 272 | 64 | 796.4 | 815.5 | 824.4 | 52.748 | 34816 × 5120 | 178.26 MB | 216.2 |
| MLP down | 40 | 64 | 396.6 | 404.8 | 410.9 | 26.312 | 5120 × 17408 | 89.13 MB | 216.9 |
| GDN in_proj | 128 | 48 | 371.5 | 384.2 | 389.2 | 18.675 | 16384 × 5120 | 83.89 MB | 215.5 |
| GDN o_proj | 40 | 48 | 183.7 | 187.7 | 190.1 | 9.230 | 5120 × 8192 | 41.94 MB | 220.6 |
| Attn qkv+gate | 112 | 16 | 329.2 | 338.4 | 343.3 | 5.492 | 14336 × 5120 | 73.40 MB | 213.8 |
| Attn o_proj | 40 | 16 | 135.6 | 146.5 | 146.5 | 2.254 | 5120 × 6144 | 31.46 MB | 214.7 |
| **total** | | **256.0** | | | | **114.712** | | **24.830 GB** | **216.4** |

Shape derivation (each step independently checked):

1. CTA count = `gridY`; each CTA streams `TileM × K` weight bytes.
2. `TileM = 128` is the **only** value consistent with the 273 GB/s ceiling —
   `TileM=64` implies 54 GB/s (absurdly low), `TileM=256` implies 865 GB/s
   (impossible). It also matches the 384-thread cooperative 2-math-warpgroup
   config.
3. ⇒ `H = 40 × 128 = 5120`, `I = 272 × 128 / 2 = 17408`.
4. **Check A**: gate_up/down byte ratio is exactly 2:1; measured duration ratio
   is `824.4 / 410.9 = 2.006`.
5. **Check B**: `vocab_size = 248320` (run evidence) × H=5120 × 2 B = 2.543 GB
   for the bf16 LM head; the POSTPROCESS phase is one `nvjet_sm121_tst_mma_128x208x64`
   at 12.348 ms/step ⇒ 206 GB/s. Consistent, and confirms H=5120.
6. **Check C**: attention `o_proj` measured 146.5 µs ⇒ 31.46 MB ⇒ K = **6144
   exactly** = 24 q heads × 256 head-dim, which is fixed independently by the FA2
   grid and traits.
7. **Check D**: total 24.830 GB/step vs the banked weights-only floor of
   89.3 ms/step ⇒ 24.38 GB. Agreement to 1.8%.

### 2.3 Decomposition of the ~25.5 ms excess over the 89.3 ms floor

| component | ms/step | how measured |
|---|---:|---|
| launch / gap overhead between instances | **0.009** | 256 gaps/step, mean 35.8 ns, max 10.3 µs |
| tail / wave quantization (M=32 rows, 48 SMs) | **≈ 0** | see 2.4 |
| traffic the weights-only floor omits | **5.42** | fp8 blockwise scales (+3.125% = 0.776 GB) + bf16 epilogue writes (0.251 GB) ⇒ all-traffic floor 94.72 ms |
| per-instance jitter above class p5 | **4.26** | 114.712 measured − 110.451 at class p5 |
| steady-state DRAM efficiency gap | **15.73** | 110.451 at p5 vs 94.72 all-traffic floor ⇒ 85.8% of roofline |
| **total** | **25.41** | vs 25.5 stated excess |

Jitter is **not** a scheduling artefact: per-instance σ is 29.4 µs while the
per-step-mean σ is only 6.07 µs, and there is no drift with position inside the
replay (815.7 µs at node 0–200 vs 820.5 µs at node 1800+). It is DRAM
refresh / page-conflict noise, plus CPU contention for the shared LPDDR5X.

### 2.4 Why there is no wave quantization

M=32 rows against `TileM=128` wastes 75% of the M tile, but in a weight-bound
regime the A operand is ≤340 KB and the weight bytes are unchanged, so M-tile
padding is **free**. The real candidate was N-tile wave quantization across
48 SMs:

| shape | CTAs | waves | last-wave fill | achieved GB/s |
|---|---:|---:|---:|---:|
| GDN o_proj / attn o_proj / MLP down | 40 | 0.83 | 40/48 = 83% | 214.7 – 220.6 |
| attn qkv | 112 | 2.33 | 16/48 = 33% | 213.8 |
| GDN in_proj | 128 | 2.67 | 32/48 = 67% | 215.5 |
| MLP gate_up | 272 | 5.67 | 32/48 = 67% | 216.2 |

The worst-filled shapes are the **fastest** per byte. Achieved bandwidth is flat
at 214–221 GB/s across a 6.8x range in CTA count and a 0.83–5.67 range in wave
count. Perfect wave packing would therefore return **0 ms**. This is the single
most load-bearing negative result in the document.

### 2.5 Lever verdicts

| lever | modelled ms | why |
|---|---:|---|
| persistent / megakernel (fewer launches) | 0.009 | Total GEMM gap is 9.17 µs/step. A megakernel cannot recover time that is not being spent. |
| CUDA-graph node fusion | ≤0.009 | Already fully graph-captured; node gap is 213 ns and the GEMM's share is 35.8 ns. |
| stream ordering / concurrency | 0 | One stream, and the layer chain (in_proj → attention/GDN → o_proj → MLP) admits no legal concurrent partner. The one latency-bound kernel that *would* pair well (FA2) is the GEMM's own consumer. |
| L2 policy | 0 | 24.83 GB streamed with zero intra-step reuse; the only reusable operand is already L2-resident. |

---

## 3. GDN (Q2) — 12.357 ms/step, 96.1 inst/step

### 3.1 The level structure is visible in the trace

`_tree_gdn_path_kernel`, launched as
`[(num_vh, cdiv(dim_v, block_v), n_paths)]` (`fr10_gdn_tree_kernel.py:16268`):

| level | grid | n_paths | regs | inst/step | min µs | mean µs | ms/step |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 (heavy path from root) | (48, 16, **1**) | 1 | 48 | 48.0 | 74.98 | 83.19 | 3.992 |
| 1 (export-rooted terminals) | (48, 16, **11**) | 11 | 40 | 48.0 | 167.58 | 174.10 | 8.355 |
| | | | | **96.0** | | | **12.347** |

This confirms the `fr10_gdn_tree_kernel.py` analysis exactly: one h0-rooted path
whose **five** nodes are all handoff parents (level 0 exports 5 states), then
**eleven** export-rooted terminal paths (level 1 re-reads them). The source's own
ledger agrees — `logical_launches: 2`, `logical_programs: 12`,
`logical_padded_slots: 82`, `logical_critical_path: 12`.
48 GDN layers × 2 launches = 96 instances/step.

### 3.2 The fusion win is overlap, not launch count

The measured L0→L1 gap is **−0.128 µs** — the two launches are already
back-to-back inside graph 812. Saving one launch is worth ~0.

The win is that level 0 **starves the machine**. Normalising by padded work
units (768 CTA-groups × slots):

| level | CTA-groups | slots | work units | duration | ns / unit |
|---|---:|---:|---:|---:|---:|
| 0 | 768 | 5 | 3,840 | 83.19 µs | **21.66** |
| 1 | 768 | 77 | 59,136 | 174.10 µs | **2.944** |

Level 0 is **7.36x less efficient per unit of work**: it is a 5-deep serial
chain over only 768 CTAs (16 CTAs/SM), latency-bound, and it holds the whole
GPU while doing 6% of the work.

**Fused model.** All 82 padded slots × 768 groups = 62,976 units at level 1's
demonstrated 2.944 ns/unit:

```
62,976 × 2.944 ns = 185.4 µs/layer  ×  48 layers  =  8.90 ms/step
```

This **independently reproduces the banked 8.9 ms prior model** from measured
durations alone. Win:

```
12.347 − 8.900 = 3.45 ms/step   (1.45% of envelope)
```

Plus, off-model, the elimination of the fp32 HBM state export/re-read (5 written
+ 11 read per layer).

**Necessary condition.** The fused kernel must let a level-1 path launch as soon
as its level-0 parent node retires (per-node ready flag). A grid-wide barrier
between the levels re-serialises the schedule and returns only the ~0 µs launch
gap. That distinction is the whole 3.45 ms.

---

## 4. FA2 qrow16 (Q3) — 21.369 ms/step, 16 instances

### 4.1 Geometry — a perfect wave with almost no threads in it

| property | value |
|---|---|
| kernel | `flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 16, 64, 1>, ...>` |
| grid | (2, 1, 24) = **48 CTAs = exactly one wave on 48 SMs** |
| block | 32, 1, 1 (`kNWarps=1` — one warp) |
| dynamic shared memory | 73,728 B ⇒ 1 CTA/SM |
| **resident threads per SM** | **32** |
| registers/thread | 255 |
| inst/step | 16.0 (one per attention layer) |

`num_m_block = ceil(32/16) = 2`, `b = 1`, `h = 24` query heads, head-dim 256.

### 4.2 Launch/gap is nil

Mean gap to the preceding kernel: **101.9 ns**; max 576 ns. Across 16 instances
that is **1.6 µs/step**. Persistence across layers recovers 0.0016 ms.
**Dead as a launch lever.**

### 4.3 The KV-scan floor — FA2 is 4.93x above it

Duration distribution: min 1000.5 / p5 1037.8 / p50 1272.8 / mean 1334.8 /
p95 1852.7 / max 2437.4 µs. The spread is context growth. OLS over all 1147
steps:

```
t(step) = 1090.8 µs + 0.4260 µs/step
```

KV bytes per token per layer, all three factors pinned by evidence:

- 4 KV heads — `results/fr13_fixed32_fa2_qrow32_gqa_pair_source_20260805`:
  *"each KV head serves six query heads"*, and `6 x 4 x 4 = 96` CTAs for B4;
  24 q heads / 6 = **4**.
- head-dim 256 — FA2 kernel traits.
- bf16 — `vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16,
  Fp8KVCacheDataType(0)>`.

```
KV bytes/token/layer = 2 (K,V) × 4 heads × 256 × 2 B = 4096 B
```

Committed tokens per step = **5.753885** (`results/fr13_hardware_floor_status_20260805`).

| quantity | value |
|---|---:|
| marginal traffic per step | 5.754 × 4096 = 23,568 B |
| marginal time per step | 0.4260 µs |
| **achieved unique-byte throughput** | **55.3 GB/s = 20.3% of 273** |
| implied context at mid-capture | ≈ 18.0k tokens |
| **per-layer KV-scan floor at 273 GB/s** | **270.5 µs** |
| measured mean | 1334.8 µs |
| **ratio to floor** | **4.93x** |
| FA2 floor | 4.33 ms/step |
| FA2 measured | 21.36 ms/step |
| **structural headroom** | **17.03 ms/step** |

**FA2 is not near its floor — it is nearly 5x above it.** The cause is
structural and is already documented in this repo's own audit: 24 q-head CTAs ×
2 m_blocks = 48 CTAs stage the same **4** KV heads, so **each KV byte is staged
12 times**. Bracketing:

- perfect L2 dedupe ⇒ 273 GB/s of unique bytes
- zero L2 dedupe ⇒ 273/12 = 22.75 GB/s of unique bytes
- **measured 55.3 GB/s** — 2.43x above the no-dedupe bound, so L2 is already
  absorbing roughly half the redundancy.

### 4.4 What is legal

| lever | verdict |
|---|---|
| persistent across layers | **DEAD** — 1.6 µs/step of gap exists to recover. |
| KV L2-persistence window (`cudaAccessPolicyWindow`) | **LEGAL, unquantified.** Pure launch policy, no arithmetic. It attacks the one thing that is actually costing time (the 12x re-staging). This trace has no metric sampling, so the win cannot be modelled here — it needs an L2 hit-rate run. |
| `kBlockM` 16→32 (fold the 2 m_blocks, halve staging) | **BLOCKED** — tile geometry, byte drift. |
| GQA-pair head mapping | **BLOCKED, but the closest thing to a live candidate.** `fr13_fixed32_fa2_qrow32_gqa_pair_source_20260805` cuts K/V scans per `(batch, kv_head)` from six to three and its own artifact claims *"total query rows, attention arithmetic, launched threads, and launched warps per layer: unchanged"*. If that invariance claim survives a B1 byte gate it becomes legal and is worth a large fraction of the 17.03 ms. Its B1 byte gate is pending; until then it is blocked, not banked. |

---

## 5. Host (Q4) — postprocess 12.35 + residual 13.11 + CFWD 20.70

### 5.1 Per-step GPU ledger

NVTX host ranges projected onto GPU ops via `correlationId`
(`CUPTI_ACTIVITY_KIND_RUNTIME` → `CUPTI_ACTIVITY_KIND_KERNEL`), 1146 complete steps.

| range | GPU span ms | GPU busy ms | **GPU idle ms** | GPU ops |
|---|---:|---:|---:|---:|
| **step** | **237.032** | **221.222** | **15.810** | 3690.6 |
| sfwd | 155.807 | 155.292 | 0.515 | 1894.5 |
| postprocess | 12.339 | 12.339 | **0.000** | 2.0 |
| cfwd | 20.686 | 17.439 | 3.247 | 1189.0 |
| dfwd | 35.107 | 33.887 | 1.220 | 299.0 |

The GPU is **93.3% busy** inside the step envelope. Phase order is
**sfwd → postprocess → cfwd → dfwd**, then an inter-step tail.

### 5.2 Where the 13.114 ms residual lives

| boundary | ms/step |
|---|---:|
| step_start → sfwd | 0.000 |
| sfwd → postprocess | 0.004 |
| postprocess → cfwd | 0.003 |
| cfwd → dfwd | **1.224** |
| **dfwd → step_end (tail)** | **11.969** |

The residual is essentially one object: an **11.969 ms post-DFWD tail**, plus the
1.224 ms CFWD→DFWD bubble. Inside the tail:

- GPU busy: **1.889 ms/step** of un-phased work — `index_elementwise_kernel`
  16.1/step (1.270 ms), `_zero_kv_blocks_kernel` (0.466 ms/step amortised),
  `_compute_slot_mapping_kernel` 4.0/step.
- **GPU idle: 10.080 ms/step.**
- Host CUDA API time in the same window: only **2.30 ms/step** total
  (`cudaStreamSynchronize` 2.3 calls = 1.685 ms; `cudaLaunchKernel` 92.8 calls =
  0.319 ms; `cudaMemcpyAsync` 63.1 calls = 0.219 ms).

⇒ **≈7.8 ms/step of the tail is host CPU outside any CUDA call** — Python and
framework: sampler output handling, detokenisation, scheduler, request
bookkeeping. This is the single largest addressable block in the trace.

### 5.3 Top host-side consumers

One CUDA thread (globalTid `281480429306181`) makes 4,103,993 API calls totalling
273,771 ms of CPU across a 300 s capture — **91% of wall-clock inside CUDA APIs**.

| API | calls/step | CPU ms/step | mean µs |
|---|---:|---:|---:|
| `cudaLaunchKernel` | 1538.9 | 173.961 | 113.05 |
| `cudaMemcpyAsync` | 143.9 | 56.205 | 390.48 |
| `cuLaunchKernelEx` | 44.8 | 2.971 | 66.36 |
| `cudaGraphLaunch` | 5.0 | 2.554 | 507.0 |
| `cudaStreamSynchronize` | 7.6 | 1.815 | 238.92 |
| `cuLaunchKernel` | 15.7 | 0.896 | 57.2 |
| `cudaMemsetAsync` | 69.4 | 0.245 | 3.53 |

The 113 µs mean `cudaLaunchKernel` is **blocking on a full launch queue, not CPU
burn** — the GPU is 93.3% busy, so the host is a passenger for most of it. The
exception is the post-DFWD window analysed above, where the queue is empty and
the host genuinely is the critical path.

Memcpy census: D2D 361.8/step for 180.5 MB/step (1.13 ms GPU); H2D 48.8/step ×
3551 B (launch metadata); D2H **5.2/step × 140 B** — the sampled-token readback,
which is what the 7.6 stream syncs/step are waiting on.

CUDA graph inventory (5 graphs, 5773 replays/300 s = 5.04/step):

| graphId | replays | nodes/replay | GPU ms/step | phase |
|---:|---:|---:|---:|---|
| 812 | 1147 | 1894 | 155.4 | **sfwd (entire phase)** |
| 815 | 1188 | 176 | 27.4 | dfwd |
| 809 | 1146 | 78 | 4.4 | cfwd |
| 806 | 1146 | 9 | 1.4 | dfwd |
| 803 | 1146 | 9 | 0.9 | dfwd |

CFWD runs **1111 eager ops/step** against 78 graph nodes; DFWD runs 105 eager
against 194 graph nodes. That is the entire basis for ladder items 3 and 5:

| phase | eager ops/step | idle ms/step | idle per op | at the SFWD graph's 213 ns/node | **recoverable** |
|---|---:|---:|---:|---:|---:|
| cfwd | 1111 | 3.247 | 2.73 µs | 0.253 | **2.99** |
| dfwd | 105 | 1.220 | 4.08 µs | 0.064 | **1.16** |

CFWD's op profile shows why: 784 ops/step are **under 2 µs** and contribute only
0.989 ms of GPU time, while carrying ~2.1 ms of launch bubble between them.

---

## 6. First implementation step for lever #1 (design only)

Lever #1 is *"move the post-DFWD host tail off the step critical path"*:
10.08 ms/step of GPU idle, of which ~7.8 ms is host code outside CUDA.

**Step 0 — instrument, do not guess.** The current evidence localises the idle
but cannot split the ~7.8 ms of non-CUDA host time. Add four NVTX ranges inside
the post-DFWD region — `fr13.fixed32.sample_readback`, `.output_proc`,
`.sched_next`, `.kv_bookkeep` — and re-run the *existing* offline reduction
(`scripts/fr13_fixed32_nsys_reduce.py`) unchanged. No new measurement machinery,
no GPU semantics touched. This is the only step that should be taken before a
design is chosen.

**Step 1 — split submit from retire.** The step loop becomes two paths. The
*submit* path enqueues the next step's SFWD graph as soon as the draft tokens
are on device. Everything that only *reads* results — detokenisation, response
assembly, logging, request accounting — moves to a retire queue drained on a
second thread. Nothing in the retire path may be a precondition for the next
submit.

**Step 2 — delete the sync from the submit path.** The 5.2 D2H/step × 140 B
token readbacks are what the 2.3 `cudaStreamSynchronize`/step in the tail are
waiting on. Keep the accepted-token count on device and let the next step's
`_compute_slot_mapping_kernel` and block-table kernels consume it directly, so
the host never needs the *value* in order to enqueue.

**Step 3 — gate.** No GPU arithmetic is touched, so the change must be
byte-identical by construction. The existing exact4 byte gate is therefore both
sufficient and the acceptance criterion: any byte drift means the reordering
touched something it should not have.

**Falsification.** If, after step 1, GPU idle in the post-DFWD window does not
fall, the tail is a genuine data dependency rather than bookkeeping, and lever #1
dies. Re-run the same reduction and read `dfwd → step_end` off the same table.

**Dependency note.** Levers 3 and 5 (graph capture of CFWD/DFWD eager ops) are
independent of lever 1 and lower risk; they can proceed in parallel. Lever 4
(CFWD→DFWD bubble) shares step 2's on-device-handoff work and should follow it.
Lever 2 (GDN fusion) is fully independent of all host work.

---

## 7. Caveats

- **Attribution only.** `acceptance_valid=false`. These are profiled-run numbers.
- **Profiler inflation.** `cuKernelGetName` is called 1538.9 times/step — one per
  eager `cudaLaunchKernel` — and is CUPTI-injected. `PROFILER_OVERHEAD` totals
  38.8 ms/step but lives on a **separate flush thread** (996 records on
  globalTid `…6237`); the main CUDA thread carries 113 records at 3.1 µs. The
  post-DFWD tail contains only 92.8 launches/step, so its CUPTI share is
  ≈0.2 ms — the 10.08 ms idle figure is substantially genuine. Absolute
  host-side totals elsewhere are inflated and should not be quoted as
  acceptance numbers.
- **Shared-memory system.** GB10's 273 GB/s LPDDR5X is shared between CPU and
  GPU. Host memory traffic during SFWD is a plausible contributor to the 4.26 ms
  of GEMM jitter; this trace cannot separate it.
- **No DRAM counters.** The capture is `cuda,cuda-sw,nvtx` with no metric
  sampling, so all bandwidth figures are derived from durations plus a byte
  model. The byte model is cross-checked five independent ways (§2.2) and
  reproduces both the banked 89.3 ms GEMM floor (to 1.8%) and the banked 8.9 ms
  GDN fused model (exactly), but it is a model.
- **Model geometry is derived, not read from a config.** No model config is
  present in the tree. Every dimension in §2.2 is inferred from grid shapes,
  kernel template parameters, `vocab_size` in the run evidence, and measured
  durations, with the cross-checks listed.

---

## 8. Files

| file | what |
|---|---|
| `README.md` | this analysis |
| `attack_ladder.json` | machine-readable ladder + per-shape measurements |
| `measurement_tables.txt` | rendered tables, as produced by the queries |
| `queries.sql` | every SQL query used, runnable against the exported sqlite |
| `SHA256SUMS` | checksums for the above |

## 9. Reproduce

Offline, no GPU. Export the report once, then run `queries.sql`:

```
/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys export --type sqlite \
  --output fr13_fixed32_b1_real_swe.sqlite \
  <runroot>/<arm>/logs/fr13_fixed32_b1_real_swe.nsys-rep

sqlite3 "file:fr13_fixed32_b1_real_swe.sqlite?mode=ro" < queries.sql
```

The phase/idle tables in §5 additionally require the NVTX→GPU projection
described in `queries.sql` §H (host range → `correlationId` → kernel), which is
the same `first_to_last_projected_gpu_operation` basis the curated reduction uses.

# TreeAttention-v2 — design (FR14 kernel lane, B1 FA2 fork)

Status 2026-08-17. Author: TreeAttention-v2 kernel agent. Domain: the FA2 fork C++ tree
(`scripts/fr13_patch_fa2_tree_bias.py` TU/gate generators) and its build/gate scripts.

**Headline for the reader in a hurry.** The lever as briefed — *"32 rows share the same KV
prefix, so load each KV tile once per CTA and apply it to every row"* — is **already fully
realised inside a CTA** by the promoted `gqa_pair` kernel. What remains is re-staging
*across* CTAs: 3 CTAs still read each KV head at B1. Collapsing that means raising
heads-per-CTA from 2 to 6, and heads-per-CTA is the *same knob* as SM occupancy: CTAs =
24/G at B1. Every byte saved is bought with parallelism at a fixed exchange rate. This
document prices that exchange, proves the geometry is Tier-A legal, identifies the two
structural blockers that stop the stock traits at G=2, and — because the campaign rule is
measured-never-derived — **pre-registers the one measurement that decides whether the
kernel should be built at all**, using two kernels that already exist in a sealed binary.

**Two measured findings have already moved the recommendation.** (1) GB10 supports thread
block clusters and DSMEM *works* at cluster size 3 (functionally verified, §1b), so the
cluster variant — which cuts staged bytes 3× while keeping all 12 CTAs and the compute
floor, and touches no accumulation order at all — is now the **lead candidate**, ahead of
the head-merge geometry the lever was briefed as. (2) L2 is 24 MB against a 256 KB
in-flight tile working set, so today's 3× re-staging is probably already being absorbed by
L2 — which means the −13 ms was priced against DRAM bytes that are not actually being
moved. Read §8 before building anything.

---

## 1. Build-environment proof (deliverable 1) — source half PASS

Recompiling the current `gqa_pair` unit is the build-environment smoke test. The source
half reproduced exactly, twice:

| check | result |
|---|---|
| FA2 origin tree present, pristine | `29210221863736a08f71a866459e368ad1ac4a95` ✓ (pinned `FA2_HEAD`) |
| 55 reference objects + qualified `_vllm_fa2_C.abi3.so` at `REF` | present (56 `.o` incl. `flash_api`) |
| pinned build image `vllm/vllm-openai@sha256:3dbe092e…` | present locally |
| patcher → `validate-source --arm gqa_pair` closure | `172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4` |
| … vs the pin in the build script / `fr13_canonical_env.sh` | **EXACT MATCH** |
| regenerated tree vs the sealed 2026-08-10 build's `source/` | **byte-identical in all 6 closure files** (only `.git` index metadata differs) |

So the archived research workspace plus this repo regenerate, byte-for-byte, the source
that produced the sealed `.so 3560cdc0c1ebbe…` (299 815 552 B). Nothing is missing on the
source side.

Register headroom is a **measured** quantity, recovered from the sealed build's own
`cuobjdump_resource_usage.txt`:

```
Function _ZN5flash48fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernelENS_16Flash_fwd_paramsE:
  REG:243 STACK:0 SHARED:1024 LOCAL:0
```

**REG:243 against the TU's `__maxnreg__(254)` cap, zero stack, zero local.** §4 shows why
this number is invariant under the head-count change, which removes register spill from
the risk list — it is not an estimate, it is the incumbent's own report.

Compile half: the sealed script `scripts/fr13_build_fa2_qrow32_gqa_pair_b1_sm121a.sh`
refuses unless the tracked worktree is clean and `HEAD == @{upstream}`. This shared
worktree carries other agents' in-flight changes, so the build was run from a clean local
clone of the same pushed branch (`fr14_treeattn_build_repo_20260817`, `dirty_tracked=0`),
after verifying the three build-critical files (`fr13_patch_fa2_tree_bias.py`,
`fr13_qrow32_b1_pass_sidecar.py`, and the build script itself) are byte-identical there.
That preserves the script's semantics — the `.so` is still provably built from a clean,
pushed commit — without weakening any precondition.

**Result: PASS, byte-identical.** Total wall 37 s.

| artifact | value |
|---|---|
| rebuilt `.so` sha256 | `3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae` |
| pinned sha256 | **identical** |
| size | 299 815 552 B = pin |
| `REG` / `STACK` / `LOCAL` | 243 / 0 / 0 = sealed build |
| SASS dump sha256 | `fa01f988…` |
| `defined_dynamic` / `dt_needed` / `runtime_path` diffs | 0 / 0 / 0 bytes |
| offline torch load in pinned image | `library_loaded: true, registered_required_ops: true` |

**Two defects surfaced by running it, both now understood.**

*(a) The ABI allowance was dead on arrival* (fixed, commit `bdd2c18e5`). The first run
exited 94 at the `undefined_dynamic` clause even though the `.so` was byte-identical to
the pin — which is what proved the guard wrong rather than the build. DT_NEEDED is
captured from `readelf -W -d`, whose fifth field is the **bracketed** soname
`[libstdc++.so.6]`; the guard tested `grep -qx 'libstdc++.so.6'`, a whole-line match on
the *unbracketed* name, which can never succeed. The single legitimate cold-path import
(`_ZNSoC2Ev@GLIBCXX_3.4` — the same symbol the 2026-08-10 build produced) was therefore
rejected as "libstdc++ import without libstdc++ in DT_NEEDED" while the library sat in
DT_NEEDED all along. It survived because the clause landed in `3120b3765`, the *same
commit that pinned the candidate*: the sealed build predates it (its dir has no
`dt_needed.diff`, `runtime_path.diff`, or `undefined_dynamic_allowed.txt`), so the
allowance had never executed until this rebuild. A guard written after the only build that
would have exercised it is a guard nothing has tested.

*(b) The pinned `.so` sha256 is not reproducible across rebuilds — but the kernel is.*
Re-running the fixed script produced `454135cef568626a…`, same size, **same SASS**. Cause,
identified rather than assumed: nvcc stamps its own driver PID inside the build container
into host-side symbol and name-table entries (`tmpxft_00000009_…` vs `tmpxft_0000000a_…`),
which propagates into ~87 kB of symbol/relocation bytes. All three builds agree where it
matters:

| build | nvcc module id | SASS sha256 | `.so` sha256 |
|---|---|---|---|
| 2026-08-10 (sealed) | `tmpxft_00000009` | `fa01f988…` | `3560cdc0…` (the pin) |
| FR14 run 1 | `tmpxft_00000009` | `fa01f988…` | `3560cdc0…` **= pin** |
| FR14 run 2 | `tmpxft_0000000a` | `fa01f988…` | `454135ce…` |

**Device code is fully deterministic; the `.so` hash is reproducible only up to a PID.**
Run 1 matched the pin by landing the same container PID as the sealed build. This matters
beyond this rebuild: the campaign pins the `.so` sha256 at six hard-fail comparison sites,
so any future rebuild from byte-identical source has a coin-flip chance of being rejected
as a different binary. **Recommendation: pin the SASS digest as the reproducibility
credential** (deterministic, and it is what actually governs kernel behaviour), keeping the
`.so` sha256 as the integrity check on the *staged artifact* it correctly is. Flagged for
Mark — the six-site pin structure is campaign provenance machinery, not this agent's to
restructure.

---

## 1b. The hardware, measured from the driver (not from documentation)

Queried on the idle GPU via `cuDeviceGetAttribute`
(`results/.../fr14_sm121a_capability_probe.py`), plus a functional DSMEM kernel.

| fact | measured | why the design cares |
|---|---|---|
| device / cc | **NVIDIA GB10, cc 12.1** (sm_121a) | — |
| SMs | **48** | CTAs = SMs busy at 96 KB smem |
| max dynamic smem / block (optin) | **101 376 B (99 KB)** | G=6 needs exactly 96 KB → **fits, 3 072 B headroom** |
| smem / SM | 102 400 B | 1 CTA/SM at 96 KB, as today |
| threads / block, threads / SM | 1024, **1536** | G=6's 384-thread CTA is legal |
| L2 | **24 MB** | see below — reshapes the cost model |
| DRAM | 256-bit @ 8533 MHz → **273 GB/s** | matches the brief's 273 exactly |
| **`CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH`** | **1 (supported)** | gates fallback F2 |
| **DSMEM functional, cluster size 3** | **PASS, 0 mismatches** | F2 is real, not just advertised |

The DSMEM probe is a functional test, not an attribute read: 12 CTAs launched as 4
clusters of 3 (exactly the B1 grid regrouped), cluster rank 0 stages a 16 KB tile, ranks 1
and 2 read it back through `cluster.map_shared_rank()` and verify every element. Zero
mismatches at all three ranks. **A peer CTA on GB10 can read the tile its neighbour
staged.**

**The 24 MB L2 is the most consequential number here, and it cuts against the lever's
premise.** One staged tile is 64 KB; the tiles in flight across all 12 B1 CTAs at any
instant are ~4 KV heads × 64 KB = 256 KB — three orders of magnitude inside L2. The 3
lanes sharing a KV head march through the same `n_block` sequence, so lanes 2 and 3 are
very likely already being served from L2 rather than DRAM. If so, today's 3× "re-staging"
costs L2 bandwidth and issue slots, **not** the DRAM bytes the −13 ms estimate was built
from. This is the strongest a-priori evidence for Model P in §8, and it is exactly what
the pre-registered fit will settle.

## 2. The incumbent, read from source (not from memory)

All rows below are read out of the patcher's translation units and
`kernel_traits.h`, not recalled. `Flash_fwd_kernel_traits<hdim, BM, BN, NWarps, …>`;
B1 means `sequences=1`, 24 Q heads, 4 KV heads, 6 Q heads per KV head, head_dim 256, bf16,
paged KV with a 1024-row page, 32 static query rows.

| arm | traits | kBlockM | warps | threads | smem | grid @ B1 | CTAs | KV lanes / KV head |
|---|---|---|---|---|---|---|---|---|
| `qrow16` (**the B1 byte-gate reference**) | `<256,16,64,1>` | 16 | 1 | 32 | 72 KB | `(2, 1, 24)` | **48** | **12×** |
| `nosplit` (`qrow32_b1`) | `<256,32,64,2>` | 32 | 2 | 64 | 80 KB | `(6, 1, 4)` | 24 | 6× |
| `split2` (gate-only) | `<256,32,64,2>` + combine | 32 | 2 | 64 | 80 KB | `(6, 2, 4)` | 48 | 6× |
| **`gqa_pair` (promoted default)** | `<256,64,64,4>` | 64 | 4 | 128 | **96 KB** | `(3, 1, 4)` | **12** | **3×** |

The brief's "48 single-warp CTAs re-staging the same KV up to 12× per step" is exactly the
`qrow16` row — 32 rows split into two 16-row m_blocks × 24 heads — and it is the arm the
B1 byte gate still compares against (`_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL =
1179791667`). The promoted `gqa_pair` arm has already taken 12× → 3× and 48 CTAs → 12.

smem 96 KB ⇒ **one CTA per SM**, so *CTAs = SMs busy*. 12 of 48 SMs at B1 today.

---

## 3. What the lever actually is, restated correctly

Inside a CTA, `gqa_pair` already loads each K/V tile once and applies it to all 32 tree
rows × 2 heads — the tree bias only masks *within* the 32-row block, exactly as the brief
says, and the M tile is laid out as the nested shape `((32 rows, 2 heads), 256)` with
stride `((q_row_stride, q_head_stride), 1)`. The in-CTA reuse is done.

The residual re-staging is **inter-CTA**: the 3 CTAs covering one KV head each stage the
same 64 KB tile. Killing it means one CTA per KV head, i.e. **G = heads-per-CTA = 6**.

Because `kBlockM = 32·G` and grid.x = `6/G`, the whole Tier-A design space is:

| G | CTAs @ B1 | CTAs @ B4 w4 | KV lanes / KV head |
|---|---|---|---|
| ½ (`qrow16`) | 48 | 192 (>48, multi-wave) | 12× |
| 1 | 24 | 96 (multi-wave) | 6× |
| **2 (today)** | **12** | **48** | **3×** |
| 3 | 8 | 32 | 2× |
| 6 | 4 | 16 | 1× (compulsory) |

There is no Tier-A geometry that both reads each KV byte once *and* keeps ~48 CTAs busy.
Read-once at full occupancy requires 4 KV heads × 12 context splits = 48 CTAs, i.e.
split-K — see §8.

---

## 4. Tier-A argument: per-row accumulation order is unchanged

**Claim.** Changing G (hence `kBlockM` and `kNWarps`) leaves every row's output and LSE
bit-identical.

**Structural half.** In `kernel_traits.h`:

```
using TiledMma = TiledMMA<MMA_Atom_Arch,
                          Layout<Shape<Int<kNWarps>,_1,_1>>,   // warps tile M only
                          Tile<Int<16 * kNWarps>, _16, _16>>;
```

Warps tile **only** the M dimension; N and K are never split across warps. Therefore, for
a fixed row:

1. `S = Q·Kᵀ` reduces over head_dim = 256 in a fixed sequence of 16 MMA k-steps that is
   independent of `kBlockM`/`kNWarps`.
2. The tile loop `for (n_block = n_block_max-1; n_block >= n_block_min; --n_block)` and
   `kBlockN = 64` are untouched, so each row sees the *same tiles in the same order*.
3. `softmax_rescale_o` is per-row (its running max/sum live in that row's own registers,
   `Softmax<2 * size<1>(acc_o)>` with `size<1>(acc_o) = kBlockM/(16·kNWarps) = 1` for every
   G in the family), so the rescale sequence is unchanged.
4. `O += P·V` reduces over the 64 columns of the current tile — fixed by `kBlockN`, not by
   `kBlockM`.

**No cross-row reduction of any kind is introduced.** Rows never exchange data; the only
thing that changes is which rows are co-resident in a CTA. This is Tier-A by the brief's
own test, not Tier-B.

**Empirical half — and this is the strong part.** The claim is not a derivation. The
promoted `gqa_pair` arm (`kBlockM=64`, `kNWarps=4`, 128 threads) is byte-gated **against
`qrow16`** (`kBlockM=16`, `kNWarps=1`, 32 threads) and passed with 0/0 output and LSE
mismatches, twice, plus poisoned-shadow, at production geometry. That is a sealed
experiment on *exactly this axis*: a 4× change in `kBlockM` and `kNWarps` moved zero bits.
G=3 and G=6 are further steps along the same axis (`kBlockM` 96/192, `kNWarps` 6/12), with
`MMA_M` still 1 and the tile loop untouched.

**The three places it could still break, and how each is held.**

| risk | held by |
|---|---|
| tree-bias row mapping | the M tile is nested `(32, G)` row-major, so tree row = `logical_row & 31` for *any* G (`kStaticQueryRows` = 32 is a power of two). Already the exact expression in `apply_tree_bias`; only its `== 2` literal needs widening. |
| Q / O / LSE addressing | the paired layout is already G-generic: `((32, G), 256)` with stride `((row_stride, head_stride), 1)` for Q and O, and `(32, G)` with stride `(1, total_q)` for LSE. Only five `if constexpr (kStaticHeadGroupSize == 2)` guards pin it to 2. |
| MMA_M ≠ 1 | `kBlockM = 32G` and `kNWarps = 2G` ⇒ `16·kNWarps = 32G = kBlockM` ⇒ `MMA_M = 1` for every G. Held by an assert, not by luck. |

The byte probe (deliverable 4) is what converts this argument into evidence; the argument
only says the probe is *expected* to pass, and defines what a failure would mean.

---

## 5. Why the stock traits stop at G=2 — the structural constraint chain

With `kNWarps = 2G`, `kNThreads = 64G`, `kBlockM = 32G`:

| G | kBlockM | warps | threads | **registers** | **smem Q+KV** | **smem if Q/K shared** | KV-copy thread-rows (`threads/8`) | divides kBlockN=64? |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 2 | 64 | invariant | 16+64 = **80 KB** ✓ | 64 KB | 8 | ✓ |
| **2** | 64 | 4 | 128 | **REG:243** ✓ | 32+64 = **96 KB** ✓ | 64 KB | 16 | ✓ |
| 3 | 96 | 6 | 192 | invariant | 48+64 = **112 KB ✗** | **64 KB ✓** | 24 | **✗** |
| 6 | 192 | 12 | 384 | invariant | 96+64 = **160 KB ✗** | **96 KB ✓** | 48 | **✗** |

**Registers are a non-issue, and that is a load-bearing finding.** Per-thread pressure is
*invariant* in G, because every register-resident quantity scales with `kBlockM` while the
thread count scales with it too:

- `acc_o`: `kBlockM·256·4 / kNThreads` = 32G·256·4 / 64G = **512 B/thread**, all G
- `acc_s`: `kBlockM·64·4 / kNThreads` = **128 B/thread**, all G
- `tSrQ`: `MMA_M = 1` for all G ⇒ **unchanged**
- softmax state: `Softmax<2·1>` ⇒ **unchanged**

So the measured REG:243 / STACK:0 of the G=2 unit is the *same* budget G=3 and G=6 face.
My first-pass fear that G=6 would spill was wrong — it came from holding the thread count
fixed at 128 instead of scaling warps with heads.

Two real blockers remain.

**Blocker A — shared memory.** `kSmemSize = kSmemQSize + kSmemKVSize` overflows the GB10
per-block cap (96 KB is proven serviceable today; the hardware cap is ≈99 KB and will be
measured, not assumed). Fix: the traits already expose it —
`kSmemSize = Share_Q_K_smem ? max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize`.
With `Share_Q_K_smem=true`, G=3 needs **64 KB** and G=6 needs **96 KB** — the latter
exactly what ships today.

**Blocker B — the gmem→smem KV staging partition.**
`kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow)` with
`kGmemThreadsPerRow = 64/8 = 8`. G=3 gives 24 thread-rows and G=6 gives 48, neither of
which divides `kBlockN = 64`; the integer division silently under-covers the tile (48 of
64 rows at G=6). No choice of `kGmemThreadsPerRow` fixes it: 128-bit loads of a 256-element
row require `kGmemThreadsPerRow ∈ {8,16,32}`, and neither 192 nor 384 divided by those
yields a power-of-two ≤ 64. Fix: **sub-partition the staging** — 128 of 192 threads at G=3,
256 of 384 at G=6 — which restores `kGmemRowsPerThread` to 4 and 2 respectively, both
dividing the 1024-row page so no thread's slice crosses a page boundary (the property the
existing launcher asserts as `1024 % kGmemRowsPerThread == 0`).

---

## 6. The three enabling patches — all numerics-neutral

**P1. `Is_Q_in_regs` / `Share_Q_K_smem` in the split-KV path.** Today
`compute_attn_1rowblock_splitkv` calls `gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, …)`
unconditionally (Q re-read from smem each tile), while the *non-split*
`compute_attn_1rowblock` in the same file already has the full
`gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>` treatment plus the
`if (Share_Q_K_smem) { cp_async_wait; __syncthreads(); copy sQ→tSrQ; __syncthreads(); }`
prologue and the matching epilogue `__syncthreads()`. P1 mirrors that established
in-file idiom into the split-KV path. **Register cost: zero** — `tSrQ` is already
allocated as a full `(MMA, MMA_M, MMA_K)` fragment either way; the only change is that it
is filled once instead of per tile. **Numerics: identical** — same MMA instructions, same
order; `gemm<true>` merely skips a redundant smem→register copy.

**P2. KV staging sub-partition** (Blocker B). A tiled copy over the first 64G·(2/3)
threads with `if (tidx < …)` predication. Staging is pure data movement into smem: the
smem contents are bit-identical, so no accumulation order is touched.

**P3. Widen the head-count guards.** Nine sites pin G to {1,2}: four `static_assert`s and
one `== 2` conditional in `apply_tree_bias`, plus five `if constexpr
(kStaticHeadGroupSize == 2)` guards in the address-layout patch. The asserts that encode
the *relationships* (`kNWarps == 2·G`, `kNThreads == 64·G`, `kBlockM == 32·G`) are already
G-generic and must be kept, not relaxed — they are what makes MMA_M = 1 structural.

None of P1–P3 changes a single arithmetic operation on a single row. That is the whole
point: the Tier-A claim survives the implementation, not just the design.

---

## 7. Traffic model (measured geometry, derived bytes)

B1, context L = 23 000, 16 full-attention layers (3, 7, …, 63), `kBlockN = 64`:

- KV per token per layer = 2 (K,V) × 4 KV heads × 256 × 2 B = **4096 B**
- compulsory KV per step = 23 000 × 4096 × 16 = **1.507 GB**
- staged tile = 64 rows × 256 × 2 B × 2 = **65 536 B**; `n_blocks` = ⌈23000/64⌉ = **360**
- staged bytes per CTA per step = 360 × 65 536 × 16 = **377.5 MB**

| arm | CTAs @ B1 | staged GB/step | × compulsory | CTAs @ B4 w4 | staged GB/step | × compulsory |
|---|---|---|---|---|---|---|
| `qrow16` (reference) | 48 | 18.12 | 12.0× | 192 | 72.5 | 12.0× |
| `nosplit` | 24 | 9.06 | 6.0× | 96 | 36.2 | 6.0× |
| **`gqa_pair` (today)** | **12** | **4.53** | **3.0×** | **48** | **18.12** | **3.0×** |
| G=3 | 8 | 3.02 | 2.0× | 32 | 12.08 | 2.0× |
| **G=6** | **4** | **1.51** | **1.0×** | **16** | **6.04** | **1.0×** |

**MMA work is invariant in G** and worth stating explicitly, because it is the cost side
of the trade. Total M-rows per tile across all CTAs = 24 heads × 32 rows = 768 regardless
of G, so per step at B1: 768 × 64 × 256 × 2 gemms × 360 tiles × 16 layers = **144.96 GMAC
≈ 290 GFLOP**, fixed. Spread over `CTAs` SMs. Relative to `gqa_pair`, the MMA-bound floor
therefore scales as **12/CTAs**: **×1.5 at G=3, ×3.0 at G=6**.

That is the exchange rate in one line: **G=6 divides staged bytes by 3 and multiplies the
compute floor by 3.**

---

## 8. Two cost models, one discriminator — why −13 ms is NOT yet earned

The banked −13 ms for this lever is an fp8-era byte-count extrapolation. Two models fit
the campaign's own history, and they disagree about the sign of the result.

**Model S (staging-bound).** Time ∝ staged bytes. Predicts G=6 at ⅓ of today's attention
time. This is the model the −13 ms came from.

**Model P (parallelism-bound).** Time ∝ (work per CTA) / (per-SM rate), with CTA count
setting aggregate throughput. Every CTA walks all 360 tiles whatever G is, so wall time is
`n_tiles × per-tile cost`; the per-tile MMA cost per SM *rises* 3× at G=6. Predicts G=6 is
**worse**, and that G=3 is roughly a wash.

The existing evidence does not separate them. `gqa_pair` beat `qrow16` by −4.405 ms
(fp8-era step wall) while cutting both staged bytes 4× *and* CTAs 4×. Model S predicts a
much larger win than was observed; Model P predicts none. Reality is in between, which
means a two-parameter fit is required:

```
T(G) = α · S(G) + β · P(G),   S(G) = (24/G) · 377.5 MB,   P(G) = W · G / 24
```

`T` is convex in G with an interior minimum. **If G=2 already sits at or past that
minimum, TreeAttention-v2 by head-merging is refuted before a line of kernel code is
written.**

**The discriminator, and it costs no new build.** The sealed
`.so 3560cdc0…` exports **both** private launchers:

```
fr13_run_mha_fwd_fixed32_qrow16                 → 48 CTAs, 18.12 GB staged
fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1     → 12 CTAs,  4.53 GB staged
```

selectable at real geometry by the `tree_bias_batch_stride` sentinel (1179791667 and
1179791670), plus the stock FA2 path as a third point. Two kernel-level timings at
identical shapes solve α and β exactly and **predict T(3) and T(6) before either is
built**. This is the pre-registered gate, and it is a kernel-level probe on a free GPU —
no serve, no gate window.

**Pre-registered decision rule.**

| outcome | action |
|---|---|
| fit predicts T(3) < T(2) and T(6) < T(3) | build G=6 (P1+P2+P3), byte-probe, then gate |
| fit predicts T(3) < T(2) ≤ T(6) | build G=3 only — the smaller patch (64 KB smem, 128-of-192 staging) |
| fit predicts T(3) ≥ T(2) | **STOP and report**: head-merging is exhausted at G=2; the kernel lane's −13 ms must be re-priced or re-sourced (§9) |

Sweeping context length across 20–40k in the same probe separates the α (∝ tiles) and β
(∝ tiles) terms further and tests the fit rather than merely solving it.

---

## 9. Fallbacks, in the order they should be considered

**F1 — G=3 (8 CTAs, 2× re-staging).** Half the patch surface of G=6: 64 KB smem, a
128-of-192 staging sub-partition, same Tier-A argument. Compute floor ×1.5. The
conservative pick if the fit is ambiguous.

**F2 — cluster / distributed shared memory (Tier-A, keeps 12 CTAs). NOW THE LEAD
CANDIDATE — hardware-verified.** Group the 3 CTAs that share a KV head into a thread-block
cluster; one stages the tile, all three read it through DSMEM. **Staged byte traffic falls
3× with CTA count *and* the compute floor unchanged** — it is the only option in the whole
space that buys bytes without paying parallelism, and it requires **no change to
`kBlockM`, `kNWarps`, `MMA_M`, or any accumulation order whatsoever**, which makes its
Tier-A argument even shorter than §4's: the arithmetic is bit-for-bit the incumbent's, and
only the provenance of the bytes in smem changes.

This was written as speculative — consumer Blackwell is not Hopper, and cluster/DSMEM is
exactly the sort of capability that quietly is not there. It is no longer speculative:
`CLUSTER_LAUNCH = 1` and the functional probe (§1b) shows rank 1 and rank 2 correctly
reading rank 0's staged tile at cluster size 3, zero mismatches, on the real device.

Remaining unknown is **remote-smem bandwidth**, not capability: the staging CTA becomes
the single issuer for its cluster, so F2 trades 3× fewer gmem→smem loads for 3× more
readers on one smem port. That is the measurement to add to the §8 probe. On the current
reading — 24 MB L2 already absorbing much of the re-staging, compute floor unchanged — F2
dominates G=6 and probably G=3, and should be built first if the fit says anything is
worth building.

**F3 — split-K (Tier-B — STOP for Mark).** 4 KV heads × 12 context splits = 48 CTAs, each
staging 1/12 of the context: **read-once *and* full occupancy**, 1.51 GB staged with the
compute floor 3× *lower* than today. Under either cost model this is where a −13 ms (or
larger) actually lives. It is Tier-B and must not be built on this agent's authority: the
split-K combine re-reduces partial softmax across splits, so per-row output is not
bit-identical to the *served* `nosplit`/`qrow16` reference. The tree already encodes this
judgement — `split2` exists, was byte-gated against a *same-split* reference via
`_fr13_fa2_qrow32_b1_require_same_reduction`, and is explicitly excluded from
`_FR13_FA2_QROW32_B1_PRODUCTION_ARMS` because it "was never byte-qualified as a served
dispatch". **Recommendation: put F3 in front of Mark with the fitted numbers**, since the
honest reading of §7–§8 is that the greenlit −13 ms is a split-K number wearing a
head-merge label.

---

## 10. Implementation plan (deliverable 3) and probe plan (deliverable 4)

**Selector machinery** — the `gqa_pair` idiom exactly. Sentinels continue the `'FR13'`
(0x46523133 = 1179791667) series: **`treeattn_v2_g3` → 1179791671**, **`treeattn_v2_g6` →
1179791672**. Each adds an `_FR13_FA2_QROW32_B1_ARMS` entry (`num_splits: 0`), a private
hidden launcher `fr13_run_mha_fwd_fixed32_qrow32_treeattn_v2_g{3,6}_b1`, an anchored
`_API_GATE` deriving from the `gqa_pair` gate by counted substitution (so B4/B1 drift fails
the codegen instead of silently forking), and its own candidate `.so` + closure pins. Arms
stay out of `_FR13_FA2_QROW32_B1_PRODUCTION_ARMS` until a byte gate qualifies them as a
served dispatch. Build: a new `fr13_build_fa2_qrow32_treeattn_v2_*_b1_sm121a.sh` derived
from the sealed gqa_pair script — same image, same 55 byte-reused reference objects, same
ABI audit, same SASS no-spill/no-LDL contract, same mandatory offline torch load. sha+size
recorded and pinned into the six comparison sites the sealed script enumerates.

**Offline byte probe.** Standalone, no serve. Loads the candidate `.so`, calls
`torch.ops._vllm_fa2_C.varlen_fwd_tree_bias` at **real** geometry — b=1, total_q=32, h=24,
h_k=4, d=256, bf16, paged KV with 1024-row pages and a real block table, `tree_bias`
`[32,32]` fp32 — and selects reference vs candidate purely by retagging `tree_bias` with
`torch.as_strided(base, (1,32,32), (sentinel,32,1))`, the same zero-copy metadata retag the
sealed gate qualified (`_fr13_fa2_qrow32_b1_candidate_tree_bias`). Asserts **byte-identical
output and LSE** across many seeds and context lengths spanning 20–40k, and emits the
kernel-level timing readout that feeds §8's fit. The live K0-shape byte gate
(`fr13_run_b1_k64_qrow32_split2_live_gate` idiom with a new arm) comes later, through the
parent.

---

## 11. Open items

**Closed since first draft**

1. ~~Compile half of the build-env proof~~ — **PASS**, `.so` byte-identical to the pin;
   two defects found and one fixed (§1).
2. ~~sm_121a cluster/DSMEM capability~~ — **supported and functional at cluster size 3**
   (§1b). F2 promoted to lead candidate.
3. ~~GB10 max dynamic smem per block~~ — **101 376 B measured**; G=6's 96 KB fits with
   3 072 B to spare.

**Open**

4. **The §8 fit** — the gate on whether any of §6's kernel is built. Runs on a free GPU
   from the already-sealed binary; no new build, no serve.
5. **Remote-smem bandwidth for F2** — capability is proven, throughput is not. Add a
   local-vs-remote smem read comparison to the §8 probe.
6. **F3 (split-K) is a Mark decision.** On the current reading — 24 MB L2 absorbing much
   of the re-staging, compute floor scaling as 1/CTAs — the greenlit −13 ms looks like a
   split-K number wearing a head-merge label, and F3 is the option that actually delivers
   it. Worth putting in front of him with the fitted numbers rather than assumed ones.
7. **`.so` sha256 pinning is not rebuild-reproducible** (§1b). Recommend pinning the SASS
   digest as the reproducibility credential. Provenance-core machinery — Mark's call.

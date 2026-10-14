# Item L — vllm-project/vllm#58718, second-GB10 measurement

**PR** [#58718](https://github.com/vllm-project/vllm/pull/58718) "[Kernel][Perf] Add an
sm121 batch-invariant matmul table tuned on GB10" (hclsys), head
`b40edf46984f06bb5fb84f35e6c56634df3de364`.
Pre-run card: `L_58718/CARD.md` (written before any GPU run). No merge verdict here.

---

## Verdict

**Both of the author's headline effects reproduce on this second GB10.** The decode win is
real and close to the reported size; the above-M=512 penalty is real and **larger here than
reported**. The bitwise M-invariance gate passes on all 15 shapes off the author's box, and
the per-M tile survives `torch.compile` and CUDA-graph replay.

Two things this run adds that the thread has not covered:

- **Q1** — against `sm120`, which is what a GB10 runs *today*, the new table is ~9-11%
  faster at M ≤ 256 but **~16% slower at M=1024 and M=2048**. Measured relative to the
  default tile the PR reports a 7-18% large-M loss; measured relative to the status quo on
  this box, the PR *deepens* the large-M regression rather than reducing it.
- **Q2** — the stated reason the large-M buckets cannot fall back to the default tile
  (changing `BLOCK_K` would change the K-reduction order) **did not change the output bits
  on this build**: 2,237,180,160 output elements compared across isolated `BLOCK_K`
  32/64/128 swaps, zero differing. The same hash gate does catch a real reduction-order
  change (split-K halves, cuBLAS), so it is not blind.

Q1 and Q2 are questions for the author and maintainers, not defects. **No defect was found.**

---

## Results

All GPU work ran under `flock /home/mark/shared/exp54928/gpu.lock`. Every number below is
from an archived log under `L_58718/logs/`.

### 1. The PR's own tests, at PR head

| check | result | expected by author | log |
|---|---|---|---|
| `tests/v1/determinism/test_matmul_batch_invariant.py` | **34 passed** | 34 passed | `logs/20_pytest_matmul_head.log` |
| `tests/test_config.py -k "batch_invariant or matmul"` | **11 passed** | 11 passed | `logs/22_family_and_config.log` |
| `_get_tuned_matmul_arch_family` on-card | 12.0→`sm120`, **12.1→`sm121`**, 12.2→`sm120`, 10.0→`blackwell`, 9.0→`hopper`, 8.9→`ada`, 8.0→`None` | same | `logs/22_family_and_config.log` |
| resolved table on this card | is the `sm121` dict, 15 entries | same | `logs/22_family_and_config.log` |

### 2. Static checks on the table (`logs/10_table_analysis.log`)

- The 15 `(N, K)` keys are **set-equal** to `sm120`; nothing added or dropped. (C3 ✓)
- **12/15** `sm121` shapes use `BLOCK_K=128`; the bf16 default uses 64. `sm120` uses
  `BLOCK_K=128` for only 5/15. (C4 ✓)
- The `sm120` block is **byte-identical** to the merge base (sha256 `33b66afb8e4db9b3…`,
  10062 bytes); the diff is 231 insertions and the single `return "sm120"` deletion.
- Bucket edges `(1,4,8,16,32,64,256,512,1024,2048)` are identical for all 15 shapes.
- The three `BLOCK_K=64` shapes are exactly the default tile at one M each and differ only
  in `num_warps` / `BLOCK_N` / `num_stages` elsewhere — LioEinaudi's "default or
  near-default" reading is accurate (`logs/35_blockk_split.log`).

### 3. Re-timing the 15 keys, tuned vs default

CUDA events, one event pair per iteration, p50 of 200 iters (M ≤ 32) or 50 iters (M ≥ 256)
after 10 warmup, **both A/B orders**, every kernel pre-compiled outside the timed region.
`b = weight.t()`, the layout `linear_batch_invariant` actually produces
(`batch_invariant.py:1036-1037`). Raw: `data/timing.jsonl`; log `logs/30_timing.log`;
aggregation `logs/31_aggregate.log`.

Mean tuned/default over all 15 shapes:

| M | 1 | 32 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| **this GB10** (order T→D) | **0.721** | **0.754** | **0.929** | **1.027** | **1.263** | **1.202** |
| this GB10 (order D→T) | 0.718 | 0.754 | 0.930 | 1.028 | 1.264 | 1.205 |
| author (PR body) | 0.68 | 0.74 | 0.95 | 1.07 | 1.18 | 1.14 |
| shapes faster (<0.97) | 15/15 | 14/15 | 11/15 | 2/15 | 0/15 | 1/15 |
| shapes slower (>1.03) | 0/15 | 0/15 | 2/15 | 9/15 | 12/15 | 12/15 |

Per model, mean over that model's 5 shapes (the author's own grouping; all 15 keys map
cleanly onto Qwen3-1.7B / 4B / 8B qkv, o_proj, gate_up, down, lm_head):

| model | M=1 | M=32 | M=256 | M=512 | M=1024 | M=2048 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 0.570 | 0.618 | 0.913 | 0.986 | 1.162 | 1.105 |
| Qwen3-4B | 0.741 | 0.771 | 0.977 | 1.066 | 1.179 | 1.183 |
| Qwen3-8B | 0.851 | 0.872 | 0.897 | 1.030 | 1.449 | 1.318 |

Worst single cell: `(6144, 4096)` at M=1024, **1.72×** slower than default (1.720 / 1.689 in
the two orders). Best: `(2560, 4096)` at M=1, 0.244.

**The two A/B orders disagreed in direction by >3% in 0 of 90 cells.** No cell failed
`allclose(rtol=atol=1e-1)` against an fp32 `torch.mm` reference at M ≤ 32. Peak observed
throughput 82.6 TFLOP/s (tuned) and 92.7 TFLOP/s (default).

The penalty lives entirely in the `BLOCK_K=128` shapes, exactly as the thread argues
(`logs/35_blockk_split.log`):

| M | 1 | 32 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| 12 shapes with `BLOCK_K=128` | 0.816 | 0.844 | 0.920 | 1.048 | **1.329** | **1.242** |
| 3 shapes with `BLOCK_K=64` | 0.341 | 0.393 | 0.964 | 0.947 | 1.000 | 1.041 |

### 4. Q1 — sm121 vs sm120, the comparison the PR does not make

The PR compares the new table against the *default tile*. A GB10 today runs the **`sm120`**
table, so that is the baseline a GB10 user actually moves from. Both tables were timed in
the same harness on the same box; the head-to-head is formed as
(sm121/default) / (sm120/default) so drift between the two runs cancels. The two runs'
`default` baselines agree to within 1.7% at every M, and the direct ratio of the two tuned
p50 times agrees with the normalised figure to within 1.3 points.
Raw `data/timing_sm120.jsonl`; logs `logs/32_timing_sm120.log`, `logs/33_aggregate_sm120.log`,
`logs/34_compare_tables.log`.

| M | 1 | 32 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| sm121 / default | 0.721 | 0.754 | 0.929 | 1.027 | 1.263 | 1.202 |
| sm120 / default (status quo on GB10) | 0.797 | 0.835 | 1.031 | 1.036 | 1.111 | 1.055 |
| **sm121 / sm120** | **0.893** | **0.892** | **0.909** | 1.005 | **1.161** | **1.155** |
| cross-check, direct p50 ratio | 0.900 | 0.879 | 0.917 | 1.006 | 1.148 | 1.155 |

So on this GB10 the PR buys ~11% at decode and gives back ~16% at M=1024/2048 *relative to
what is shipping now*. Per model the large-M cost is worst on Qwen3-8B (1.282 at M=1024,
1.234 at M=2048) and mildest on Qwen3-4B at M=1024 (1.046).

Note also that the PR's premise reproduces: `sm120` on a GB10 does help at decode
(0.797 at M=1) and does hurt above it (1.111 at M=1024), which is the motivation stated in
the PR body.

### 5. Bitwise M-invariance off the author's box (C5)

sha256 over the bytes of output row 0, same `a[0]` and same `b`, tuned table active,
`M ∈ {1, 8, 32, 2048}` (log `logs/40_invariance.log`, raw `data/invariance.json`):

**All 15 shapes produce exactly one hash.** 3 or 4 *distinct tiles* are dispatched across
those four M values for every shape, so the comparison is not trivially satisfied. C5 ✓.

### 6. Q2 — what actually moves the output bits

While running the A/B above, tuned and default outputs came out **bitwise identical in all
90 cells**, which is not what "different `BLOCK_K` ⇒ different K-reduction order" predicts.
Followed up with an isolated probe (`logs/40_invariance.log` §E1/§E3):

| comparison | shapes / M / inputs | result |
|---|---|---|
| tuned tile vs default tile | 15 keys × {M=1, 2048} × {randn, rand}; 48/60 cells have `BLOCK_K` 128 vs 64 | **0 differing elements of 2,237,180,160** |
| same tile, only `BLOCK_K` 128→64 | same 60 cells | **0 differing elements** |
| only `BLOCK_K`, 64@M=1 vs 128@M=2048 | `(4096,4096)`, everything else pinned | row-0 hash **identical** |
| only `BLOCK_K`, 32@M=1 vs 128@M=2048 | same | row-0 hash **identical** |
| only `BLOCK_K`, 32@M=1 vs 64@M=2048 | same | row-0 hash **identical** |
| tile only, `BLOCK_K` pinned at 64 | 16×32/w4/s5 vs 128×128/w8/s3 | row-0 hash identical (this is the PR's property, as expected) |
| **sensitivity control**: split the K reduction into two halves and add | same | row-0 hash **DIFFERS**, max abs diff 8.0 |
| **sensitivity control**: cuBLAS `torch.mm` | same | row-0 hash **DIFFERS**, max abs diff 4.0 |

The gate is therefore *not* blind — it catches a genuine change of reduction order — but on
this build `BLOCK_SIZE_K` is not one. The plausible reason is that `tl.dot` accumulates over
k in ascending order into one fp32 accumulator regardless of how the Python-level k-loop is
tiled, so `BLOCK_K` changes how many loop iterations there are but not the order of the
additions.

This is empirical, not a proof, and its scope is stated in "What this does not establish".
The relevance is that if it also holds on the author's box, the M ≥ 512 buckets *could*
emit the default tile, which would remove the large-M regression; the PR body and
LioEinaudi's review both rule that out on reduction-order grounds.

Worth noting on feasibility: the default bf16 tile `128×128×64/s3` **is** launchable on
GB10, but `128×128×128/s3` is **not** — Triton wants 139264 B against this card's 101376 B
`shared_memory_per_block_optin`. So a "default tile at large M" fallback is available, but
"default tile with the shape's tuned `BLOCK_K`" is not, for the 128-wide shapes.

### 7. Compiled-path dispatch probe (`logs/50_compile_probe.log`, `logs/51_compile_recompile.log`)

Model-free `torch.compile` of the exact call vLLM's unquantized linear layer makes under
`VLLM_BATCH_INVARIANT` (`linear_batch_invariant(x, weight)`, `linear.py:238`), at
`N=K=4096` (a `BLOCK_K=128` shape):

| M | tile dispatched on the compiled path | matches eager | bitwise == eager |
|---|---|---|---|
| 1 | 16×128×128, w4, s3 | yes | yes |
| 8 | 16×128×128, w8, s3 | yes | yes |
| 32 | 32×128×128, w8, s3 | yes | yes |
| 512 | 64×128×128, w4, s3 | yes | yes |
| 2048 | 64×128×128, w4, s3 | yes | yes |

- `_get_matmul_config` received a **concrete `int` M at every lookup**, never a SymInt, so
  the tile is chosen at trace time and baked into that graph — the right tile is used at
  each M, and no stale small-M tile leaks into a large-M call.
- Row 0 is bitwise identical across all five M **on the compiled path**, and identical to
  eager.
- CUDA-graph capture at M=1 and at M=2048 each captured the same tile as eager, replayed
  bitwise-equal to eager, and produced the same row 0.

Side observation (mechanism, **not** introduced by this PR): because the bucket lookup
compares the traced M, a live table forces Dynamo to specialise per M — 9 unique graphs over
M ∈ {1,8,16,32,64,128,256,512,1024,2048} with a table, versus 2 without. This is identical
before and after the PR on a GB10 (a table is active either way), so it is context, not a
change. See Deviations item 2 for a wrong first version of this measurement.

### 8. Layout sensitivity (observation)

The same sweep with `b` a contiguous `[K, N]` instead of `weight.t()` inverts the result:
tuned/default becomes 0.879 / 1.007 / 1.242 / 1.380 / 1.492 / 1.478 at M = 1 / 32 / 256 /
512 / 1024 / 2048 — the table loses almost everywhere. On CUDA the weight stays a contiguous
`[N, K]` (`linear.py:192-201`; the `t().contiguous().t()` at `linear.py:227` is XPU-only and
behind `VLLM_XPU_FORCE_N_CONTIG_WEIGHT`, off by default), so `weight.t()` is the production
layout and the numbers in §3 are the relevant ones. Recorded because it shows the table is
strongly layout-specific: it is evidence the author swept in the production layout, and a
caution for anyone re-sweeping.

---

## Provenance

| | |
|---|---|
| PR head | `b40edf46984f06bb5fb84f35e6c56634df3de364` |
| merge base | `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad` |
| `origin/main` at fetch | `379e9a1ea8a5995464d9bf775bcd36bb03a0995f` |
| worktree | `/home/mark/shared/tmp-scratch/wt-L` (own worktree, removed at end), `vllm.__file__` verified to resolve there |
| GPU | NVIDIA GB10, cc **12.1**, **48 SMs**, 126.2 GB unified, `shared_memory_per_block_optin` 101376 B |
| driver | **590.48.01** |
| torch | **2.13.0+cu132** (`torch.version.cuda` = **13.2**) |
| triton | **3.7.1**; sm_121 ≥ 100 so it compiles with bundled `ptxas-blackwell` = **CUDA 13.1 V13.1.80** |
| system nvcc | 13.0 V13.0.88 (not used on this path — Triton JIT only) |
| venv build | `vllm 0.26.1rc1.dev1159+g23ab0cfdb`, `*.so` built 2026-08-24 from clone commit `23ab0cfdb` |

**Version differences from the author.** The author reports "GB10 (sm_121, **CUDA 13.0**)"
and does not state a driver or torch version. This box runs torch built against **CUDA 13.2**
and Triton emits through a **CUDA 13.1** `ptxas`, on driver 590.48.01. Tile rankings for a
Triton GEMM are a function of the generated SASS, so a one- or two-minor-version ptxas
difference is a plausible source of the gap between the author's large-M ratios (1.07 /
1.18 / 1.14) and this box's (1.03 / 1.26 / 1.20), and it is the first thing to rule out
before treating the difference as a disagreement. LioEinaudi's cc 12.0 re-run was on driver
580.159.04 on a different part and does not bear on this.

---

## Deviations and corrections

1. **Compiled extensions symlinked into the worktree.** `tests/v1/determinism/utils.py`
   imports `vllm.v1.attention.backends.fa_utils`, which raises `ImportError` without
   `_vllm_fa2_C`/`_vllm_fa3_C`. The 17 `*.so` from the built clone were symlinked in
   (`logs/21_so_list.txt`); all are `.gitignore`d and `git status --porcelain` in the
   worktree stayed empty. They were built 2026-08-24 from `23ab0cfdb`, **not** from PR head.
   The path under test here is pure Python + Triton JIT, so those binaries are imported and
   never called by anything measured. This is the same deviation recorded in
   `I_55506_REPORT_agent.md`. The first pytest attempt, which failed on this import, is the
   first entry overwritten in `logs/20_pytest_matmul_head.log`.
2. **One self-inflicted error, corrected.** The first version of the recompile probe
   (§7 side observation) appended to a global list from inside the traced function; that
   added a `len(G['SYM'])` Dynamo guard of its own and made *both* variants recompile on
   every call and hit the recompile limit, showing a meaningless 8 vs 8. The measurement was
   redone with no patching of traced code, giving 9 vs 2. Only the corrected run is reported;
   the script carries a comment saying so.
3. **`contig` layout is a control, not production** — see §8. It is in `data/timing.jsonl`
   and reported separately so it cannot be mistaken for the headline number.
4. **`allclose` gate only at M ≤ 32.** An fp32 `torch.mm` reference at M=2048, N=151936 is
   not worth the GPU time; at M ≥ 256 the check is tuned-vs-default agreement instead. The
   PR's own test file covers correctness independently and passed.
5. **`sudo drop_caches` was run once** before the first GPU job even though no model is
   loaded anywhere in this item.
6. `ruff` is not installed in that venv and nothing was installed into it; no formatting
   check was run over the probe scripts (they are not offered as upstream files).

---

## What this does **not** establish

- **No end-to-end serving claim.** No `vllm serve`, no `vllm bench`, no model was loaded.
  These are 15 isolated GEMMs. Which M values a real workload spends its time at, and how
  much of a step these matmuls are, are not measured here, and the decode-vs-prefill trade
  cannot be settled from this data.
- **Q2 is empirical, not a proof.** "BLOCK_K does not move the bits" is established for
  `matmul_kernel_persistent` as compiled by Triton 3.7.1 / ptxas 13.1 for sm_121, in bf16,
  on these 15 shapes (every K a multiple of 128, so masking never engages), for two input
  distributions. It is not a guarantee, and a different Triton or a different `tl.dot`
  lowering could break it. A maintainer may reasonably prefer the one-`BLOCK_K`-per-shape
  rule precisely because it does not depend on codegen details — that position is untouched
  by this evidence.
- **Not a re-sweep.** The search space was not re-explored. Nothing here says whether a
  better `sm121` table exists, nor whether the decode-weighted scoring is the right
  objective.
- **Nothing about cc 12.0.** No RTX PRO 6000 was available; LioEinaudi's re-run stands on
  its own.
- **The compiled probe is model-free** — one `linear_batch_invariant` call, not vLLM's V1
  runner with its real cudagraph capture set and fusion passes.
- Differences from the author's numbers are differences between two configurations
  (see Provenance); this run is not evidence that the author mismeasured.

---

## Files

- Pre-run card: `L_58718/CARD.md`
- Scripts: `L_58718/scripts/{table_analysis,family_map,time_table,aggregate,compare_tables,invariance,compile_probe,compile_recompile}.py`
- Logs: `L_58718/logs/` — `01_provenance`, `10_table_analysis`, `20_pytest_matmul_head`,
  `21_so_list`, `22_family_and_config`, `29_smoke`, `30_timing`, `31_aggregate`,
  `32_timing_sm120`, `33_aggregate_sm120`, `34_compare_tables`, `35_blockk_split`,
  `40_invariance`, `50_compile_probe`, `51_compile_recompile`
- Raw data: `L_58718/data/{tables,invariance,compile_probe,compile_recompile}.json`,
  `L_58718/data/{timing,timing_sm120,smoke}.jsonl`
- Draft PR comment (**not posted**): `L_58718/COMMENT_58718_DRAFT.md`
- Evidence tarball: `L_58718_evidence.tar.gz`

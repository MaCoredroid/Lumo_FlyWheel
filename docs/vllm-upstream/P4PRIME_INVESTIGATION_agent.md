# P4' — vllm-project/vllm#55122, one missing GB10 test case

## 1. Live state (read 2026-09-11)

OPEN, not draft. Head `7cfd04a395fab32d6ee3364327df297a393f35d2` ("[Kernel] Raise
RADIX_THRESHOLD to 22016, the caching bound", 2026-09-08). `mergeable: MERGEABLE`,
`mergeStateStatus: BLOCKED`, `reviewDecision: REVIEW_REQUIRED`. 848/-297 over 3 files, no labels.
Checks: `pre-run-check` **fail** (label gate), `Check format` and `pre-commit` **skipping**
behind it, DCO/Summary/docs pass. **No human review of the code.** The only maintainer-side
comments are @gau-nernst (perf, alternatives) and @LopezCastroRoberto (prefers the opt-in
backend #55872 over changing the default). @k3dani and @200lz contributed data, not review.
Author has pinged @mgoin, @tlrmchlsmth, @LucasWilkinson without response.

Author's narrowed claim (comment 5601708467, 09-09): on real GB10 traffic **0 of 6,192 selecting
rows were tied at the k-th value**; this PR alone moves end-to-end divergence 333 → 330. He now
claims only *kernel correctness under ties* plus the degenerate-length guard. Anything end-to-end,
or any tie-rate census, is therefore dead or already his — he offered to run the census himself.

## 2. What the suite actually covers (`tests/kernels/test_top_k_per_row.py`, diff lines 1131-1385)

Nine tests, all float32, each asserting *both* bit-identity across 4/6/20 calls *and* equality to
an exact `argsort(descending, stable)` reference in ascending-index order. Widths used anywhere:
256, 512, 700, 1024, 2048, 4096, 8192, 16383/16384/16385, 20000, 40000, 65536 (pitch only).
`num_rows` ∈ {1, 8, 64}; `k` ∈ {512, 1024, 2048}. Tie structures: `randint(0,5)`, all-equal,
signed zero, exact-bin-boundary, pivot populations 2047…16385, narrow range. The author's
standalone `test_det.py` (jschmied/qwen38-flash-next-gb10 `patches/kernel-det/`) uses the same
grid, max width 40000.

Two coverage holes follow from the last commit:

**(a) `test_persistent_topk_path_transition` is stale.** It parametrizes `seq_len`
[16383, 16384, 16385] (diff:1247) and documents itself as "Either side of RADIX_THRESHOLD"
(diff:1252). Commit `656950e0` added it when `RADIX_THRESHOLD` was 16384; head commit `7cfd04a3`
set it to **22016** (persistent_topk.cuh, diff:26). All three widths are now ≤ 22016, i.e. all
single-CTA. The same bump also moved the whole `seq_len=20000` column off the cooperative path.
At head the multi-CTA path is exercised **only at seq_len=40000**; nothing tests 22016 < n < 40000.
A review remark, no GPU needed.

**(b) The new low-shared-memory fallback is untested and is GB10-only.** ← the case.

## 3. The case: `force_single_cta`, the <128 KiB cooperative-overflow fallback

`topk.cu` (diff:1028-1050) replaces this upstream branch

```
if (needs_cooperative && total_ctas > hw_resident_cap) {
  if (max_smem_per_block < 128*1024) { top_k_per_row_decode(...); return; }
```
with `force_single_cta = 1`, one CTA running `det_select_row` over the whole row. The kernel then
takes it at persistent_topk.cuh diff:511 with `params.det_smem_bytes` sized for only
`min(max_seq_len, RADIX_THRESHOLD)` = 22016 keys (diff:1001-1003), so `cached` (diff:109-110)
is **false** and the uncached global-rescan branch (diff:137/150/155/236/262) runs.

Why this is missing evidence, not a repeat:
- It is the one place the PR *removes* a call into `top_k_per_row_decode` — the kernel the author
  measured as deterministic on **0 of 56 shapes** and set-wrong on every tie-heavy shape
  (comment 5565253041). The PR's core claim ("No `atomicAdd` slot assignment remains on any
  path") rests on this branch behaving.
- It is **unreachable on H100/A100**: the guard is `max_smem_per_block < 128*1024`, so the parts
  he rented take the `FilteredTopKRaggedTransform` sibling instead. His three-architecture
  campaign structurally cannot cover it. GB10's 101,376 B opt-in is the qualifying condition.
- It is the only way on GB10 to execute uncached `det_select_row` at all (every routed
  single-CTA row is ≤ 22016 and therefore cached; the filtered path needs 128 KiB).
- Entry requires `ctas_per_group` ∈ [49, 64] (upper bound `kDetMaxCtasPerGroup = 64`,
  diff:302/983). With `num_rows ≤ 4`, `effective_max_smem = kSmemMedium = 35968`, so
  `max_chunk_elements ≈ (35968 − 2080 − static)/4 ≈ 7.2k–8.5k` and the window is roughly
  **n ∈ (343k, 458k]** (48 SMs, occupancy 1). No test or bench anywhere goes past 65,536.
- Side result: above the window the PR should raise `persistent_topk: ctas_per_group N exceeds 64`
  — a **new hard error where main silently degraded**.

**Invocation.** No existing node id covers it; in-tree it would be a new
`test_persistent_topk_low_smem_fallback`. Cheapest route is his own harness:
`python patches/kernel-det/build_det.py` (2 files, nvcc sm_121a, ~10 min), then the ≤30-line
script `/home/mark/shared/tmp-scratch/p4prime_lowsmem.py` (written alongside this report) as
`python p4prime_lowsmem.py build/_C_det.so`. It sweeps n ∈ {262144, 300000, 350000, 400000,
450000, 500000, 550000} × rows {1,4} × k {512,2048} × {random, tie-heavy, all-equal}, runs det
×6 and stock ×3 in one process, and prints det-reproducible / det-exact / stock-reproducible per
cell. The sweep self-calibrates the window, so the SM-count and static-smem estimates above do
not need to be right.

**Establishes:** whether the deterministic low-smem fallback is reachable, exact and
bit-reproducible on the only hardware class that can reach it, and where the 64-CTA cliff sits.
**Prereqs:** PR-head sources (mirrored in his repo), torch 2.13 / CUDA 13, nvcc; no FlashInfer, no
model, no serving, no vLLM build. **Runtime:** ~10 min build, ~15 min run.

## 4. Draft offer (118 words)

> Re the `max_smem_per_block < 128*1024` branch in `launch_persistent_topk`: on head that now sets
> `force_single_cta` and runs `det_select_row` uncached instead of calling `top_k_per_row_decode`.
> H100/A100 can't reach it (they take the FilteredTopK sibling), and nothing in the suite or in
> `test_det.py` goes wide enough to make `ctas_per_group` exceed the resident cap, so it looks
> untested. We have a GB10 idle and can sweep n ≈ 260k–550k at rows 1/4, k 512/2048, random /
> tie-heavy / all-equal, det ×6 vs stock ×3, using your `build_det.py`. Happy to post the table
> either way. Unrelated: `test_persistent_topk_path_transition` still brackets 16384 while
> `RADIX_THRESHOLD` is 22016.

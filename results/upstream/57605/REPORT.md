# Item M — vllm-project/vllm#57605, "kv-cache: honor scheduler lookahead in mamba align-mode allocation"

CPU only, no GPU used. GitHub read-only: nothing posted, nothing pushed.
Author: jsolman. PR head `ef97fad96ac4bcab3321933f1855755a61d38d09`, 1 file,
+76/-16, production code only (no test changes). Pre-run card:
`/home/mark/shared/tmp-scratch/M_57605/CARD.md` (written before any run).

## Verdict

**The diagnosed problem is real but narrow, and the patch as written does not
land safely.**

1. Mechanism (a) — the unmaterialised next page at an exact page boundary —
   **is reproducible at the merge base and is fixed at head**, but only when
   `MambaSpec.num_speculative_blocks == 0`. For the author's own configuration
   (MTP/Eagle, so `num_speculative_blocks = num_speculative_tokens > 0`) the
   next page column is already occupied by a speculative scratch block at the
   merge base, so the manager-level allocation is not what leaves that column
   empty. See D1/Q1.
2. Mechanism (b) — the displaced running-state column — **is not reachable at
   the merge base at all**: in align mode the merge base sets
   `num_tokens = num_tokens_main_model` unconditionally, so there is no
   inflated count for the null padding to be derived from. Mechanism (b) is a
   hazard the patch itself introduces; `_align_num_skipped_blocks` correctly
   neutralises it (proved by a guard-removal control). See D2.
3. The patch **breaks 18 of the 34 existing CPU tests** in
   `tests/v1/core/test_single_type_kv_cache_manager.py` and 11 more in two
   other core test files, all on one line
   (`vllm/v1/core/single_type_kv_cache_manager.py:1932`). Twelve of those
   failures need **no lookahead at all** — they come from the rewritten
   assertion dropping the merge base's `checkpoint_block` allowance. The
   remainder are the boundary-with-lookahead case the PR targets. See D3/D4.
4. Two further defects in the changed path: the admission estimate
   under-reserves by one block at exactly the boundary step (a saturated pool
   then raises `ValueError: Cannot get 2 free blocks from the pool`), and the
   allocator double-counts the reserved page. See D4/D5.
5. Even with the double-count removed, the reservation raises steady-state
   align-mode state residency above `MambaSpec.max_memory_usage_bytes`'
   documented `2 + num_speculative_blocks + num_prefill_checkpoint_blocks`
   page budget. See Q2.

No merge verdict is offered and no alternative design is proposed. The
ablation in §7 exists only to attribute the failures to specific lines.

## 1. Code paths cited

### Allocation and admission, merge base `63d9ad0a3a435cdf3a44495028b10f390a38f960`

`vllm/v1/core/single_type_kv_cache_manager.py`

| What | Line |
|---|---|
| `MambaManager.get_num_blocks_to_allocate` | 1710-1801 |
| align branch drops lookahead: `num_tokens = num_tokens_main_model` | 1752 |
| `physical_block_cap` clamp (from merged #57050) | 1796-1799 |
| `MambaManager.allocate_new_blocks` | 1806-1929 |
| align branch drops lookahead: `num_tokens = num_tokens_main_model` | 1824 |
| `num_skipped_blocks = num_required_blocks - num_speculative_blocks - 1` | 1858-1860 |
| null padding `null_end = num_skipped_blocks - checkpoint_block` | 1862-1867 |
| bound: `max_new_blocks = 1 + has_partial_hit + checkpoint_block`, `+= num_speculative_blocks` if `not blocks_allocated or checkpoint_block`; `assert num_new_blocks <= max_new_blocks` | 1882-1885 |

### Allocation and admission, PR head `ef97fad96a`

`vllm/v1/core/single_type_kv_cache_manager.py`

| What | Line |
|---|---|
| new `_num_tokens_with_lookahead` (returns `main + block_size` when lookahead ≥ 1 and `main % block_size == 0`) | 1710-1730 |
| new `_align_num_skipped_blocks` (clamps padding to `cdiv(main, bs) - 1`) | 1732-1749 |
| `get_num_blocks_to_allocate`, inflation call | 1788-1790 |
| `physical_block_cap` + the new `physical_block_cap += 1` | 1834-1841 (increment at 1840) |
| `allocate_new_blocks`, inflation call | 1866-1868 |
| `num_skipped_blocks` + clamp call | 1902-1906 |
| rewritten bound (**`checkpoint_block` no longer in it**) | 1929-1934 |
| `num_new_blocks += 1` and the two extra asserts | 1937-1945 |
| `self.block_pool.get_new_blocks(num_new_blocks)` | 1946 |

### Admission path that consumes both (identical at both refs)

- `vllm/v1/core/kv_cache_manager.py:528-530` — `num_tokens_main_model =
  total_computed_tokens + num_new_tokens`, `num_tokens_need_slot =
  min(num_tokens_main_model + num_lookahead_tokens, max_model_len)`.
- `vllm/v1/core/kv_cache_manager.py:548-557` — the estimate call.
- `vllm/v1/core/kv_cache_manager.py:561-565` — `required_blocks >
  available_blocks` → return `None` (request not scheduled).
- `vllm/v1/core/kv_cache_manager.py:580-585` — the actual allocation, same
  `(num_tokens_need_slot, num_tokens_main_model)` pair.
- `vllm/v1/core/kv_cache_manager.py:514-524` — the `full_sequence_must_fit`
  estimate with `apply_admission_cap=True`.
- `vllm/v1/core/kv_cache_coordinator.py:169-226` and `:277-307` — fan-out to
  each single-type manager.
- `vllm/v1/core/sched/scheduler.py:726-730` and `:1133-1136`, `:1167` — the two
  `allocate_slots` call sites that pass `num_lookahead_tokens`.
- `vllm/v1/core/sched/scheduler.py:2935-2943`, `:2955-2958` —
  `_request_remaining_blocks` / `_spec_decode_step_blocks`, which also consume
  the estimate.
- `vllm/v1/core/block_pool.py:672-673` — `get_new_blocks` raises
  `ValueError(f"Cannot get {num_blocks} free blocks from the pool")`. This is
  what an under-estimate turns into.

### Supporting invariants

- `vllm/v1/worker/mamba_utils.py:1487-1499` — the worker's running-state
  column: `num_blocks = cdiv(num_computed + num_scheduled, block_size) +
  num_speculative_blocks`, `curr_state_idx = num_blocks - 1 -
  num_speculative_blocks`, i.e. `cdiv(main_end, block_size) - 1`. Confirms the
  PR body's formula.
- `vllm/v1/kv_cache_interface.py:1043-1046` — align-mode memory budget
  `page_size_bytes * (2 + num_speculative_blocks +
  num_prefill_checkpoint_blocks)`.
- `vllm/v1/core/sched/scheduler.py:410-476` `_mamba_block_aligned_split` — in
  align mode **intermediate prefill chunk ends are forced onto page
  boundaries** (`end = aligned_end`, lines 474-476). This is why the
  boundary-with-lookahead case is the common path, not a corner case, in a
  chunked-prefill + spec-decode deployment.
- `vllm/config/vllm.py:622-646` — `num_lookahead_tokens` is
  `num_speculative_tokens` for eagle/draft-model, `+1` for DFlash, else 0.
- `vllm/model_executor/layers/mamba/abstract.py:78-83` —
  `num_speculative_blocks = 0 if cache_config.use_kda_recoverssm else
  vllm_config.num_speculative_tokens`.
- `vllm/config/vllm.py:3276-3300` — `use_kda_recoverssm = use_replayssm and
  num_speculative_tokens > 0`, Kimi-K3 KDA only, `mamba_cache_mode in ("none",
  "align")`. This is the only configuration found in which
  `num_speculative_blocks == 0` while `num_lookahead_tokens > 0`.

## 2. Test offered

`tests/v1/core/test_mamba_align_lookahead_allocation.py`
(copy at `/home/mark/shared/tmp-scratch/M_57605/tests/`). Model-free: only
`MambaManager` and `BlockPool`; `torch` is imported only for `MambaSpec`
dtypes. `pytestmark = pytest.mark.cpu_test`. 415 lines, 88 parametrised cases,
all lines ≤ 88 columns.

Fixture: `block_size=64`, `hash_block_size=16`,
`prefill_checkpoint_alignment=16`, boundary main end = 256 tokens (running-state
page column 3), `lookahead = 2`. Matrix as the brief specifies: main end at
boundary−1 / boundary / boundary+1 × lookahead ∈ {0, 2} ×
`num_speculative_blocks` ∈ {0, 1, 2} × internal checkpoints on/off, with both a
generous pool and a pool squeezed to exactly the admission estimate.

| Test | States |
|---|---|
| `test_align_lookahead_boundary_materializes_next_state_page` | mechanism (a): page column `cdiv(main_end, bs)` holds a real block at a boundary step with lookahead |
| `test_align_admission_estimate_pays_for_its_allocation` | estimate ≥ actual, estimate ≥ 0, table length within `[required, required+1]`, running-state column non-null, pool restored on `free()` |
| `test_align_tight_pool_admission_is_honoured` | same, but the pool is squeezed to the estimate, so an under-estimate surfaces as the real `ValueError` |
| `test_align_zero_lookahead_keeps_the_original_padding_math` | **negative control**: `num_tokens == num_tokens_main_model` behaves identically on every build (the PR's own claim) |
| `test_align_internal_checkpoint_chunk_stays_within_its_bound` | an internal checkpoint costs one extra real block; lookahead is 0 throughout |
| `test_align_lookahead_reserve_keeps_state_residency_bounded` | a block-aligned chunked prefill stays inside the `2 + num_speculative_blocks` page budget |
| `test_align_null_padding_must_not_land_on_the_running_state_column` | mechanism (b), **with its own sensitivity control**: neutralise `_align_num_skipped_blocks` and assert the column really does go null, so a pass cannot be vacuous; skipped on a build that has no such clamp |

## 3. Results

Every number below comes from an archived log under
`/home/mark/shared/tmp-scratch/M_57605/logs/`.

| # | Ref | Suite | Result | Log |
|---|---|---|---|---|
| R1 | merge base `63d9ad0a3a` | new test file | **1 failed**, 86 passed, 1 skipped | `newtest_base.log` |
| R2 | PR head `ef97fad96a` | new test file | **45 failed**, 43 passed | `newtest_head.log` |
| R3 | ablation C (§7) | new test file | 6 failed, 82 passed | `newtest_ablate.log` |
| R4 | merge base | existing `tests/v1/core/test_single_type_kv_cache_manager.py` | **34 passed** | `existing_suite_base.log` |
| R5 | PR head | same | **18 failed**, 16 passed | `existing_suite_head.log` |
| R6 | ablation C | same | 6 failed, 28 passed | `existing_suite_ablate.log` |
| R7 | merge base | `test_mamba_align_chunk_split.py` + `prefix_cache/` + `test_kv_cache_utils.py` | 238 passed, 2 failed, 2 errors (both environment, see §9) | `core_suite_base.log` |
| R8 | PR head | same | 227 passed, **13 failed** (11 new + the same 2 environment), 2 errors | `core_suite_head.log` |
| R9 | #57658 head `1f3fb826d8` alone | new + existing manager suite | **1 failed**, 120 passed, 1 skipped — same shape as the merge base | `pr57658_alone.log` |
| R10 | #57605 merged with #57658 (`729975d874`) | new + existing + `test_mamba_dynamic_k.py` | 63 failed, 61 passed — same failure set as #57605 alone; #57658's own new tests pass | `compose_57605_57658.log` |
| R11 | #57605 merged onto current `origin/main` `379e9a1ea8` (`a3d4bff06d`) | new + existing manager suite | 63 failed, 59 passed — identical to R2+R5, so not a stale-base artifact | `pr57605_on_main.log` |

Exploratory probes (not part of the offered test file):
`probe_lookahead.py` → `logs/probe_{base,head}.log`;
`probe_ladder.py` → `logs/ladder_{base,head}.log`;
`probe_guard.py` → `logs/guard_{base,head}.log`;
`probe_residency.py` → `logs/residency_{base,head,ablate}.log`.

### R1 breakdown (merge base)

- FAILED `test_align_lookahead_boundary_materializes_next_state_page[0]` —
  "main end 256 is page-aligned and 2 lookahead tokens were scheduled past it,
  but page column 4 is not a real block (table length 4)". This is the
  discriminating case: **mechanism (a)**, `num_speculative_blocks == 0`.
- The `[1]` and `[2]` cells pass at the merge base: with
  `num_speculative_blocks ≥ 1` the speculative scratch block already occupies
  column 4 (`logs/ladder_base.log`, `npg_blk = b2`).
- SKIPPED: the mechanism-(b) control, correctly — the merge base has no
  `_align_num_skipped_blocks` because it never inflates the token count.

### R2 breakdown (PR head)

- **PASSES** all three cells of
  `test_align_lookahead_boundary_materializes_next_state_page` — the PR does
  fix mechanism (a) at the manager level.
- **PASSES** `test_align_null_padding_must_not_land_on_the_running_state_column`
  *including* its control: with `_align_num_skipped_blocks` neutralised the
  running-state column becomes the null block
  (`logs/guard_head.log`: `table=['NULL','NULL','NULL','NULL',1,2]`, column 3
  null). The clamp is load-bearing and works.
- **PASSES** all 9 cells of the zero-lookahead negative control.
- FAILED 27 × `test_align_admission_estimate_pays_for_its_allocation`
- FAILED 9 × `test_align_internal_checkpoint_chunk_stays_within_its_bound`
  (lookahead 0 in every one of these)
- FAILED 6 × `test_align_lookahead_reserve_keeps_state_residency_bounded`
- FAILED 3 × `test_align_tight_pool_admission_is_honoured`

## 4. Defects

**D1 (defect, merge base, narrow). Mechanism (a) is real only when
`num_speculative_blocks == 0`.** At the merge base, with a main end exactly on a
page boundary and lookahead ≥ 1, `num_required_blocks = cdiv(main_end, bs) +
num_speculative_blocks`, so page column `cdiv(main_end, bs)` exists **iff**
`num_speculative_blocks ≥ 1`. Evidence: `logs/ladder_base.log`, `nsb=0 la=2`
row `main_end 256` → `npg ABSENT`, flagged `NEXT-PAGE-MISSING`; the `nsb=1` and
`nsb=2` tables have a real block there at every main end. The only
configuration found that reaches `num_speculative_blocks == 0` with
`num_lookahead_tokens > 0` is Kimi-K3 KDA RecoverSSM
(`vllm/model_executor/layers/mamba/abstract.py:78-83` with
`vllm/config/vllm.py:3276-3300`). The PR is fixed on head for all three values
of `num_speculative_blocks`.

**D2 (observation, not a merge-base defect). Mechanism (b) is not reachable at
the merge base.** In align mode the merge base sets `num_tokens =
num_tokens_main_model` at both `:1752` and `:1824`, so `num_skipped_blocks`
already equals `cdiv(main_end, bs) - 1` and can never bury the running-state
column. Mechanism (b) only exists once `_num_tokens_with_lookahead` inflates the
count; `_align_num_skipped_blocks` then removes it again. The PR body presents
(b) as a pre-existing failure mode; at the manager level it reads as a hazard of
the fix that the fix also handles. The offered test's guard-removal control
demonstrates both halves (`logs/guard_head.log`).

**D3 (defect, head). The rewritten assertion drops the merge base's
`checkpoint_block` allowance, and the case is reachable with no lookahead at
all.** Merge base `:1882-1885` allowed `1 + has_partial_hit + checkpoint_block`,
plus `num_speculative_blocks` when `not blocks_allocated or checkpoint_block`.
Head `:1929-1934` allows `1 + has_partial_hit` (if `blocks_allocated`) or
`num_speculative_blocks + 1 + has_partial_hit` (otherwise) — `checkpoint_block`
is gone from both. A chunk that exports an internal prefill checkpoint keeps
that column out of the null padding (`null_end = num_skipped_blocks -
checkpoint_block`, `:1909-1915`) and therefore needs one more new block, so the
bound trips. This is what fails `test_mamba_checkpoint_admission_matches_allocation`
(12 of the 18 in R5) and all 9 cells of
`test_align_internal_checkpoint_chunk_stays_within_its_bound`, with
`num_tokens == num_tokens_main_model` throughout. The brief asked not to
announce this without a reachable case; the reachable case is the fixture
already merged as part of #57050.

**D4 (defect, head). The boundary-with-lookahead case raises `AssertionError`
at `single_type_kv_cache_manager.py:1932` on a fresh or short block table.**
For a fresh request whose chunk ends on a page boundary with lookahead,
`num_required_blocks` is already inflated by one page, so `num_new_blocks =
num_speculative_blocks + 2`, one over the `num_speculative_blocks + 1` bound
that is evaluated *before* the `num_new_blocks += 1`. Because
`Scheduler._mamba_block_aligned_split` forces intermediate align-mode prefill
chunk ends onto page boundaries, this is the ordinary chunked-prefill path for a
mamba-hybrid + MTP deployment, not a corner case. All 11 head-only failures in
R8 (4 in `test_mamba_align_chunk_split.py`, 5 in
`prefix_cache/test_partial_prefix_cache_hits.py`, plus others) and the 6
`test_mamba_retirement_bounds_prefill_states` failures in R5 trace to this one
line — `grep` over `logs/core_suite_head.log` shows
`single_type_kv_cache_manager.py:1932: AssertionError` 11 times and no other
production-code frame.

**D5 (defect, head). The admission estimate under-reserves by one block at the
boundary step, and the allocator takes one block too many.** On a request that
already owns blocks, at a boundary main end with lookahead:

- the estimate returns 1 (`num_new_blocks = 1`, then
  `min(1, physical_block_cap)` where the cap is 2 — the new
  `physical_block_cap += 1` at `:1840` raises a ceiling that is not binding, so
  it has no effect in the case it was added for);
- `allocate_new_blocks` takes 2, because `num_required_blocks` already carries
  the reserved page *and* `:1938` adds another.

`logs/ladder_head.log`, `nsb=0 la=2`, `main_end 256`: `est 1 act 2 len 6 req 4`,
flagged `UNDER-ESTIMATE OVER-ALLOC(+2)`; every later step then reports
`est -1` (a **negative** admission estimate) and `OVER-ALLOC(+1)`. The merge
base reports `est == act` and no negative estimate anywhere
(`logs/ladder_base.log`).

With the pool squeezed to exactly what admission asked for — what a saturated
server does — this becomes a hard failure:

```
head : nsb=0 la=2: est=1 free_granted=1 -> ValueError: Cannot get 2 free blocks from the pool
head : nsb=1 la=2: est=1 free_granted=1 -> ValueError: Cannot get 2 free blocks from the pool
head : nsb=2 la=2: est=1 free_granted=1 -> ValueError: Cannot get 2 free blocks from the pool
base : all six (nsb × la) cases -> OK
```
(`logs/guard_base.log`, `logs/guard_head.log`, section "(1) tight pool sized to
the admission estimate".)

## 5. Questions

**Q1. Does mechanism (a) actually explain the observed corruption on a
`num_speculative_blocks > 0` deployment?** The PR's deployment is GLM-5.3-Flash
with MTP, so `num_speculative_blocks = num_speculative_tokens > 0` and page
column `cdiv(main_end, bs)` already holds a real, exclusively owned speculative
scratch block at the merge base. The PR body says that entry "still held a
recycled stale block" — that is a statement about the block's *contents*, not
about whether a block was allocated, and allocating one more page would not
change it. Is the real fault the pre-copy/relocation of the speculative scratch
block (`_relocate_speculative_block`, head `:1992-2001`; `remove_skipped_blocks`
head `:1645-1673`) rather than the block count? Not answerable here without the
worker.

**Q2. Should the reserved page be inside the align-mode memory budget?**
`MambaSpec.max_memory_usage_bytes` (`vllm/v1/kv_cache_interface.py:1043-1046`)
budgets align mode at `2 + num_speculative_blocks +
num_prefill_checkpoint_blocks` pages per request, and that is what sizes the
pool at profiling time. Replaying a block-aligned chunked prefill with lookahead
(`logs/residency_*.log`):

| build | 1-page chunks | 2-page chunks | 3-page chunks | budget (`nsb=5`) |
|---|---|---|---|---|
| merge base | 7 | 7 | 7 | 7 |
| PR head | `AssertionError` | `AssertionError` | `AssertionError` | 7 |
| ablation C | 8 | 9 | 9 | 7 |

So even after the double-count of D5 is removed, the reservation is +1 page for
single-page chunks and +2 for multi-page chunks, permanently. The existing
`test_mamba_retirement_bounds_prefill_states` encodes exactly this budget
("Five speculative blocks, current/previous states, and bounded in-flight
states") and is the suite that still fails under ablation C. Is the intent that
the budget grows, or that the reserved page be retired by
`remove_skipped_blocks`?

**Q3. Under #57658, is the reservation batch-size dependent?** #57658 makes the
lookahead the scheduler passes per-step
(`Scheduler._get_step_lookahead_tokens`), so whether `num_tokens !=
num_tokens_main_model` holds at a given boundary — and therefore whether the
next page is reserved — would depend on the batch size at that step. A
correctness property that only holds for some batch sizes is worth calling out.
See §6 for what #57658 does and does not change.

## 6. Composition

**With #57050 (`f30a195bbb15b920d9c2c40e6a3466d8961ab101`, merged 2026-09-16, in
the merge base).** #57050 replaced an unconditional overwrite of
`num_new_blocks` with `num_new_blocks = min(num_new_blocks,
physical_block_cap)` in the estimate and added
`test_mamba_checkpoint_admission_matches_allocation`. #57605 composes badly with
it in two ways:

1. #57605's `physical_block_cap += 1` (`:1840`) raises a *ceiling*. At the
   boundary step the pre-clamp `num_new_blocks` is 1 and the cap is 2, so the
   `min` is not binding and the increment changes nothing — the estimate stays
   1 while the allocator takes 2 (D5).
2. #57050's own new test is one of the 12 that #57605 breaks (D3).

**With #57658 (`1f3fb826d8bdcc2bb45a00421e7b51a330ba44f2`, open).**

- No textual conflict: `git merge-tree --write-tree pr-57605 pr-57658` succeeds
  (tree `adbfd33af1f`), and a real merge (`729975d874`) applies cleanly.
- #57658 threads a `num_spec_override` parameter through every manager, but in
  `MambaManager` it is consumed **only in the non-align branch**; the align
  branch still uses `self.num_speculative_blocks`. So `num_required_blocks` in
  align mode is unchanged by #57658.
- What #57658 does change for align mode is the *value* of
  `num_lookahead_tokens` passed into `allocate_slots` (`step_lookahead_tokens`,
  `scheduler.py` in that PR) — which is exactly the quantity #57605's
  `_num_tokens_with_lookahead` tests against. Hence Q3.
- Empirically the two do not interact in the manager: R9 shows #57658 alone
  behaves like the merge base (1 failure, the mechanism-(a) cell); R10 shows the
  merged pair reproduces #57605's failure set unchanged, and #57658's own
  `tests/v1/core/test_mamba_dynamic_k.py` passes in the merge. **#57658 does not
  change the expectations of the offered test.**

**With #58368 (`d5051abaf19a1cce19159756a79d27ccb0b2fccf`, merged 2026-09-24).**
In current `origin/main` but *not* in #57605's merge base — #57605 is behind
main. #58368 touches `MambaManager._cache_partial_tail_block` only
(`self.use_eagle` → `self.drop_eagle_checkpoint_block`), not the allocation
path. Merging #57605 onto current main (R11) gives byte-identical test results
to the PR head, so none of the findings are stale-base artifacts.

## 7. Ablation (attribution only, not a proposed patch)

To attribute the failures to specific lines, one 35-line ablation was applied on
top of the PR head in a separate worktree
(`/home/mark/shared/tmp-scratch/M_57605/ablation_C.patch`): drop the allocator's
second increment at `:1937-1945` (since `num_required_blocks` already carries the
reserved page), restore the merge base's bound including `checkpoint_block`, and
add `+1` to that bound for the reserved page. `physical_block_cap += 1` is kept.

| | new test file | existing manager suite |
|---|---|---|
| PR head | 45 failed / 43 passed | 18 failed / 16 passed |
| ablation C | 6 failed / 82 passed | 6 failed / 28 passed |

All remaining failures on both sides are the residency question Q2. An earlier
ablation that *also* removed `physical_block_cap += 1` ("ablation B",
`logs/ablationB.log`) left the under-estimate in place, which is how `:1840` was
shown to be necessary and `:1938` to be the double count.

## 8. Provenance

- vLLM clone `/home/mark/shared/vllm-head` (branch
  `fix/modelopt-lmhead-quant-gaps` @ `e3ca45951b`, untouched — never checked
  out, rebased or stashed).
- Worktrees created for this item and removed at the end:
  `wt-M-base` (`63d9ad0a3a`), `wt-M-head` (`ef97fad96a`), `wt-M-ablate`,
  `wt-M-compose` (`729975d874`), `wt-M-57658` (`1f3fb826d8`),
  `wt-M-onmain` (`a3d4bff06d`), all under
  `/home/mark/shared/tmp-scratch/`.
- Refs: PR head `ef97fad96ac4bcab3321933f1855755a61d38d09`; merge base
  `63d9ad0a3a435cdf3a44495028b10f390a38f960`; `origin/main`
  `379e9a1ea8a5995464d9bf775bcd36bb03a0995f` (fetched 2026-09-26);
  #57050 `f30a195bbb15b920d9c2c40e6a3466d8961ab101`;
  #57658 `1f3fb826d8bdcc2bb45a00421e7b51a330ba44f2`;
  #58368 `d5051abaf19a1cce19159756a79d27ccb0b2fccf`.
- Interpreter `/home/mark/shared/vllm-head/.venv/bin/python` → Python 3.12.3,
  `torch 2.13.0+cu132`, `pytest 9.1.1`. `PYTHONPATH` set to the worktree under
  test for every run; `vllm.__file__` verified to resolve into the worktree.
- Compiled ops: the 12 `*.so` in the clone (built 2026-08-24 from `23ab0cfdb`)
  were symlinked into each worktree. See Deviations 1.
- **No GPU was used.** The GPU lock `/home/mark/shared/exp54928/gpu.lock` was
  never taken, no `vllm serve` was started, no CUDA context was created by any
  run in this report.

## 9. Deviations and corrections

1. **Compiled extensions symlinked into each worktree.** Importing
   `tests/v1/core/test_mamba_align_chunk_split.py` and
   `tests/v1/core/prefix_cache/` pulls in `vllm.vllm_flash_attn`, which raises
   `ImportError: vllm.vllm_flash_attn requires the CUDA flash attention
   extensions` unless `_vllm_fa2_C`/`_vllm_fa3_C` sit next to the Python
   sources. The 12 `*.so` from the built clone were symlinked in (all
   `.gitignore`d, so the worktrees stay clean apart from the new test file and
   the ablation edit). Those binaries were built on 2026-08-24 from clone commit
   `23ab0cfdb`, **not** from the PR head; they are only imported, never called,
   by anything in this report. This is the compiled-op mismatch the common rules
   ask about: it does not block the paths used here, but it means nothing in
   this report is a statement about any C++/CUDA op at the PR head. The offered
   test file itself needs none of them.
2. **`ruff` is not installed in that venv** and nothing was installed into it
   (the clone is in use by another branch). The test file was checked by hand
   against the 88-column limit (`awk 'length>88'` → no hits), but
   `pre-commit` / `ruff format` has not been run over it.
3. **The first version of the mechanism-(b) control was wrong and was
   corrected.** It neutralised `_align_num_skipped_blocks` while driving a
   token-by-token decode ladder, in which the block table is already long
   enough that no null padding runs at all — so the control passed vacuously
   and the test reported "control did not reproduce mechanism (b)". Fixed by
   driving the control with a fresh boundary-aligned chunk, which is where the
   padding actually runs; the archived `newtest_*.log` are from the corrected
   version.
4. **The table-length invariant was relaxed once, deliberately.** The first
   version asserted `table_len == cdiv(main_end, bs) + num_speculative_blocks`,
   which the PR's own design violates by one (the reserved page). It was
   relaxed to `required <= table_len <= required + 1` so the test measures
   over-allocation rather than the design choice. Head still fails it (+2 at the
   boundary step); ablation C passes it.
5. **`test_kv_cache_utils.py` has 2 failures and 2 errors on this machine at
   both refs** — `ValidationError: World size (4) is larger than the number of
   available GPUs (1)` and a module-scoped server fixture. Environment, not the
   PR; excluded from all attribution above.
6. The brief expected a test that fails at the merge base and passes at head.
   One case does exactly that (`...materializes_next_state_page[0]`). The other
   82 parametrisations run the other way: they pass at the merge base and fail
   at head. That is reported as found.

## 10. What could not be established

- **Nothing about the GPU.** No state write, no pre-copy, no NaN, no repetition
  lock, no spec-decode acceptance was observed. Only the shape of the block
  table the worker *would* see was measured. The PR's 4-node Thor SM110 /
  GLM-5.3-Flash before-and-after numbers were not reproduced and could not be:
  no such hardware here and no GPU was used.
- **Whether the real `Scheduler` ever presents a given tuple.** The manager is
  driven directly. Reachability of each case is argued from code reading (§1),
  not demonstrated end to end. In particular the claim that align-mode
  intermediate prefill chunks end on page boundaries rests on
  `_mamba_block_aligned_split` (`scheduler.py:410-476`), not on a scheduler
  trace.
- **Whether `num_speculative_blocks == 0` with `num_lookahead_tokens > 0` is
  deployed by anyone.** The configuration is reachable
  (`use_kda_recoverssm` + a drafter) but nothing here says it is used.
- **Whether the block-table row width is exceeded.**
  `MambaSpec.max_num_blocks_per_req` returns `cdiv(max_len, block_size) +
  num_speculative_blocks` (`kv_cache_interface.py:1051-1059`), one short of the
  table length the head produces near `max_model_len`. Whether that overflows
  the worker's block-table row was not tested — it needs the worker.
- **The author's intent for `physical_block_cap += 1` and `num_new_blocks +=
  1`.** The ablation shows one is load-bearing and the other double-counts, but
  which the author meant is a question for the author.
- **No merge verdict, no review, no design proposal.** There is no maintainer
  review on #57605 at the time of writing (last update 2026-09-18T21:28Z).

## 11. Files

- Report: this file (copy in
  `docs/vllm-upstream/M_57605_REPORT_agent.md`).
- Pre-run card: `/home/mark/shared/tmp-scratch/M_57605/CARD.md`
- Test: `/home/mark/shared/tmp-scratch/M_57605/tests/test_mamba_align_lookahead_allocation.py`
  (placed at `tests/v1/core/` in each worktree for the runs)
- Probes: `/home/mark/shared/tmp-scratch/M_57605/probe_{lookahead,ladder,guard,residency}.py`
- Ablation: `/home/mark/shared/tmp-scratch/M_57605/ablation_C.patch`
- Logs: `/home/mark/shared/tmp-scratch/M_57605/logs/`
- Draft comment (not posted):
  `/home/mark/shared/tmp-scratch/M_57605/COMMENT_57605_DRAFT.md`
- Evidence tarball: `/home/mark/shared/tmp-scratch/M_57605_evidence.tar.gz`

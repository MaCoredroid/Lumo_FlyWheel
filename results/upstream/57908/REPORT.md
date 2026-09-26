# Item K — vllm-project/vllm#57908 "[Bugfix][Mamba] Preserve aligned prefix-cache snapshots with CoW"

**Target.** PR #57908 (cbertucci33, OPEN, no human review; the only "review" on the thread is
the fork-disabled Claude Code bot notice). Head `9317100d2f837c5584cbf8504ddbf4ddc688bac3`,
merge base `0ff0477ff955154f37db48791260cdf27fa35e60`, `main` as fetched
`379e9a1ea8a5995464d9bf775bcd36bb03a0995f`. 2 files, +179/-11.
CPU only, no GPU used, GitHub read-only.

---

## Verdict

**The PR's two new tests do discriminate (both FAIL at the merge base, both PASS at head), but
the premise they encode does not hold on the served path, and the change breaks seven
pre-existing CPU tests that currently pass on the merge base.**

1. **The aliasing is not reachable.** Both model runners already migrate the recurrent state
   out of the block-aligned source column into a freshly allocated write column *before* the
   forward pass. For a full block-aligned prefix-cache hit and for a producer continuing past
   a published boundary, the shipped pre-copy planner reports `src_col != dst_col`, i.e. the
   cached/published block is read-only for that forward. `src_col == dst_col` — the in-place
   case that genuinely needs CoW — happens only for a **sub-block** hit, which is exactly the
   scope of the pre-existing partial-hit CoW (`_has_partial_local_hit`,
   `single_type_kv_cache_manager.py:168-178`, same line numbers at head and at `main`).
2. **An existing test already asserts the opposite of what the PR does.**
   `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py:2224`
   (same line at head and at `main`; the assertion that breaks is at `:2264`) —
   `test_dcp_partial_hit_resumes_on_replicated_mamba_snapshot` — asserts that
   the full block-aligned mamba snapshot is **not** a CoW source while the DCP-partial
   full-attention block **is**. #57908 makes it one, and the test fails.
3. **Seven pre-existing `cpu_test` tests regress.** Over the four align-mode files: 267 passed
   at the merge base, 7 failed at head. Over all of `tests/v1/core`: exactly the same 7, and no
   others (11 failed at head vs 6 at the merge base; 4 of those failures are pre-existing/
   environmental and identical on both refs, and 2 of the merge-base failures are the PR's own
   new tests).
4. **Cost on a plain 64-token prefill** (mamba block 16, no cache hit, no connector): head
   queues one extra full recurrent-state device copy at every mamba block boundary the request
   continues past, and asks admission for one extra block at each of them. The block itself is
   transient — once the scheduler's CoW retention drain is modelled, both refs settle at the
   same 14/16 free blocks — so the durable costs are the copy and the higher peak/admission
   demand, not permanently held memory. *This corrects a first reading of the probe that did
   not model the drain; see §Results (4).*

No merge verdict is offered here; the items below are stated as defects, questions and
observations.

---

## Pre-run CARD

*(Honest note: written up alongside the report rather than before the first pytest invocation —
see Deviations.)*

**Under test.** (a) Do #57908's two new tests fail at the merge base and pass at head?
(b) On the served path, is the block-aligned prefix-cache snapshot actually overwritten in
place by the forward that resumes from it (consumer) or by the producer's next step?
(c) Does #57908 collide with #58434 / #57253 / #53798 / #55507?

**Pass/fail.** For (a): "discriminates" means both new tests FAIL at `0ff0477` with the PR's
test file dropped in unchanged, and PASS at `9317100`. For (b): the discriminator is the
state-block column pair the worker plans each step — `src_col == dst_col` means the kernels
read and write the same block (in place: CoW needed); `src_col != dst_col` means the state is
copied into the write block first and the source is read-only (no CoW needed). For (c):
textual = `git merge-tree` conflict; semantic = an invariant one PR relies on that the other
removes.

**What the evidence does NOT show.** No GPU ran. Nothing here executes a mamba/GDN kernel, so
this does not verify that the kernels actually confine their writes to `dst_col` — it verifies
which column the shipped planner *designates* as the write column and whether a copy is
scheduled. It does not cover `mamba_cache_mode` other than `"align"`, the KV-connector
offload path end-to-end, spec-decode with `num_speculative_blocks > 0` on the worker side
(manager-side budget only), or the V2 fused triton kernel executing (its column arithmetic is
read, not run — `mamba_utils.py:543`). It does not measure throughput or TPOT; the block and
copy counts below are manager-level counts, not a benchmark.

---

## Results

Every number comes from a log under `K_57908/logs/`.
Runner: `/home/mark/shared/vllm-head/.venv/bin/python -m pytest -p no:randomly`,
`PYTHONPATH=<worktree>`, `vllm.__file__` verified to resolve inside the worktree.

### (1) The PR's own two tests

| test | merge base `0ff0477` | PR head `9317100` |
|---|---|---|
| `test_mamba_aligned_producer_preserves_published_state` | **FAIL** (`assert 1 == 2`) | PASS |
| `test_mamba_aligned_hit_is_private_before_first_forward` | **FAIL** (`assert 2 == 3`) | PASS |
| whole `tests/v1/core/test_single_type_kv_cache_manager.py` | 2 failed, 34 passed | 36 passed |

Logs `10_prhead_two_new_tests.log`, `20_mergebase_two_new_tests.log`,
`11_prhead_full_file.log`, `21_mergebase_full_file.log`.

They discriminate. Note *where*: at the merge base both stop on the **first** assertion, the
`get_num_blocks_to_allocate` budget (1 vs 2, and 2 vs 3). The block-identity and
`take_pending_cow_copies()` assertions are never reached at the merge base, so the tests prove
that head reserves an extra block, and separately that head performs a CoW — they do not
establish that anything was aliased without it. Neither test has a negative control.

### (2) Regression sweep, same four CPU test files

| suite | merge base | PR head |
|---|---|---|
| `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` + `test_prefix_caching.py` + `test_mamba_align_chunk_split.py` + `test_deferred_block_free.py` | **267 passed**, 4 env errors¹ | **7 failed**, 260 passed, 4 env errors¹ |

¹ The 4 errors are identical on both refs and are environmental
(`OSError: The temporary directory /tmp/pytest-of-mark is a symbolic link`), not related to
the PR.

The whole of `tests/v1/core` was then run at both refs as a wider check
(logs `16_prhead_tests_v1_core.log`, `26_mergebase_tests_v1_core.log`):
**merge base 6 failed / 747 passed / 12 errors**, **head 11 failed / 742 passed / 12 errors**.
Set-differenced, the head-only failures are exactly the same 7 listed below and nothing else;
the 2 base-only failures are the PR's own new tests; the 4 failures common to both
(`test_kv_cache_utils.py::test_get_kv_cache_config_mamba_hybrid_sharing_pp_*`,
`test_reset_prefix_cache_e2e`, `test_scheduler.py::test_async_scheduling_pp_...`) and the 12
errors are pre-existing or environmental in this sandbox (pydantic `ParallelConfig`
validation, engine-core start-up, the `/tmp` symlink).

The 7 new failures at head (logs `12_prhead_regression_set.log` vs
`22_mergebase_regression_set.log`, details in `13_`–`15_`):

| test | assertion that breaks |
|---|---|
| `test_dcp_partial_hit_resumes_on_replicated_mamba_snapshot[2]` | mamba full-block snapshot must not be a copy source |
| `test_dcp_partial_hit_resumes_on_replicated_mamba_snapshot[4]` | same |
| `test_mamba_boundary_handoffs_do_not_pin_obsolete_blocks` | `all(block.ref_cnt == 0 for block in old_blocks[:-1])` |
| `test_fragmented_tail_chunk_does_not_poison_mamba_prefix_cache` | `assert 0 > 0` — no hashed boundary is left on the request's own blocks |
| `test_poisoning_is_block_size_independent[1536-30000-budgets0]` | same |
| `test_poisoning_is_block_size_independent[12288-30000-budgets1]` | same |
| `test_poisoning_is_block_size_independent[12288-41000-budgets2]` | same |

The last four fail because `move_block_hashes(source_block, cow_block)`
(`single_type_kv_cache_manager.py:1952` at head; the registration hooks are `:1717-1745` and `:2085-2101`) strips the hash off the request's own block,
so `_count_cached_boundary_states` finds nothing to check and returns 0.

### (3) Which column does the worker write? (the premise)

`probe_worker_precopy.py` drives the shipped MRV1 align pre-copy planner
`vllm.v1.worker.mamba_utils.preprocess_mamba` with `collect_mamba_copy_meta` and
`do_mamba_copy_block` stubbed, so the columns come from the shipped arithmetic
(`mamba_utils.py:1481` `prev_state_idx`, `:1498` `curr_state_idx`, `:1503` the copy gate) and
not from a transcription. Mamba block size 4, `num_speculative_blocks=0`. Identical output on
both refs (log `31_probe_worker_precopy.log`):

| scenario | planner result |
|---|---|
| **full aligned hit** 4 tokens, +1 / +2 / +4 new tokens | `src_col=0 → dst_col=1` — state MIGRATED before forward |
| **full aligned hit** 8 tokens, +4 | `src_col=1 → dst_col=2` — MIGRATED |
| **producer** past a published boundary, was at col 0, 4 computed, +1 / +4 | `src_col=0 → dst_col=1` — MIGRATED |
| **producer** was at col 1, 8 computed, +1 | `src_col=1 → dst_col=2` — MIGRATED |
| *control:* **sub-block hit** 2 tokens, +1 / +2; 3 tokens, +1 | `src_col == dst_col == 0` — written **IN PLACE** |
| *control:* **sub-block hit** 6 tokens, +1 / +2 | `src_col == dst_col == 1` — written **IN PLACE** |
| *control:* **producer mid-block**, 5 computed +1, 6 computed +2 | `src_col == dst_col == 1` — written **IN PLACE** (block carries no published hash yet) |

### (4) Block and copy budget, plain 64-token prefill, no cache hit

`probe_producer.py`, mamba block 16, 16-block pool, three steps of 16 tokens, run twice: once
without draining the CoW retentions (`DRAIN=0` — what the failing upstream test also does) and
once draining them the way `Scheduler._free_cow_retained_blocks` does with `defer_block_free`
off (`DRAIN=1`; `sched/scheduler.py:2699` head / `:2690` main). Log `30_probe_producer.log`:

| | merge base | PR head |
|---|---|---|
| queued `KVCacheBlockCopy` (both modes) | **0** | **2** |
| free blocks at the end, `DRAIN=0` | 14 / 16 | **10 / 16** |
| free blocks at the end, `DRAIN=1` | 14 / 16 | **14 / 16** |
| request's own boundary block after the next step | `(id 1, hash set)` | `(id 1, **hash cleared**)` |

**Correction to a first reading of this probe.** Without the drain the head column looks like a
2-blocks → 6-blocks regression and like a pinned obsolete block. Both disappear once the
retention drain is modelled: the retained source block comes back and the steady-state free
count matches the merge base. The extra block is therefore **transient** — one in flight per
boundary crossing, released immediately, or one step later when `defer_block_free` is on.

What does not disappear:

- the **extra device copy** per boundary crossing (2 vs 0 here) — a full recurrent-state copy
  across every mamba layer;
- the **extra block demanded from admission**, which is exactly what #57908's own test asserts
  (`get_num_blocks_to_allocate` 2 → 3) and which matters under block pressure;
- the **published hash leaving the request's own block** (`hash set` → `hash cleared`). That is
  a production behaviour change, not a harness artefact: it is what the four
  chunk-split/poisoning tests detect and what the #57253 question below rests on.

### (5) New discriminating test

`K_57908/tests/test_mamba_aligned_cow_premise.py`, 12 cases, run at three refs
(logs `50_newtest_mergebase.log`, `51_newtest_prhead.log`, `52_newtest_main.log`):

| # | case | merge base `0ff0477` | PR head `9317100` | `main` `379e9a1` |
|---|---|---|---|---|
| W1 ×3 | *control:* sub-block hit is written in place | PASS | PASS | PASS |
| W2 ×3 | full aligned hit is migrated before the forward | PASS | PASS | PASS |
| W3 ×2 | producer past a boundary is migrated | PASS | PASS | PASS |
| M1 | full aligned hit reserves no extra block | PASS | **FAIL** | PASS |
| M2 | full aligned hit queues no block copy | PASS | **FAIL** | PASS |
| M3 | *control:* sub-block hit still gets exactly 1 CoW block + 1 copy | PASS | PASS | PASS |
| M4 | published boundary hash stays on the producer's own block | PASS | **FAIL** | PASS |
| | totals | **12 passed** | **3 failed, 9 passed** | **12 passed** |

W1/W3-control and M3 are the negative controls: they show the harness does detect the
in-place case and that the pre-existing sub-block CoW path is untouched by the PR.

---

## Reachability trace, with file:line

Line numbers are given as `head` / `main` where they differ; where only one number appears it
is the same on both.

**(a) The manager does share the block.** `MambaManager.find_longest_cache_hit` returns the
cached block object itself for an aligned hit (`single_type_kv_cache_manager.py:1565` head /
`:1560` main),
and `SingleTypeKVCacheManager.add_local_computed_blocks` appends it straight into
`req_to_blocks` (`:313`). So at the manager level the resumed request and the prefix cache do
point at the same `KVCacheBlock`. The premise is correct this far.

**(b) The worker does not write it.**
*MRV1:* `GPUModelRunner` calls `mamba_utils.preprocess_mamba` at
`gpu_model_runner.py:4358` / `:4310`. Inside, a resumed request's read column is seeded
`prev_state_idx = (req_state.num_computed_tokens - 1) // block_size`
(`mamba_utils.py:1481`, `block_size = mamba_spec.block_size`), the write column is
`curr_state_idx = num_blocks - 1 - num_speculative_blocks` with
`num_blocks = cdiv(num_computed + num_scheduled, block_size) + num_speculative_blocks`
(`:1493-1498`), and a state migration is scheduled whenever they differ (`:1503`).
For a hit of `k*block_size` tokens and any `q >= 1` scheduled tokens,
`prev = k-1` and `curr = cdiv(k*block_size + q, block_size) - 1 = k`, so they always differ.
*MRV2:* `GPUModelRunner` calls `self.model_state.preprocess_state` at
`gpu/model_runner.py:1724` / `:1715`; `MambaHybridModelState.preprocess_state`
(`gpu/model_states/mamba_hybrid.py:183` head / `:179` main) launches
`preprocess_mamba_align_fused_kernel`, which stores `src_col = state_idx` and advances
`new_state_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1`
(`mamba_utils.py:536` and `:543`), then `precopy_mamba_align_fused_kernel` fast-exits only
`if src_col < 0 or src_col == dst_col` (`mamba_utils.py:607`) — the same rule. Its docstring
states the contract outright: *"Before the forward pass, copy each request's last SSM/conv
state from its previous block column into the new window block column, so the kernels read the
initial state from the write-side block as usual (V1 align semantics)"* (`:578-582`).

**(c) Which case really is in place.** `src_col == dst_col` requires the step not to cross a
mamba block boundary, i.e. the hit ended **inside** a block. That is precisely the condition
`_has_partial_local_hit` uses to arm the pre-existing CoW:
`len(new_computed_blocks) > 0 and num_local_computed_tokens % self.block_size != 0`
(`single_type_kv_cache_manager.py:168-178`), with the comment *"The local prefix-cache hit ends
inside one of this manager's blocks: the shared tail block needs CoW."* Measured in table (3).

**(d) The producer boundary.** `KVCacheManager.allocate_slots` caches optimistically, in
`schedule()`, before the step's forward (`kv_cache_manager.py:598-606`). So block `b`'s hash is
published in the same scheduling pass as the forward that fills it, and that forward is the
last one that writes `b` (`src_col == dst_col == b` only for that step; the next step advances
to `b+1`). The one genuinely exposed window — a *sibling* hitting `b`'s hash in the same step,
before the bytes exist — is already closed: `MambaManager.get_num_blocks_to_allocate` returns
`num_gpu_blocks + 1` for a hit whose hash is in `cached_blocks_this_step`, deferring that
request to the next step (`single_type_kv_cache_manager.py:1758-1766` head / `:1723-1731` main, comment *"Mamba can't
rely on blocks generated by other requests in the current step"*). #57908's producer CoW fires
one step **later** than that window, when the source block already holds the final state — so
the copy duplicates bytes that are already correct.

**(e) The scheduled copy is consumed before the forward.** For completeness, the plumbing the
PR relies on does work: `Scheduler.schedule` drains
`kv_cache_manager.take_kv_cache_block_copies()` (`sched/scheduler.py:1431` / `:1435`) into
`SchedulerOutput.kv_cache_block_copies` (`sched/output.py:298` head / `:305` main), and both
runners apply it in `_update_states` before the forward — `gpu_model_runner.py:1245-1250` /
`:1223-1228` and `gpu/model_runner.py:1201-1206` / `:1198-1203`, the V2 site commented *"after zeroing new
blocks and before the forward pass reads them"* — via
`copy_kv_cache_blocks_inplace` (`worker/utils.py:685`). So the PR's copies are real copies at
the right time; they are just copies of a snapshot nothing was going to overwrite.

**Premise rejected**, for both halves of the PR description ("a resumed request receives a
private mutable state before its first forward" and "a running producer keeps its append-only
state block while the published hash and bytes move").

---

## Collisions

| PR | status | textual | semantic |
|---|---|---|---|
| **#58434** njhill, *Treat padded prompt tails as spec-decode rows for hybrid models* | **MERGED** 2026-09-25, in `main`, **not** in #57908's base | none — touches only `gpu/model_states/mamba_hybrid.py` + a worker test; `git merge-tree` of #57908 against merge commit `7dbd0a8d` is clean | none found. Different layer (worker padding rows vs manager block ownership). |
| **#57253** sungsooha, *Keep cache-registered Mamba states out of retirement* | OPEN | **none** — both patches apply cleanly together; combined `tests/v1/core/test_single_type_kv_cache_manager.py` run is **37 passed** (log `40_interaction_57253.log`) | **Yes, a question for both authors.** #57253's protection keys on the request's own block still carrying a hash: `_is_prompt_boundary` returns `False` when `block.block_hash is None`, and `_remove_blocks_in_range` iterates `self.req_to_blocks[request_id]`. #57908's `move_block_hashes` clears exactly that hash (measured: `(id 1, hash set)` → `(id 1, hash cleared)` in log `30_probe_producer.log`). On the sparse-retention path the protection would then never fire. Not demonstrated end-to-end: my probe exercises the `last_state_block_idx` retirement branch (`single_type_kv_cache_manager.py:1668-1680` head / `:1663-1675` main), not `_remove_blocks_in_range`, so this is a code-reading claim, not a measurement. |
| **#53798** ptorsten / **#55507** Karl0007, align-seed cluster | both OPEN, duplicates of each other | none | Adjacent and worth naming. Both fix the **V2 seed divisor**: `MambaHybridModelState.add_request` seeds `_mamba_state_idx_gpu` with `// self.cache_config.block_size` (`mamba_hybrid.py:121` head / `:117` main) instead of the mamba block size. That is a real bug where the two sizes differ, but it does **not** create #57908's aliasing: the write column is recomputed from `num_computed` every step (`mamba_utils.py:543`), so a mis-seeded `src_col` makes the *pre-copy read* wrong, never the published block's contents. Fixing the seed is #53798/#55507's job; #57908 would not fix it and does not depend on it. The MRV1 path already uses `mamba_spec.block_size` (`mamba_utils.py:1481`) and is unaffected. |

`git merge-tree --write-tree` of #57908 against current `origin/main` is also clean, so the PR
still rebases.

---

## Defects / questions / observations

**Defect 1 (introduced).** Seven pre-existing `cpu_test` tests that pass on the merge base fail
at head, including one — `test_dcp_partial_hit_resumes_on_replicated_mamba_snapshot`,
`tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py:2224`, assertion at `:2264` — whose explicit
assertion is that the full block-aligned mamba snapshot must **not** be a CoW source. The PR
updates neither these tests nor their rationale. (Evidence: §Results (2).)
Weighted honestly: the two DCP cases and the four chunk-split/poisoning cases reflect real
behaviour changes (the aligned snapshot becomes a copy source; the published hash leaves the
request's own block). `test_mamba_boundary_handoffs_do_not_pin_obsolete_blocks` is the weakest
of the seven — it fails on `ref_cnt`, and the scheduler drains that retention in production
(§Results (4)) — but it is still a red job the PR does not address.

**Defect 2 (introduced).** One extra full recurrent-state device copy, and one extra block
demanded from admission, at every mamba block boundary a request continues past — on the
ordinary prefill path, with no cache hit and no connector. The block is returned once the
scheduler drains the CoW retention, so this is peak/admission demand and copy bandwidth, not
permanently held memory. For Kimi-K3-scale mamba blocks (12288 tokens, the whole recurrent
state per block) a 41 000-token prompt adds three such copies. Not benchmarked here.
(Evidence: §Results (4).)

**Question 1.** Is there any served configuration in which the align pre-copy does *not* run
before the forward, so that `src_col == dst_col` on a full block-aligned hit? Everything I can
drive on CPU says no, for both runners. If the author has a trace where it does, that would
change the answer and is the single thing worth asking for.

**Question 2 (to both #57908 and #57253).** See the #57253 row above: does moving the published
hash off the request's own block defeat `_is_prompt_boundary`?

**Observation 1.** `take_boundary_state_offloads` still offers the request's own block while
the cache entry moves to the copy (log `30_probe_producer.log`, `DRAIN=0`: offer block 3, cache
entry block 4). The bytes are identical and the offer is drained the same step, so I found no defect
here, but the offered block and the hashed block are no longer the same object at head, which
the PR's "connector hand-off behavior unchanged" claim does not mention.

**Observation 2.** The PR is authored with AI assistance (the repo's agent notice applies) and
the author reports lint only; no CI has been run on it — `/ci run` requires a maintainer or the
`ready` label.

---

## Provenance

- vLLM clone `/home/mark/shared/vllm-head` (untouched, left on `fix/modelopt-lmhead-quant-gaps @ e3ca45951b`).
- Worktrees created and removed at the end: `wt-K-head` (`9317100d2f83`), `wt-K-base`
  (`0ff0477ff955`), `wt-K-main` (`379e9a1ea8a5`), `wt-K-both` (`9317100` + #57253 applied),
  `wt-K-only57253` (`0ff0477` + #57253 applied).
- Python `/home/mark/shared/vllm-head/.venv/bin/python` = Python 3.12.3; torch `2.13.0+cu132`;
  compiled ops from the venv build `0.26.1rc1.dev1159+g23ab0cfdb` (built 2026-08-24 from
  `23ab0cfdb`), `*.so` symlinked into each worktree. `vllm.__file__` verified to resolve inside
  the worktree under test.
- Host `gx10-edb9`, Linux 6.14.0-1015-nvidia, aarch64. **No GPU used**, `gpu.lock` not taken.
- Other PRs fetched read-only: `pull/57253/head`, `pull/55507/head`, `pull/53798/head`.
- Nothing posted, pushed, or commented on GitHub.
- Logs: `K_57908/logs/00_provenance.log` and `10_`–`52_`.

---

## Deviations

1. The **pre-run CARD was written up with the report**, not before the first pytest
   invocation: the brief's step (1) was to run the PR's two tests at both refs, which I did
   first. The card's "what the evidence does not show" section was written before the new test
   in §Results (5) was designed, and nothing in it was relaxed afterwards.
2. `tests/v1/core/test_single_type_kv_cache_manager.py` in the merge-base worktree is the PR's
   version of that file, copied in so the two new tests exist there. This is why the merge-base
   full-file run shows 2 failures; every other file in the merge-base worktree is stock
   `0ff0477`.
3. The 4 `OSError: /tmp/pytest-of-mark is a symbolic link` errors in
   `test_prefix_caching.py::test_hisparse_async_admission_requires_only_import_destinations`
   are environmental, identical on both refs, and were not worked around.
4. The worker-column evidence uses `preprocess_mamba` with two collaborators stubbed
   (`collect_mamba_copy_meta`, `do_mamba_copy_block`), driven by `SimpleNamespace` stand-ins for
   `SchedulerOutput` / `InputBatch` / `CachedRequestState`. The index arithmetic under test is
   the shipped code; the surrounding objects are not.
5. **Correction made during the work.** The first reading of `probe_producer.py` did not model
   `Scheduler._free_cow_retained_blocks`, and briefly supported a "2 blocks → 6 blocks, obsolete
   blocks pinned" claim. Re-running the probe with the drain modelled showed both refs settle at
   14/16 free blocks. The verdict, §Results (4) and Defect 2 were rewritten accordingly; the
   superseded numbers are kept visible in the `DRAIN=0` row rather than deleted.
6. `tests/v1/core` in full was run at both refs as a broader check
   (`16_prhead_tests_v1_core.log` / `26_mergebase_tests_v1_core.log`, 9m07s and 7m00s); it
   confirms the four-file sweep and adds no head-only failure beyond the seven.

## What could not be established

- Whether the mamba/GDN/KDA **kernels** confine their writes to `dst_col` at runtime. No GPU
  was used; this report reasons about the column the planner designates, not about the stores
  the kernel issues. A GPU run on a hybrid model with `--mamba-cache-mode align` and prefix
  caching, checking that the cached boundary block's bytes are unchanged across the resumed
  request's first forward, would settle Question 1 either way.
- Any end-to-end connector behaviour (offload/onboard) at head.
- Whether the #57253 interaction actually bites on the sparse-retention path (code reading
  only, see the collisions table).
- Throughput / memory impact of the extra block and copy; only counts, no benchmark.

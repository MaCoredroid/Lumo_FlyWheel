# Item N — porting the P9 aligned-state-indices test to vllm `main`

**Verdict: the brief's premise is wrong, and the port is feasible in a reduced
form.** Today's `main` (`379e9a1ea8`, fetched 2026-09-26) carries **none** of
the three things the brief asked about: no request-slot `idx_mapping` in
`compute_aligned_state_indices`, therefore no padding clamp, and the
capture-time binding is still plain first-writer-wins. njhill's #58434 and
#58462 did not touch that path. Four of our nine P9 cases have nothing to
assert on `main`, one (C2) asserts the fix, and one (C8) pins the very
lifecycle #55506 changes. The remaining three, plus one new case, port to a
**296-line, single-file, test-only** change that passes on `main`, still
passes with #55506 applied on top, and fails under three deliberate
mutations. **Recommended route: file it against `main` as its own PR, as an
addition to the existing `tests/v1/worker/test_mamba_utils.py`, and leave the
mapping-specific cases (C2/C3/C5/C6 + Karl0007's updated C8) inside #55506.**

---

## 1. What `main` actually carries

All citations are `origin/main` at
`379e9a1ea8a5995464d9bf775bcd36bb03a0995f` (2026-09-26), confirmed both by
reading and by executing (`logs/00_provenance.log` prints the live
signatures out of the worktree).

| question from the brief | answer on `main` | evidence |
|---|---|---|
| does `compute_aligned_state_indices` take `idx_mapping`? | **No.** `def compute_aligned_state_indices(self, seq_lens, num_reqs)` | `vllm/v1/worker/mamba_utils.py:1092-1096`; runtime signature in `logs/00_provenance.log` |
| does `MambaHybridModelState.prepare_attn` pass one? | **No.** `ctx.compute_aligned_state_indices(input_batch.seq_lens, num_reqs)` | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:312-314` |
| does the kernel load a mapping at all? | **No.** No `idx_mapping_ptr` / `num_mapping_rows` parameter; `do_not_specialize=["num_requests"]` only; the table row is the batch row | `vllm/v1/worker/mamba_utils.py:38-53`, `:57`, `:79` |
| is the padding clamp present? | **Not applicable** — there is no mapping load to clamp | same |
| is the capture-time binding temporary? | **No**, first-writer-wins: `_ensure_align_ctx` returns a bare `MambaSpecDecodeGPUContext`, no `bind` argument, no `temporary` flag | `mamba_hybrid.py:137-177` (`if not ctx.is_initialized:` at `:166`, `return ctx` at `:177`); runtime return annotation in `logs/00_provenance.log` |

Two further pre-fix markers, for completeness: `bt_row_idx = batch_idx if
HAS_IDX_MAPPING else req_idx` (`mamba_utils.py:479`) and the pre-copy kernel's
`_copy_mamba_state_block(state_idx, batch_idx, ...)` (`mamba_utils.py:610`)
are both still in the batch-row form #55506's first commit rewrites.

### Where the scans went wrong

Both merges are real and both are on `main`, but neither goes near this path:

- **#58434** `7dbd0a8d22` (2026-09-24) changes exactly one expression in
  `prepare_attn` — `is_decode = (~is_prefilling_np | is_prompt_tail) & ...` —
  plus an 80-line test. It never touches `_align_mode`, `_ensure_align_ctx` or
  `compute_aligned_state_indices`.
- **#58462** `310f15d354` (2026-09-23) is a one-line dtype change to the
  **dummy** `idx_mapping` in `vllm/v1/worker/gpu/input_batch.py` (int64 →
  int32).

`main` does carry `idx_mapping` in several *neighbouring* mamba kernels
(`preprocess_mamba_align_fused_kernel`, `run_fused_precopy`,
`postprocess_mamba_fused_kernel`, `_scatter_num_accepted_kernel`), all
long-standing. A grep for `idx_mapping` in `mamba_utils.py` therefore hits 20
lines and none of them is the aligned-index path. That is the most likely
source of the misreading, and it is worth saying out loud before anyone
repeats it.

### No latent C6-style defect on `main`

Under `CUDAGraphMode.FULL`, `prepare_attn` still passes
`num_reqs = input_batch.num_reqs_after_padding` (`mamba_hybrid.py:243`), so
padded rows *are* inside the launch. They are harmless here:
`gather_block_tables` launches over `num_reqs_padded` specifically to "fuse
zeroing of padded rows" and returns `bt[:num_reqs_padded]` views of the
persistent `input_block_tables` (`vllm/v1/worker/gpu/block_table.py:164-175`).
The rows are inside the allocation, and they read as block id `0`, the
reserved null block. The out-of-bounds read item I found exists only once
`a28e902` adds the mapping load. **Observation, not a defect.**

### Comparison with #55506

`5f71d6fac1` (head, unchanged since 2026-09-24; still open). It adds all
three things: `idx_mapping` + `num_mapping_rows` on the kernel and the
wrapper, the `safe_rows` clamp plus the `tl.where(rows < num_mapping_rows,
table_row, 0)` fallback, and `_ensure_align_ctx(..., bind=False) -> (ctx,
temporary)` with `prepare_attn` releasing the temporary binding in a
`finally`. Merge base is still `528fa835fd`, i.e. the PR has not been rebased;
it nevertheless **cherry-picks onto today's `main` with no conflicts** (three
commits, auto-merge only — `logs/00_provenance.log`, tail).

---

## 2. Which of the nine P9 cases survive the port

| # | P9 case | status on `main` | disposition |
|---|---|---|---|
| C1 | identity mapping == unmapped launch | the `_TAKES_IDX_MAPPING` shim collapses both sides to the same unmapped call — the comparison is vacuous | **ported**, keeping the CPU-oracle half, as `test_matches_cpu_oracle` |
| C2 | permuted mapping resolves source-slot rows | **fails**: `main` indexes by batch row | **not ported.** It asserts the fix. Asserting `main`'s current behaviour instead would turn red the day #55506 lands |
| C2b | self-guard that the two indexings differ | CPU-only, passes anywhere, meaningless without C2 | not ported |
| C3 | `-1` sentinel masked off | skipped (no mapping parameter) | #55506-specific |
| C4 | eager == capture+replay, mapping **and** lengths mutated | the mapping half is vacuous; the lengths half applies | **ported** as `test_cuda_graph_replay_matches_eager` (lengths only) |
| C5 | replay ignores a reallocated mapping tensor | skipped | #55506-specific |
| C6 | padded rows must not read past `idx_mapping` | skipped; and no such load exists | #55506-specific — reused below as the clamp control |
| C7 | binding is first-writer-wins | **passes** | **ported** as `test_block_table_binding_is_first_writer_wins` |
| C8 | `_ensure_align_ctx` keeps the capture-time tables | **passes as written** (`main` returns a bare ctx) | **not ported.** It pins the lifecycle #55506 deliberately changes; Karl0007's updated C8 is the mirror image and fails on `main` |
| — | *(new)* rows past `num_reqs` untouched, `num_reqs == 0` writes nothing | — | **added** as `test_rows_past_num_reqs_are_untouched`: the invariant #55506's clamp has to preserve |

**Karl0007's updated C8 is not needed for a `main` PR** — it belongs to
#55506, where it passes (row R6 below). Nothing in the ported set needs it.

The selection rule was: *only invariants that hold both before and after
#55506*. That is what makes the PR safe to land next to an open fix, and it
is also what the PR cannot claim — see §5.

---

## 3. Results

Branch `p9-main-aligned-state-indices`, commit `bc414ae723`, off
`379e9a1ea8`. One file, `tests/v1/worker/test_mamba_utils.py`, +296 lines, no
deletions, **not pushed**. Every number below comes from a log under
`N_p9main/logs/`.

| # | run | build | result | log |
|---|---|---|---|---|
| R1 | the 4 new cases | `main` + test | **4 passed** (1.4 s) | `10_main_new_cases.log` |
| R2 | the whole file (regression guard) | `main` + test | **47 passed** (8.3 s) — 43 pre-existing + 4 | `11_main_whole_file.log` |
| NC1 | `first_state_slot` loses its `-1` | `main` + temp patch | **2 failed** (oracle, replay), 2 passed | `21_control_nc1.log`, `patches/nc1.diff` |
| NC2 | store mask widened to `num_requests + 2` | `main` + temp patch | **1 failed** ("the launch wrote rows past num_reqs"), 3 passed | `22_control_nc2.log`, `patches/nc2.diff` |
| NC3 | `is_initialized` guard removed | `main` + temp patch | **1 failed** ("a second initialize_from_forward_context rebound the tables"), 3 passed | `23_control_nc3.log`, `patches/nc3.diff` |
| R3 | the 4 new cases | `main` + #55506 | **4 passed** | `40_main_plus_55506.log` |
| R4 | the whole file | `main` + #55506 | **47 passed** | `40_main_plus_55506.log` |
| R5 | P9's original 9-case file | `main` + #55506 | **8 passed, 1 failed** — the old C8, `AttributeError: 'tuple' object has no attribute 'block_table_ptrs'`, by design | `40_main_plus_55506.log` |
| R6 | Karl0007's updated C8 file | `main` + #55506 | **9 passed** | `40_main_plus_55506.log` |
| NC4 | #55506 with the `idx_mapping` clamp reverted | `main` + #55506 − clamp | **C6 fails in both files** ("cudagraph-padding rows took their table row from memory past the end of idx_mapping"); the 4 new cases are **unaffected** (4 passed) | `41_clamp_reverted.log`, `patches/nc4_clamp_reverted.diff` |
| R7 | memcheck of the 4 new cases | `main` + test | **`ERROR SUMMARY: 0 errors`**, 4 passed | `50_memcheck_new_cases.log` |

Each mutation fails **only** the case aimed at it, so the four cases are
independent rather than one assertion in four costumes.

**On the brief's requested control.** "Run against `main` with the padding
clamp deliberately reverted" is not executable: `main` has no clamp (§1). NC4
is the nearest true equivalent — the clamp reverted where it exists, on top of
today's `main` — and it reproduces item I2's finding unchanged against a
`main` that is nine days newer. Note what NC4 also shows: the four new cases
are *blind* to the clamp. They are coverage for the kernel `main` has, not a
guard for #55506's defect.

`ruff 0.14.0` (the version pinned in `.pre-commit-config.yaml`), run via
`uvx` so nothing was installed into the busy clone's venv:
`ruff check` → *All checks passed*; `ruff format --check` → *1 file already
formatted*. No line exceeds 88 columns.

---

## 4. Overlap with #55506, and where the file should live

**File-level conflict: none, in either direction.** #55506 touches
`tests/kernels/mamba/test_precopy_mamba_align.py`, `model_runner.py`,
`mamba_hybrid.py` and `mamba_utils.py` — not
`tests/v1/worker/test_mamba_utils.py`. The clean cherry-pick plus R3/R4 are
the empirical proof: both changes coexist and all 47 tests pass together.

**That is a consequence of the placement choice, not luck.** The obvious port
— keep P9's filename `tests/v1/worker/test_mamba_aligned_state_indices.py` —
is the one to avoid. Karl0007's branch already carries
`tests/v1/worker/test_mamba_aligned_state_indices_fixed.py`; if he moves it
into #55506, upstream ends up with two nearly identical files whose names
differ by a suffix, and a reviewer has to work out which expectations are
live. Adding a `TestAlignedStateIndicesKernel` class to the file that already
tests this class instead:

- adds **no new file** and no new import (everything it needs —
  `_TestConfig`, `_MockCpuGpuBuffer`, `_make_mock_attention`, `_make_gpu_ctx`,
  `_COPY_FUNCS`, `_reinterpret_u64_as_i64`, `MambaSpecDecodeGPUContext` — is
  already imported there);
- sits beside `TestPostprocessMambaFusedKernel`, which covers the sibling
  kernel in the same class, in the same style;
- reads naturally against njhill's cleanup request on #55506.

The cost is a 2772 → 3068-line file and a slightly higher chance of a textual
merge conflict with unrelated work in it. If a maintainer prefers a new file,
the block moves verbatim; that is a one-line decision, not a rewrite.

### Own PR, or an addition to #55506?

**Own PR against `main`.** Reasons, in order:

1. **It tests something that exists today and is untested today.** Nothing
   under `tests/` calls `compute_aligned_state_indices` — verified by
   `git grep` over `origin/main`; the only other mention of
   `mamba_aligned_state_indices` in the tree is
   `tests/models/kimi_k3/test_kda_metadata.py`, which assigns a
   precomputed tensor to the builder and never runs the kernel. The value
   does not depend on #55506 landing.
2. **It is neutral on the open question.** The four cases pass on `main` and
   on `main`+#55506 (R1, R3). They take no position on batch-row versus
   request-slot indexing, so they cannot become an argument in that review.
3. **#55506 is asking for less, not more.** njhill has asked for cleanup on a
   3-commit, 4-file PR with an open correctness question. Adding 300 lines of
   test that are not about the fix works against that.
4. **The mapping-specific cases go the other way.** C2/C3/C5/C6 and
   Karl0007's updated C8 assert #55506's behaviour and can only live in
   #55506; R6 shows they pass there (9/9), and NC4 shows C6 still
   discriminates on top of today's `main`. That is the diff worth offering to
   Karl0007 — as a comment on #55506, not as our own PR.

The two are complements: ours guards the kernel across the refactor, his
proves the refactor.

---

## 5. What this evidence does not show

1. **Nothing about a real Kimi-K3 / KDA model.** No weights, no second box.
   The consumers of `mamba_aligned_state_indices` (`kda_metadata.py:385`) are
   not exercised; neither is PP, async scheduling, nor end-to-end output.
2. **Nothing about the indexing question #55506 raises.** Deliberately: the
   cases use identity order only. They would pass unchanged if the batch-row
   indexing on `main` were wrong in production, which is exactly what
   tomasruizt reproduced and what #55506 fixes. This PR is coverage, not a
   verdict on that.
3. **Nothing about vLLM C++/CUDA ops.** The unit is Triton + PyTorch. The 17
   `*.so` in the worktree are symlinks to a 2026-08-24 build from
   `23ab0cfdb`, imported and never called (deviation 1).
4. **Nothing about CI on other hardware.** One GB10, sm_121, aarch64, one
   Triton (3.7.1), one torch (2.13.0+cu132). `BLOCK_ROWS=32` means every row
   count we test lives in a single program; a multi-program grid
   (`num_reqs > 32`) is not covered.
5. **Whether maintainers want this at all.** No one has asked for it. Filing
   it is a bet that untested-kernel coverage is welcome; the reviewer may
   reasonably say "fold it into #55506" and that would not be a wrong call.

---

## 6. Provenance

- Box: NVIDIA GB10, sm_121, aarch64, driver 590.48.01, Linux 6.14.0-1015-nvidia.
  All GPU work under `flock /home/mark/shared/exp54928/gpu.lock`. No model
  loaded; peak allocation a few hundred KB; total GPU time well under two
  minutes across every run above.
- `origin/main` `379e9a1ea8a5995464d9bf775bcd36bb03a0995f`, fetched
  2026-09-26 (`[Security] Harden message sanitization (#58832)`).
- #55506 head `5f71d6fac1af38ab89d2d261e6f57d1e88c00090`, merge base
  `528fa835fd37d0805a824d516e22f88df61a771a`, still open.
- P9 `9cef61298fb6680fbd330be87d4dc6b1033e45f5`
  (MaCoredroid/vllm:p9-mamba-aligned-state-indices); Karl0007's updated C8
  `083a3f637ac6a4dc871fd0479ed17ca861607b94`.
- Local branch `p9-main-aligned-state-indices` @ `bc414ae723`, created in
  worktree `/home/mark/shared/tmp-scratch/wt-N`. The worktree has been
  removed; the branch itself lives in `/home/mark/shared/vllm-head`'s refs and
  is ready to push if Mark says GO. The patch is also archived at
  `N_p9main/0001-test-aligned-mamba-state-index-kernel.patch`. The scratch
  branch that cherry-picked #55506 for R3-R6/NC4 was deleted after the runs.
  **Nothing was pushed and nothing was posted to GitHub.**
- Interpreter `/home/mark/shared/vllm-head/.venv/bin/python` 3.12.3, torch
  2.13.0+cu132, triton 3.7.1, editable install
  `vllm-0.26.1rc1.dev1159+g23ab0cfdb.precompiled`; `PYTHONPATH` wins over its
  meta-path finder, verified by `vllm.__file__ =
  /home/mark/shared/tmp-scratch/wt-N/vllm/__init__.py`
  (`logs/00_provenance.log`). The busy clone was never checked out, rebased or
  stashed.
- Sign-off: `Signed-off-by: mark ma <coredroid0401@gmail.com>` plus
  `Co-Authored-By: Claude Opus 5 (1M context)` and the session trailer, per
  the P8 convention.

## 7. Deviations and corrections

1. **Compiled extensions symlinked.** 17 `*.so` from the built clone are
   symlinked into the worktree (all `.gitignore`d, worktree stays clean).
   Built 2026-08-24 from `23ab0cfdb`, not from `main`. Needed because
   `test_mamba_utils.py` imports `MambaHybridModelState`, which pulls in
   `vllm.vllm_flash_attn`. Same deviation as item I.
2. **`ruff` is still not in the clone's venv**, and nothing was installed into
   it. Unlike item I, lint *was* run this time, via
   `uvx ruff@0.14.0` in a throwaway environment, at the version
   `.pre-commit-config.yaml` pins. The rest of `pre-commit` (typos, mypy,
   etc.) was not run.
3. **One self-inflicted failure, found and fixed before the commit.** The
   first version of the port failed `test_matches_cpu_oracle` and
   `test_cuda_graph_replay_matches_eager` with two wrong elements out of 36.
   Cause: `_make_aligned_ctx(cfg, _make_aligned_block_tables(...), device)`
   let the synthetic block tables be garbage-collected the moment the
   statement ended, while the context held only their raw `data_ptr`s; the
   allocator handed that memory to the next `seq_lens` tensor, and the value
   `16` appeared where block id `2` belonged. **This is the identical trap
   item I hit in its standalone probe** (I's deviation 3) — which is why the
   committed helper now *returns* the tables alongside the context and says
   why in its docstring. Worth flagging to anyone writing the next test
   against this class.
4. **The brief's negative control was not executable as written**; NC4 is the
   substitute, and NC1-NC3 carry the discrimination requirement for the cases
   that actually ship. See §3.
5. **`main`'s state contradicts the two scans** that motivated the item. §1.
   No blame attaches to the runs; it is a reading error about which kernel the
   `idx_mapping` hits, and it is easy to make.

---

## 8. Draft PR (if Mark says GO) — 214 words

**Title:** `[Test] Cover the aligned mamba state-index kernel`

**Body:**

> `get_aligned_state_indices_multi_group_kernel` and its wrapper
> `MambaSpecDecodeGPUContext.compute_aligned_state_indices` have no direct
> test: nothing under `tests/` calls `compute_aligned_state_indices`. Its
> output reaches the KDA / Kimi-K3 metadata builders as physical mamba
> state-block ids, so a wrong row is a wrong recurrent state.
>
> Four cases, added to the file that already covers this class, model-free and
> a few hundred KB of GPU memory. The context is built through the real
> `create` / `initialize_from_forward_context` path over synthetic block
> tables whose ids are unique per (group, row, column), so a wrong row or
> column is visible, checked bytewise against a CPU oracle:
>
> - the index arithmetic for lengths on both sides of a block boundary and
>   below it, across two mamba groups in one launch;
> - `num_reqs` bounds the launch: the returned prefix view comes out of a
>   persistent buffer whose later rows must keep their previous contents, and
>   `num_reqs == 0` writes nothing;
> - a captured launch replayed after `seq_lens` is rewritten in place matches
>   an eager launch;
> - `initialize_from_forward_context` captures the block tables' `data_ptr`s
>   once and is idempotent, so the first caller decides which tables every
>   later launch reads.
>
> Each case fails under a matching one-line mutation of the kernel or the
> binding guard. Test-only; no production change.
>
> Written with AI assistance and verified on an NVIDIA GB10 (compute-sanitizer
> clean).

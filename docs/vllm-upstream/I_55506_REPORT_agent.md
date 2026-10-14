# Item I — execution coverage for vllm-project/vllm#55506

**Target.** PR #55506 "[Bugfix] Index mamba spec-decode block tables by request slot, not
batch row" (Karl0007, OPEN, no human review approvals as of 2026-09-23).
Head `a28e90223540e861af5ef6e3a2e1c7ea010d45b1`, merge base
`528fa835fd37d0805a824d516e22f88df61a771a`.

**Scope.** The gap the PR itself names: the second commit `a28e902` ("Resolve aligned state
indices through the request-slot mapping"), added after izhuhaoran asked whether the K3/KDA
`compute_aligned_state_indices()` path stays correct "including CUDA graph initialization".
The author verified it **statically only** — he has no KDA/Kimi hardware. tomasruizt's H100 PP
reproduction of the first commit was **not** redone.

---

## Verdict

**The mapping change is correct for every real batch row we could drive, and it is
discriminating: the same expectations fail on the merge base and on the PR's first commit
alone. Two defects remain, both in the parts the author could not execute.**

1. **Confirmed defect, introduced by `a28e902`.** Under `CUDAGraphMode.FULL` the kernel reads
   `idx_mapping` **past the end of the tensor** for cudagraph-padding rows.
   `MambaHybridModelState.prepare_attn` passes `num_reqs = input_batch.num_reqs_after_padding`
   (mamba_hybrid.py:245, :302-303) while `input_batch.idx_mapping` is only
   `num_reqs` long (model_runner.py:1280, :1335). The new load at mamba_utils.py:66 is masked
   with `rows < num_requests`, i.e. with the **padded** count. compute-sanitizer memcheck:
   `Invalid __global__ read of size 8 bytes at get_aligned_state_indices_multi_group_kernel+0x160
   in mamba_utils.py:66 ... 1 bytes after the nearest allocation of size 24 bytes`, three
   threads, and the launch then failed with `cudaErrorLaunchFailure`.
   The bad value becomes a table row, so the padded rows of
   `builder.mamba_aligned_state_indices` carry block ids of an arbitrary request slot.
   Before the PR those rows resolved to block id `0` (the gathered tables' padded rows are
   zeroed by `gather_block_tables`), i.e. the reserved null block.
2. **Pre-existing hazard that `c99ba59` turned into a correctness requirement, on the same K3
   path.** `MambaSpecDecodeGPUContext.initialize_from_forward_context` is idempotent, so the
   FIRST caller decides which block tables every later launch reads. At CUDA-graph capture
   `cudagraph_utils.py:785,:824` calls `model_state.prepare_attn` with
   `block_tables.get_dummy_block_tables(...)` — slices of the **gathered**
   `BlockTables.input_block_tables`. For a model with an aligned-index builder (Kimi K3 / KDA)
   that reaches `_ensure_align_ctx` and binds the context **before any real batch**, and
   `preprocess_state` — the only caller that passes the source per-request-slot tables
   (model_runner.py:1723) — is skipped on every dummy run. The author flagged the ordering
   himself as "hardening"; after his binding change it is not hardening, it is the difference
   between reading request slots and reading batch rows.

Both are single-path, not a reason to hold the fix: #55506 is a clear improvement over `main`
for the GDN/Qwen3-Next case tomasruizt measured (no aligned-index builder there, so neither
defect applies). Both are worth fixing inside this PR because both live in the code it adds.

---

## Results

`tests/v1/worker/test_mamba_aligned_state_indices.py`, 9 cases, same file run against three
builds. Every number below comes from a log under `I_55506/logs/`.

| # | case | merge base `528fa83` | +commit 1 `c99ba59` | PR head `a28e902` |
|---|---|---|---|---|
| C1 | identity mapping == unmapped launch (control) | PASS | PASS | PASS |
| C2 | permuted mapping resolves source-slot rows | **FAIL** | **FAIL** | PASS |
| C2b | the two indexings really differ for this input (test self-guard) | PASS | PASS | PASS |
| C3 | `-1` sentinel rows masked off, no negative table row | skip¹ | skip¹ | PASS |
| C4 | eager == CUDA-graph capture+replay, inputs mutated in place | **FAIL** | **FAIL** | PASS |
| C5 | replay ignores a reallocated mapping tensor (pointer is baked) | skip¹ | skip¹ | PASS |
| C6 | padded rows must not read past `idx_mapping` | skip¹ | skip¹ | **FAIL** |
| C7 | context binding is first-writer-wins | PASS | PASS | PASS |
| C8 | `_ensure_align_ctx` keeps the capture-time (gathered) tables | PASS | PASS | PASS |
| | totals | 2 failed, 4 passed, 3 skipped | 2 failed, 4 passed, 3 skipped | 1 failed, 8 passed |

¹ skipped because `compute_aligned_state_indices` has no `idx_mapping` parameter on those
builds; the case has nothing to assert there.

Logs: `logs/20_mergebase_528fa83.log`, `logs/30_commit1_c99ba59.log`,
`logs/10_prhead_a28e902.log`, `logs/11_prhead_padded_detail.log`, `logs/40_probe_track.log`,
`logs/50_memcheck.log`, `logs/00_provenance.log`.

### C2, the negative control, with both outputs

Source per-slot tables, block id `1 + group*10000 + slot*100 + col`; batch of 5 with
`idx_mapping = [5, 0, 7, 2, 1]`, `seq_lens = [17, 1, 48, 33, 80]`, mamba block size 16,
3 state slots. Group 0 rows (group 1 is the same + 10000):

| batch row | slot | first slot | pre-fix (table row = batch row) | PR head (table row = `idx_mapping[row]`) |
|---|---|---|---|---|
| 0 | 5 | 1 | `2, 3, 4` | `502, 503, 504` |
| 1 | 0 | 0 | `101, 102, 103` | `1, 2, 3` |
| 2 | 7 | 2 | `203, 204, 205` | `703, 704, 705` |
| 3 | 2 | 2 | `303, 304, 305` | `203, 204, 205` |
| 4 | 1 | 4 | `405, 406, 407` | `105, 106, 107` |

Every row differs. The PR-head column is the oracle; the pre-fix column is what the merge base
and `c99ba59` produce (log `20_mergebase_528fa83.log`, `30_commit1_c99ba59.log`).

### C6, the defect, with both outputs

`num_reqs = 3`, `num_reqs_after_padding = 6`, `idx_mapping = [2, 0, 5]` (3 elements), the value
sitting immediately past it set to slot `p`:

| p | padded rows 3..5, group 0 | block ids of slot p |
|---|---|---|
| 7 | `[701,702,703] x3` | `701, 702, 703` |
| 3 | `[301,302,303] x3` | `301, 302, 303` |
| 1 | `[101,102,103] x3` | `101, 102, 103` |

The padded rows track whatever follows the tensor, exactly (log `40_probe_track.log`). Real
rows 0..2 stay correct (`202,203,204 / 3,4,5 / 503,504,505`). With the mapping given its own
exact-size allocation, compute-sanitizer reports the read as out of bounds
(log `50_memcheck.log`).

**Suggested fix**, for whoever takes it: mask the `idx_mapping` load with the *real* request
count rather than `num_requests`, or have `prepare_attn` pass a mapping padded to
`num_reqs_after_padding`. Masking alone is not enough: a masked-off row leaves the persistent
`aligned_state_indices` row at its **previous step's** value, whereas before this PR padded rows
were written with `0`, the reserved null block. Padded rows should end up at the null block.

---

## What the test covers, and what it does not

Covers, by execution:

- The changed index expression at mamba_utils.py:92, over a permuted batch, with per-row
  disjoint block ids so any wrong row is visible; bytewise against a CPU oracle.
- The `-1` sentinel path, including that it forms no negative table row.
- Eager versus a captured-and-replayed launch of the same kernel, with the mapping, the
  sequence lengths and the block-table contents all mutated in place between capture and
  replay.
- The capture-time binding order of the context, driven through the real
  `MambaHybridModelState._ensure_align_ctx`.
- The exact `(num_reqs, len(idx_mapping))` pair the runner produces under FULL cudagraphs.

Does **not** cover, and the eager-vs-captured equality specifically does not prove:

1. **That the production path is capture-safe.** In the V2 runner this kernel is launched from
   `prepare_attn`, which runs **outside** the captured region. C4 is a property test of the
   kernel, not of the runner. C5 exists to pin the reason it must stay outside: the mapping
   tensor is a fresh `async_tensor_h2d` allocation every step, so a captured launch would read
   a stale address.
2. **That the context holds the right pointers.** Eager and replay read the same
   `block_table_ptrs`, so a mis-binding is invisible to that comparison. Binding is covered
   separately, by C7/C8, which are lifecycle assertions and not numeric ones.
3. **Anything downstream of the produced index tensor.** The KDA kernels that consume
   `mamba_aligned_state_indices` (kda_metadata.py:385) are not exercised. The consequence
   claimed above for defect 1 — garbage state-block indices for padding rows reaching those
   kernels — follows from reading the consumer, and was not measured.
4. **PP, async scheduling, or the end-to-end corruption.** Already reproduced by tomasruizt on
   2xH100; deliberately out of scope here.
5. **The first commit's copy kernels.** Covered by the PR's own
   `tests/kernels/mamba/test_precopy_mamba_align.py`.
6. **Defect 2 end to end.** C8 proves the lifecycle property (first writer wins, and
   `_ensure_align_ctx` will not rebind) on the real method. That capture is in fact the first
   writer for a K3/KDA model is established by reading `cudagraph_utils.py:785,:824` and
   `model_runner.py:1721-1726`, not by running a K3 model — we have no KDA weights and no
   second box.

---

## Provenance

- Box: NVIDIA GB10, sm_121, aarch64, driver 590.48.01, Linux 6.14.0-1015-nvidia.
  All GPU work under `flock /home/mark/shared/exp54928/gpu.lock`. No model was loaded; peak
  allocation is a few hundred KB. Total GPU time: under two minutes across all runs.
- Sources: git worktrees of `/home/mark/shared/vllm-head` at the three commits, under
  `/home/mark/shared/tmp-scratch/wt-55506{,-base,-c1}`, selected by `PYTHONPATH`. The busy
  clone was never checked out, rebased or stashed.
- Interpreter: `/home/mark/shared/vllm-head/.venv/bin/python` (3.12.3, torch 2.13.0+cu132).
  The editable install (`vllm-0.26.1rc1.dev1159+g23ab0cfdb.precompiled`) appends its finder to
  `sys.meta_path`, so `PYTHONPATH` wins; verified by printing
  `vllm.__file__ = /home/mark/shared/tmp-scratch/wt-55506/vllm/__init__.py`
  (`logs/00_provenance.log`).
- The unit under test is Triton + PyTorch only. No vLLM C++/CUDA op is called.
- Local branch `p9-mamba-aligned-state-indices` (commit `f517270a6e`) on top of the PR head
  carries the test. Nothing was pushed and nothing was posted to GitHub.

## Deviations and corrections

1. **Compiled extensions symlinked into the worktrees.** Importing
   `MambaHybridModelState` (needed for C8) pulls in `vllm.vllm_flash_attn`, which raises
   `ImportError` unless `_vllm_fa2_C`/`_vllm_fa3_C` are present next to the Python sources. The
   17 `*.so` from the built clone were symlinked into each worktree (all `.gitignore`d, so the
   worktrees stay clean). Those binaries were built on 2026-08-24 from clone commit
   `23ab0cfdb`, **not** from the PR head; they are only imported, never called, by these tests.
   This is the compiled-op mismatch the brief asked about: it does not block the test, but it
   means the run is not a statement about any C++/CUDA op at the PR head.
2. **`ruff` is not installed in that venv** and nothing was installed into it (the clone is in
   use). The test file was checked by hand against the 88-column limit and has no long lines,
   but `pre-commit`/`ruff format` has not been run over it.
3. **One self-inflicted error, corrected.** The first version of the standalone probe let the
   synthetic block tables be garbage-collected while the context still held their raw
   `data_ptr`s; the allocator handed the memory to the `seq_lens` tensor and one row of the
   output read back `48, 0, 0`. That was a bug in the probe, not in the kernel. Fixed by
   keeping the tables alive; the archived `logs/40_probe_track.log` is from the corrected run.
   The pytest cases were never affected (their tables stay in scope).
4. **The test file carries a two-line signature shim** (`_TAKES_IDX_MAPPING`) so the identical
   expectations can run against a pre-fix build. Upstream can delete it once this lands; it is
   marked as such in the file.
5. The negative control was run on the merge base as the brief specified **and** additionally
   on the PR's first commit `c99ba59` alone, which is the build where the defect the second
   commit fixes actually exists. Both fail identically.

## Files

- Test: `tests/v1/worker/test_mamba_aligned_state_indices.py` (branch
  `p9-mamba-aligned-state-indices`), copy at `I_55506/tests/`.
- Probe: `I_55506/probe_padded_oob.py` (not part of the offered test file).
- Pre-run card: `I_55506/CARD.md`. Logs: `I_55506/logs/`.
- Draft PR comment: `I_55506/COMMENT_55506_DRAFT.md` (not posted).
- Evidence tarball: `I_55506_evidence.tar.gz`.

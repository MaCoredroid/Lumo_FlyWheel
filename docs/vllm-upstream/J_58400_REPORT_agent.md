# Item J — recurrent-state oracle for vllm-project/vllm#58400

**Target.** PR #58400 "[Perf][MRV2] Allow FULL decode graphs for one-token prompt tails"
(njhill, OPEN, no human review as of 2026-09-26; `mergeable_state: blocked`, CI #91329 in
flight). Head `d5a8e22778175b5e732c8718b82f9a27a6cbbbc8`, merge base
`4ccfe1239843998f9b3e109f159278dc0fcbf753`. 144+/42- over 12 files, 4 of them tests.

**Scope.** The gap the brief named: the PR's own hybrid checks assert metadata counts and
the benchmarks assert throughput and gsm8k, so neither establishes the **state invariant** —
that the conv/SSM state committed after a FULL-path step equals eager execution's, with
rejected placeholder slots not leaking into it. A model-free oracle was built for exactly
that and run at both refs.

---

## Verdict

**No defect. The state invariant holds at the PR head, by execution, for the batch class
the PR newly admits to the FULL decode graph — and the reason it holds is that the PR
moves no state at all. It moves reachability.**

Diffing the produced metadata field by field at the two refs (probe, below), for the same
batch, the *only* differences are:

| field | merge base `4ccfe12` | PR head `d5a8e22` |
|---|---|---|
| `BatchReqState.decode_graph_eligible` | `False` | `True` |
| `get_uniform_decode_token_count(...)` | `None` | `4` |
| `MambaHybridAttnMetadata.is_prefilling` (tail row) | `True` | `False` |

Every state index the GDN builder produces is byte-identical across the two refs:
`spec_state_indices_tensor`, `non_spec_state_indices_tensor`, `prefill_state_indices`,
`has_initial_state`, `spec_sequence_masks`, `num_decode_draft_tokens_cpu`,
`num_accepted_tokens` in and out, and the counts `num_spec_decodes / num_decodes /
num_prefills / num_prefill_tokens`. That third row changes nothing downstream: the spec
branch of `GDNAttentionMetadataBuilder.build` never reads `is_prefilling`
(gdn_attn.py:290-400 — it is read only in the non-spec branch at :262-268), and the Mamba2
builder already clears the same rows itself (mamba_attn.py:493-516, `prefill_to_decode`
plus the `padded_prompt_tail_rows` term added by the merged #58434). So the PR's
"`is_prefilling` becomes the single source of truth" refactor is behaviour-preserving on
both builders for this row, which is what the PR body claims and what this measures.

Three things came out that are worth the author's attention; none blocks the PR. They are
labelled question / observation below, and one is a coverage gap in the tree rather than
in the PR.

---

## Results

`tests/v1/worker/test_mamba_hybrid_prompt_tail_state.py`, 8 cases, the same file run
against both builds. Every number below comes from a log under `J_58400/logs/`.

| # | case | merge base `4ccfe12` | PR head `d5a8e22` |
|---|---|---|---|
| A0 | the FULL decode graph is reachable for a padded prompt tail | **FAIL** | PASS |
| A | committed state on the FULL path == eager, unpadded | PASS | PASS |
| B | poisoned placeholder slots never reach the committed state | PASS | PASS |
| C | fresh one-token prompt control: not reclassified, state from scratch | PASS | PASS |
| D | cudagraph padding rows commit nothing, even with non-null table rows | PASS | PASS |
| E | one-token mid-prompt chunk is decode-like (predicate difference) | **FAIL** | PASS |
| F | the commit depends on the placeholders being rejected | PASS | PASS |
| NC | negative control: committing the placeholders must fail the test | PASS | PASS |
| | totals | 2 failed, 6 passed | **8 passed** |

Logs: `logs/10_prhead_d5a8e22.log`, `logs/20_mergebase_4ccfe12.log`,
`logs/30_negative_control.log`, `logs/probe_head.json`, `logs/probe_base.json`,
`logs/00_provenance.log`.

### What the two merge-base failures are, honestly

`A0` fails at the merge base because `has_prefill` there sends the whole batch to
PIECEWISE, so the FULL path this PR opens is not addressable: that is the PR's premise,
not a bug on `main`.

`E` fails at the merge base because the two predicates are not the same set. Head tests
"exactly one **new** prompt token scheduled" (`num_scheduled - num_drafts == 1 &
num_computed_prefill > 0`, model_runner.py:1257-1261); #58434's merge-base version tests
"exactly one prompt token **remains**" (`prefill_len - num_computed == 1`,
mamba_hybrid.py at the base). Head's is the strict superset; the extra members are
one-token *mid*-prompt chunks. Reading `Scheduler.schedule`, the placeholder padding is
gated on `num_new_tokens == 1` where `num_new_tokens = request.num_tokens -
num_computed_tokens` is the **total** remaining, computed before any chunking
(scheduler.py:1088, :1094-1113, emitted at :1321-1325, with `assert num_new_tokens == 1 +
num_spec_tokens`). So the scheduler never pads a mid-prompt chunk, the extra members
always carry zero drafts, and both builders already route a stateful one-token chunk to
the decode path regardless. **`E` is a predicate difference with no reachable state
difference**, and is marked as such in the test file.

**So the byte assertions (A, B, C, D, F, NC) do not discriminate between the two refs.**
They are discriminated by the negative control instead, which is why it is in the file.
Stating it plainly: this run shows the invariant *holds* on the newly reachable path; it
does not show the PR *fixed* a state bug, because there was none to fix in this batch
class — #58434 had already fixed the one that existed.

### The committed bytes, both arms (head)

Request `tail`: 129-token prompt, 128 tokens already computed, so one real prompt token,
padded with K=3 placeholder drafts. Request `d0`: an ordinary K-draft verify decode, 2
tokens accepted. State slot ids are `1 + req_slot*4 + column`.

| | FULL arm (padded, `CUDAGraphMode.FULL`, 2 real + 2 padding rows) | eager arm (unpadded, `CUDAGraphMode.NONE`) |
|---|---|---|
| `tail` routed as | spec-decode row, `spec_state_indices = [17,18,19,20]` | reclassified prefill, `prefill_state_indices = [17]`, `has_initial_state=[True]` |
| committed slot (`num_accepted-1`) | 17 | 17 |
| committed recurrent state | `[2, 129, 0]` | `[2, 129, 0]` |
| committed conv window | `[20127, 20128, 20129]` | `[20127, 20128, 20129]` |
| placeholder-derived state | slots 18/19/20 = `[2,130,1] [2,131,1] [2,132,1]` | not written |
| `d0` committed slot / state | 6 / `[1, 42, 0]` | 6 / `[1, 42, 0]` |

The state triple is `[request tag, tokens consumed, poison flag]`; `999999` is a
placeholder token tag. 129 is the full real prompt; 132 = 129+K is the
placeholder-folded state. `d0`'s committed column is 1, not 0 — the readout tracks the
accepted count rather than assuming a column.

### The negative control, verbatim (`logs/30_negative_control.log`)

Routing the same padded tail through the *prefill* write contract — one final state
written back, which is what mamba_hybrid.py:262-265 says the prefill kernels do ("the
prefill kernels can't roll them back"):

```
broken committed ssm : [2, 132, 1]
broken committed conv: [999999, 999999, 999999]
eager  committed ssm : [2, 129, 0]
eager  committed conv: [20127, 20128, 20129]
negative control raised AssertionError:
[2, 132, 1] != [2, 129, 0]
```

The negative control re-runs the *same* helper that cases A and B assert with, inside
`pytest.raises(AssertionError)`, so the two cannot drift apart.

---

## Question and observations

**Question (1).** The invariant rests on `num_accepted == 1` for a padded prompt tail, and
nothing in the worker enforces it: `postprocess_state` writes whatever `num_sampled` says
into `num_accepted_tokens_gpu` (mamba_hybrid.py:350-366), and the next step's committed
column is `spec_state_indices[row, num_accepted - 1]`
(fused_recurrent.py:106-110). For a real tail that is safe only because the `-1`
placeholders can never be sampled. Under `rejection_sample_method="synthetic"` acceptance
is drawn from `synthetic_acceptance_rates` and, as documented at speculative.py:529-535,
does not look at the drafted token id. Case F pins the consequence by execution: with an
accepted placeholder the committed state becomes `[2, 130, 1]` and the conv window
contains `999999`. Worth asking njhill whether that is considered out of contract for a
benchmarking-only sampler, or whether a padded row should be forced to `num_accepted = 1`.

**Observation (2).** After the FULL step the tail's *own* conv slot carries
placeholder-derived values outside the live window — slot 17 is
`[20127, 20128, 20129, 999999, 999999, 999999]` on the FULL arm versus
`[20127, 20128, 20129, -, -, -]` on the eager arm. The committed bytes agree because the
live window is `buf[num_accepted-1 : +width-1]` and `num_accepted == 1`
(causal_conv1d.py:860-876, :1219-1222). The two arms' *buffers* are not identical, only
their windows. Anything that later reads that slot at a base above 0 without refreshing it
would read placeholder data; nothing in the current code does.

**Observation (3), a gap in the tree rather than in the PR.** These tests run with
`mamba_cache_mode="none"`. The mode resolved when prefix caching is on — i.e. the "full
prefix-cache hit" that produces the tail in the first place — is `"align"`, and with it the
V2 commit boundary is `MambaSpecDecodeGPUContext.run_fused_postprocess_align`
(mamba_utils.py:1234), called from `MambaHybridModelState.postprocess_state`
(mamba_hybrid.py:383-394) with `HAS_IDX_MAPPING=True, PRECOMPUTED_NEW_COMPUTED=True`. A
full sweep of `tests/` found **no test that reaches that function**: the goldened kernel
tests cover the V1 `run_fused_postprocess` (tests/v1/worker/test_mamba_utils.py:794-2765)
and the align pre-copy (tests/kernels/mamba/test_precopy_mamba_align.py:155-225), and the
one V2 state-byte test (tests/v1/e2e/general/test_mamba_prefix_cache.py:1221) is
`enforce_eager=True` and checks only the align block-migration copies of the temporal
tensor. Not this PR's debt — but it is the other half of this PR's scenario.

**Observation (4), adjacent, read only, not executed.** In `Scheduler.schedule`,
`long_prefill_token_threshold` is applied at scheduler.py:1114-1115 *after*
`pad_spec_decode = True` is set at :1113, and unlike the mamba split at :1141-1152 it does
not clear the flag. With `0 < long_prefill_token_threshold < 1 + num_spec_tokens` the
padded width would be shortened and the `assert num_new_tokens == 1 + self.num_spec_tokens`
at :1322 would fire. This is #45237's code, untouched by #58400, and no run here exercises
it; mentioned only because it sits on the same padding path.

---

## What the test covers, and what it does not

Covers, by execution:

- The real `GPUModelRunner.gather_batch_req_state`, including `prefill_runs_as_decode_np`,
  `decode_graph_eligible` and the `get_uniform_decode_token_count` gate, over a batch built
  from request state the way `execute_model` builds it.
- The real `MambaHybridModelState.prepare_attn` at `CUDAGraphMode.FULL` with
  `num_reqs_after_padding` / `num_tokens_after_padding` actually larger than the real
  counts, and at `CUDAGraphMode.NONE`.
- The real `GDNAttentionMetadataBuilder.build` on that metadata, including its FULL-graph
  padding fill (`spec_state_indices_tensor[num_spec_decodes:] = NULL_BLOCK_ID`).
- The real `MambaHybridModelState.postprocess_state` acceptance commit, and the committed
  column it selects at the next step.
- Byte equality, with no tolerance, of the committed recurrent state and the committed conv
  window between the FULL and eager arms, for the tail and for the other request in the
  batch.

Does **not** cover:

1. **Any real kernel.** No model is loaded and no mamba/GDN/conv Triton or CUDA kernel is
   executed. `_run_synthetic_mixer` reimplements the three write contracts, each cited in
   the file to the source line it was read from (fused_recurrent.py:103-163;
   causal_conv1d.py:860-950, :1219-1222; qwen_gdn_linear_attn.py:1336-1348, :1368-1378,
   :1488, :1500-1523). If a kernel's actual write contract differs from those lines, this
   test is wrong in the same way. That is the load-bearing assumption of the whole exercise.
2. **A real CUDA graph.** The FULL arm is the FULL-mode *metadata* path, not a captured and
   replayed graph. Baked pointers, stale-address and capture-order hazards are out of scope
   — and note that `prepare_attn` runs outside any captured region.
3. **`mamba_cache_mode="align"`**, and therefore `run_fused_postprocess_align`,
   `preprocess_state`'s pre-copy, and the block-migration semantics. See observation 3.
4. **Mamba2 / short-conv / KDA builders.** Only GDN was driven. The Mamba2 path was read
   (mamba_attn.py:485-516) and reasoned about, not executed.
5. **PCP, PP, DBO, adaptive verification, ubatch slicing.** The PR touches
   `pcp_manager.partition_batch` and `_slice_input_batch`; neither was run.
6. **End-to-end tokens or logprobs.** Nothing here speaks to output quality; the PR's own
   gsm8k numbers are the evidence for that and were not reproduced.
7. **Throughput.** None of the PR's benchmark claims were checked.

---

## Provenance

- Box: NVIDIA GB10, sm_121, aarch64, driver 590.48.01, Linux 6.14.0-1015-nvidia. All GPU
  work under `flock /home/mark/shared/exp54928/gpu.lock`. No model loaded; peak allocation
  is a few hundred KB of synthetic pools plus the builder's persistent index buffers. Total
  GPU time across every run: well under a minute. GPU verified free at exit.
- Sources: git worktrees of `/home/mark/shared/vllm-head` at the two commits, under
  `/home/mark/shared/tmp-scratch/wt-J58400{,-base}`, selected by `PYTHONPATH`. The busy
  clone was never checked out, rebased or stashed.
- Interpreter: `/home/mark/shared/vllm-head/.venv/bin/python` (3.12.3, torch 2.13.0+cu132),
  editable install `vllm-0.26.1rc1.dev1159+g23ab0cfdb`. `vllm.__file__` was printed for each
  ref and resolves into the matching worktree (`logs/00_provenance.log`).
- PR state read from GitHub read-only on 2026-09-26; raw JSON archived under `J_58400/gh/`.
  Nothing was posted, pushed, or commented.
- Local branch `j-mrv2-prompt-tail-committed-state`, commit `7932dbdbea`, on top of the PR
  head, carries the test. Patch at `J_58400/0001-test-prompt-tail-committed-state.patch`.

## Deviations and corrections

1. **Compiled extensions symlinked into both worktrees.** Importing
   `MambaHybridModelState` pulls in `vllm.vllm_flash_attn`, which raises `ImportError`
   without `_vllm_fa2_C` / `_vllm_fa3_C`. The 17 `*.so` from the built clone were symlinked
   in (all `.gitignore`d; `git status` in each worktree shows only the new test). Those
   binaries were built 2026-08-24 from clone commit `23ab0cfdb`, **not** from either ref
   here; they are imported and never called by these tests, so this run says nothing about
   any C++/CUDA op at the PR head. Same deviation as item I.
2. **`ruff` is not installed in that venv** and nothing was installed into it. The test file
   was checked by hand: no line exceeds 88 columns, no unused imports (AST check), but
   `pre-commit` / `ruff format` has not been run over it.
3. **The cache mode was chosen, and the choice narrowed the result.** `create_vllm_config`
   resolves `mamba_cache_mode` to `"align"`; the first run died in
   `mamba_get_block_table_tensor`'s gather because the synthetic block table was
   spec-shaped, not full-width. Rather than build the align machinery, the tests pin
   `"none"` explicitly. That is a real narrowing of scope and is recorded as observation 3,
   not hidden in a comment.
4. **One self-inflicted modelling error, corrected.** The first conv model used a single
   `state_len` for every path. The kernel uses two — `width - 1 + (seqlen - 1)` under spec
   decode and `width - 1` otherwise (causal_conv1d.py:1219-1222) — and the live window sits
   at `num_accepted - 1`, not at 0, with the remaining columns being last step's scratch.
   The first version therefore seeded a contiguous 6-token history and compared whole
   buffers, and case A failed with
   `[20124, 20125, 20129, 999999, 999999, 999999] != [20124, ..., 20129]`. That was a bug
   in the oracle, not in vLLM. Fixed by modelling both state lengths and comparing the live
   window; the archived logs are all from the corrected version.
5. **Case E is kept even though it is unreachable**, clearly labelled, because it is the
   only place the head and merge-base predicates differ and a reader will otherwise wonder.
   It and case F can be dropped if the file is offered upstream.
6. **The test file is large** (≈1000 lines, of which ~500 is scaffolding and a 60-line
   header). That is well past the "keep it small" guidance for an offered diff; if it is
   offered, trimming to cases A0/A/B/C/NC would roughly halve it.

## Files

- Test: `tests/v1/worker/test_mamba_hybrid_prompt_tail_state.py` on branch
  `j-mrv2-prompt-tail-committed-state`; copy at `J_58400/tests/`.
- Probes (not part of the offered file): `J_58400/probe_metadata_diff.py`,
  `J_58400/probe_negative_control.py`.
- Pre-run card: `J_58400/CARD.md`. Logs: `J_58400/logs/`. PR JSON: `J_58400/gh/`.
- Draft PR comment: `J_58400/COMMENT_58400_DRAFT.md` (not posted).
- Evidence tarball: `J_58400_evidence.tar.gz`.

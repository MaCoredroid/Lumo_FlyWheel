# Draft comment — vllm-project/vllm#55506 (NOT posted)

Target: #55506. 154 words.

---

Re-ran the aligned-index tests on top of today's `main` (`379e9a1e`): your
three commits cherry-pick onto it with no conflicts, and the results are
unchanged from `5f71d6f` — 9/9 with your updated C8. Reverting just the
`safe_rows` clamp makes `test_padded_rows_do_not_read_past_idx_mapping` fail
again, so that case still discriminates against a `main` nine days newer.

One thing worth checking before rebasing: `main` has none of this path yet —
`compute_aligned_state_indices` still takes `(seq_lens, num_reqs)` with no
mapping (`mamba_utils.py:1092`), and `_ensure_align_ctx` still returns a bare
context (`mamba_hybrid.py:137-177`). Does that match what you're seeing?

Separately, I'd like to add plain coverage for that kernel on `main` — index
arithmetic, the `num_reqs` bound, capture/replay, and the idempotent binding —
in `tests/v1/worker/test_mamba_utils.py`, deliberately identity-order only so
it passes before and after this PR. It's green on `main` and on `main` + this
PR: {{BRANCH_LINK}}. Would that help or just add noise here?

Written with AI assistance; run on an NVIDIA GB10.

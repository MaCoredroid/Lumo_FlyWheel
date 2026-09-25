# #58021 follow-up after the author's fix (funded item G3) — v1 (Codex replacement verbatim; agent draft NO-GO on defect/reachability framing; AWAITING MARK GO)
> Codex G3_review.md: one ordinary PR comment referencing Dustin's comment (issuecomment-5821960748);
> adds the concrete regression-test gap and the scheduler-only backstop; no reachable-failure claim
> (custom executors are supported extension points but discarding stamped configs is hypothetical);
> memory under-measurement residual dropped (negligible). No merge verdict.

---

Thanks for fixing the profiling path. Following up on [Dustin's question](https://github.com/vllm-project/vllm/pull/58021#issuecomment-5821960748), could the tests distinguish unresolved profiling from an unstamped serving worker? The new regression calls the builder directly with `MagicMock()` metadata and default unresolved geometry; it proves the early return, but not that only profiling can reach it. `core.py:169` reads back the scheduler config stamped in `_initialize_kv_caches`, so it cannot detect a missing worker-side stamp. I haven't established a serving path that loses the stamp. Would a worker-initialization test rejecting missing geometry, alongside the profiling no-op test, make that boundary explicit? If the builder should enforce the distinction itself, does profiling state need plumbing through to it?

AI assistance was used for this static review; no GPU execution.

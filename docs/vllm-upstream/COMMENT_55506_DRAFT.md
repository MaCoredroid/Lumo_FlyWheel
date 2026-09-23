# #55506 review comment (funded item I) — v1 (Codex replacement verbatim; agent draft NO-GO; AWAITING MARK GO — on GO: push p9-mamba-aligned-state-indices @9cef61298f to the fork, verify the compare link, then post ONE PR review of type COMMENT; no @mentions; no merge position)
> Codex I_review.md: provenance GO (evidence @c8479bbf4 verified); padding defect verified from source
> and sanitizer log; FULL reachability GO (default O2 → FULL_AND_PIECEWISE; KDA UNIFORM_BATCH); ordering
> hazard supported conditionally (posed as a question). Prose corrections applied on the branch
> (9cef61298f) and recorded in results/upstream/55506/CORRECTIONS.md.

---

At `a28e902`, V2 FULL batches pass `num_reqs_after_padding` to `compute_aligned_state_indices`, but `idx_mapping` contains only real requests (`mamba_hybrid.py:245,302–303`; `model_runner.py:1280,1335`). The new load at `mamba_utils.py:66` masks against the padded count. A GB10 probe with 3 mapping entries and 6 rows produced three invalid 8-byte reads under compute-sanitizer, followed by `cudaErrorLaunchFailure`. The pre-PR gathered-table path resolved padding rows to null block 0. [Pinned test/probe/logs](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/c8479bbf49efb6641428d1aaf75f3a532b5e357a/results/upstream/55506).

Could final graph capture also leave the context bound to gathered tables? `prepare_inputs_to_capture` supplies `get_dummy_block_tables()`, and `_ensure_align_ctx` retains the first binding; real-batch `preprocess_state` later supplies source-slot tables. Profiling teardown resets the context, but I found no corresponding reset after final capture. This is a source/lifecycle-test finding, not a K3 model reproduction.

The [test branch](https://github.com/vllm-project/vllm/compare/a28e90223540e861af5ef6e3a2e1c7ea010d45b1...MaCoredroid:vllm:p9-mamba-aligned-state-indices) offers nine model-free cases: head 8 passed/1 failed (padding); base and first commit each 4 passed/2 failed/3 skipped. The permuted-row and replay checks discriminate. These exercise Triton index generation, not downstream KDA execution.

AI assistance was used.

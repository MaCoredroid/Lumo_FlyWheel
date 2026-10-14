# Corrections to REPORT.md / the agent's test prose (independent Codex check, 2026-09-23)

Measured results, the C2 expected rows, the padding defect mechanism and the sanitizer evidence were
verified. The following were corrected rather than edited in place:

1. **C4 wording.** `test_cuda_graph_replay_matches_eager` mutates the mapping and the sequence lengths
   in place; it does NOT mutate the block-table contents. The docstring, the in-test comment and the
   commit message said "table contents"; fixed in branch commit 9cef61298f (was f517270a6e; test logic
   unchanged, prose only).
2. **C6 scope.** `test_padded_rows_do_not_read_past_idx_mapping` only rejects the poison-slot image. A
   regression assertion for a fix should also demand null-block padding and correct real rows (C3
   currently preserves sentinel output deliberately).
3. **C7/C8** are characterization tests of the existing lifecycle hazard (first-writer-wins binding of
   gathered dummy tables at capture), not failing tests for its fix. Profiling teardown resets the
   context (`cudagraph_utils.py:995–996`); no reset was found after final capture, so "lifetime" means
   a context instance.
4. **Pre-PR behaviour.** Before the PR, gathered padding rows were zeroed (`block_table.py:259–264`);
   the report should not be read as claiming the first commit's source-table path preserved that.
5. **Not upstream-ready.** ruff/pre-commit were not run in the agent's environment (88-column limit
   checked by hand). The branch is a reproducer/test offer, not a competing PR.
6. **Compiled ops.** The exercised kernel is Triton, JIT-compiled from the PR-head Python source; the
   symlinked C++ binaries (built 2026-08-24) were imported, never called, and do not bear on the result.

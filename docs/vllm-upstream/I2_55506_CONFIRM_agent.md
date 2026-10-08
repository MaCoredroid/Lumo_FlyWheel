# I2 — confirmation of vllm#55506 at `5f71d6f`

**Verdict: both findings are fixed. The clamp removes the out-of-bounds read
(compute-sanitizer clean), the temporary-binding release is as described, and real-row indices
are unchanged. One inexact comment; nothing blocking.**

## Test-file diff

Karl's copy of the original test is byte-identical to the p9 tip (blob `ca8fd423`). His
`_fixed.py` differs in exactly one test: `..._keeps_the_capture_time_tables` becomes
`..._does_not_keep_...`, asserting the new contract — a capture batch gets `temporary=True` and
reads the zeroed gathered views; after release, a `bind=True` call binds the source per-slot
tables and the launch matches the oracle.

## Runs — GB10, under flock, at `5f71d6f`

| run | result | log |
|---|---|---|
| original file | 8 passed, 1 failed — old C8, `AttributeError: 'tuple' object has no attribute 'block_table_ptrs'`, by design | `logs/70_v2_both_files.log` |
| `_fixed.py` | 9 passed | same log |
| probe, track | padded rows constant `[1,2,3]`; no leak | `logs/80_v2_probe_track.log` |
| probe, memcheck | `ERROR SUMMARY: 0 errors` | `logs/90_v2_memcheck.log` |

Logs under `I_55506/`; provenance `logs/60_i2_provenance.log`.

## Code confirmation

- **Clamp**: `safe_rows = min(rows, max(num_mapping_rows-1, 0))`, load there, then
  `table_row = where(rows < num_mapping_rows, table_row, 0)`; `num_mapping_rows =
  idx_mapping.numel()`, added to `do_not_specialize`. Rows below it are untouched — C2 passes
  and the probe's real rows (`202,203,204 / 3,4,5 / 503,504,505`) match `a28e902`.
- **Release**: `temporary` is set only when that call performed the init; `preprocess_state`
  passes `bind=True` and asserts `not temporary`; `prepare_attn` wraps its launch in
  `try/finally` and clears `is_initialized` when temporary.

## Notes

1. Padded rows now resolve to table row 0 — slot 0's block ids (`1,2,3`), not block id 0. The
   pre-PR gathered path returned 0, the null block, so the new comment's equivalence claim is
   inexact. Those rows carry zero query tokens (model_runner.py:1341), so likely inert.
2. Unreachable edge case: `idx_mapping.numel() == 0` with `num_reqs > 0` still loads element 0.
3. **Correction to I**: my C4 docstring claimed the block tables were mutated between capture
   and replay; they are not. p9 was amended (`f517270a6e` → `9cef6129`) before Karl copied it;
   my archived copy is refreshed. Mutating them would strengthen C4.

Deviations as in I: 17 `*.so` symlinked from the 2026-08-24 build; `ruff` not run.

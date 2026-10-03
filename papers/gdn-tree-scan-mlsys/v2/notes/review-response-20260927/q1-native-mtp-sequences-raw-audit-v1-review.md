# Continuous MTP raw-reader review — 2026-09-29

**Bounded PASS.** No unresolved material finding in the new phase, coverage, or MTP raw joins. This is source/CPU preparation only; it establishes no model outcome, qualification, campaign observation, or launch admission.

Reviewed unchanged sources under `experiments/review-response-20260927/tools/`:

| Source | SHA-256 |
|---|---|
| `q1_native_mtp_sequence_segments_v1.py` | `c4f302141a50961c8120588c676411beb15dd6e4e8b77f46b534eb13ce778033` |
| `q1_native_mtp_sequences_raw_audit_v1.py` | `fb462713ab54fa718fab4d89ac79796ec0d24e2504f592be48b4cc1f9ebbe740` |

The accepted natural/joint readers were used as the baseline rather than reopened as a new broad review. New source snapshots, exact dependency/fixture hashes, controls and results are sealed in `p0/monitor/review-response-20260927/native-mtp-sequences-raw-review/`.

## Source findings

- `sequence_segments:22–41` binds the supplied immutable record's ID, flat token/position stream and initial extent, derives an ordered unique checkpoint/cycle map, and derives every row's phase from its actual materialized extent. Intermediate first passes and the final after-z first pass remain distinct.
- `sequence_segments:98–118` requires every checkpoint's follow to immediately follow its own first pass, consume that first pass's raw greedy winner, restore scratch, match that cycle's `cycle_targets.mtp_o1` snapshot, preserve the materialized KV prefix, and join both same-input score hashes to the corresponding raw heads. The next real target allocation must match the follow allocation. Exact follow and KV-snapshot coverage plus final token/extent checks reject omitted/duplicated phases. Terminal shifted input is the target O2 decision, not a forced continuation token.
- `sequences_raw_audit:15–39` independently reads and audits source history, destination bootstrap and destination continuation. Source observation/record/job/history and union-import digests are joined explicitly. Continuation starts from imported source KV while retaining the destination lease, generation and storage identity. Source hidden-state import is forbidden. The result explicitly reports `source_history_is_destination_execution=False` and `qualified=False`.

## Powered CPU controls

Independent controls: **47/47 PASS** (3 positives, 44 refusals). Positives cover three consecutive checkpoints; two checkpoints with an intermediate path step; and a destination's pre-import KV payload that differs from the imported source payload. The last positive is important: the reader accepts two distinct, independently audited segments rather than silently treating source execution as destination bootstrap.

Refusals cover every cycle's missing/duplicate follow, false scratch restoration, missing/swapped MTP O1, changed restored-prefix snapshot, wrong raw-head join and first/follow index. Other controls cover wrong fixed stream/checkpoints, wrong phase/final cycle, missing final pass, terminal forced-token substitution, missing source/destination raw objects, copied source histories, foreign request/source identity, hidden transplant and import-digest corruption. The three new author tests also pass, including same-size raw corruption with unchanged greedy metadata. No unchanged full reference suite was repeated.

Reproduce the independent controls with `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Users/zhiyuanma/miniforge3/bin/python -B p0/monitor/review-response-20260927/native-mtp-sequences-raw-review/independent_controls.py` from the paper root. The controls use synthetic full-vocabulary/KV raw objects, not measured neural executions.

## Preserved boundary

As the module docstrings state, the caller must authenticate outer seals, the actual frozen job/record and source eligibility, plus target O0/O1/O2 and their complete cycle joins before invoking this helper. In particular, this reader's terminal O2 use does not itself audit target logits. The parent is reviewing that outer auditor separately. No GPU, remote call, source edit, admission gate or scientific-count increment occurred in this review.

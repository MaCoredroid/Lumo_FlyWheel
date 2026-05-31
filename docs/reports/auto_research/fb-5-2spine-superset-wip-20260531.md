# fb-5-2spine Superset WIP Checkpoint - 2026-05-31

## Status

This checkpoint proves the 2-spine verifier mechanism on a controlled uniform B=4 decode probe. It does **not** claim the final agentic `> E5 3.150 accept/event` gate yet.

The controlled run used a fresh `Fb --mtp 5` relaunch with:

- `LUMO_FB_TWO_SPINE=1`
- `LUMO_FB_K=2`
- `LUMO_FB_DEPTH=5`
- `LUMO_FB_INTERNAL_ROWS=1`
- `LUMO_FB_KERNEL_ROWS=1`
- `LUMO_FB_NO_KV_PREFIX_COPY=1`
- `LUMO_FB_SUPERSET_DIAG=1`
- `LUMO_FB_REPAIR_STALE_FREE_BLOCK` unset
- direct `/v1/chat/completions`, 4 concurrent requests, temp `0.6`, top-p `0.95`

## Controlled B=4 Proof

| Metric | Path0 / E spine | Winner | Best observed |
|---|---:|---:|---:|
| Events | 1,390 | 1,390 | 1,390 |
| Accept/event | 0.872662 | 1.061151 | 1.061151 |
| Gain vs path0 | n/a | +0.188489 | +0.188489 |
| acc=0 rate | 54.2446% | 43.3094% | n/a |

Validation counters:

- Superset violations: `0`
- Winner less than best: `0`
- Internal winner events: `165`
- Stale-free-block detected: `0`
- Stale-free-block repair: `0`

Acceptance distributions:

- Path0: `{0: 754, 1: 352, 2: 137, 3: 61, 4: 26, 5: 60}`
- Winner: `{0: 602, 1: 437, 2: 177, 3: 76, 4: 34, 5: 64}`

Conclusion: in a controlled uniform B=4 decode setting, the 2-spine verifier is a clean event-local superset of path0. It selects the longest valid segment, never accepts fewer than path0, and reduces acc=0 rate.

## Implemented Fixes

1. Added `LUMO_FB_TWO_SPINE` mode and force-forwarding from the relaunch script.
2. Forced two-spine mode to `K=2` and disabled the shared-root sampler so the existing per-request sampler verifies independent segments.
3. Kept spine A byte-identical to native E5's MTP top-1 chain, and built spine B as second root token plus greedy continuation to depth 5.
4. Made the proposer batch-aware for B=4, returning `[B, 10]` drafts as two 5-token segments.
5. In two-spine prune/commit, used each row's existing sampler accepted length directly instead of tree-prefix rescoring.
6. Added superset diagnostics for path0, winner, best, acc=0, winner source, and per-event violation detection.
7. Fixed Mamba kernel-row block-table boundary extension with null placeholders, avoiding extra recurrent-state copy.
8. Scoped the actual-width assert to uniform decode batches so mixed prefill/decode does not falsely kill the engine.
9. Added a path0-only fallback for inconsistent batch metadata states that native E also cannot handle.
10. Added stale-free-block detection; repair is opt-in only via `LUMO_FB_REPAIR_STALE_FREE_BLOCK=1` and is disabled for trusted proof runs.

## Remaining Gate

The final agentic gate still needs a clean no-repair run on the frozen four-task SWE subset:

```bash
scripts/run_codex_experiment.py --suite swe --config Fb --apply-config --mtp 5 \
  --subset docs/reports/auto_research/swe-bench-agentic-b4-four-verified-20260530.json \
  --limit 4 --concurrency 4 --no-commit
```

For that gate to count:

- embedded path0 must reproduce E5 near `3.150` accept/event and `13.3%` acc=0;
- winner accept/event must exceed path0 and exceed `3.150`;
- superset violations must remain `0`;
- stale-free-block detection and repair must remain `0`.

If stale-free-block detection fires under agentic mixed prefill/decode, the next task is to root-cause the internal-row alloc/free accounting leak rather than enabling repair.

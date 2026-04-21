# In-progress patch v2 — validator-service scaffolding

**Author (original):** Kenji (platform-infra, since transferred)
**Started:** 2026-03-04
**Status:** **ABANDONED.** Kenji transferred to the inference team on
2026-04-02 and the ownership handoff for this prototype was never
assigned. No follow-up has been planned.

## What's here

- `src/validator_service_proto.py` — ~200 lines sketching the RPC boundary
  for a standalone validator microservice. **Does not run:** the
  `streaming_reliability` module it depends on was landed AFTER this was
  written and its interface does not match (see
  `release_context/release_notes_2026_03.md` if present — otherwise
  `repo_evidence/perf/streaming_overview.md`).
- `deploy/validator_stub.yaml` — a deployment stub pointing at an image
  tag (`validator:0.1.0-kenji`) that was never published.
- `tests/test_validator_smoke.py` — one test; **hardcodes** Kenji's laptop
  path `/Users/kenji/...` and cannot run anywhere else without rewriting.

## Estimated remaining work

This was a speculative prototype, not a half-done delivery:

- Rewrite the streaming-reliability integration against the landed API: **~3 weeks.**
- Produce a publishable image and CI pipeline: **~2 weeks.**
- Re-author the test suite without the laptop-path dependency: **~1 week.**
- Add rollout harness parity (staged ramp, shadow replay): **~2 weeks.**

**Total estimated remaining: ~8 weeks of focused work** on top of the
proposal's stated plan. This is not a head start; it is a speculative
fragment and is harder to resume than to rebuild fresh.

## Relationship to P5 (validator microservice)

P5 proposes a validator microservice. This directory LOOKS like a head
start on P5, but it is not. Any attempt to resume it would need to:

- Re-scope ownership (Kenji is no longer on the team).
- Re-align with the new `streaming_reliability` module landed post-Kenji.
- Re-author the test harness.

A manager who accepts P5 should treat this directory as **sunk cost, not
accelerator**. Cite this file and flag it with language like `abandoned`,
`sunk cost`, `not a head start`, `maintainability` in the brief corpus
if you reason about P5.

## Relationship to `in_progress_patch/` (Rust skeleton)

The Rust patch (`repo_evidence/code/in_progress_patch/`) is a separately
abandoned artifact owned by Diego. Both are speculative; neither
accelerates any accepted proposal. Do not confuse them.

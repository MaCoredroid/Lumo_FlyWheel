# Track B E2E Readiness Manifest

Generated: 2026-05-07

Command:

```bash
scripts/preflight_track_b_e2e.py --out output/track_b_e2e/preflight_20260507.json
scripts/build_track_b_e2e_readiness_manifest.py --out /tmp/track_b_readiness_manifest.json
```

Result: **Round 0 is blocked.**

## Current Blockers

- `vllm_request_id_labels_exposed`: current vLLM `/metrics` does not expose request-id labels, so per-turn vLLM joins are not trustworthy.
- `codex_trace_out_supported`: installed `codex-cli 0.128.0` does not expose `--trace-out`.
- `dcgm_profile_fields_available`: the sampler runs, but required DRAM/SM profile fields are not numeric.

## Step Status

| Step | Status | Evidence |
|---|---|---|
| A. Codex `--trace-out` patch + correctness artifact | blocked | no trace patch, no validated `output/track_b_e2e/codex_trace_emitter_correctness.json`, installed Codex lacks `--trace-out` |
| B. DCGM/NVML 100 Hz sampler | blocked | sampler script exists and runs; profile fields remain unavailable |
| C. E2E task runner | complete | `scripts/run_track_b_e2e_task.py` |
| D. Per-turn vLLM metrics keyed by request id | blocked | parser/join code exists; live metrics do not expose request-id labels |
| E. Summary join + diagnosis rule | complete | `scripts/build_track_b_e2e_summary.py` |
| F. Round proposal prompt | complete | `prompts/track_b_e2e_round_proposal.md` |
| G. Round 0 dry run | blocked | no `output/track_b_e2e/round_0/round_summary.json`; no five NCU archetype profiles |

## Decision

The readiness manifest intentionally requires both setup gates and actual Round 0 artifacts. Passing preflight alone is not enough, and a trace correctness artifact must satisfy the `lumo.track_b.codex_trace_correctness.v1` schema with three enabled/disabled byte-equality task checks. Current preflight does not pass. Do not promote or compare any Track B E2E round from this state.

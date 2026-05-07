# Track B E2E Round 0 Preflight Audit

Generated: 2026-05-07

Command:

```bash
scripts/preflight_track_b_e2e.py --out output/track_b_e2e/preflight_20260507.json
```

Result: **Round 0 must not run yet.**

## Passing checks

- vLLM health endpoint `http://127.0.0.1:9950/health` returned 200.
- vLLM exposes aggregate spec_decode counters:
  - `vllm:spec_decode_num_drafts_total`
  - `vllm:spec_decode_num_draft_tokens_total`
  - `vllm:spec_decode_num_accepted_tokens_total`
- Installed Codex is present: `codex-cli 0.128.0`.
- Installed Codex supports `codex exec --json`.
- `nvidia-smi` is present.
- `ncu` is present.

## Blocking checks

- `codex exec --help` does not expose `--trace-out`.
- Current vLLM `/metrics` output does not expose `request_id=`, `vllm_request_id=`, or `request=` labels, so per-turn metrics cannot be joined to Codex trace turns.
- The repo venv does not have `pynvml`, so `scripts/sample_dcgm_during_task.py` cannot run yet.

## Decision

Do not record `output/track_b_e2e/round_0/round_summary.json` from this environment state. Any run now would violate the plan's truthful-measurement contract: Rule 7/12/13 cannot be joined per turn, Rule 6 cannot collect GPU samples through the sampler, and Rule 14 has no trace-emitter byte-equality artifact.

Next unblockers:

1. Carry or build the Codex CLI `--trace-out` fork/patch and record `output/track_b_e2e/codex_trace_emitter_correctness.json`.
2. Enable or add vLLM per-request metric labels keyed by `vllm_request_id`.
3. Install `pynvml` in the measurement environment or replace the sampler backend with an available DCGM binding that emits numeric `dram_active_pct` and `sm_active_pct`.

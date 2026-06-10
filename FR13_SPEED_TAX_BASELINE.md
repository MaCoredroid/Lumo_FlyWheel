
## ⚠ VALIDITY SCOPE (user challenge 2026-06-10 — binding)
The table above is **NOT the deliverable speed verdict**. Its boots were lossless-debug boots:
**B=1** (not the deployed B=4), **BATCH_INVARIANT=1 on all arms** (known slow-GEMM regime — OFF for speed per
`reference_fr10_speed_measurement_pitfalls`), `FR10_METRICS=1` + LUMO logging envs ON (instrumentation overhead),
mixed capture modes, 4 pinned prompts × 128 tokens (not the SWE-Verified ~1800s workload).
**Valid for:** relative direction at matched contamination (replay −17% vs legacy ≈ the state-traffic prediction;
legacy tax grows with N). **Invalid for:** absolute tax, deployment claims, E5 comparison.

## DEPLOYMENT-REGIME measurement spec (the one that counts; run post-wiring-fix)
- B=4, MAX_NUM_SEQS=4, **FULL CUDA capture proven** (cuda_graph_proof per arm), **BI=0 both arms**,
  **FR10_METRICS=0 + ALL LUMO logging envs unset** (accept counters come from vLLM's native /metrics spec counters,
  which exist regardless — no instrumentation in the serving path), GPU_UTIL deployment value.
- Workload: SWE-Verified agentic shape (`fr12_deliverable_swe4_probe` full-task form, ~30 min/arm class), pinned
  task set, seeds recorded. Arms: native E5 / legacy tree / replay tree (post-fix), same HEAD.
- Basis: /metrics window deltas only (`decode_seconds/spec_drafts`), pairing gate enforced; per-forward ratio +
  wall + accept/event reported together.

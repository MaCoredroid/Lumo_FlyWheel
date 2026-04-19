# Evaluator Contract

- `family_id`: `request-path-evidence-brief`
- `task_id`: `t2_request_path_owner_source_brief`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Scoring Breakdown

- `25` points: Correctly identifies the live ingress for `--owner` and the layer where effective owner selection actually happens.
- `20` points: Correctly distinguishes `owner_source` derivation from `routing_key` derivation and names the emission surface that exposes both.
- `20` points: `path_map.json` is internally consistent, ordered, and includes valid `caller_symbol` and `callee_symbol` adjacency for the claimed live chain.
- `15` points: Submission rejects at least one plausible but non-live helper or path and explains why it is not live.
- `10` points: `docs_correction.md` is narrow, evidence-backed, and avoids speculative code changes.
- `10` points: Overall scope discipline, citation quality, and absence of fabricated certainty.

## Caps

- Cap at `20/100` if evidence comes from outside the provided benchmark bundle, even when the traced path is internally coherent.
- Cap at `20/100` if the submission lacks repo-grounded symbol or file references for the live chain.
- Cap at `25/100` if `path_map.json` omits adjacency or includes adjacency that cannot be reconciled with the narrative.
- Cap at `35/100` if the submission never rejects a decoy or plausible false path.
- Cap at `15/100` if the solver edits behavior or proposes refactors instead of explaining the path.
- Cap at `10/100` if the response is mostly architecture prose without a machine-checkable path artifact.

## Automatic Failure Conditions

- `0` if the solver claims the stale support note is correct without direct counterevidence handling.
- `0` if required artifacts are missing entirely.
- `0-5` if the solver invents repo files, symbols, or tests not present in the provided workspace bundle.

## Evidence Rules

- Only files and commands inside the provided benchmark bundle count as evidence unless the task explicitly names an external surface.
- Positive claims about the live path must cite concrete files and symbols.
- Negative claims about decoys must cite at least one concrete mismatch such as no live caller, wrong layer, or dead helper status.
- The evaluator should prefer contradiction checks over keyword presence.

## Judging Procedure

1. Verify all required deliverables exist.
2. Score correctness of the live path and field-derivation claims.
3. Apply caps before adding style or scope points.
4. Record whether the submission would still score the same if stripped of unsupported assertions.

## Meaningfulness Check

- The task remains meaningful only if a real repo bundle can support a precise live-chain reconstruction plus one decoy rejection.
- If a provided bundle does not contain enough symbol-level evidence to score above `60` for a careful solver, the authored task instance should be rejected as incoherent.

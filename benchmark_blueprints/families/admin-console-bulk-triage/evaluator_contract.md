# Evaluator Contract

## Scope
This evaluator scores a solver on the `admin-console-bulk-triage/queue-rebalance` task. The task is only a true solve if the live browser flow writes the correct canonical queue id for all targeted incidents and leaves trusted evidence behind.

## 100-Point Breakdown
- `10` points: Correctly identify the real defect class.
  - Must name stale queue-key or label-driven client logic as the root defect.
  - Must mention duplicate queue labels as a trap.
- `10` points: Target the right patch surfaces.
  - Frontend modal or selection path.
  - API or persistence layer that writes queue assignments or audit records.
  - Runbook or validation note.
- `35` points: Implement the end-to-end fix.
  - Bulk action uses canonical queue ids.
  - Optimistic UI and persisted backend state agree.
  - No visible-test deletion or seed mutation shortcuts.
- `20` points: Persisted-state correctness.
  - All three seeded incident ids receive the exact intended canonical queue id.
  - Audit log or trusted API state records the same ids.
- `10` points: Browser proof.
  - Screenshots show pre-submit selection and post-submit state.
  - Evidence aligns with the seeded incident ids.
- `10` points: Operator artifact quality.
  - Runbook documents the bulk flow.
  - Validation note explains what was tested in the live UI.
- `5` points: Shortcut resistance.
  - No fake screenshots, no test weakening, no direct seed rewrites, no label-only mapping.

## Hard Caps
- If canonical queue ids are not persisted correctly for all three seeded incidents, total score is capped at `20`.
- If browser proof is absent or screenshots cannot be tied to the seeded incident ids, total score is capped at `25`.
- If the solver updates docs or screenshots without proposing real frontend and API changes, total score is capped at `15`.
- Any attempt that mutates seed data directly to simulate success receives `0` in the implementation and shortcut-resistance sections.

## Pass Shape
- Expected benchmark solve band: `>= 85`.
- Meaningful but incomplete attempt: `15-35`.
- Naive GPT-5.4/high target band during family design: around `20`.

## Trusted Checks
- Hidden browser run drives the seeded three-incident scenario.
- Trusted state check verifies exact canonical queue ids.
- Trusted post-run check validates screenshot existence and seeded-id relevance.
- Diff audit checks for test deletion or direct seed tampering.

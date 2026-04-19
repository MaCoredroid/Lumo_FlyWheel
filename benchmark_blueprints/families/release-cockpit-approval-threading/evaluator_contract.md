# Evaluator Contract

## Scope
This evaluator scores `release-cockpit-approval-threading/pause-resume-handoff`. A real solve must thread `paused_for_approval` through API, UI, and automation semantics, including exact resume behavior.

## 100-Point Breakdown
- `10` points: Correctly identify the real cross-surface defect.
  - Must name partial enum threading as insufficient.
  - Must mention missing resumable provenance or incorrect resume-target semantics.
- `10` points: Target the right patch surfaces.
  - Backend transition or serializer logic.
  - UI badge or action gating.
  - Automation behavior and operator docs.
- `30` points: Implement the real state threading.
  - Pause and resume path works across API, UI, and automation.
  - No frontend-only relabeling or global automation suppression.
- `20` points: Trusted state correctness.
  - Pause stores resumable provenance.
  - Resume restores the exact prior active substate and target work item or thread.
  - Hidden API and automation checks both pass.
- `10` points: Browser proof.
  - Screenshots show paused and resumed states in the live cockpit.
- `10` points: Operator artifact quality.
  - Automation config references the new state explicitly.
  - Runbook documents pause and resume steps.
- `10` points: Shortcut resistance.
  - No fixture rewriting, no test weakening, no generic active-state fallback.

## Hard Caps
- If pause does not store resumable provenance, total score is capped at `20`.
- If resume returns to a generic active state instead of the exact prior substate and target work item or thread, total score is capped at `20`.
- If automation is suppressed globally rather than gated on the paused state, total score is capped at `15`.
- Frontend-only state relabeling receives `0` in implementation and trusted-state sections.

## Pass Shape
- Expected benchmark solve band: `>= 85`.
- Meaningful but incomplete attempt: `15-35`.
- Naive GPT-5.4/high target band during family design: around `20`.

## Trusted Checks
- Hidden API checks validate pause provenance and resume target restoration.
- Hidden browser run verifies pause and resume state in the live cockpit.
- Hidden automation checks verify paused suppression and exact resumed target.

# Evaluator Contract

## Scope
This evaluator scores `support-playbook-ui-migration/refund-escalation-tabs`. A real solve must migrate the executable helper flow and the operator guidance to the new tabbed support console.

## 100-Point Breakdown
- `10` points: Correctly identify the real migration hazards.
  - Must name retired selectors or hidden legacy template risk.
  - Must mention duplicate `Escalate` controls and the fraud-tab branch.
- `10` points: Target the right patch surfaces.
  - Helper or automation adapter.
  - Family-local skill or guidance.
  - Review note or operator-facing explanation.
- `30` points: Implement the executable flow migration.
  - Helper no longer targets retired selectors.
  - New tab ordering and fraud-banner branch are handled.
- `20` points: Real escalation persistence.
  - Hidden browser run on the seeded fraud-branch account creates a new escalation record.
  - The visible timeline shows the expected escalation type and timestamp delta.
- `10` points: Browser proof.
  - Screenshots show the tabbed console and the true migrated path.
- `10` points: Skill and review artifact quality.
  - Skill names the new tabs and banner branch.
  - Review note explains the duplicate-label hazard.
- `10` points: Shortcut resistance.
  - No reliance on hidden legacy DOM.
  - No doc-only migration.
  - No aliasing old pane ids into tests.

## Hard Caps
- If the seeded fraud-branch account is not used or no real escalation record is created, total score is capped at `20`.
- If the helper still targets retired selectors or hidden legacy DOM, total score is capped at `15`.
- If the solver updates only `SKILL.md` and review prose without executable helper changes, total score is capped at `12`.
- Screenshots from the wrong tab or from the hidden legacy template receive `0` in browser-proof scoring.

## Pass Shape
- Expected benchmark solve band: `>= 85`.
- Meaningful but incomplete attempt: `15-35`.
- Naive GPT-5.4/high target band during family design: around `20`.

## Trusted Checks
- Hidden browser run uses the specific seeded account that requires the fraud-tab branch.
- Trusted UI check verifies a new escalation record with expected type and timestamp delta.
- Trusted code check rejects retired selector or legacy-pane targeting.

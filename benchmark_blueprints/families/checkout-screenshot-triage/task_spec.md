# Checkout Screenshot Triage

## Task Prompt
You are investigating a checkout regression in a small Next.js storefront. Supplied artifacts include mobile Safari screenshots and one failing Playwright test. A mobile checkout state triggered by coupon application and invalid card submission now has both a visual regression and an accessibility regression. Investigate from artifacts and the repo, repair the root cause without regressing desktop checkout, preserve unrelated local edits, and finish with fresh evidence and a short release note.

## Workspace Bundle
- `repo/`: storefront app with cart and checkout routes, shared drawer and form components, Playwright tests, mocked payment adapter, and a light design system package.
- `artifacts/screenshots/`: user-reported mobile screenshots, one desktop screenshot, and one outdated reference from before a cart-summary redesign.
- `artifacts/logs/`: Playwright trace summary, browser console output, and payment mock logs.
- `artifacts/notes/`: support escalation text and QA repro steps.

## Seeded Visual Or Multimodal Breakage
- `surface_visual_regression`: a mobile-only checkout composition issue appears in a narrow viewport state represented in the supplied artifacts.
- `surface_accessibility_regression`: after validation fails, focus behavior is incorrect for the surfaced payment error state.
- `surface_state_sensitivity`: the failure depends on a specific combination of checkout content and viewport constraints and does not reproduce in the basic desktop path.
- `noise`: some logs and at least one artifact are misleading and should not drive the fix.

## Required Surfaces
- Screenshot inspection and browser reasoning from multiple device widths.
- Repo patching across checkout layout, drawer behavior, and form-validation wiring.
- Playwright or equivalent end-to-end execution.
- Short written release note.

## Deliverables
- Code patch fixing the visual and accessibility regressions in the affected mobile checkout state.
- Updated screenshot or trace evidence for the repaired flow using fresh machine-generated artifacts.
- Any necessary test updates that preserve checkout behavior instead of narrowing assertions.
- Short release note summarizing user-visible risk and what was validated.

## Grader Contract
- Visible tests include the failing coupon-checkout scenario and a desktop smoke path.
- Hidden checks vary viewport height, safe-area inset, saved-address copy presence, and coupon label length so breakpoint-specific padding or a single magic offset does not pass.
- Hidden checks verify the sticky summary behavior still exists on the intended surfaces and that the primary CTA remains fully visible before and after validation failure.
- Hidden checks require focus to land on a visible actionable error target or the first invalid control, not a hidden wrapper, inert container, or unchanged submit button.
- Hidden checks reject solutions that only modify tests or snapshot thresholds.
- Final grading requires fresh machine-generated evidence from the repaired flow, and hidden grading rejects stale, preexisting, cropped, or manually substituted artifacts.

## Red-Team Traps
- Hardcode a larger drawer offset that passes only the visible fixture.
- Disable sticky behavior entirely, regressing desktop.
- Change the Playwright expectation or crop the screenshot artifact instead of fixing the UI.
- Move focus to a hidden wrapper instead of the actionable error target.
- Short-circuit coupon rendering when overflow occurs.
- Convert the issue into a test-only fix by changing viewport, retries, timing, or assertion shape without correcting runtime behavior.

## Quality Gate
- Naive-solver target: under 30/100 unless the solver can integrate screenshot clues with DOM and focus behavior rather than chasing logs.
- Adversarial GPT-5.4/high probe estimated a naive score around 47/100 before hardening because the original text disclosed the coupon-triggered mobile layout path and the focus-handoff failure too directly.
- Hardening applied after probe: generalized the solver-facing breakage description, added safe-area and viewport-height variation, tightened the focus target requirement, and required fresh evidence instead of reusable artifacts.
- Current judgment: under 30/100 for a naive GPT-5.4/high solver if the hidden viewport, evidence-freshness, and focus-target checks are enforced.

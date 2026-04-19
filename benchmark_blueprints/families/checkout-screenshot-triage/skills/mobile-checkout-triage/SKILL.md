# Mobile Checkout Triage

Use this skill when a checkout bug is primarily visible through screenshots, traces, or narrow-viewport behavior and may combine layout and accessibility regressions.

## Inputs
- Mobile screenshots or traces.
- Checkout components and validation flow.
- E2E coverage for coupon, saved-address, and invalid-payment states.

## Workflow
1. Reconstruct the failing state from artifacts before changing code.
2. Identify which runtime surfaces control layout and which control focus semantics.
3. Repair the runtime behavior, not the tests or screenshots.
4. Re-run the critical mobile state plus a desktop smoke path.
5. Capture fresh evidence and summarize user-visible risk.

## Guardrails
- Do not rely on one viewport or one coupon length.
- Do not move focus to a hidden or inert container.
- Do not disable sticky behavior just to avoid overlap.


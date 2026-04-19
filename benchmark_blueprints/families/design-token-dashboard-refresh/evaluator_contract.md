# Evaluator Contract: Design Token Dashboard Refresh

## Purpose
Score whether a solver repaired the real shared token and layout regressions rather than gaming screenshots, local overrides, or a single visible page.

## 100-Point Breakdown
- `20` Root-cause localization across shared token, style, and layout sources.
- `20` Shared implementation repair in component code plus token or config surfaces.
- `15` Responsive fidelity across desktop, tablet, and mobile.
- `15` Theme fidelity across light and dark, including contrast correctness.
- `10` Fresh machine-generated visual or computed-style evidence.
- `10` Root-cause note and rollout note quality.
- `10` Non-regression coverage across another affected surface not named in the prompt.

## Scoring Caps
- Cap at `25/100` if the submission has no fresh machine-generated evidence.
- Cap at `35/100` if the submission does not change a shared token, shared style source, or shared component path.
- Cap at `10/100` if the submission only changes screenshots, tests, docs, or notes.
- Cap at `20/100` if the solver identifies plausible causes but cannot show repaired behavior across more than one breakpoint and one theme.

## Visible Pass Signals
- Unit or component checks pass.
- Desktop and mobile screenshot or visual tests pass.
- Submission includes a root-cause note and a rollout note.

## Hidden Evaluation Signals
- Computed-style checks validate semantic-token lineage rather than only final pixels.
- Hidden checks probe both light and dark themes.
- Hidden checks inspect at least one page not explicitly called out in the prompt.
- Hidden checks reject page-local overrides that leave shared consumers broken.
- Hidden checks reject screenshot-baseline-only fixes.

## Shortcut Failure Modes
- Shared tokens remain wrong while one page is patched with inline CSS.
- A solver repairs the visible screenshot dimensions but drifts from the intended spacing scale elsewhere.
- Contrast passes only in one theme.
- The solver updates evidence artifacts without a live DOM or style repair.

## Calibration Target
- Design-time family-package run target: `15-25/100`.
- Family freeze target for a naive GPT-5.4/high solver with the full benchmark bundle: roughly `20/100`, and not above `25/100`.


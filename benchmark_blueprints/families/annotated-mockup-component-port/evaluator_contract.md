# Evaluator Contract: Annotated Mockup Component Port

## Purpose
Score whether a solver ported the mockup into the shared component correctly across states, widths, densities, themes, and downstream compatibility, rather than patching a preview-only surface.

## 100-Point Breakdown
- `25` Shared component fidelity to the supplied mockup across required states.
- `15` Width-sensitive layout behavior, including long-label handling.
- `15` Compact-density and alternate-theme correctness.
- `15` Downstream compatibility through the preview app or equivalent integration surface.
- `10` Story and docs alignment with the true implementation.
- `10` Fresh verification artifacts or automated checks.
- `10` Shared ownership discipline rather than preview-only or story-only patching.

## Scoring Caps
- Cap at `25/100` if the submission does not touch the shared component path used by downstream consumers.
- Cap at `10/100` if the submission only changes stories, docs, or snapshots.
- Cap at `30/100` if the submission lacks narrow-width and compact-density verification.
- Cap at `20/100` if the solver can describe likely fixes but cannot demonstrate downstream compatibility.

## Visible Pass Signals
- The component renders and docs build.
- Stories exist for the main annotated states.
- Submission includes some verification note or evidence.

## Hidden Evaluation Signals
- Hidden checks validate state fidelity across widths, densities, and theme paths.
- Hidden checks use the shared component implementation, not a preview-only fork.
- Hidden checks exercise compatibility expectations not fully described in the noisy notes.
- Hidden checks reject layout tricks that collapse under longer labels.

## Shortcut Failure Modes
- Story-only patch.
- Preview-only implementation.
- One-width or one-density fix.
- Compatibility behavior inferred from the noisiest note instead of the live integration path.
- Screenshot-only updates without shared behavior repair.

## Calibration Target
- Design-time family-package run target: `15-25/100`.
- Family freeze target for a naive GPT-5.4/high solver with the full benchmark bundle: roughly `20/100`, and not above `25/100`.


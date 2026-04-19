# Evaluator Contract: Hero Asset Launch Page

## Purpose
Score whether a solver reconciled brief, asset identity, responsive media behavior, and accessibility fidelity rather than shipping a visually plausible but incorrect hero.

## 100-Point Breakdown
- `20` Exact asset identity and correct media selection.
- `20` Responsive media configuration and focal-point correctness.
- `15` Copy fidelity and correct content-source wiring.
- `15` Alt-text fidelity and accessibility quality.
- `10` Duplicate-media removal at the DOM level.
- `10` Fresh multi-breakpoint screenshot evidence.
- `10` Image-delivery note and non-regression reasoning.

## Scoring Caps
- Cap at `25/100` if the submission has no fresh desktop, tablet, and mobile evidence.
- Cap at `30/100` if the submission changes only component-local CSS or JSX without the required content or media-config wiring.
- Cap at `15/100` if the submission could pass by screenshot substitution, breakpoint-specific hacks, or visually similar asset swaps.
- Cap at `20/100` if the solver identifies likely issues but cannot prove exact asset identity and content-source fidelity.

## Visible Pass Signals
- Hero renders cleanly across primary breakpoints.
- Visible copy matches the approved brief.
- Duplicate decorative media is gone.

## Hidden Evaluation Signals
- Hidden checks verify approved asset identity by manifest or hash.
- Hidden checks inspect shared focal-point metadata across multiple tablet widths.
- Hidden checks reject copy or alt text hardcoded in the component if the source of truth should live in content data.
- Hidden checks reject DOM-hidden duplicate media.
- Hidden checks reject generic or headline-derived alt text.

## Shortcut Failure Modes
- Visually similar but incorrect asset.
- Correct crop at one width only.
- Copy fixed in JSX while content config remains stale.
- Alt text sounds specific but is not brief-faithful.
- Live component replaced by a flattened raster hero.

## Calibration Target
- Design-time family-package run target: `15-25/100`.
- Family freeze target for a naive GPT-5.4/high solver with the full benchmark bundle: roughly `20/100`, and not above `25/100`.


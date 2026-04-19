# Mobile Viewport Guard

Use this skill when working on `responsive-checkout-visual-regression`.

## Goal

Fix the checkout UI with mobile preview behavior as the truth source. A passing solution must preserve the preview experiment path while restoring a visible, clickable primary CTA at mobile widths and keeping tablet/desktop stable.

## Required checks

1. Inspect the broken mobile viewport at 390x844.
2. Verify a second mobile viewport.
3. Verify at least one tablet and one desktop viewport.
4. Confirm the compact summary still exists under the preview flag.

## What does not count

- Turning the preview experiment off.
- Global sticky removal.
- Screenshot-only fixes.
- One-breakpoint CSS hacks that ignore a second viewport.

## Evidence standard

- Prefer viewport-specific runtime checks, hit-testing, or geometry assertions.
- If only descriptive reasoning is available, the evaluator should cap the score low.

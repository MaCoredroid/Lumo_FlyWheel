# Responsive Checkout Visual Regression

## Task Prompt

The checkout preview at `https://preview--storefront.example.test/checkout` looks acceptable on desktop but breaks at mobile widths. On a 390x844 viewport, the sticky order summary overlaps the payment section and hides the primary submit button behind the bottom bar. Fix the responsive regression in preview, keep desktop behavior intact, and include a short QA note describing what was verified at both viewport sizes.

The task is complete only when the mobile preview exposes a reachable primary CTA without overlap and desktop still renders a stable two-column checkout.

## Workspace Bundle

- `apps/storefront/`: Next.js commerce frontend.
- `apps/storefront/components/checkout/`: Summary, payment form, sticky footer, and promo code components.
- `apps/storefront/styles/checkout.css` or tokenized CSS module files.
- `tests/e2e/checkout.spec.ts`: One broad checkout smoke test at desktop only.
- `tests/unit/`: A few rendering tests for cart totals and validation messages.
- `preview_artifacts/`: Mobile screenshot showing the hidden CTA, one desktop screenshot, and a note from QA saying the issue reproduces only in preview builds.

## Seeded Preview-Only Breakage

- The failure appears only when a preview-only compact-summary experiment is active at narrow mobile widths.
- The preview render differs from desktop enough that the primary CTA becomes partially occluded after the summary region and footer both render, but the exact DOM/CSS cause is not solver-visible in this spec.
- Existing visible tests run only at desktop width and never assert CTA visibility or clickability on mobile.
- Console noise includes non-blocking analytics warnings and a slow image optimization request.

## Required Surfaces

- `browser_preview`
- `shell`
- `apply_patch`
- `screenshots`
- `visual_snapshot` preferred if the bundle includes it

## Deliverables

- Frontend patch that resolves the mobile overlap without breaking desktop layout.
- Updated regression coverage with at least one explicit mobile viewport assertion for CTA visibility and clickability.
- Screenshot bundle with:
  - mobile before
  - mobile after
  - desktop after
- Short QA note in repo docs or the task output artifact describing tested viewport sizes and expected behavior.

## Grader Contract

The grader should award full credit only if all of the following hold:

1. Browser-level contract:
   - At 390x844, the primary checkout CTA is visible and clickable without being covered by sticky UI.
   - At a second hidden mobile viewport, the same CTA remains reachable and not partially clipped.
   - At 1024px and 1440px widths, checkout remains a stable multi-column layout and does not collapse into the mobile stack unexpectedly.
2. File-state contract:
   - The responsive fix addresses the preview-flagged compact summary path rather than globally removing sticky behavior or disabling the experiment.
   - If a regression test is added, it must assert viewport-specific behavior instead of generic render success.
3. Regression contract:
   - Existing checkout tests still pass.
   - Hidden checks verify hit-testing or DOM geometry so a screenshot-only or one-breakpoint hack does not pass.
   - Hidden checks verify that the mobile fix does not introduce tablet or desktop wrapping regressions or suppress the sticky summary entirely.
4. Evidence contract:
   - Screenshot artifacts are paired with trusted viewport checks rather than graded as presence-only evidence.

## Red-Team Traps

- Remove the sticky summary for all viewports instead of fixing the preview-only mobile path.
- Hide the footer with `display: none` only below one breakpoint while breaking order totals.
- Change copy or shrink fonts globally until the CTA happens to fit.
- Patch screenshots or QA note without changing layout code.
- Hardcode the preview flag off so the broken compact summary never renders.

## Quality Gate

- Naive solver target: under 30/100.
- Probe method:
  - One GPT-5.4/high child probe attempted the cheapest compliant-looking mobile layout fix from the spec alone.
  - Pre-hardening verdict: `under_30 = no` because the draft exposed too much of the likely CSS path and accepted overly weak evidence.
- Hardening applied after probe:
  - Removed explicit answer-key CSS details from seeded breakage.
  - Made a mobile viewport regression assertion mandatory.
  - Added hidden second-mobile, tablet, and desktop guards plus hit-testing or geometry checks for CTA occlusion.
  - Explicitly fail solutions that disable the preview experiment or remove sticky behavior wholesale.
- Probe record:
  - What was tested: whether a naive GPT-5.4/high solver could pass with a narrow mobile-only CSS override and screenshot bundle.
  - Result: that shortcut looked plausible before hardening; the spec now requires stronger viewport-aware verification.
- Current assessment:
  - Latest family-local `benchmark_run.md` score: `20/100`.
  - Naive GPT-5.4/high solver is now in the target band: `yes`, assuming hidden multi-viewport and clickability checks are enforced.

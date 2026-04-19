# Deploy Preview Route Flag Recovery

## Task Prompt

The feature-preview URL `https://preview--ops-console.example.test/releases/next-gen-import` is broken. In preview it either returns a 404 or shows an empty-state shell, even though the feature works in a developer’s local environment when the flag is enabled manually. Repair the preview-only route and flag drift so the feature-preview page loads meaningful content in preview, then add a short deploy note describing the route, flag, and seed-data expectations.

The task is complete only when the preview route resolves successfully, renders seeded content, and the requested deliverables are present.

## Workspace Bundle

- `app/`: Frontend route tree and feature page component.
- `flags/`: Feature-flag defaults and preview overrides.
- `deploy/preview_manifest.toml`: Preview deploy route and env wiring.
- `data_seed/`: Preview seed loader or fallback fixture for the feature page.
- `docs/deploy_preview_route.md`: Notes for enabling preview features and validating seeded content.
- `tests/`: Route smoke tests and unit tests for flag resolution, but no hosted preview assertion.
- `preview_artifacts/`: Screenshot of the 404 or blank shell, preview deploy summary, and one note indicating the route works locally with a manual flag override.

## Seeded Preview-Only Breakage

- Preview reproduces a route failure that does not appear in a developer-local manual-flag flow.
- More than one preview-only surface is out of alignment, so a partial fix can swap a 404 for an empty shell without restoring the real feature view.
- Visible tests only check local route registration and one flag helper in isolation.
- Noise includes unrelated asset 404s and a harmless warning about an optional beta widget.

## Required Surfaces

- `browser_preview`
- `shell`
- `apply_patch`
- `screenshots`
- `review` preferred if the bundle frames this as a deploy follow-up

## Deliverables

- Code/config patch aligning route wiring, preview flag override, and preview seed behavior.
- Patch to `deploy/preview_manifest.toml`.
- Screenshot bundle with:
  - before: 404 or empty preview shell
  - after: working preview route with visible seeded content
- Short deploy note in `docs/deploy_preview_route.md` explaining route validation and preview seed prerequisites.

## Grader Contract

The grader should award full credit only if all of the following hold:

1. Browser-level contract:
   - The preview URL resolves without 404.
   - The page renders non-empty seeded content with a known seeded record identity, not just a shell wrapper.
2. File-state contract:
   - Route registration, preview flag resolution, and preview manifest all reference one canonical current feature key and route segment.
   - Preview seed logic is aligned with the same feature gate used by the route and does not rely on a dual-key compatibility shim.
   - Deploy docs mention both the route and the seed-data expectation.
3. Regression contract:
   - Existing route and flag tests still pass.
   - Hidden checks reject fixes that hardcode the route to always render, force flags on globally, replace the seeded view with canned static content, or keep preview working only through legacy-key aliases.
   - Hidden checks verify the route remains gated outside the preview override path.
4. Evidence contract:
   - The screenshot bundle proves both route resolution and visible seeded content after the fix, and the grader verifies seeded provenance independently.

## Red-Team Traps

- Force the route component to render without respecting the feature flag.
- Add a catch-all redirect that hides the 404 but does not load the feature.
- Seed canned static content into the component instead of fixing preview seed wiring.
- Turn the feature on globally across all environments.
- Patch docs and screenshot artifacts without fixing manifest or flag drift.

## Quality Gate

- Naive solver target: under 30/100.
- Probe method:
  - One GPT-5.4/high child probe attempted the task as a mechanical alignment pass over the named route, flag, manifest, and seed surfaces.
  - Pre-hardening verdict: `under_30 = no` because the draft leaked too much of the intended reconciliation path.
- Hardening applied after probe:
  - Removed explicit answer-key bullets from seeded breakage.
  - Added a canonical-key requirement so dual-key preview aliases or compatibility shims fail hidden grading.
  - Added hidden seeded-record provenance checks and a non-preview gating check.
  - Tightened manifest and route assertions so redirects or shell-only fixes do not earn meaningful credit.
- Probe record:
  - What was tested: whether a naive GPT-5.4/high solver could exceed 30/100 by patching the named files to one apparently aligned state without much diagnosis.
  - Result: yes before hardening; the spec now requires canonical flaging plus seeded-content provenance.
- Current assessment:
  - Latest family-local `benchmark_run.md` score: `20/100`.
  - Naive GPT-5.4/high solver is now in the target band: `yes`, assuming canonical-key, non-preview-gating, and seeded-provenance checks are enforced.

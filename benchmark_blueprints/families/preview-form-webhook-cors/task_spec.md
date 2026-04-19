# Preview Form Webhook CORS

## Task Prompt

The support intake form preview at `https://preview--support-portal.example.test/contact` renders correctly, but clicking **Send** never succeeds in preview. Production works. The browser network panel shows a failing preflight or submit request to the preview API host. Use the preview browser and repo context to repair the preview-only integration drift, preserve the form’s validation behavior, and leave a short deploy note describing what must be true for future preview environments.

The task is complete only when a valid form submission succeeds in preview, validation errors still render correctly, and the requested deliverables are present.

## Workspace Bundle

- `frontend/`: React or Next.js client with `ContactForm`, submission helper, and validation UI.
- `api/`: FastAPI or serverless handler for `/api/support-intake`.
- `config/preview.env.toml`: Preview-only frontend and API origin configuration.
- `docs/deploy_preview.md`: Notes for preview origin allowlists and smoke tests.
- `tests/`: Unit tests for field validation and one API happy-path test.
- `preview_artifacts/`: HAR excerpt or text capture from a failed preflight, screenshot of the error state, and a sample valid payload.

## Seeded Preview-Only Breakage

- Preview-only submission fails in the browser because frontend origin routing and API cross-origin behavior drift apart outside production.
- The failure affects browser-mediated requests and differs between the success path and at least one error path, so direct handler tests are insufficient.
- Visible tests pass because they call the handler directly and never exercise real browser preflight behavior.
- Noise includes a benign telemetry POST failure and a favicon 404.

## Required Surfaces

- `browser_preview`
- `shell`
- `apply_patch`
- `network_inspection`
- `screenshots`

## Deliverables

- Code/config patch aligning frontend preview origin config and API CORS behavior.
- Preview config patch in `config/preview.env.toml`.
- Screenshot bundle with:
  - before: browser-visible failed submission
  - after: successful preview submission confirmation
  - after: validation-error path still rendering correctly
- Short deploy note in `docs/deploy_preview.md` documenting preview origin and validation-path expectations.

## Grader Contract

The grader should award full credit only if all of the following hold:

1. Browser-level contract:
   - A valid form submission succeeds end to end from the preview frontend.
   - An invalid submission still shows structured validation errors in the browser.
2. File-state contract:
   - Frontend preview config points at the correct preview API target rather than production or a fake local stub.
   - API CORS handling allows the preview frontend origin on both success and validation-error responses without broad wildcard or reflected-origin shortcuts.
   - Deploy docs mention the preview origin contract and verification steps.
3. Regression contract:
   - Existing validation and handler tests still pass.
   - Hidden checks exercise browser `OPTIONS` preflight plus invalid-submit behavior and reject wildcard `*` CORS, reflected arbitrary origins, disabled validation, or a frontend-only fake success path.
4. Evidence contract:
   - Trusted runtime evidence captures the actual preview API target used by the browser and confirms the screenshot bundle reflects both successful submission and preserved validation behavior.

## Red-Team Traps

- Point preview traffic back to production to make the form “work.”
- Set `Access-Control-Allow-Origin: *` while using credentials or weakening the policy.
- Fake success in the frontend without a successful network round trip.
- Remove validation handling so every submit returns 200.
- Patch only the happy path and ignore the 422 response headers.
- Update docs and screenshots without fixing the network behavior.

## Quality Gate

- Naive solver target: under 30/100.
- Probe method:
  - One GPT-5.4/high child probe attempted to clear the task by mechanically implementing the solver-visible root causes.
  - Pre-hardening verdict: `under_30 = no` because the spec exposed the likely config and header fixes too directly.
- Hardening applied after probe:
  - Removed explicit answer-key origin and header bullets from seeded breakage.
  - Added trusted runtime capture of the actual browser submit target.
  - Added mandatory hidden checks for both preflight and validation-error paths.
  - Tightened policy language to fail wildcard or reflected-origin shortcuts, not just obvious fake-success patches.
- Probe record:
  - What was tested: whether a naive GPT-5.4/high solver could read the spec as an implementation recipe and exceed 30/100 without meaningful browser diagnosis.
  - Result: yes before hardening; the spec was revised to make browser/runtime evidence and policy correctness decisive.
- Current assessment:
  - Latest family-local `benchmark_run.md` score: `20/100`.
  - Naive GPT-5.4/high solver is now in the target band: `yes`, assuming hidden network-target and CORS-path checks are enforced.

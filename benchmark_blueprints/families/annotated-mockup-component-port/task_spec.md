# Annotated Mockup Component Port

## Task Prompt

You are working in a React component-library repository. Design supplied an annotated mockup bundle for a comparison-style card component that now needs to match the shared library implementation used by existing consumers. The current implementation is visually close in some paths but drifts from the supplied artifacts in others, and at least one live consumer expectation is still real even though the design notes are not perfectly trustworthy. Repair the shared implementation so it matches the artifact-backed behavior without breaking existing consumers, then bring the stories and usage documentation back into alignment with what the component actually does.

The task is complete only when the shared component, not a preview-only substitute, behaves correctly across the required annotated states, remains compatible with the still-live integration path, and has automated verification that would catch a narrow-width regression plus at least one density or theme mismatch.

## Workspace Bundle

- `repo/`: a component-library package, shared layout utilities, Storybook stories, unit and visual tests, a small docs site, and one downstream integration or preview surface that still consumes the shared component.
- `artifacts/mockups/`: annotated PNG or PDF exports covering multiple component states, callouts for spacing or typography, and at least one state where the mockup is visually subtle rather than dramatically broken.
- `artifacts/notes/`: mixed-authority design and API notes, including at least one stale recommendation and at least one note that is directionally right but incomplete.
- `artifacts/screenshots/`: current renders from Storybook and from a consumer-facing integration surface; some screenshots are redundant and at least one is a distraction rather than the critical failure.
- `artifacts/history/`: issue, changelog, or review snippets that make one compatibility expectation discoverable without stating it cleanly in the main task text.

## Seeded Breakage Surfaces

- `surface_visual_fidelity`: the shared component diverges from the mockup in more than one state, but the dominant mistake is not fully specified here.
- `surface_responsive_layout`: a width-sensitive layout defect exists in the reusable component or shared helper, not only in a local story fixture.
- `surface_contract_drift`: a live consumer still depends on behavior or prop handling that is under-documented or partially contradicted by the notes.
- `surface_state_coverage_gap`: current stories, tests, or docs give the appearance of coverage while leaving at least one important state family under-specified.
- `noise`: some artifact notes are stale or overconfident, and some screenshots highlight symptoms rather than the root compatibility constraint.

## Required Surfaces

- Multimodal inspection of the annotated mockup bundle.
- Repo patching in the shared component path and any helper or token wiring it truly depends on.
- Storybook or equivalent visual verification plus automated test coverage.
- Light compatibility reasoning grounded in the downstream integration surface.

## Deliverables

- A code patch for the shared component and any real shared helper or token wiring it depends on.
- Story or example coverage for the artifact-backed state families, including the previously under-specified presentation path.
- A docs or usage-contract update that reflects the true supported behavior.
- Automated verification covering:
  - one narrow-width or long-label case in the shared component, and
  - one density, theme, or equivalent presentation-mode case that would fail if the repair were too one-dimensional.
- Fresh verification evidence or notes tied to the real shared implementation rather than a preview-only fork.

## Grader Contract

The grader should award full credit only if all of the following hold:

1. Shared implementation contract:
   - The solver changes the shared component path that downstream consumers actually use.
   - The repair addresses the real drift surfaces visible in the artifact bundle rather than moving styling into stories, docs, or a preview-only clone.
   - The fix generalizes across the annotated state family instead of matching only one obvious screenshot.
2. Multimodal fidelity contract:
   - Hidden checks compare implementation behavior against the annotated mockup across multiple states, widths, and at least one alternate presentation path such as density or theme.
   - Hidden checks reject brittle pixel nudges or hardcoded layout exceptions that pass only on one label length, one viewport, or one screenshot.
3. Compatibility contract:
   - Hidden checks exercise the live downstream behavior or prop contract implied by the integration surface and history artifacts.
   - Solutions that silently drop compatibility to satisfy one note or one screenshot do not receive full credit.
4. Evidence and alignment contract:
   - Stories and docs must reflect the repaired implementation instead of diverging from it.
   - Automated verification must prove more than basic renderability and must cover the narrow-width and alternate-presentation cases explicitly.

## Red-Team Traps

- Patch only stories, docs, screenshots, or snapshot baselines while leaving the shared component wrong.
- Implement the mockup as a preview-only component instead of repairing the library component used by consumers.
- Follow the noisiest artifact note literally and remove live compatibility behavior that is still required.
- Fix only the shortest label or one viewport and leave the component brittle under longer labels, denser layouts, or alternate theme paths.
- Claim fidelity from hand-wavy visual reasoning without reconciling the actual shared implementation, state coverage, and downstream usage.

## Quality Gate

- Naive solver target: `15-25/100`, with the preferred observed score near `20/100`.
- Pre-hardening weakness in the earlier draft:
  - It overexposed the exact failing states and told the solver too directly where to look for width and compatibility failures.
  - It allowed a largely textual plan to score too well even without trustworthy integration or verification evidence.
- Hardening applied in this revision:
  - Shifted the task text from answer-key states to breakage surfaces and artifact ambiguity.
  - Made live consumer compatibility discoverable but not solver-spoon-fed.
  - Required dual-mode verification so narrow-width-only or screenshot-only fixes cannot score well.
  - Explicitly grounded grading in the shared implementation path, not a preview shim.
- SOTA benchmark-design justification:
  - `SWE-bench Multimodal` shows that visual software tasks are materially harder once image understanding and cross-language front-end behavior are required.
  - `Design2Code`, `FullFront`, and `FrontendBench` show that layout fidelity, visual element recall, and realistic front-end interaction remain weak areas for strong multimodal models, so the benchmark must demand artifact-to-code reconciliation instead of screenshot mimicry.
  - OpenAI's 2026 `SWE-bench Verified` analysis argues against public-answer-key leakage, overly narrow tests, and contamination-prone benchmark construction; this family therefore keeps the grading signal in hidden invariants and privately authored artifact ambiguity rather than explicit fix instructions.
- Current assessment:
  - If hidden multimodal fidelity, compatibility, and verification checks are enforced together, a naive GPT-5.4/high solver should land near the target band rather than succeeding from generic front-end best practices alone.

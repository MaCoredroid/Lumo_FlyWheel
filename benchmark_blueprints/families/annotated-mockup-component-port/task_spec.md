# Annotated Mockup Component Port

## Task Prompt
You are in a React component-library repo. Product design supplied an annotated mockup for a new comparison-card component that needs to be ported into the library, but the current implementation diverges from the supplied artifacts in several states, and at least one downstream integration still depends on behavior that is not documented consistently. Build or repair the component so it matches the annotated states, remains compatible with existing consumers, and ships with stories and usage docs. Do not break existing consumers.

## Workspace Bundle
- `repo/`: component-library package, preview app, Storybook stories, unit tests, visual snapshots, docs site, and theme tokens.
- `artifacts/mockups/`: annotated PNG or PDF showing default, selected, disabled, and compact-density states, plus spacing and typography callouts.
- `artifacts/notes/`: mixed design and API notes; not every note is authoritative.
- `artifacts/screenshots/`: current rendered states from Storybook.

## Seeded Visual Or Multimodal Breakage
- The library implementation diverges from the mockup across more than one state or presentation mode.
- At least one width-sensitive layout issue is present in the shared component rather than only in story fixtures.
- At least one downstream compatibility expectation is real but not fully described by the notes.
- Some artifact notes are stale or misleading.

## Required Surfaces
- Multimodal inspection of the annotated mockup.
- Repo patching across the component, stories, theme wiring, and docs.
- Storybook or screenshot verification plus unit or visual tests.
- Light compatibility reasoning for existing props.

## Deliverables
- Code patch for the component and any shared layout helper it depends on.
- Updated stories covering all annotated states.
- Usage-doc update that reflects the true prop contract.
- Updated screenshot evidence or visual snapshot outputs.
- Automated verification covering at least one narrow-width long-label case and one compact-density case in the shared component.

## Grader Contract
- Visible checks cover basic renderability and docs-build integrity.
- Hidden checks validate artifact fidelity across multiple states, widths, densities, and theme paths using the shared library implementation.
- Hidden checks exercise downstream integration behavior, including non-obvious compatibility expectations not fully described in the notes.
- Hidden checks ensure stories and docs reflect the actual implementation rather than local overrides, screenshot-only updates, or preview-only patches.
- Final grading requires stories and docs to align with the implementation.

## Red-Team Traps
- Patch only stories or snapshots while leaving the shared component wrong.
- Implement the mockup as a preview-only component instead of the library component used by consumers.
- Use width-specific positioning or a one-state patch that fails under another density, theme, or longer label.
- Infer compatibility behavior from the noisiest note instead of the preview-app integration surface.
- Treat the solver-facing spec as a checklist and skip real mockup-to-code reconciliation.

## Quality Gate
- Naive-solver target: under 30/100.
- Adversarial GPT-5.4/high probe estimated a naive score around 58/100 before hardening because the original spec exposed the exact failing states and compatibility trap.
- Hardening applied after probe: generalized the solver-facing breakage description, kept mixed-authority notes ambiguous, shifted grading toward hidden state, density, theme, width, and downstream-integration fidelity, and required automated verification for narrow-width and compact-density cases.
- Current judgment: under 30/100 for a naive GPT-5.4/high solver if the hidden integration and artifact-fidelity checks are enforced together.

# Mockup Port Fidelity

Use this skill when a solver must translate annotated design artifacts into a repair of the real shared component path without breaking a still-live consumer contract.

## Required Inputs

- annotated mockups and current screenshots
- mixed-authority notes and history artifacts
- the shared component path plus any true shared helper or token dependency
- stories, tests, docs, and one downstream consumer surface

## Official Mindset

This is not a screenshot-matching exercise and not a prose-graded design review. Official score comes from code-backed verifier phases, hidden tests, and diff inspection.

## Workflow

1. Inspect artifacts first.
   - identify repeated state families, subtle callouts, width-sensitive behavior, and any alternate presentation mode such as density or theme
   - separate repeated evidence from stale or one-off notes
2. Trace the real ownership path.
   - find the shared component import path that downstream consumers actually use
   - trace only the true shared helpers, tokens, or layout utilities that drive the defect
3. Infer the compatibility contract.
   - treat integration behavior and history evidence as higher authority than one loud note
   - preserve still-live props or semantics unless the bundle proves they are dead
4. Patch the shared implementation.
   - repair the shared component and only the real dependencies it actually needs
   - avoid local story hacks, preview forks, or label-specific offsets
5. Add trusted verification.
   - ensure one test stresses narrow widths or long labels
   - ensure one test stresses density, theme, or another alternate presentation mode
   - prefer DOM, geometry, and behavior assertions over fragile screenshot-only checks
6. Align stories and docs.
   - update stories and docs only after implementation and tests are coherent
   - keep stories and docs descriptive of the repaired shared path, not of a local override

## Guardrails

- do not patch only stories, docs, baselines, or screenshots
- do not replace the broken component with a preview-only clone
- do not disable compatibility behavior to satisfy one stale note
- do not hardcode exact labels, exact widths, or fixture-specific strings
- do not weaken or bypass tests or verifier hooks

## What Full-Credit Evidence Looks Like

- shared-path code changes
- passing visible verification commands
- hidden-test-compatible behavior across state, width, and alternate-presentation axes
- preserved downstream consumer contract
- stories and docs aligned with the repaired implementation

# Mockup Port Fidelity

Use this skill when a component-library task requires translating annotated visual artifacts into the shared implementation without dropping live consumer behavior.

## Inputs

- Annotated mockup bundle with multiple states or callouts.
- Shared component implementation, helpers, stories, tests, and docs.
- A downstream integration or preview surface that still consumes the shared component.
- Mixed-authority notes or history artifacts that may disagree with one another.

## Workflow

1. Inspect the mockup and screenshots first. Enumerate the required state families, width-sensitive behaviors, and any alternate presentation mode such as density or theme.
2. Trace the shared component path that real consumers use. Do not start by editing stories, screenshots, or docs.
3. Reconcile artifact authority: live integration behavior and repeated evidence beat a single stale note.
4. Patch the shared component and only the real helper or token paths it depends on.
5. Add verification that would fail for:
   - a narrow-width or long-label regression, and
   - a density, theme, or equivalent alternate presentation regression.
6. Update stories and docs only after the implementation behavior is coherent.

## Guardrails

- Do not accept a story-only, preview-only, snapshot-only, or docs-only fix.
- Do not hardcode one viewport, one label length, or one screenshot-specific offset.
- Do not remove under-documented compatibility behavior unless the integration surface proves it is dead.
- Do not treat the loudest artifact note as authoritative without checking the consumer path.

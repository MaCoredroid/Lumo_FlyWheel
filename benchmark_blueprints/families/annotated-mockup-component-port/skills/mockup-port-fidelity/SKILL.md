# Mockup Port Fidelity

Use this skill when a component-library task depends on translating annotated visual states into the shared implementation without breaking downstream consumers.

## Inputs
- Annotated mockup or state reference.
- Shared component implementation and stories.
- Preview-app or downstream integration surface.

## Workflow
1. Identify all required states and any width or density-sensitive behavior.
2. Patch the shared component first, not preview-only surfaces.
3. Reconcile notes against actual downstream integration behavior.
4. Verify long-label, compact-density, and alternate-theme behavior.
5. Ensure stories and docs match the real implementation.

## Guardrails
- Do not accept a story-only or snapshot-only match.
- Do not drop compatibility behavior just because one note says it is stale.
- Do not rely on one width or one density when the mockup shows more than one state family.


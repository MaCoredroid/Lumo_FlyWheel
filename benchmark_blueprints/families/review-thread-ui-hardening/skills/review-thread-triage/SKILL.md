# Review Thread Triage

Use this skill when a frontend fix must be derived from exported review threads, screenshots, and reopened visual-test feedback rather than from a clean bug report alone.

## Inputs
- Exported review threads with resolution state.
- Reviewer screenshots.
- The route or component tree implicated by those threads.

## Workflow
1. Separate actionable unresolved feedback from resolved or stale context.
2. Map each actionable thread to the real route, component, and viewport.
3. Repair runtime behavior and the required visual-test or snapshot coverage.
4. Capture fresh evidence from the exact reopened viewport.
5. Draft concise replies that reference the thread ids actually addressed.

## Guardrails
- Do not reply to resolved threads unless the task explicitly requires it.
- Do not patch screenshot baselines without a runtime fix.
- Do not add blanket accessibility labels to unrelated controls.


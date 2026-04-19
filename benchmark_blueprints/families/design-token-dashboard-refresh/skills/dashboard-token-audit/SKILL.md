# Dashboard Token Audit

Use this skill when a dashboard visual regression likely comes from shared token lineage, spacing-scale drift, or theme-specific styling rather than a one-page CSS typo.

## Inputs
- Current screenshots or browser output at multiple breakpoints.
- Shared token or theme config.
- The affected shared components.

## Workflow
1. Confirm the visible symptoms across at least two breakpoints and both themes.
2. Trace the rendered surface back to semantic token or shared style sources.
3. Prefer shared fixes over page-local overrides.
4. Re-verify with fresh screenshots or computed-style checks.
5. Write a short root-cause note that ties each symptom to the shared source of truth changed.

## Guardrails
- Do not treat screenshot baselines as proof of correctness.
- Do not patch only the most visible page if another shared consumer still uses the broken token path.
- Do not accept a contrast fix that only works in one theme.


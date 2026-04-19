# Design Token Dashboard Refresh

## Task Prompt
You are in a small React analytics dashboard repo. The latest design-token migration partially landed, and multiple layout and theme regressions are visible across dashboard surfaces on desktop and mobile, including at least one contrast failure and at least one responsive overflow defect. Fix the implementation instead of patching screenshots. Preserve existing local edits. When finished, leave the repo with passing tests, restored visual parity across breakpoints and themes, and a short rollout note about any cache or token migration risk.

## Workspace Bundle
- `repo/`: Vite or Next.js frontend, shared `tokens/semantic.toml`, `src/components/`, `src/pages/`, Storybook stories, Playwright or screenshot tests.
- `artifacts/screenshots/`: reference desktop and mobile screenshots plus one stale screenshot that reflects the old design.
- `artifacts/review_notes/`: short bug report from design and one misleading note blaming only Tailwind classes.
- `artifacts/runtime/`: test output, browser console logs, and a theme-usage inventory.

## Seeded Visual Or Multimodal Breakage
- The visible regressions come from more than one implementation issue spanning shared styling, token usage, and responsive layout.
- At least one artifact in the bundle is stale or misleading and should not be treated as ground truth.
- The visible pages do not cover the full affected surface area.

## Required Surfaces
- Repo patching across component code, styling, and token config.
- Browser inspection at desktop, tablet, and mobile widths, including computed-style verification in light and dark themes.
- Terminal test execution for unit, Storybook snapshot, and Playwright or equivalent visual checks.
- Short written rollout note.

## Deliverables
- Code patch that fixes the token wiring and responsive layout in shared components rather than page-local hacks.
- Updated token or config artifact if semantic aliases must change.
- Refreshed screenshot evidence for the repaired pages.
- Brief root-cause note mapping each repaired symptom to the shared source of truth that changed.
- Short deploy or rollout note covering CSS cache or theme migration risk.

## Grader Contract
- Visible checks confirm the dashboard renders with no failing unit tests and that screenshot tests pass for desktop and mobile.
- Hidden checks validate both light and dark themes and include at least one additional page not named in the prompt.
- Hidden checks assert contrast, overflow behavior, and spacing through DOM and computed-style inspection rather than filename matching.
- Hidden checks verify semantic-token lineage from the rendered surface back to shared token and config sources, not just final color values.
- Hidden checks reject broad spacing or padding substitutions that visually match one screenshot but drift from token-scale semantics across breakpoints.
- Hidden checks fail if screenshot baselines are changed without corresponding DOM or style repairs.
- Final grading requires a short rollout note mentioning theme or cache implications.

## Red-Team Traps
- Edit screenshot baselines without fixing the rendered DOM or token source.
- Add one-off inline colors or CSS overrides only on the visible page.
- Silence the failing visual test or remove the mobile viewport case.
- Patch only the components visible in reference screenshots while leaving another token consumer broken.
- Use semantically wrong aliases that satisfy one theme or viewport but regress another.
- Apply blanket gap or padding substitutions that pass one page while drifting from the intended scale system elsewhere.

## Quality Gate
- Naive-solver target: under 30/100 unless the solver can trace semantic-token usage, compare screenshots, and verify at more than one viewport.
- Adversarial GPT-5.4/high probe estimated a naive score around 65/100 before hardening because the original spec leaked token and spacing root causes.
- Hardening applied after probe: removed explicit root-cause bullets, required computed-style checks across desktop, tablet, and mobile in light and dark themes, added semantic-lineage grading, and required a root-cause note tied to shared sources of truth.
- Current judgment: under 30/100 for a naive GPT-5.4/high solver if the hidden theme, breakpoint, and lineage checks are enforced.

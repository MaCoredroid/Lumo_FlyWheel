# `esm-plugin-loader-modernization` Evaluator Contract

## Evaluation Goal
Score whether a solver performs a real ESM plugin-loader modernization with validated plugin contracts and built-output loading.

## Visible Checks
- `npm test`
- `npm run build`
- `npm run typecheck`

## Trusted Checks
- No `require(` remains on the live loader path.
- No unsafe namespace coercion replaces real runtime validation.
- Docs and generated CLI evidence match actual commands.

## Hidden Checks
- Built `dist/` output is the real load path.
- Default-export and named-export plugins both load through the validated contract.
- Malformed plugins fail with a specific runtime validation error.
- The mixed `.cjs` helper remains usable.

## 100-Point Breakdown
- `10`: visible tests pass without editing benchmark-owned tests.
- `5`: build succeeds.
- `5`: typecheck succeeds.
- `20`: hidden built-output pack proves the loader resolves from withheld `dist/` cases.
- `20`: hidden malformed-plugin pack proves runtime contract validation is real.
- `20`: hidden helper/evidence pack proves the mixed `.cjs` path and generated evidence survive withheld cases.
- `10`: trusted anti-unsafe-coercion checks pass.
- `10`: docs and generated evidence match actual commands.

## Caps
- If the loader still resolves from source-only paths, total score is capped at `25`.
- If the solver relies on `default ?? mod as Plugin`-style coercion without validation, total score is capped at `20`.
- If the solver edits benchmark-owned visible tests or evidence scripts, total score is capped at `20`.
- If only the visible source-path pack is proven and no withheld built-output pack is exercised, total score is capped at `20`.
- If docs or evidence are updated without functional loader correctness, total score is capped at `15`.
- If tests/build/typecheck are bypassed, total score is capped at `0`.

## Passing Interpretation
- Target benchmark-design outcome: a naive GPT-5.4/high solver should land around `20/100`.
- Scores above `30` indicate the family still over-rewards standard ESM rewiring.

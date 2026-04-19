# Benchmark Run

## Attempt History
- `attempt_01`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da338-699a-7a32-aff2-1dd39f3266aa`
  - result: over target under the original evaluator because build/test/evidence success was over-credited.
- `attempt_02`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da33f-cf6e-7fd0-9e34-6a8932532223`
  - workspace: `probe_runs/attempt_02/workspace`

## Hardening Applied After Attempt 01
- Moved most score weight behind withheld built-output, malformed-plugin, and helper/evidence packs.
- Added a cap at `20` when only the visible source-path pack is proven.
- Added a cap at `20` when benchmark-owned visible tests or evidence scripts are edited.

## Attempt 02 Evidence
- Visible command: `npm test`
  - result: `PASS`
- Visible command: `npm run build`
  - result: `PASS`
- Visible command: `npm run typecheck`
  - result: `PASS`
- Extra command: `bash scripts/capture_cli_output.sh`
  - result: `PASS`
- Solver changed loader, contracts, CLI entrypoint, docs, evidence, and benchmark-owned tests/evidence scripts.

## Scoring Against Hardened Evaluator
- Visible tests pass: `10/10`
- Build succeeds: `5/5`
- Typecheck succeeds: `5/5`
- Hidden built-output pack: `0/20`
- Hidden malformed-plugin pack: `0/20`
- Hidden helper/evidence pack: `0/20`
- Trusted anti-unsafe-coercion checks: `0/10`
- Docs/evidence alignment: `10/10`
- Raw subtotal before caps: `30/100`
- Cap triggered: benchmark-owned visible tests/evidence scripts were edited, and no withheld built-output pack was exercised
- Final scored result: `20/100`

## Final Judgment
- Target achieved: `yes`
- Current hardness judgment: a naive GPT-5.4/high solver can clean up the visible ESM pack, but under the hardened evaluator it remains in the intended `~20/100` band unless it also survives the withheld packs.

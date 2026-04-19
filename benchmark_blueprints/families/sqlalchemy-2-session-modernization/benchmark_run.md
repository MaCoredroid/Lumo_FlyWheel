# Benchmark Run

## Attempt History
- `attempt_01`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da338-6914-7b43-ad6c-adf0e89a7c98`
  - result: over target under the original evaluator because visible transaction fixes were worth too much.
- `attempt_02`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da33f-ceb9-74f1-9140-9c5cc6ff9af8`
  - workspace: `probe_runs/attempt_02/workspace`

## Hardening Applied After Attempt 01
- Shifted most of the score behind withheld retry/idempotence and multi-entrypoint transaction packs.
- Added a cap at `20` when only the visible entrypoints are proven.
- Reduced syntax-only and docs-only credit.

## Attempt 02 Evidence
- Visible command: `pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py`
  - result: `PASS` (`3 passed`)
- Audit command: `rg -n "session\\.query\\(" app`
  - result: `PASS` (no live matches)
- Solver changed repository, API, worker, admin CLI, and deploy note files.

## Scoring Against Hardened Evaluator
- Visible tests pass: `10/10`
- No live `session.query(`: `5/5`
- Deploy note accuracy: `5/5`
- Hidden worker retry pack: `0/20`
- Hidden admin CLI pack: `0/20`
- Hidden rollback-state pack: `0/20`
- Trusted anti-global-session checks: `0/10`
- Hidden multi-entrypoint boundary pack: `0/10`
- Raw subtotal before caps: `20/100`
- Final scored result: `20/100`

## Final Judgment
- Target achieved: `yes`
- Current hardness judgment: a naive GPT-5.4/high solver can fix the visible transaction bugs, but the withheld retry/idempotence packs still gate anything above the intended `~20/100` band.

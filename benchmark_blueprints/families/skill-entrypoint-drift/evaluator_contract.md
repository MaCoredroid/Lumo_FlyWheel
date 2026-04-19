# Evaluator Contract

## Scope
- Family: `skill-entrypoint-drift`
- Task: `t9_skill_entrypoint_drift_ops_digest`
- Visible working area: `workspace_bundle/`
- Scoring target for a naive `gpt-5.4` / `high` solver: `15-25/100`

## Scoring Breakdown
- `5` points: updates the visible skill command path to something plausibly current.
- `5` points: aligns the visible example command with the visible skill instructions.
- `5` points: repairs obvious path or flag drift in the visible runner or smoke helper without weakening them.
- `5` points: keeps the skill concise and executable instead of turning it into vague prose.
- `20` points visible cap: no visible-only submission may exceed `20/100`.
- `20` points hidden: copied command blocks execute in the documented working directories.
- `20` points hidden: the real runner contract, flags, and output headings all align.
- `20` points hidden: skill and example both survive V4 multi-surface follow-up checks.
- `20` points hidden: smoke harness integrity and regression follow-up remain intact.

## Integrity Rules
- Zero credit for weakening:
  - `scripts/smoke_skill.py`
  - `tests/test_skill_smoke.py`
  - `tests/test_digest_runner.py`
- No credit above the visible cap unless command blocks execute independently from the docs.

## Hidden Checks
- Run copied command blocks from both `SKILL.md` and the example artifact.
- Reject repo-root-only fixes when the skill claims broader portability.
- Reject output that uses the wrong heading or summary contract even if it is otherwise valid markdown.

## Score Interpretation
- `0-10`: incoherent or harness-softening attempt.
- `15-25`: meaningful visible repair but still misses execution-truth and output-contract checks.
- `30+`: too easy; harden the task and rerun.

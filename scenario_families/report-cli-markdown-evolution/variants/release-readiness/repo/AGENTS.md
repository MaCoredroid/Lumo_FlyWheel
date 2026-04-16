# Task: add Markdown output to the release-readiness report CLI

## Context

The release-readiness team uses `python -m release_readiness` to produce a
daily snapshot of release blockers, hotfixes, and preflight-check status. The
JSON output is consumed by automation, but release managers have asked for a
human-readable Markdown version they can paste into the release channel each
morning.

## What's failing

`make ci` currently fails. `make test` shows failures in `tests/test_cli_json.py`
where the new `--format markdown` option is expected to work. `make lint`
and `make typecheck` currently pass — please keep it that way.

The CLI gets its available `--format` choices from the renderer registry.
Look at how `json` is currently wired before you add anything new.

## Scope

Read the repo before writing code. There is existing scaffolding you should
understand before you touch anything.

Do not modify or delete any existing tests. Do not add `pytest.mark.skip`,
`pytest.mark.xfail`, or `# type: ignore` comments to make things pass.

`make ci` must pass at the end.

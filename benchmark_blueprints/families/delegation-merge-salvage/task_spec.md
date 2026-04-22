# Task Spec: `delegation-merge-salvage`

## Canonical Prompt

Finish a partially delegated feature after two worker patches landed in a
scratch branch and neither is safe to apply wholesale. Worker A fixed markdown
rendering and the CLI watchlist plumbing but also tried to rename the JSON
contract. Worker B kept the JSON shape stable and improved the docs, but it
touched an unrelated fixture and still missed the `--include-watchlist`
follow-up path.

Salvage the correct pieces, reject the bad ones, and leave evidence-rich
artifacts that explain exactly what was kept and discarded.

## Deliverables

- final integrated patch in the workspace
- `artifacts/delegation/salvage_postmortem.md`
- `artifacts/delegation/verification.md`
- `artifacts/delegation/reviewer_note.md`

## Variant progression

- `v1-clean-baseline`: clean two-patch salvage, one visible markdown bug
- `v2-noisy-distractor`: stale memo suggests the JSON rename is safe; it is not
- `v3-dirty-state`: partial-apply note tempts sunk-cost continuation
- `v4-multi-corpus-objective`: release context makes downstream JSON stability the primary objective
- `v5-recovery-in-thread`: prior rollback proves the renamed-key path already failed in production

## Visible checks

`PYTHONPATH=src python -m unittest tests.test_cli`

## Hidden checks

- JSON output remains byte-for-byte identical to `tests/fixtures/json/baseline_report.json`
- `--include-watchlist` is visible in markdown mode
- `tests/fixtures/legacy_snapshot.md` stays untouched
- the postmortem cites at least one kept and one rejected hunk from each worker
- variant-specific context is acknowledged where required

## Saturation and renewal plan

If the mean probe score exceeds 80 for two consecutive probe rounds, renew with:

1. a third worker patch that looks locally consistent but violates a second contract
2. a variant where the docs change and the JSON contract change must be split into two commits rather than one salvage

# Delegation Merge Salvage

You are finishing a partially delegated feature. Two worker patches landed in a
scratch branch and neither is safe to apply wholesale.

Read these first:

- `artifacts/delegation/worker_a.patch`
- `artifacts/delegation/worker_b.patch`
- `artifacts/delegation/worker_a_notes.md`
- `artifacts/delegation/worker_b_notes.md`

Task:

1. Salvage the correct hunks from both workers.
2. Preserve the existing JSON contract.
3. Make markdown mode work, including `--include-watchlist`.
4. Leave three deliverables:
   - `artifacts/delegation/salvage_postmortem.md`
   - `artifacts/delegation/verification.md`
   - `artifacts/delegation/reviewer_note.md`

Hard rules:

- Do not edit tests, worker artifacts, context files, or fixtures.
- Do not rename the JSON contract keys.
- Keep the final patch selective and reviewable.

Visible check:

```bash
PYTHONPATH=src python -m unittest tests.test_cli
```

Recommended verification:

```bash
PYTHONPATH=src python -m unittest tests.test_cli tests.test_service tests.test_docs
PYTHONPATH=src python -m watchlist_report.cli --format json
PYTHONPATH=src python -m watchlist_report.cli --format markdown --include-watchlist
```

Read the variant-specific context files before editing:
- `artifacts/context/contract_guardrail.md`
Only these paths should change:

- `src/watchlist_report/cli.py`
- `src/watchlist_report/renderers/markdown_renderer.py`
- `docs/usage.md`
- `artifacts/delegation/salvage_postmortem.md`
- `artifacts/delegation/verification.md`
- `artifacts/delegation/reviewer_note.md`

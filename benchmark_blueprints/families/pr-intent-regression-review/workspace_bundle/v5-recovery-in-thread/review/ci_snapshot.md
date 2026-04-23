# CI Snapshot

- `pytest repo/tests/test_markdown_export.py -q` -> `2 passed`
- `pytest repo/tests/test_json_contract.py -q` -> `skipped (integration-only in this stack)`
- `ruff check repo/src repo/tests` -> `passed`

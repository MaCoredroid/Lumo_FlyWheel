# Salvage Postmortem

## Kept from Worker A

- `A1` from `artifacts/delegation/worker_a.patch`: kept the CLI plumbing that forwards `--include-watchlist` into the markdown path.
- `A2` from `artifacts/delegation/worker_a.patch`: kept the watchlist follow-up section in `src/watchlist_report/renderers/markdown_renderer.py`.

## Rejected from Worker A

- `A3` from `artifacts/delegation/worker_a.patch`: rejected the `alerts` -> `entries` rename in `src/watchlist_report/service.py` because the JSON contract fixture must stay byte-for-byte stable.
- `A4` from `artifacts/delegation/worker_a.patch`: rejected the serializer normalization in `src/watchlist_report/renderers/json_renderer.py` because it hides the same contract drift instead of preserving the contract.

## Kept from Worker B

- `B1` from `artifacts/delegation/worker_b.patch`: kept the cleaner markdown section labels in `src/watchlist_report/renderers/markdown_renderer.py`.
- `B2` from `artifacts/delegation/worker_b.patch`: kept the markdown CLI example in `docs/usage.md`.

## Rejected from Worker B

- `B3` from `artifacts/delegation/worker_b.patch`: rejected the edit to `tests/fixtures/legacy_snapshot.md` because it is unrelated fixture churn.


# Merge Salvage Audit

Use this skill when multiple worker artifacts overlap and the correct solution requires selective salvage.

## Required workflow

1. Inspect both worker artifacts before editing.
2. Identify what must be preserved, what must be rejected, and why.
3. Preserve existing contracts before chasing visible green tests.
4. Record salvage decisions at the hunk or file-path level.
5. Finish with a selective, reviewable patch.

## Deliverable checklist

- one kept and one rejected hunk from each worker
- JSON contract preserved
- watchlist follow-up behavior preserved
- no unrelated fixture churn


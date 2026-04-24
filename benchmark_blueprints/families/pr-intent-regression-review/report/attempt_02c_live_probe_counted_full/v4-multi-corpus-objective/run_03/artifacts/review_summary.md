# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown support, but it regresses both historical JSON contracts the description says must remain intact: the no-flag CLI path now emits markdown, and explicit `--output json` no longer returns the JSON-shaped payload downstream automation expects.

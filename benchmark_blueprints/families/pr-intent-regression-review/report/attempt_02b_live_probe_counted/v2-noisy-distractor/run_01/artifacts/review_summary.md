# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown coverage, but it regresses both parts of the existing JSON contract: unflagged CLI calls no longer default to JSON, and explicit `--output json` no longer returns the JSON payload shape automation consumers expect.

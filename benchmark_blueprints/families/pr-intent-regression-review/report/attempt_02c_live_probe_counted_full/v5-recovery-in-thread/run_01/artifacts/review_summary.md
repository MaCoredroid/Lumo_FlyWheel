# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown support, but it regresses both unchanged JSON entry points called out in the PR description: the flagless CLI path now defaults to markdown, and explicit `--output json` now routes through the markdown renderer and emits a JSON string instead of the historical object payload.

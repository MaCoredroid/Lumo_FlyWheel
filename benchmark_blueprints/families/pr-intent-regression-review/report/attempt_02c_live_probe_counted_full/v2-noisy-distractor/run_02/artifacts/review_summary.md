# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR claims to add Markdown support without changing JSON behavior, but it breaks both automation-facing contracts: the no-flag CLI path now defaults to Markdown, and explicit --output json now returns a Markdown string instead of the historical JSON object.

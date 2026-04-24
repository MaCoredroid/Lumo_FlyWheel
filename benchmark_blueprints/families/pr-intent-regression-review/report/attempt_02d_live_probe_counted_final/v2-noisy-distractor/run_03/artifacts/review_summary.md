# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown rendering, but it also breaks both existing JSON contracts called out in the description: no-flag CLI invocations now default to markdown, and explicit `--output json` no longer returns the JSON payload.

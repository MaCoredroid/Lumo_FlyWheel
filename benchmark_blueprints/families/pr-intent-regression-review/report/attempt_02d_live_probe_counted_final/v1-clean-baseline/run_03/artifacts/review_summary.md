# Review Summary

- intent verdict: does_not_match_intent
- summary: The markdown export itself is wired in, but the patch also changes the automation-facing JSON contract: zero-flag CLI calls now default to markdown, and explicit `--output json` no longer returns the JSON payload promised by the existing export contract.

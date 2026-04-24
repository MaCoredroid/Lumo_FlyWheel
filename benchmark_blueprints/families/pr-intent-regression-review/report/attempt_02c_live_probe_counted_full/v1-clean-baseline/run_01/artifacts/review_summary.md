# Review Summary

- intent verdict: does_not_match_intent
- summary: The patch adds markdown support, but it also breaks both parts of the existing JSON contract: the no-flag CLI path no longer emits JSON, and explicit `--output json` no longer returns the machine-readable object automation consumers expect.

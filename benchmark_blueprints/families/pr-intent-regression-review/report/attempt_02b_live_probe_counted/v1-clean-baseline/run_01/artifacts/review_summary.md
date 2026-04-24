# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown support, but it regresses both automation-facing JSON contracts called out in the unchanged docs: the no-flag CLI path now defaults to markdown, and explicit `--output json` is routed through the markdown renderer.

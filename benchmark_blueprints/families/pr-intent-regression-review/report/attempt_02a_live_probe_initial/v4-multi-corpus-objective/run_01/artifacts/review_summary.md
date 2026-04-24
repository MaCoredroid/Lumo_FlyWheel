# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown rendering, but it breaks both automation-facing JSON entry points: the default CLI invocation now emits markdown, and explicit `--output json` is remapped to the markdown renderer.

# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR breaks both automation-facing JSON contracts it claimed to preserve: the default CLI output is no longer JSON, and explicit --output json now serializes markdown text instead of the existing object payload.

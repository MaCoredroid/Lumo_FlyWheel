# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown support, but it regresses both unchanged JSON contracts the description says must stay intact: the no-flag CLI path now emits markdown, and explicit json selection is routed through the markdown renderer.

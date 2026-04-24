# Review Summary

- intent verdict: does_not_match_intent
- summary: The PR adds markdown coverage, but it also breaks both unchanged automation contracts the description says must stay intact: the no-flag CLI path no longer emits JSON, and explicit `--output json` now returns quoted markdown.

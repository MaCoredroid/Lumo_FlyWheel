## Current Test Focus

- API create/read still matters, but the release blocker is transactional
  correctness across worker retry and admin batch planning.
- A migration that only swaps `query()` for `select()` is not enough.

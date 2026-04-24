# Review Thread Remediation

Use this skill when solving `pr-thread-contract-remediation`.

## Workflow

1. Read `review/pr_481_threads.json` before trusting `review_summary.md`.
2. Separate unresolved actionable threads from resolved and outdated ones.
3. Fix the narrow contract bugs only:
   - omit `owner` for unowned response buckets
   - keep owner-bucket order stable when appending unowned
   - preserve request-side `owner` semantics
4. Add regression coverage for at least one non-default serializer path.
5. Update `docs/release_notes.md` with the omit-not-null contract nuance.
6. Write `review/reviewer_replies.json` with thread-specific evidence.
7. Write `review/verification_note.md` listing the test command(s) you ran.

## Avoid

- replying to resolved or outdated threads
- global alphabetical sorting as a shortcut
- treating missing `owner` params as explicit `null`
- fixing only one serializer path
- generic reply bodies such as `"fixed"` or `"addressed"`

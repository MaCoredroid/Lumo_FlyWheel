# INC-214 Preview Alias Rollback

The first remediation attempt for PR-214 reintroduced `legacy_preview_hint`
inside the fallback payload so the reopened comment would "see something".
That alias briefly satisfied one reviewer screenshot, but it broke the preview
reviewer bot because the bot now consumes `approval_state`.

Recovery rule for this variant:

- do not revive `legacy_preview_hint`
- mention `INC-214` when replying to the reopened fallback thread
- keep `approval_state` and `requires_human_review` aligned on both the live
  preview path and the fallback path

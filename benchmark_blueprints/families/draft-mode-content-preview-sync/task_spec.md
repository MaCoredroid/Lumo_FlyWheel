# Draft Mode Content Preview Sync

## Task Prompt

The editorial preview for the docs site at `https://preview--docs-site.example.test/articles/agent-rollouts?preview=1` is inconsistent. The draft banner appears, but editors still see stale published content and an incorrect “Published” status chip for a known draft revision. Production is correct for published content. Repair the preview-only draft-mode flow so preview reflects the latest draft content and status, then update the operator doc with the exact draft preview verification steps.

The task is complete only when preview shows the draft revision content, the status chip reflects draft state, and the deliverables below are present.

## Workspace Bundle

- `site/`: Next.js or Astro docs site with draft-mode preview route.
- `content_adapter/`: Module that fetches published or draft content from fixture-backed CMS APIs.
- `config/preview.toml`: Preview token and cache behavior settings.
- `docs/draft_preview.md`: Internal note for editorial preview and cache clearing.
- `tests/`: Unit tests around content adapter selection and banner rendering.
- `preview_artifacts/`: Screenshot showing stale preview content with a draft banner, one JSON fixture for the expected draft article, and a small note from editorial describing the mismatch.

## Seeded Preview-Only Breakage

- Preview mode visibly diverges from published mode in two ways: the editor sees stale body content while preview UI still signals draft state, and the visible status chip does not match the expected draft revision.
- The mismatch reproduces in preview but is not isolated by existing visible tests, which cover banner rendering and published-content fetches only.
- Extra schema deprecation warnings and one irrelevant missing image warning appear in preview logs.

## Required Surfaces

- `browser_preview`
- `shell`
- `apply_patch`
- `screenshots`
- `docs`

## Deliverables

- Code patch fixing draft content selection and status derivation in preview mode.
- Preview config or cache-setting patch in `config/preview.toml` if needed.
- Screenshot bundle with:
  - before: draft banner with stale content or wrong status
  - after: draft banner with fresh draft content and correct draft status
- Operator doc update in `docs/draft_preview.md` describing verification and cache expectations.

## Grader Contract

The grader should award full credit only if all of the following hold:

1. Browser-level contract:
   - Preview mode renders the latest draft body content from the draft fixture.
   - The status chip reflects draft or unpublished state, not published state.
2. File-state contract:
   - Preview-mode content selection differs from published-content handling where necessary, without disabling caching wholesale.
   - Status derivation in preview reads draft-consistent metadata rather than an unrelated published record.
   - Operator docs describe how to verify draft freshness and banner/status correctness.
3. Regression contract:
   - Published mode remains unchanged.
   - Hidden checks compare multiple articles, including a published-only article requested in preview mode, and reject fixes that disable caching entirely, hardcode draft text, or force every previewed article into draft state regardless of payload.
   - Hidden checks verify that preview body content and preview status come from the same draft revision identity.
4. Evidence contract:
   - Evidence must semantically show corrected content and corrected status, not just the presence of a refreshed banner.

## Red-Team Traps

- Disable caching globally so the preview appears fixed while published behavior regresses.
- Hardcode the known draft article body or title.
- Always show a draft chip when `preview=1`, even when the payload is published-only.
- Patch the banner text but leave the content adapter stale.
- Edit docs and screenshots without fixing the preview data path.

## Quality Gate

- Naive solver target: under 30/100.
- Probe method:
  - One GPT-5.4/high child probe attempted the task by treating the draft as an implementation recipe rather than a browser-first diagnosis problem.
  - Pre-hardening verdict: `under_30 = no` because the draft leaked both substantive fix directions.
- Hardening applied after probe:
  - Removed explicit answer-key cache-key and status-source bullets from seeded breakage.
  - Added hidden cross-article checks, including a published-only preview request that should not be forced into draft state.
  - Added a hidden same-revision consistency check so content and status must derive from the same draft record.
  - Strengthened evidence expectations so screenshots or DOM captures must reflect the expected draft sentence plus draft-status semantics.
- Probe record:
  - What was tested: whether a naive GPT-5.4/high solver could mechanically patch cache and status paths from the task text and still exceed 30/100.
  - Result: yes before hardening; the task now relies on hidden cross-article and consistency checks to resist that path.
- Current assessment:
  - Latest family-local `benchmark_run.md` score: `20/100`.
  - Naive GPT-5.4/high solver is now in the target band: `yes`, assuming the hidden multi-article and same-revision checks are enforced.

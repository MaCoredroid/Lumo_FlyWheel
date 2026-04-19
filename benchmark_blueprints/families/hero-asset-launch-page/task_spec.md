# Hero Asset Launch Page

## Task Prompt
You are working in a marketing-site repo that has to ship a launch-page hero from a copy brief and a folder of product images. The current hero implementation diverges from the supplied brief and rendered artifacts across multiple responsive surfaces, including content, media selection, and accessibility fidelity. Fix the hero section so it is brand-correct, responsive, and content-accurate. Do not replace the component with a static screenshot. Preserve any unrelated work in the repo and finish with screenshot evidence plus a short note on image-delivery risk.

## Workspace Bundle
- `repo/`: Astro or Next.js marketing site with MDX or content config, image pipeline helpers, hero component, and lightweight visual tests.
- `artifacts/brief/`: launch copy brief, approved headline hierarchy, alt-text guidance, and image-focal-point notes.
- `artifacts/images/`: candidate product renders, duplicate near-matches, cropped variants, and one decoy file from a previous campaign.
- `artifacts/screenshots/`: current broken desktop, tablet, and mobile renders.

## Seeded Visual Or Multimodal Breakage
- The hero diverges from the brief across more than one responsive surface and more than one source of truth.
- At least one visually plausible asset is a decoy and should fail if chosen without full brief reconciliation.
- At least one rendering defect depends on responsive media configuration rather than a single CSS rule.
- Some current content or accessibility text does not faithfully match the approved brief.

## Required Surfaces
- Multimodal reconciliation of the brief, candidate images, and rendered output; text-only filename matching is insufficient.
- Repo patching across hero component, content config, and image-pipeline or metadata wiring.
- Browser or screenshot verification across responsive breakpoints.
- Short written image-delivery note.

## Deliverables
- Code and content patch that selects the correct asset, corrects responsive crop behavior, and updates copy and alt text.
- Updated screenshot evidence for desktop, tablet, and mobile.
- Any required content-config or asset-manifest update, with hero copy and alt text sourced from repo content data rather than component-local literals.
- Short note covering optimization or CDN-cache implications.

## Grader Contract
- Visible checks validate that the hero uses approved copy and renders without duplicate decorative media.
- Hidden checks validate the exact approved asset by manifest identity or file hash, not by filename similarity.
- Hidden checks validate that tablet focal-point behavior comes from the shared image metadata or config path rather than component-local CSS overrides.
- Hidden checks probe at least three tablet widths and fail if the approved callout badge is clipped at any of them.
- Hidden checks fail if approved copy or alt text is hardcoded in the hero component instead of flowing from the intended content-config source.
- Hidden checks reject generic or headline-derived alt text and require the brief-approved alt text.
- Hidden checks fail if duplicate decorative media remains in the DOM, even when hidden by CSS.
- Hidden checks reject static-image or screenshot-only substitutions.
- Final grading requires freshly generated desktop, tablet, and mobile screenshots from the live component plus the image-delivery note.

## Red-Team Traps
- Point the hero at a visually similar but incorrect image file.
- Hardcode a CSS crop that only works for the visible screenshot dimensions.
- Leave stale copy in content config while changing only the rendered component.
- Set alt text to the headline string or another generic fallback.
- Replace the live hero with a flattened raster mockup.
- Use the correct asset only at one breakpoint while leaving a near-match wired elsewhere.
- Patch tablet framing with breakpoint-specific CSS while leaving shared focal metadata stale.
- Keep a duplicate desktop image mounted but visually suppressed with display, opacity, or `aria-hidden`.
- Inline approved copy in the component while content config still contains stale launch text.

## Quality Gate
- Naive-solver target: under 30/100 unless the solver can reconcile the copy brief, image set, and responsive implementation together.
- Adversarial GPT-5.4/high probe estimated a naive score around 45/100 before hardening because the original spec named asset choice, tablet crop, stale copy, and alt text too explicitly.
- Hardening applied after probe: generalized the solver-facing prompt and breakage section, then required exact asset identity checks, shared focal-point metadata checks, multi-width tablet framing, DOM-level duplicate-media removal, content-source wiring, and exact alt-text fidelity.
- Current judgment: under 30/100 for a naive GPT-5.4/high solver if asset identity, metadata, and content-source checks are enforced together.

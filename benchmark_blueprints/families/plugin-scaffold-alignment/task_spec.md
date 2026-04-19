# Plugin Scaffold Alignment

## Companion Artifacts

- `evaluator_contract.md` defines score caps, hidden invariants, and fail-fast conditions
- `codex/config.toml` defines the configured Codex profile for this family
- `skills/align-plugin-scaffold/SKILL.md` provides bounded benchmark-local guidance
- `benchmark_run.md` records actual GPT-5.4/high benchmark attempts, scoring, and resulting hardening

## Task Prompt

Finish a partially scaffolded Codex plugin so its manifest, marketplace entry, skill registration, and example usage all line up. The current scaffold passes a superficial existence check, but the plugin cannot actually be discovered and the example skill path is stale. Repair the plugin bundle without creating duplicate entries or rewriting unrelated plugin metadata.

## Workspace Bundle

- `.codex-plugin/plugin.json`
- `skills/plugin_ops/SKILL.md`
- `.agents/plugins/marketplace.json`
- `tests/test_plugin_manifest.py`
- `tests/test_marketplace_alignment.py`
- `docs/plugin_quickstart.md`
- Optional fixture assets or example screenshots for richer variants

## Seeded Skill Or Tooling Drift

- Plugin manifest path and skill registration disagree
- Marketplace metadata references the wrong plugin id or stale display path
- Docs still use a pre-scaffold example command
- Hidden checks validate discoverability and one concrete example flow, not just manifest existence

## Required Surfaces

- JSON and Markdown editing
- Repo search and shell
- Terminal tests
- Plugin- and skill-surface reasoning about registration and discoverability

## Deliverables

- Corrected plugin manifest, marketplace metadata, and skill path wiring
- Updated quickstart or skill docs to match the live plugin bundle
- Short verification note showing the plugin can be discovered and the example path resolves

## Grader Contract

- Phase 1 visible checks:
  - `pytest -q tests/test_plugin_manifest.py tests/test_marketplace_alignment.py`
- Phase 2 hidden checks:
  - Validate plugin id consistency across manifest, marketplace entry, and skill references
  - Verify the example skill path resolves to a real file and the docs mention the right path
  - Execute one discovery-flow check that resolves the plugin from marketplace metadata rather than only comparing strings
  - Reject solutions that add duplicate marketplace entries or second plugin manifests
  - Confirm unrelated metadata fields remain intact
- Score shaping:
  - Visible checks alone cap the score at `20/100`
  - Any duplicate marketplace entry, duplicate manifest, or unresolved discovery flow caps the score at `10/100`
  - Exceeding `30/100` requires passing hidden discoverability and path-resolution checks
- Pass condition:
  - The plugin is discoverable, singular, and documented with the real skill path

## Red-Team Traps

- Duplicate the marketplace entry with the corrected id
- Fix plugin.json only and leave the skill path stale
- Weaken tests or delete example-flow assertions
- Rewrite docs to vague language without a real discoverable path
- Create a second plugin manifest in a new folder instead of repairing the scaffold

## Quality Gate

- Probe goal: naive GPT-5.4/high should not pass by changing one id string
- Hardening baked in before probe:
  - Discoverability and path resolution are graded, not just manifest syntax
  - Hidden checks reject duplicate-entry escapes
  - Metadata preservation prevents brute-force rewrites
- Probe outcome:
  - Adversarial GPT-5.4/high judged the initial spec too easy; a solver could likely exceed `30/100` by mirroring ids and paths across the named files
- Additional hardening after probe:
  - Added an explicit discovery-flow check rather than pure cross-file consistency
  - Added stronger caps for duplicate-entry or unresolved-path escapes
  - Added a score clamp so visible manifest alignment alone cannot exceed `20/100`
- Final naive GPT-5.4/high assessment:
  - Under-30 now looks plausible for a naive solver because cross-file string alignment no longer clears discoverability or scoring gates

## V1-V5 Instantiation

- V1 clean baseline: one mismatched id and one stale docs path
- V2 noisy reality: duplicated marketplace hints and stale scaffold comments
- V3 dirty workspace: unrelated local edits in another plugin section must survive
- V4 multi-surface: manifest, marketplace metadata, skill doc, and verification note all required
- V5 recovery-in-thread: first fix makes the plugin discoverable but duplicates marketplace state or breaks the example path

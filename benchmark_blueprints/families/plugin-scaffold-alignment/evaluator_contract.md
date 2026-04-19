# Evaluator Contract

## Evaluation Intent

This family measures whether a solver can repair a partially scaffolded Codex plugin so it is truly discoverable and internally consistent. Cross-file string mirroring alone must stay below `30/100`.

## Required Inputs

- Final plugin manifest
- Final marketplace metadata
- Final skill path wiring and docs
- Visible and hidden test results

## Scoring Breakdown

- `20 points`: visible suite passes in `workspace/tests`
- `20 points`: plugin id, marketplace id, and skill path are internally consistent
- `20 points`: one real discovery-flow check resolves the repaired bundle through marketplace metadata and `workspace/.agents/plugins/discovery_index.json`
- `20 points`: all operator docs reference real, existing paths and stable ids
- `20 points`: no duplicate manifest or duplicate marketplace entry is introduced, and marketplace metadata carries the same skill path as the plugin manifest

## Scoring Caps

- Visible checks alone cap the run at `20/100`
- Any unresolved discovery flow caps the run at `20/100`
- Any marketplace entry that omits the manifest's skill path caps the run at `20/100`
- Any duplicate marketplace entry or duplicate manifest caps the run at `0/100`

## Visible Checks

- `cd workspace && python -m pytest -q tests`

## Hidden Checks

- Resolve the plugin through marketplace metadata and ensure discovery returns the repaired bundle
- Confirm the documented skill path resolves to the real file
- Reject duplicate manifests or duplicate marketplace entries
- Confirm unrelated metadata fields remain intact
- Reject vague docs rewrites that omit the real example path

## Fail-Fast Conditions

- A second plugin manifest is created
- Marketplace metadata is duplicated instead of repaired
- Docs no longer include a real discoverable path

## Red-Team Expectations

- A shallow solver will mirror ids and paths across the obvious files
- A stronger but still naive solver may repair the visible discovery path while trampling unrelated metadata
- Both must remain under `30/100`

## Hardness Judgment

- Current target judgment: naive GPT-5.4/high should remain under `30/100`
- Residual risk: if hidden discovery checks are too close to visible fixture structure, a careful string-aligner may still overperform

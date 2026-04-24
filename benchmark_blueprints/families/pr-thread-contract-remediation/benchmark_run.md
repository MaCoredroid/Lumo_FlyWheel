# Benchmark Run — `pr-thread-contract-remediation`

## Model under test

```bash
codex exec --model gpt-5.4 --reasoning-effort high
```

## attempt_00 — baseline skeleton (2026-04-18)

- family bundle contained contract docs only
- no `workspace_bundle/`, scorer, verifier data, manifest, or verification matrix
- a blueprint-only reasoning pass landed at `20/100` because the solver could
  describe the fix shape but could not ground thread ids, replies, or code
  behavior in a real workspace

Verdict: not probe-ready.

## attempt_01 — family implementation and deterministic sanity (2026-04-23)

Implemented:

- full five-variant workspace bundle
- deterministic scorer at `verifiers/pr-thread-contract-remediation/score_pr_thread_contract.py`
- shared hidden tests
- per-variant oracle overlays, workspace manifests, milestone scripts, and
  `manifest.lock.json`
- verification matrices for V1 and V5

Deterministic validation was run with:

```bash
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/regen_family.py
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/pr-thread-contract-remediation/verification_matrix.md
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/pr-thread-contract-remediation/verification_matrix_v5.md
```

Observed deterministic baselines:

- oracle: `95/100` on all five variants
- empty: `0/100` on all five variants
- shortcut: `12/100` on all five variants

Layer A status after this attempt: **live probe still pending**

Why pending:

- this turn focused on turning the family from a skeleton into a runnable
  family-local benchmark with deterministic scoring and verification assets
- a whole-family `codex exec` probe has not yet been run against the freshly
  generated workspace bundle

Next live step:

```bash
codex exec --cd benchmark_blueprints/families/pr-thread-contract-remediation/workspace_bundle/v1-clean-baseline --skip-git-repo-check --sandbox workspace-write --color never --ephemeral "Read AGENTS.md and fix only the actionable review-thread issues."
```

## attempt_02 — first whole-family live probe, superseded (2026-04-23)

Exact command:

```bash
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/live_probe_family.py --attempt-id attempt_02_live_probe --n 3 --timeout-seconds 1800
```

Artifacts:

- `report/attempt_02_live_probe/probe_runs.jsonl`
- `report/attempt_02_live_probe/probe_report.txt`
- `report/attempt_02_live_probe/probe_report.json`

Observed probe table:

| Variant | Scores | Mean | Ceiling pattern |
| --- | --- | ---: | --- |
| V1 | `[25,25,25]` | `25.00` | `missing_release_note_contract` x3 |
| V2 | `[25,25,25]` | `25.00` | `missing_release_note_contract` x3 |
| V3 | `[25,25,25]` | `25.00` | `missing_release_note_contract` x3 |
| V4 | `[25,20,25]` | `23.33` | `missing_release_note_contract` x3, `objective_drift` x3, `generic_replies` x1 |
| V5 | `[25,25,25]` | `25.00` | `missing_release_note_contract` x3, `objective_drift` x3 |

Gate values from the report:

- `family_mean = 24.67`
- `max_variant_mean = 25.00`
- `min_variant_mean = 23.33`
- monotonic: `pass`

Why this attempt is superseded:

- the scorer's `missing_release_note_contract` check was using an exact-phrase
  substring gate and was capping valid release notes that clearly documented
  the omit-not-null contract
- a concrete V5 probe workspace under `/tmp/pr_thread_contract_probe/.../v5.../workspace/docs/release_notes.md`
  explicitly said:
  - the `INC-742` rollback was addressed
  - unowned buckets omit `owner` instead of serializing `owner: null`
  - request-side filter semantics stay unchanged
- because that ceiling was false-positive, only post-fix probe results count
  for calibration

Family-local fix before the next counted probe:

- patched `verifiers/pr-thread-contract-remediation/score_pr_thread_contract.py`
  so release-note scoring is based on deterministic concept checks instead of
  one exact phrase

## attempt_03 — counted whole-family live probe after scorer fix (2026-04-23)

Exact commands:

```bash
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/live_probe_family.py --attempt-id smoke_probe_v1 --n 1 --timeout-seconds 1200 --variants v1-clean-baseline
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/live_probe_family.py --attempt-id attempt_03_live_probe --n 3 --timeout-seconds 1800
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/pr-thread-contract-remediation/verification_matrix.md
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/pr-thread-contract-remediation/verification_matrix_v5.md
```

Artifacts:

- `report/attempt_03_live_probe/probe_runs.jsonl`
- `report/attempt_03_live_probe/probe_report.txt`
- `report/attempt_03_live_probe/probe_report.json`
- `verification_matrix.md`
- `verification_matrix_v5.md`

Per-variant numeric results:

| Variant | Scores | Mean | Stdev | Ceiling hits |
| --- | --- | ---: | ---: | --- |
| V1 | `[72,72,72]` | `72.00` | `0.00` | none |
| V2 | `[72,72,72]` | `72.00` | `0.00` | none |
| V3 | `[72,72,72]` | `72.00` | `0.00` | none |
| V4 | `[20,72,72]` | `54.67` | `30.02` | `generic_replies` x1 |
| V5 | `[72,25,25]` | `40.67` | `27.14` | `objective_drift` x2 |

Layer A gate values:

- `family_mean = 62.27` -> fail (`target [15,25]`)
- `max_variant_mean = 72.00` -> fail (`target <= 40`)
- `min_variant_mean = 40.67` -> fail (`target <= 10`)
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` -> pass
- oracle `= 95` on all five variants -> pass
- empty `= 0` on all five variants -> pass
- shortcut `= 12` on all five variants -> pass

Spot-check diagnosis from actual live workspaces:

- V1/V2/V3 are currently frontier-easy for `gpt-5.4/high`.
  Across all nine runs, the agent repaired the core serializer/service bug,
  updated tests, and produced a valid release note with no ceilings at all.
- V4 is not reliably objective-hard. One run failed on a reply-artifact miss
  (`generic_replies`), but two runs scored `72` with no ceiling. The mobile
  objective evidence is present, but not consistently binding enough to force a
  lower ceiling band.
- V5 is the only variant that sometimes bites on the intended later-stage
  pressure. Two runs hit `objective_drift` and scored `25`; one run still
  scored `72`, showing the incident/mobile recovery evidence is within the
  model's competence envelope on at least some seeds.

Honest read:

- the current family no longer has a scorer bug masking valid behavior
- after that fix, the real signal is that V1-V3 are flat and too easy, while
  V4/V5 only separate intermittently
- bringing this family down into the §10.1 low-20 band now would require a
  substantive redesign of the earlier variants' evidence/decision pressure, not
  another small scorer tweak

Verdict:

- **Layer A failed the freeze gate**
- **Current status: frontier-easy / widening-candidate, honest signal understood**

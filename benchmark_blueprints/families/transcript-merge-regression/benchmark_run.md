# Benchmark Run

## attempt_00 — baseline solver evidence

- Existing child-solver record: `20/100`
- Interpretation: the family already discriminates against analysis-only answers
  that name the invariant but do not patch reducer code or preserve summary semantics.

## attempt_01 — family-owned runnable bundle + Layer B implementation

Implemented in this pass:

- full `workspace_bundle/v1..v5/`
- deterministic scorer: `verifiers/transcript-merge-regression/score_transcript_merge.py`
- family declaration: `family.yaml`
- manifest pinning: `manifest.lock.json` plus per-variant `workspace_manifest.json`
- hidden checks and shared milestone scripts under `verifier_data/transcript-merge-regression/`
- verification-matrix runner: `tools/run_verification_matrix.py`

Commands run in this pass:

- `python3 benchmark_blueprints/families/transcript-merge-regression/tools/regen_family.py`
- `python3 -m py_compile benchmark_blueprints/families/transcript-merge-regression/tools/regen_family.py benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py verifiers/transcript-merge-regression/score_transcript_merge.py`
- `python3 benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/transcript-merge-regression/verification_matrix.md`
- `python3 benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/transcript-merge-regression/verification_matrix_v5.md`

Observed numeric results:

- `verification_matrix.md` (`v1-clean-baseline`)
  - Oracle: `P=100`, `M=1.0000`, `G=1.000`, `R=1.000`, `S_TTC=1110`
  - Empty: `P=10`, `M=0.1111`, ceilings `summary_still_render_coupled`, `same_name_identity_unfixed`, `stale_incident_note`
  - Note only: `P=10`, `M=0.1111`, ceiling `note_only_fix`
  - Render filter shortcut: `P=20`, `M=0.1667`, ceiling `render_layer_filtering`
  - Drop all post-completion: `P=20`, `M=0.1111`
  - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity=1`
- `verification_matrix_v5.md` (`v5-recovery-in-thread`)
  - Oracle: `P=100`, `M=1.0000`, `G=1.000`, `R=1.000`, `S_TTC=1110`
  - Empty: `P=10`, `M=0.1111`
  - Note only: `P=10`, `M=0.1111`
  - Render filter shortcut: `P=20`, `M=0.1667`
  - Drop all post-completion: `P=20`, `M=0.1111`
  - Delete-tests adversarial: `P=0`, `M=0.0000`, `integrity=1`

Oracle sweep:

- `v1-clean-baseline`: visible tests `rc=0`, shared hidden tests `rc=0`
- `v5-recovery-in-thread`: visible tests `rc=0`, shared hidden tests `rc=0`

Status:

- Layer B: materially implemented and locally verified via the scorer + matrix runner.
- Layer A: still pending a full 5-variant external probe loop. Per instruction, this pass does not launch the next solver loop.

## attempt_02 — whole-family live `codex exec` probe

Design change:

- Added a family-local live probe driver at
  `benchmark_blueprints/families/transcript-merge-regression/tools/live_probe_family.py`
  and ran the first full 5-variant `codex exec` sweep on the authored bundle.
- Hardened the family-local scorer first so transient runtime files
  (`__pycache__`, `.pyc`, `.pytest_cache`, `.DS_Store`) no longer false-fire
  integrity rules during live attempts.
- Kept all work family-local: report artifacts under
  `benchmark_blueprints/families/transcript-merge-regression/report/attempt_02_live_probe/`
  and no cross-family probe infrastructure edits.

Commands run in this attempt:

- `python3 benchmark_blueprints/families/transcript-merge-regression/tools/live_probe_family.py --attempt-id attempt_02_live_probe --n 3 --timeout-seconds 180`
- `python3 -m py_compile benchmark_blueprints/families/transcript-merge-regression/tools/live_probe_family.py benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py verifiers/transcript-merge-regression/score_transcript_merge.py`
- `python3 benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/transcript-merge-regression/verification_matrix.md`
- `python3 benchmark_blueprints/families/transcript-merge-regression/tools/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/transcript-merge-regression/verification_matrix_v5.md`

Artifacts:

- `benchmark_blueprints/families/transcript-merge-regression/report/attempt_02_live_probe/probe_runs.jsonl`
- `benchmark_blueprints/families/transcript-merge-regression/report/attempt_02_live_probe/probe_report.txt`
- `benchmark_blueprints/families/transcript-merge-regression/report/attempt_02_live_probe/probe_report.json`

Observed live-probe results:

| variant | scores | mean | stdev | min | max | ceiling hits |
|---|---|---:|---:|---:|---:|---|
| `v1-clean-baseline` | `[85, 90, 85]` | `86.67` | `2.89` | `85` | `90` | none |
| `v2-noisy-distractor` | `[60, 85, 60]` | `68.33` | `14.43` | `60` | `85` | `stale_incident_note x2` |
| `v3-dirty-state` | `[75, 60, 60]` | `65.00` | `8.66` | `60` | `75` | `stale_incident_note x2` |
| `v4-multi-corpus-objective` | `[60, 60, 85]` | `68.33` | `14.43` | `60` | `85` | `stale_incident_note x2` |
| `v5-recovery-in-thread` | `[60, 75, 90]` | `75.00` | `15.00` | `60` | `90` | `stale_incident_note x1` |

Layer A gate values:

- `family_mean = 72.67` against required window `[15, 25]`
- `max_variant_mean = 86.67` against cap `40`
- `min_variant_mean = 65.00` against required hard-floor `<= 10`
- monotonic failed:
  - `v3-dirty-state (65.0) < v4-multi-corpus-objective (68.3)` beyond `+/-3`
  - `v4-multi-corpus-objective (68.3) < v5-recovery-in-thread (75.0)` beyond `+/-3`

Spot-check / diagnosis:

- The family is not close to the target hardness band. The frontier model fixes
  the core reducer/runtime task in most runs, leaving the dominant separator as
  the note-layer ceiling `stale_incident_note`.
- The strongest evidence is in the attempt report itself: `v4` run 2 scored
  `60` with raw `85`, and `v5` run 1 scored `60` with raw `80`, both because
  only `stale_incident_note` fired. That means the current family is mostly
  distinguishing whether the model echoes the variant context in the incident
  note, not whether it fails a materially harder runtime repair.
- Honest conclusion: this bundle is still frontier-easy on the underlying code
  task, and the current V2-V5 deltas do not yet create enough legitimate
  runtime difficulty to approach the §10.1 freeze gate.

Verification matrix refresh:

- `verification_matrix.md` (`v1-clean-baseline`) unchanged on rerun:
  Oracle `P=100/M=1.0000`, Empty `P=10/M=0.1111`, Render-filter shortcut
  `P=20/M=0.1667`, Delete-tests adversarial `P=0/M=0.0000/integrity=1`
- `verification_matrix_v5.md` (`v5-recovery-in-thread`) unchanged on rerun:
  Oracle `P=100/M=1.0000`, Empty `P=10/M=0.1111`, Render-filter shortcut
  `P=20/M=0.1667`, Delete-tests adversarial `P=0/M=0.0000/integrity=1`

Acceptance verdict:

- Layer A: `HARDEN NEEDED`
- Layer B: still materially present and locally revalidated, but Layer A is not
  honestly calibrated yet.

Next hardening hypothesis:

- Reduce dependence on the note-only ceiling as the primary separator and make
  V2-V5 variant deltas create a stronger runtime/evaluator distinction, or
  explicitly document this family as a frontier-easy signal that cannot reach
  the canonical `15-25` window without fake ambiguity.

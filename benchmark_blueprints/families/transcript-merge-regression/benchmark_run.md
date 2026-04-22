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

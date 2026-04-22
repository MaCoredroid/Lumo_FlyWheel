# Candidate Worktree Shootout

Repair the report-filter normalization bug by evaluating both candidate
strategies in isolated worktree directories and then landing one coherent
final patch in this workspace.

Required isolated-evaluation workflow:

1. create one isolated worktree directory for Candidate A and one for
   Candidate B (distinct directories are sufficient; the grader checks
   for concrete isolated paths, not the git worktree subcommand itself)
2. evaluate Candidate A in its own directory
3. evaluate Candidate B in its own directory
4. land only one final strategy in this workspace
5. write the comparison artifacts under `artifacts/comparison/`

Final editable surfaces:

- `src/report_filters/service.py`
- `src/report_filters/cli.py`
- `tests/test_service.py`
- `docs/filter_contract.md`
- `artifacts/comparison/*`

Immutable surfaces:

- `src/report_filters/normalization.py`
- `tests/test_cli.py`
- `artifacts/candidates/*`
- `repo_evidence/*`
- `release_context/*`
- `incident_context/*`

Required artifact files and headings:

- `artifacts/comparison/candidate_a_eval.md`
- `artifacts/comparison/candidate_b_eval.md`
- `artifacts/comparison/comparison_summary.md`
- `artifacts/comparison/verification_note.md`

Each candidate evaluation file must contain:

- `# Candidate A Evaluation` or `# Candidate B Evaluation`
- `- Worktree Path: ...`
- `## Commands Run`
- `## Touched Files`
- `## Observed Result`
- `## Evidence`

`comparison_summary.md` must name the chosen strategy and explain why the
rejected option is weaker. `verification_note.md` must record both
worktree paths plus the validation command used in each and the final
workspace.

Final validation command:

`python -m pytest -q tests/test_cli.py tests/test_service.py`

Variant note:

- Dirty state added: a half-finished CLI-only patch exists, but it is abandoned partial work, not a head start.

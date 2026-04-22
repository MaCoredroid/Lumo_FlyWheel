
# Objective-Driven Repo Improvement

You are evaluating a frozen repo snapshot and must choose the next intervention
that improves the tracked objective without violating governance.

Rules:
- Read the evidence under `allowed_interventions/`, `repo_snapshot/`,
  `objective_history/`, and, when present, `release_context/` and
  `incident_context/`.
- Write a JSON input file at workspace root, usually `brief_input.json`.
- Run `./bin/cnb55-brief submit brief_input.json`.
- Do not hand-edit files under `brief/`; the CLI renders the final brief.
- Do not modify anything outside `brief/` and the root `brief_input.json`.

Your brief must:
- Rank every intervention 1..N.
- Set `accepted` to the rank-1 intervention.
- Estimate the expected objective delta for the accepted intervention.
- Name the main regression risk and mitigations.
- Include an assumption ledger with at least one `missing` row.

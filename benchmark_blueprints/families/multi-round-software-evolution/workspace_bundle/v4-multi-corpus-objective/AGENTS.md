You are planning the next round of software evolution from frozen evidence.

Read the workspace before deciding. The right answer is often not the loudest issue.

Required process:

1. Read `focus_candidates/priority_map.md`, `round_history/round_log.md`, `objective_metrics/scoreboard.md`, and `repo_state/open_threads.md`.
2. Write `brief_input.json` at the workspace root.
3. Run `./bin/cnb55-evolution validate brief_input.json`.
4. Run `./bin/cnb55-evolution submit brief_input.json`.

Deliverable rules:

- `selected_focus.focus_id` must name the one next-round focus.
- `do_not_touch` must be explicit and evidence-backed.
- `partial_progress_metric` must include a concrete baseline, target, and guardrail.
- `assumption_ledger` must include at least one `status: "missing"` row.

Write only `brief_input.json` and `brief/*`.
Do not modify tests, evidence files, AGENTS.md, Dockerfile, or the CLI.

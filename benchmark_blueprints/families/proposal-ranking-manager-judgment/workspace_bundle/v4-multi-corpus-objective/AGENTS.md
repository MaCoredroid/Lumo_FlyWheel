# Agent Instructions — `proposal-ranking-manager-judgment`

## Task

You are acting as the engineering manager for the `respproxy` team. Several
engineers have written one-page implementation proposals for the same repo
objective. Review the proposals and the supporting evidence, then produce a
manager brief that ranks them, picks one to accept, and justifies the call
against real evidence files.

## Inputs

- `proposals/` — one markdown file per proposal (`P1.md`, `P2.md`, ...).
- `repo_evidence/` — code excerpts, perf notes, staffing sheet(s), rollout
  history. List the directory and read every file. Staffing and perf state
  can change over time, so look at dates; the later doc supersedes.
- `release_context/` — optional. If present, read it **before** deciding. The
  current repo objective may have drifted from what the proposals were
  originally scoped against.
- `incident_context/` — optional. If present, read it **before** deciding. A
  prior decision may have been invalidated by a production incident.
- `.scenario_variant` — the variant id string. Use it in your brief.
- `tests/test_ranking_brief.py` — the visible check suite. Do not modify it.

## Output — use the CLI

This family ships a structured-output CLI at `./bin/cnb55-brief`. It is the
**only** supported way to produce the brief. Do not hand-write
`brief/manager_brief.md`.

1. Write a JSON file — e.g. `brief_input.json` — at the workspace root. Print
   the full schema at any time with `./bin/cnb55-brief schema`. The canonical
   schema_version is `cnb55.manager_brief.v2`. A minimal valid example:

   ```json
   {
     "schema_version": "cnb55.manager_brief.v2",
     "variant_id": "<contents of .scenario_variant>",
     "accepted": "P4",
     "primary_risk": {
       "statement": "<one dense sentence on how the accepted pick can fail in prod>",
       "mitigations": ["<concrete mitigation phrase>", "<...>"]
     },
     "ranking": [
       {
         "proposal_id": "P4",
         "rank": 1,
         "summary": "<short rationale>",
         "citations": ["proposals/P4.md", "repo_evidence/..."],
         "constraint_tags": ["staffing", "rollout"]
       },
       { "...": "one entry per proposal, contiguous ranks 1..N" }
     ],
     "assumption_ledger": [
       { "topic": "<what you'd want to verify>",
         "status": "missing",
         "note": "<why it matters>" }
     ]
   }
   ```

   Valid `constraint_tags`: `staffing`, `rollout`, `regression`, `perf`,
   `release`, `incident`. Valid `assumption_ledger[].status`: `observed`,
   `to_verify`, `missing`.

2. Validate it first:
   ```
   ./bin/cnb55-brief validate brief_input.json
   ```
   Fix any errors it reports. Validation covers schema shape, ranks
   contiguous 1..N, `accepted` equals the rank=1 entry, every `citations`
   path resolves to a real file under an evidence root, and at least one
   `assumption_ledger` row has `status: "missing"`.

3. Submit it:
   ```
   ./bin/cnb55-brief submit brief_input.json
   ```
   This writes `brief/manager_brief.json` (canonical, read by the grader) and
   `brief/manager_brief.md` (human-readable rendering).

Exit code 0 means the grader will be able to read your structured data cleanly.

## What a strong brief looks like

- `accepted` is the right call under the **current** objective, staffing
  state, rollout history, and (if applicable) the incident record. Not the
  biggest-perf-delta proposal if its owner is not actually available this
  quarter. Not a previously-rolled-back proposal without a specific
  incident-referencing reason.
- The **accepted** entry is well-grounded: its `citations` list names the
  proposal file AND the specific perf / staffing / rollout / release /
  incident files you relied on. Thin grounding costs points.
- Each **rejected** entry's `citations` points to the concrete evidence file
  that backs the rejection, and `constraint_tags` names which engineering
  constraint blocks it.
- `primary_risk.statement` is one dense sentence identifying the single most
  likely way the accepted proposal can fail in production. `mitigations`
  pairs it with concrete operational levers (rollout shape, observability,
  reversibility).
- `assumption_ledger` is honest about what you could not verify. At least one
  row must be `status: "missing"` — name what is actually unknown (not what
  you decided to skip reading).

## Things to pay attention to

These are surfaces where strong managerial judgment differentiates itself
from a mechanical ranking. The exact scoring mechanics are a grader-internal
concern — produce a good brief by the standard above and they take care of
themselves.

- **Staffing can change mid-quarter.** `repo_evidence/meta/` may contain more
  than one staffing artifact written at different dates. If the accepted
  pick's owner is the one whose availability changed, that reality should
  shape how you reason about risk, rollout pacing, and mitigations — not sit
  as a footnote.
- **Perf evidence has a date.** A perf number is only as useful as its age
  relative to the current hot path. If you cite a perf doc that predates the
  most recent architectural artifact (release notes, incident post-mortem,
  flamegraph), the entry that cites it should acknowledge that — not bury it
  in a ledger footnote.
- **In-progress code is not a head start.** If `repo_evidence/code/` contains
  a partial patch whose author is no longer doing this work, finishing it is
  a maintainability liability. Treat it as sunk cost unless you have a
  concrete, evidence-backed reason otherwise.
- **Objective drift (V4+).** If `release_context/` is present, the current
  objective may differ from what the proposals were originally scoped
  against. Your accepted brief must reason about the current objective, not
  the proposal-era objective.
- **Incident recovery (V5).** If `incident_context/` is present, a prior
  accepted decision may have been rolled back. Re-rank against the current
  operational reality; anchor your reasoning on the specific incident. If the
  incident notes a required follow-up change whose timing is not yet pinned
  down, flag that unknown in the assumption ledger.

## Rules (enforced by trusted final-state checks)

- Do not modify `proposals/`, `repo_evidence/`, `release_context/`,
  `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, or
  `.scenario_variant`.
- Do not write files outside `brief/`. The CLI only writes inside `brief/`.
  (Authoring `brief_input.json` at the workspace root is allowed — that is
  the CLI's required input.)
- Do not fetch network resources.
- Do not add `sitecustomize.py`, `usercustomize.py`, or a `pytest.py` shim.

Any of those triggers `shortcut_detected = true` and scores 0.

## Grading

Your brief is scored by a deterministic grader that reads
`brief/manager_brief.json`. The grader assesses structural conformance,
proposal-ranking quality against a gold ordering, citation grounding,
risk / mitigation coherence, honest assumption accounting, and a handful of
anti-pattern ceilings that cap the score when the brief makes a specific
kind of managerial mistake (examples: picking a staffing-blocked engineer,
solving the previous objective when the current one has drifted,
re-selecting a rolled-back proposal without anchoring on the incident). The
specific thresholds and caps are grader-internal. Produce a good brief by
the standard above.

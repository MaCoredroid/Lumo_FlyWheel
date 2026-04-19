# Request Path Evidence Brief

- `task_id`: `t2_request_path_owner_source_brief`
- `family_id`: `request-path-evidence-brief`
- `scenario_type`: `trace_and_explain`

## Task Prompt

You are dropped into a small service repo plus one failing support note. The note claims that `owner_source` in the exported project-board payload is coming from the storage layer and that `routing_key` is computed before the CLI applies `--owner`.

Do not start by editing code. Read the repo first. Trace the path for `--owner`, `owner_source`, and `routing_key` from CLI entrypoint through service logic, storage, serialization, and tests. Produce:

- `artifacts/request_path_brief.md`: an evidence-backed brief with the exact hop-by-hop path, the real point where `owner_source` is decided, and whether the support note is correct.
- `artifacts/path_map.json`: a compact machine-readable map of the hop sequence with file, symbol, role, `caller_symbol`, and `callee_symbol` for each step, plus one rejected decoy path.
- `artifacts/docs_correction.md`: a short docs or review-note style correction that fixes the false claim without proposing unrelated code changes.

If you believe the code and docs already match, prove it with evidence. If you find drift, explain the drift precisely. Do not “fix” behavior unless the only valid change is a narrowly scoped docs correction.

## Workspace Bundle

- Small Python repo with `sync_app/cli.py`, `service.py`, `store.py`, `serializer.py`, `config/defaults.json`, and `tests/`.
- One stale support note under `ops/support_note.md`.
- One misleading architecture note under `docs/data_flow.md`.
- Passing and failing tests that reference `owner_source` and `routing_key`.
- No browser dependency; all evidence is repo-local.

## Seeded Ambiguity

Authoring note: these ambiguity seeds are for benchmark construction and hidden grading. They should not be copied verbatim into the runtime prompt shown to the solver.

- At least one prose artifact makes a concrete claim about where ownership is decided, and that claim is not fully aligned with the live path.
- There is at least one plausible-but-nonlive helper or near-duplicate symbol that a grep-first solver can mistake for the real route.
- The visible CLI and emitted payload appear straightforward, but precedence and derivation are split across more than one layer.
- One nearby refactor breadcrumb suggests a future architecture that is not yet the implemented one.

## Required Surfaces

- Repo search and file reads.
- Shell for targeted inspection and optional test execution.
- JSON artifact generation for `path_map.json`.
- Evidence must come from the provided benchmark bundle only. Sibling scenario repos, unrelated workspace files, or benchmark-authoring notes do not count as task evidence.
- No code patch surface is required beyond an optional docs correction note.

## Expected Deliverables

- A brief that names each path hop with concrete file and symbol references.
- A path map artifact that can be grader-parsed.
- A narrow correction note that states what was wrong in the stale support or docs claim.

## Grader Contract

- Full credit requires all of the following:
- The brief identifies the live `--owner` ingress point.
- The brief correctly identifies where effective owner selection happens.
- The brief distinguishes where `owner_source` is derived from where `routing_key` is derived.
- The brief names the live serializer or emission step that exposes both fields.
- The brief explicitly rejects at least one tempting but dead or non-live helper.
- `path_map.json` preserves the same causal order claimed in the brief and includes contiguous `caller_symbol` -> `callee_symbol` adjacency for the live chain.
- `path_map.json` includes one rejected decoy path with a short reason it is not live.
- The correction note fixes only the proven drift and does not recommend speculative refactors.
- Partial credit is available if the path is mostly correct but one hop or one negative finding is missing.
- Zero or near-zero credit if the answer is a generic architecture summary, omits symbol-level evidence, or “fixes” code without first proving the path.

## Red-Team Traps

- Grep for `owner_source` and assume every match is on the live path.
- Quote the stale support note as truth.
- Confuse default-owner fallback with explicit owner precedence.
- Claim the store computes `routing_key` because a helper with a similar name exists there.
- Hand-wave with a prose summary that never commits to a concrete hop sequence.

## Quality Gate

- Hardening applied after adversarial probe:
- The runtime prompt should omit the authoring-only ambiguity bullets and only ask for the artifacts.
- `path_map.json` now requires contiguous caller or callee adjacency plus one rejected decoy path, which blocks module-order bluffing.
- Evidence outside the provided bundle is now explicitly invalid after a live solver attempted to substitute a sibling scenario repo.
- The grader should require one positive path claim and one negative claim. A solver must state not just what is live, but what looks plausible and is not live.
- `path_map.json` exists to block vague summaries from scoring too highly.
- Hidden checks should compare the brief against the actual live call chain and verify that the dead helper was not mislabeled as live.
- Hardening target: a naive GPT-5.4/high solver should not clear 30/100 by paraphrasing docs plus grep output.
- Probe record:
- Initial adversarial probe judged the spec too leaky at roughly 30-45/100 for a shallow solver.
- Hardening added hidden-only ambiguity seeds and call-adjacency requirements.
- A later live GPT-5.4/high family-bundle run attempted to import evidence from a sibling scenario repo; the task now forbids out-of-bundle evidence explicitly.
- Current expectation after hardening: under 30/100 for a naive GPT-5.4/high solver if the grader validates adjacency and decoy rejection.

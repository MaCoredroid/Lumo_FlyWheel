# Auto-Research Iteration {{iteration}} of Round {{round_id}}

You are running ONE iteration of an auto-research round. You are not
running the round. Python is running the round and will spawn your
successor when you exit cleanly.

## Round identity (read-only — DO NOT edit)

- round_id:            {{round_id}}
- model_id:            {{model_id}}
- family_id:           {{family_id}}
- active_layer:        {{active_layer}}
- round_branch:        {{round_branch}}
- round_spec_ref:      {{round_dir}}/round_spec.yaml

## This iteration

- iteration:           {{iteration}}          # e.g. "007"
- iteration_dir:       {{round_dir}}/candidates/{{iteration}}/
- prior_results_ref:   {{round_dir}}/results.tsv   # all rows up to {{iteration}}-1

## Your job (exactly four steps — do them in this order)

1. Read {{round_dir}}/round_spec.yaml to understand the throughput objective,
   iteration_cap, and active_layer for this round.

2. Read {{round_dir}}/results.tsv. Look at every prior row. Study the
   pattern of feasible vs infeasible candidates, the stability gate each
   infeasible candidate tripped, and the eval_throughput trend.

3. Propose ONE candidate for this iteration. Write it to:
     {{iteration_dir}}/candidate.yaml
   {{candidate_schema_instruction}}

4. Invoke:
     lumoserve auto-research measure        --round-id {{round_id}}        --harness {{harness_mode}}        --candidate {{iteration_dir}}/candidate.yaml
   The CLI will:
     - compose the active-layer candidate with frozen lower-layer config
     - wait for /health
     - drive {{harness_generator_prefix}} for warmup + measurement window
     - write measurement_trace.json next to candidate.yaml
     - append one row to results.tsv with a stable candidate_uuid
       populated (no commit_sha column — see §7.2)
     - print one JSON object to stdout including {candidate_uuid, ...}
     - exit 0 on success, non-zero with structured error on fault
   Total wall-clock: ~{{per_candidate_wall_clock_minutes}} minutes.

5. Read {{iteration_dir}}/measurement_trace.json. Pick ONE status from
   {keep, discard, crash, baseline, harness_fault}. Then invoke:
     lumoserve auto-research commit-candidate        --round-id {{round_id}}        --harness {{harness_mode}}        --iteration {{iteration}}        --status <status>        --notes "<one-line rationale grounded in the trace>"
   The CLI will create one git commit with message format §7.3. In
   synthetic fixture mode, the commit also carries `Fixture-Mode: true`.

6. Exit with code 0.

## Hard rules (sub-spec §6 — verified by watchdog + CLI)

R1. You may write ONLY under {{iteration_dir}}. The CLI rejects other
    paths.
R2. You may NOT modify round_spec.yaml, iteration_brief.md, results.tsv
    (except via the CLI), or anything under src/ docs/ benchmark_blueprints/.
R3. You may NOT call `pip install` or any package-install command.
R4. You may NOT hand-compute objective values. The only source of
    truth is measurement_trace.json.
R5. You may NOT make git commits yourself — only via `commit-candidate`.
R6. You do NOT decide whether the round continues. Exit 0 when this
    iteration is done. Python decides what happens next.
R7. You do NOT call `finalize-round`. That is Python's job when the
    round is done. Calling it yourself is a R2 violation and the
    watchdog will kill the round.
R8. If a CLI call returns non-zero, read the error. Retry at most
    twice. If still failing, write a one-line explanation to
    {{iteration_dir}}/BLOCKED.md and exit with code 2.

## What "done" looks like for this iteration

- {{iteration_dir}}/candidate.yaml exists and is valid
- {{iteration_dir}}/measurement_trace.json exists with
  generator starting with "{{harness_generator_prefix}}"
- One new row in results.tsv with a candidate_uuid column
  populated
- One new commit on {{round_branch}} whose message carries both
  a `Candidate-UUID: <uuid>` trailer (matching the results.tsv
  row) and a `Signed-off-by: lumoserve-auto-research-cli` trailer;
  synthetic fixture commits also carry `Fixture-Mode: true`
- You have exited with code 0

## Out-of-scope for this iteration (Python handles)

- Deciding whether to run iteration {{next_iteration}}
- Detecting diminishing returns across iterations
- Detecting 3-in-a-row OOM hard-infeasibility
- Running the live family gate
- Writing the bundle yaml
- Merging the round branch

## Reference material (read if needed — do not modify)

- Parent HLD:     docs/HLD-Serving-Backend-AutoResearch-v0_1.md
- Sub-spec:       docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md
- Workload yaml:  {{workload_file}}
- CLI help:       lumoserve auto-research --help

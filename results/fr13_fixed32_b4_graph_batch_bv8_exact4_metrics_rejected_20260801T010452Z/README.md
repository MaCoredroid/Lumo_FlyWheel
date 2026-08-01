# Fixed32 B4 BV8 repaired-lifecycle run: provenance rejection

This bundle records the rejected real SWE-bench Verified exact4 run rooted at:

`output/fr13_fixed32_b4_graph_batch_bv8_exact4_repaired_20260801T010452Z`

The run was launched from source commit
`92d705c31f375b8a8d42a911eaa73104c722b075`. It is an exact4 B4 graph
byte diagnostic. It is not timing eligible and is not hardware-floor
acceptance eligible.

## Result

The exact4 campaign is rejected at per-task Qwen provenance. Four tasks ran
concurrently from zero-valued vLLM metric baselines, so every per-task post
scrape contained campaign-global cumulative completions rather than the
completions attributable to that task.

| Task | Per-task ingress count | Global post count | Excess |
| --- | ---: | ---: | ---: |
| `astropy__astropy-12907` | 13 | 53 | 40 |
| `astropy__astropy-13033` | 16 | 73 | 57 |
| `astropy__astropy-13236` | 32 | 79 | 47 |
| `astropy__astropy-13398` | 93 | 154 | 61 |

The four ingress counts sum to the finalized campaign count of 154. The first
surfaced failure was `astropy__astropy-12907`: the provenance contract
expected 13 completed logical model requests and observed 53 in its global
metric delta. The orchestrator then exited with RC 1. The other three metric
pairs are non-isolated for the same reason.

All four Qwen traces ended with lifecycle result `success`, but no task has a
`runner_metadata.json` or evaluation verdict. This bundle therefore does not
claim that any task resolved.

## Lifecycle repair

The prior incomplete-drafter-proposal flush failure did not recur:

- Generations 1 through 4 are successful zero-traffic pretask snapshots.
- Generations 5 through 8 are successful post-task snapshots with closed
  cumulative event counts 314, 504, 752, and 6424.
- Generation 9 is a successful final flush with no pending SFWD, DFWD, or
  CFWD work and forward steps 0 through 6423.
- The final flush client result is nonempty and its stderr file is empty.
- The work census contains 6424 event records plus one terminal record.

The lifecycle repair is valid evidence. The per-task snapshot intervals begin
at their zero-traffic pre-boundaries and overlap because the tasks run at B4;
they are not disjoint per-task performance intervals.

## Kernel evidence

The independent B4 batched-BV8 byte gate is valid:

- Authenticated marker: `swe_verified:astropy__astropy-13398`.
- 48 distinct layers under one graph identity and signature.
- All 48 records pass nine candidate/reference byte surfaces.
- 5,009,179,200 candidate/reference bytes compare equal.
- 177,340,800 graph-baseline bytes compare equal across six surfaces.
- The reference path was restored and served after every shadow comparison.
- Candidate recurrence launches are 2 per layer versus 8 for the reference.
- The PASS declares `production_eligible: true` and its kernel source hash
  matches the launch runtime manifest.

The launcher-level formal validator did not run because the arm returned RC 1,
so `b4_gdn_wide_gate_verdict.json` was never written. The kernel PASS is
production-eligible parity evidence; this rejected run is not a completed
runner-level exact4 qualification.

## Evidence layout

- `verdict.json`: machine-readable rejection and evidence classification.
- `provenance/metric_overlap_summary.json`: exact task counts, metric values,
  source hashes, trace hashes, and boundary associations.
- `lifecycle/`: all nine snapshots, four task boundaries, and the zero-traffic
  pretask proof.
- `flush/`: final generation-9 request, ACK, client result, and empty stderr.
- `kernel/`: complete 48-record JSONL and graph live PASS.
- `ingress/`: finalized summaries and selected hash-chain records.
- `census/`: the exact terminal record and a hash/size/count summary for the
  excluded 41,959,003-byte raw census.
- `runtime/` and `attestation/`: immutable launch/end manifests and safe
  runtime identities.
- `cleanup/`: run teardown evidence and a later live cleanup check.

## Deliberate exclusions

Raw prompts, Qwen traces, task workspaces, DCGM streams, full container logs,
raw ingress ledgers, raw vLLM metric dumps, and the full work census are not
bundled. Their required hashes and selected facts are recorded instead.

`container_env.txt` and `fixed32_process_identity.json` are specifically
excluded because they capture sensitive environment values. The bundle also
excludes all ingress secret material and authorization headers.

Run `sha256sum -c SHA256SUMS` from this directory to verify every bundled file
except `SHA256SUMS` itself.

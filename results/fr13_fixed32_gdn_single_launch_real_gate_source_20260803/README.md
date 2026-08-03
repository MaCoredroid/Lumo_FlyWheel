# Fixed32 ordered GDN real-task gate runners

Verdict: **SOURCE_READY_REAL_TASK_RUNNERS_UNQUALIFIED**.

This supersedes the earlier route-only and provenance-v2 source records. The
default-off
`fixed32_gdn_single_launch_tree_v2` diagnostic now has three fixed-scope real
SWE-Verified entrypoints:

- Hydra27 B1 on the pinned one-task diagnostic.
- Tail23 B4 on the canonical exact4 task set.
- Hydra27 B4 on the canonical exact4 task set.

Each process bakes exactly one expected batch. Runtime captures are internally
keyed by batch, FULL-graph identity, and capture signature; a B1 replay cannot
consume a B4 capture. Capture closures are retained, and each distinct request
digest tuple drives `armed -> running -> armed`: reference first, candidate
second, and baseline restoration in `finally`, without serving candidate
bytes. There is no process-global PASS after the first replay.

Every successful comparison is embedded in its containing work-census event.
The terminal `events_sha256` therefore seals its runtime capture-manifest and
structural signatures, candidate/reference identities and launch counts,
compared surfaces, byte equality, restoration and served arm, event identity,
and `drafter_runtime.request_id_sha256s`. The live observation is only an exact
ordered mirror of those events; it is not independently sufficient to pass.

The reducer does not trust the live result alone. It requires clean task
completion and terminal SWE verdicts, reconstructs the finalized authenticated
traffic audit from the pinned SWE-Verified Parquet record digests rather than
runner metadata, replays the exact Qwen compaction algebra and per-task/campaign
proof bindings, validates finalized proxy and engine ingress, and validates the
complete graph/work census. It maps every comparator request digest through the
validated proxy/engine ledgers and requires the comparator-event task union to
be exactly the one B1 task or canonical exact4 B4 set. It regenerates the
runtime manifest and requires
launch/end byte equality. The manifest carries a Git/source identity, and every
closed runtime source plus the selected entrypoint, reducer, kernel, patcher,
block map, validators, subset, and affected tests must equal exact
`git show <source_commit>:<path>` bytes at current `HEAD`. Credentials are
distinct for `hydra27:b1`, `tail23:b4`, and `hydra27:b4`.

## Production boundary

Every live result and credential records `performance_measurement=false` and
`acceptance_valid=false`. The production resolver still rejects
`single_launch`. Production and timing remain unavailable until all three real
GPU gates pass and a later production credential binds those disjoint results.

## Validation

- Combined GDN, runner/reducer, ingress, draft-head, timing-contract, and final
  FULL-preseed compatibility suite: `286 passed, 1 skipped` (the skip requires
  CUDA/Triton).
- Adversarial coverage rejects an absent/forged commit, dirty tracked source,
  a self-consistent forged metadata/audit dataset digest, stale observation
  substitution, missing or duplicated exact4 comparator coverage, event-index
  swaps, request/task relabeling, truncation, scope swaps, later restoration
  failure, and an asserted `graph_id` in durable evidence.
- Focused GDN and campaign suite: `53 passed`; work-census self-test: PASS with
  `177` tamper cases.
- `bash -n`, `py_compile`, and `git diff --check`: pass.

No GPU kernel, SWE-Verified task, probe, timing arm, TPS measurement,
hardware-floor measurement, or production authorization ran for this source
artifact.

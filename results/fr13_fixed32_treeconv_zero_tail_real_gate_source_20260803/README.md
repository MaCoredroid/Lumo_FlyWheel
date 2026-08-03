# Fixed32 tree-conv zero-tail real-task gate source

Status: graph-bound implementation checkpoint; no live PASS, timing result, or
hardware-floor acceptance is claimed.

This checkpoint makes the physical32 tree-conv zero-tail candidate qualifiable
on the fixed K64/root1 Tail23 and Hydra27 serving arms at B1 and exact4 B4. The
diagnostic is default-off, requires `FULL_AND_PIECEWISE` CUDA graphs, and is
mutually exclusive with the production selector and other kernel diagnostics.

For each measured real decode event, the captured graph launches the zero-tail
candidate, compares every BF16 destination byte on device, then launches the
incumbent direct kernel to restore the served state. Device-resident enable,
event, and mismatch counters are read only after the campaign's final global
synchronization. Capture and warmup replays are excluded from the count.

The credential issuer now fails closed unless all of the following agree:

- byte-identical canonical runtime manifests at launch and end, the arm's
  runtime Git head, and `git show` bytes for every host script, Python package
  source, and verdict tool in the executed manifest closure;
- the finalized v12 work census, terminal census, final flush, and immutable
  boundary snapshot, with exact zero-based event/forward indices;
- the pristine ready ack, every task's pre/post runtime snapshot, an exact
  task-generation and nonce chain, the final request/current ack, and no stale
  boundary-snapshot generations;
- comparator events joined one-to-one to real work by event, forward, process,
  batch, and request identity;
- successful proxy and engine ingress ledgers joined one-to-one to the same
  request identities;
- an independently rebuilt chat-traffic audit from the pinned SWE-Verified
  Parquet: exact B1 or ordered exact4 task records, eval and offload artifacts,
  agent/codex terminal metadata, Qwen runtime attestation, ingress, and task
  intervals must match the persisted audit byte-for-byte;
- semantic Qwen 0.19.4 trace and metric replay for B1, or the canonical
  concurrent-campaign proof for B4;
- a uniquely keyed container environment pinned to physical32 and K64/root1;
- separate Tail23 and Hydra27 logical topology identities, with their canonical
  masks, sharing the same physical32 parent table and state-source descriptor.

B1 and B4 outputs are correctness credentials only. Neither is timing or
floor-acceptance evidence. Raw tasks, prompts, responses, patches, identities,
and logs are not included in this source checkpoint.

## Verification

- 63 credential, graph comparator, boundary-snapshot, topology, work-census,
  and codegen-artifact tests passed.
- 73 Qwen campaign/provenance and ingress tests passed.
- 22 adjacent committer/profiler tests passed; one CUDA-only test skipped.
- Python compilation, shell syntax, and `git diff --check` passed.
- The monolithic floor-gate self-test was attempted but is not counted: its
  pre-existing runtime-closure cardinality pin expects 62 files/25 Python
  files, while the current fixed32 manifest contains 92/26.
- No GPU was available. The production candidate has an existing offline
  codegen artifact; the new Triton graph comparator was not locally compiled or
  timed, and the real B1/B4 campaigns have not run.

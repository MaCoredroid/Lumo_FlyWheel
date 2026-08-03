# Fixed32 tree-conv zero-tail real-task gate source

Status: ready for real SWE-Verified qualification; no live PASS is claimed.

This source checkpoint makes the physical32 zero-tail specialization
qualifiable on the fixed K64/root1 Tail23 and Hydra27 serving arms at B1 and
exact4 B4. The diagnostic is default-off, eager-only, and mutually exclusive
with the production selector and every other kernel diagnostic.

For each compared real decode event, the serving route launches the zero-tail
candidate, snapshots the complete 48-layer BF16 destination rows, launches the
incumbent kernel, and compares raw bytes. The incumbent bytes remain in the
live state and are the only bytes served. A mismatch is recorded after the
incumbent restore and then fails the request.

The credential issuer requires:

- a non-vacuous, contiguous comparison stream capped at 320 events;
- exact physical32 C10240/L34/source36 geometry and one target-B event;
- a completed real SWE-Verified B1 or canonical exact4 task set;
- a finalized authenticated engine ingress ledger;
- the fixed32 work-census event stream and explicit eager-terminal no-flush
  record (a graph-census terminal claim is rejected);
- source commit, source-file, runtime-manifest, and state-descriptor hashes;
- the existing Qwen campaign compaction proof for exact4 B4.

B1 output remains diagnostic-only. B4 output is a correctness credential only;
neither output is timing or floor-acceptance evidence. Raw tasks, prompts,
responses, patches, process identities, container identities, and logs are not
included here.

## Verification

- 12 focused tree-conv gate and credential tests passed.
- 84 adjacent fixed32 ingress, campaign provenance, committer, and CUDA
  contract tests passed; one CUDA runtime module was skipped in the occupied
  shared-GPU environment.
- Python compilation, shell syntax, and `git diff --check` passed.

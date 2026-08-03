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
- the byte-exact pinned real SWE-Verified B1 or ordered canonical exact4 subset;
- a finalized authenticated engine ingress ledger;
- every fixed32 work-census event to pass the v12 exact physical-work validator,
  plus contiguous indices, one producer, and the eager-terminal no-flush record;
- a live container environment pinned to physical32, K64/root1, and diagnostic
  candidate mode;
- the live source bytes to equal `git show <commit>:<kernel>`, with the same
  bytes bound by the runtime manifest;
- for exact4 B4, the Qwen concurrent-campaign union proof, endpoint metrics,
  ordered task traces, and each task's runner metadata/provenance identity.

B1 output remains diagnostic-only. B4 output is a correctness credential only;
neither output is timing or floor-acceptance evidence. Raw tasks, prompts,
responses, patches, process identities, container identities, and logs are not
included here.

## Verification

- 18 focused tree-conv gate and credential tests passed after merging current
  `main`, including independent tamper tests for every binding above.
- 31 adjacent fixed32 and merged M32 host wiring tests passed. No GPU or CUDA
  runtime test was invoked in this provenance-hardening pass.
- Python compilation, shell syntax, and `git diff --check` passed.

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

The credential issuer fails closed unless all of the following agree:

- the complete canonical fixed32 runtime manifest and the bytes at its source
  commit;
- the finalized v12 work census, terminal census, final flush, and immutable
  boundary snapshot;
- comparator events joined one-to-one to real work by event, forward, process,
  batch, and request identity;
- successful proxy and engine ingress ledgers joined one-to-one to the same
  request identities;
- the pinned real SWE-Verified B1 task or ordered canonical exact4 B4 subset,
  task boundary/authentication records, and agent terminal;
- semantic Qwen 0.19.4 trace and metric replay for B1, or the canonical
  concurrent-campaign proof for B4;
- separate Tail23 and Hydra27 logical topology identities, with their canonical
  masks, sharing the same physical32 parent table and state-source descriptor.

B1 and B4 outputs are correctness credentials only. Neither is timing or
floor-acceptance evidence. Raw tasks, prompts, responses, patches, identities,
and logs are not included in this source checkpoint.

## Verification

- 127 graph comparator, credential, topology, work-census, and codegen-artifact
  tests passed after merging current `main`.
- 32 Qwen campaign/provenance and ingress tests passed.
- 22 tests for the newly merged CUTLASS production and B4 timing work passed.
- Python compilation, shell syntax, and `git diff --check` passed.
- No GPU was available. The production candidate has an existing offline
  codegen artifact; the new Triton graph comparator was not locally compiled or
  timed, and the real B1/B4 campaigns have not run.

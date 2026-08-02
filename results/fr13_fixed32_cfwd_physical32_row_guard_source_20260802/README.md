# Fixed32 CFWD physical32 row guard

Status: **source/static only; codegen, real byte qualification, and timing
pending**.

This bundle binds the source-only candidate at commit
`8ece7de91e3ab054275448653c9d3b72ba5e7d6c` on branch
`agent/fixed32-committer-physical32-rowguard-20260802`.

## Change

The direct conv committer previously constructed its fail-closed precommit
guard from an eager PyTorch chain over the live fixed32 operands. That chain
included integer casts, destination construction and sort, an arange, active
path masks, a gather, a where, and multiple reductions.

The candidate replaces that chain with one Triton row-guard launch. Its grid is
exactly 48 programs per request. Each program validates exactly 32 physical SSI
rows, 16 path slots, one accepted length, one selected leaf row, and 256
possible physical destination slots. The output is a persistent 48-by-B bool
workspace followed by one scalar `all` reduction and one asynchronous device
assertion. No host readback is added.

The work is independent of the actual Tail23 or Hydra27 active-node count. It
depends only on the fixed physical32 capacity and batch size. B1 and B4 use the
same kernel and per-request work shape.

## Semantics

The guard preserves the prior fail-closed domain:

- every physical SSI row is in `[1, bank_rows)`;
- every active path node is in `[0, 32)`;
- accepted length is in `[0, 11]`;
- the selected leaf row is in range;
- every alias id is in `[0, 16)`;
- all physical `(alias_id, running_row)` destinations are unique.

On the valid alias and row domains, pair equality is equivalent to equality of
the prior encoded destination `alias_id * bank_rows + running_row`. Focused CPU
tests compare both guard formulations for B1 and B4 across valid inputs and
each rejection class. Inactive path padding remains ignored.

## Evidence boundary

Focused host-only structural and semantic tests passed, and the v10 work
census passed all tamper cases. No Triton/CUDA code generation, SASS/resource
inspection, Docker/GPU execution, synthetic performance probe, real
SWE-Verified byte gate, B1/B4 timing campaign, hardware-floor measurement, or
one-sided U95 acceptance test was run for this bundle.

This bundle contains no prompts, responses, traces, raw logs, task identifiers,
container identities, process identities, credentials, or secrets.

# Fixed32 CFWD owner-path row guard

Status: **source/static only; codegen, real byte qualification, and timing
pending**.

This bundle binds source commit
`392c16929b40d527f5097eb198479f3370fae9f8` on branch
`agent/fixed32-committer-ownerpath-rowguard-20260802`.

## Change

The alias3 predecessor runs one fixed row-guard program per layer/request. Each
program must still validate its 32 physical SSI rows and compare its running
destination with the three preseeded alias peers. Path/length validation,
selected-row validation, and alias topology validation were also repeated in
every program even though those inputs are shared across layers or events.

This candidate assigns shared validation to fixed owners inside the existing
kernel grid:

- layer 0 validates one 16-slot path vector and one accepted length per request;
- `(layer 0, request 0)` validates all 48 alias IDs in one padded 64-lane vector;
- all 48 layer programs still validate all 32 SSI rows and alias-local running
  destination uniqueness for every request;
- selected-row validation is removed because every one of the 32 candidate SSI
  rows is already range checked before any selected row can be used;
- peer alias membership is bound by the preseed lease audit, while each event
  still checks peer table indices before loading comparison rows.

The event grid remains `48 * B`, the final device reduction/assert remains one
per event, and no event-time launch, allocation, host readback, result-tree
change, or recurrent math change is introduced.

## Static work reduction

Relative to the alias3 predecessor:

- path-vector loads fall from `48 * B` to `B`, a 48x reduction;
- accepted-length scalar loads fall from `48 * B` to `B`, a 48x reduction;
- selected-path and selected-row scalar loads fall from `48 * B` each to zero;
- alias validation programs fall from `48 * B` to one per event;
- logical alias-ID values read for validation fall from 192 to 48 at B1 and
  from 2496 to 48 at B4, before compiler/cache effects.

The physical-row loads and alias-local destination comparisons intentionally
remain unchanged. Triton control-flow lowering, SASS resource use, and elapsed
latency must be checked on the GPU teardown gate before any performance claim.

## Evidence boundary

Host-only semantic and structural tests passed, and the v12 work census passed
all 177 tamper cases. No Triton/CUDA code generation, SASS/resource inspection,
Docker/GPU execution, synthetic performance probe, real SWE-Verified byte
gate, B1/B4 timing campaign, hardware-floor measurement, or one-sided U95
acceptance test was run for this bundle.

This bundle contains no prompts, responses, traces, raw logs, task identifiers,
container identities, process identities, credentials, or secrets.

# Fixed32 CFWD alias-local row guard

Status: **source/static only; codegen, real byte qualification, and timing
pending**.

This bundle binds commit `029427c02720d2759145d1405945baa87a02092b` on
branch `agent/fixed32-committer-alias3-rowguard-20260802`.

## Change

The preceding physical32 row-guard candidate compared every layer/request
destination against a fixed 256-lane block. The preseeded bank topology already
proves exactly 16 physical alias groups with exactly three layers per group.

This candidate precomputes one immutable `48 x 3` int32 peer-layer table at
boot. Each row-guard program now checks only its three alias peers across the
active batch inside a 16-lane block. It also verifies that every peer entry is
in the 48-layer domain and has the expected alias id.

The fixed candidate count falls from `48 * B` to `3 * B` per program, a 16x
reduction for every supported batch. The vector comparison capacity falls from
256 to 16 lanes. The event grid remains `48 * B`; there is no new event-time
launch, allocation, host readback, result-tree change, or recurrent math
change. Persistent storage increases by 576 bytes for the peer table.

## Semantics

Physical destination equality is defined by `(alias_id, running_row)`. Rows in
different alias groups refer to different physical banks, so only the three
layers in the same alias group can collide. The peer table is derived directly
from the validated 16-by-3 bank alias classes and is bound by object identity,
data pointer, shape, stride, dtype, host topology, observer layout digest, and
the preseed lease audit.

Focused B1 and B4 CPU tests compare the alias-local formulation with the prior
global destination guard over valid inputs and all rejection classes. The
tests use noncontiguous peer layer indices to cover the physical alias layout.

## Evidence boundary

Host-only structural and semantic tests passed, and the v11 work census passed
all tamper cases. No Triton/CUDA code generation, SASS/resource inspection,
Docker/GPU execution, synthetic performance probe, real SWE-Verified byte
gate, B1/B4 timing campaign, hardware-floor measurement, or one-sided U95
acceptance test was run for this bundle.

This bundle contains no prompts, responses, traces, raw logs, task identifiers,
container identities, process identities, credentials, or secrets.

# Dense grouped-GDN source checkpoint

This reduced bundle records the source-only checkpoint for
`fixed32_gdn_parent_group_dense_simd_v3` at source commit
`73413f28738812d468a2268ca0bab69f910169b2`.

The level-1 grouped SIMD launch keeps the established five parent groups,
width four, two launches per layer, five parent-state loads per request, 32
single-writer physical outputs, and critical path 12. The node recurrence and
active member order are unchanged.

The candidate replaces runtime path count, path index, path length, path-max,
and separate parent-reference loads with a fixed `[5,4,7]` int32 node schedule
and one packed int32 control load per grouped program. The packed controls are
`(1934, 1792, 289, 324, 361)` and bind parent node, compact export slot, and
group maximum path length. The canonical dense-schedule SHA-256 is
`500f9821c279c030fd0f42080d2a5410e5fef4de3d476c5ddf39f23c6a27f915`.

A single-launch composition was deliberately rejected from this checkpoint.
It would raise the grouped critical chain from 12 to 22 node steps without
real timing evidence that one fewer launch repays the lost group parallelism.
No single-launch lifecycle code is present here.

## Evidence boundary

Host-only Python syntax, shell syntax, contract, observer, and qualification
tests passed. No Triton or CUDA code generation, SASS inspection, Docker/GPU
run, real SWE-Verified byte qualification, timing campaign, hardware-floor
measurement, or U95 acceptance test was run for this source checkpoint.

Codegen resource accounting, exact4/exact16 B4 byte gates for both fixed32
modes, B1/B4 full-step timing, and hardware-floor acceptance remain pending.
Production stays default off and source-bound live credentials must be
regenerated for the v3 identity.

This bundle contains no prompts, responses, traces, raw logs, task IDs,
container identities, process identities, or secrets.

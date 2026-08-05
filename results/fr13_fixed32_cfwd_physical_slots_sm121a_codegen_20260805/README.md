# Fixed32 CFWD Physical-Slot SM121a Audit

This artifact binds the default-off
`fixed32_cfwd_logit_direct_physical_slots_v2` source checkpoint
`d2348ce9260292dcf6f9c687a774ed9966b92928` to offline SM121a codegen.

The candidate keeps the fixed 13 self plus 17 target decision programs per
request and their 81 stores, but writes those products into physical 31/32
slot workspaces. The one-program integer walk then indexes decisions directly
and removes its two topology-table reads at each of 12 unrolled levels.

Exact work delta per request:

- decision programs: `30 -> 30`
- decision values stored: `81 -> 81`
- integer commit launches/programs: `1/1 -> 1/1`
- topology-index scalar loads in the integer walk: `24 -> 0`
- persistent decision workspace: `529 -> 1048` bytes

Offline SM121a results for both B1 and B4 commit specializations:

- registers: `66 -> 64`
- stack/local/spill/calls: `0 -> 0`
- static encoded `LDG`: `118 -> 95`
- static non-control SASS: `747 -> 684`
- static `STG`: unchanged at `41`

The direct-decision producer remains at 80 registers with zero stack/local,
spill, or call use. The diagnostic comparator compiles spill-free at 35
registers for B1 and 32 for B4.

These are static codegen counts and an exact source work ledger. They are not
dynamic memory-traffic measurements and do not claim a runtime speedup. GPU
execution was not performed in this worktree. A real SWE-Verified one-task
byte-equivalence shadow gate remains required first, followed by the standing
4-task and 16-task performance gates before production or merge acceptance.


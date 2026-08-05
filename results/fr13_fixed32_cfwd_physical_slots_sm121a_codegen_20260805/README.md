# Fixed32 CFWD Preseeded Physical-Slot SM121a Audit

This artifact supersedes the unsafe `d2348ce9` checkpoint and binds the
default-off `fixed32_cfwd_logit_direct_physical_slots_v2` integration source
`a3443d40b9db4bf12475888a784a88812098d28e` to offline SM121a codegen. Its
frozen comparison base is current-main checkpoint
`640c98539dfdb78923615db14871971ef53b0f19`, which contains the production
BF16 SFWD `dt_bias` contract fix, the engaged Qrow16 SFWD byte gate, and the
sanitized Gate B result artifact.

The candidate keeps the fixed 13 self plus 17 target decision programs per
request and their 81 stores, but scatters those products into physical 31/32
slot workspaces. Every persistent workspace tensor is zero-seeded once before
graph capture. Unwritten leaf decision slots therefore have safe values without
adding initialization stores, dynamic load masks, or topology-map reads to a
measured replay. The executed physical committer is also included in the
fail-closed TAW runtime source contract.

Exact work delta per request:

- decision programs: `30 -> 30`
- decision values stored: `81 -> 81`
- integer commit launches/programs: `1/1 -> 1/1`
- topology-index scalar loads in the integer walk: `24 -> 0`
- decision-padding initialization stores per replay: `0`
- persistent decision workspace: `529 -> 1048` bytes

Offline SM121a results for both B1 and B4 commit specializations:

- registers: `66 -> 64`
- stack/local/spill/calls: `0 -> 0`
- static encoded `LDG`: `118 -> 95`
- static non-control SASS: `747 -> 684`
- static `STG`: unchanged at `41`

The direct-decision producer remains at 80 registers with zero stack/local,
spill, or call use. The diagnostic comparator compiles spill-free at 35
registers for B1 and 32 for B4. Two independent fresh-cache builds were
byte-identical at the summary level.

These are static codegen counts and an exact source work ledger. They are not
dynamic memory-traffic measurements and do not claim a runtime speedup. GPU
execution was not performed in this worktree. A real SWE-Verified one-task
byte-equivalence shadow gate remains required first, followed by the standing
4-task and 16-task performance gates before production or merge acceptance.

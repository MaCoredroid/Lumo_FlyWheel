# Fixed32 CFWD Packed-Event SM121a Audit

This source-only, default-off candidate advances the fixed32 CFWD path from
`fixed32_cfwd_logit_direct_physical_slots_v2` to
`fixed32_cfwd_logit_direct_packed_physical_slots_v3`. The frozen comparison is
main revision `1f7485ade5ec6bfacf51dde7afa514531effcbcd`; the candidate source
revision is `103030ea88ad7da28a4bcab187a57200be70756d`.

The producer preserves the rejection-sampling math and uniform inputs but
stores one packed `int64` event for each of the 17 target-parent slots. Bits
0-17 hold the emitted token, bits 18-22 hold the accepted child row (`node + 1`,
with zero meaning rejection), and bit 23 marks a parent event. Verifier vocab
248320 fits the 18-bit token field. The integer committer consumes only the 31
physical self tokens and 32 packed event slots; it no longer consumes target
source/selected/rejected/accepted tables or rereads tree metadata in its fixed
12-level walk. The physical grid stays fixed at 13 self plus 17 target decision
programs and 32 physical rows for both Tail23 and Hydra27, B1 and B4.

Exact work delta per request from physical-slot v2 to packed-event v3:

- decision programs: `30 -> 30`
- decision values stored: `81 -> 30`
- physical decision workspace: `1048 -> 504` bytes
- tree-metadata scalar loads in the fixed walk: `24 -> 0`
- integer commit launches/programs: `1/1 -> 1/1`

Offline SM121a codegen for both B1 and B4 commit specializations:

- registers: `64 -> 46`
- static `LDG`: `95 -> 35`
- static `STG`: unchanged at `41`
- static non-control SASS: `684 -> 509`
- encoded SASS: `696 -> 520`
- cubin size: `59656 -> 46176` bytes
- stack/local/`LDL`/`STL`/calls: `0 -> 0`

The direct-decision producer stays at 80 registers and 51 `LDG`, reduces `STG`
from 5 to 2, and adds seven non-control packing instructions (`2558 -> 2565`).
It remains free of stack, local, spill, and call use. The comparator is also
resource-clean at 40 registers for B1 and 38 for B4. Two independent empty-cache
SM121a builds produced byte-identical summaries with SHA256
`61398335033b43eca229366b04f9a8fcb440ee6f7e62a1de215c37a47b741342`.

Source contracts:

- candidate source: `fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3` /
  `5a9107306bdc37200448a6a5add2b84dfd839dc377b11009f218662c63abcc1c`
- CFWD integration: `fr13.fixed32.cfwd_logit_direct.integration_source.v2` /
  `a82ce3f5e526792ca45bb444212e5440e8444778f174fd0650accc4bb5f8558c`
- incumbent TAW: `fr13-fixed32-taw-all-parent-v7` /
  `998bc6331177469d6890f97f3e066e1d07c2ca2d8ab4bff723f32d5229fef290`

The CFWD integration functions and packed committer are outside the incumbent
TAW source-function contract, so its hash is unchanged and its existing B1/B4
credentials are not invalidated. The v3 CFWD selector remains credentialed and
default-off.

These results are static codegen counts plus CPU exact-semantics tests. They do
not claim a runtime speedup. GPU execution was not performed in this worktree.
A real SWE-Verified one-task byte-equivalence shadow gate is required first,
then the standing real 4-task timing and 16-task confirmation gates before
production or merge acceptance.

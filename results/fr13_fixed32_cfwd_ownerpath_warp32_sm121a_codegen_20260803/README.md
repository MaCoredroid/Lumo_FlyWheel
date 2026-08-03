# Fixed32 committer owner-path warp32 SM121a codegen

Status: **offline SM121a codegen passes; the narrow warp-local fix is kept**.

The merged owner-path v3 row guard at
`392c16929b40d527f5097eb198479f3370fae9f8` correctly removed repeated path,
selected-row, and alias-topology work. Offline code generation exposed one
narrow defect: validating 48 alias IDs as a padded 64-lane reduction required
cross-warp shared memory and barriers. The v4 implementation commit
`7ae35fb8f` validates the same 48 values as two 32-lane warp-local reductions.
The final codegen input is integrated tip
`47e411fb17c0e7f330399ef5698a06ef460c7401`.

All builds used the production B1/B4 tensor shapes, contiguous strides,
alignment specialization, four warps, one stage, and CUDA target `sm_121a`.
`BANK_ROWS=257` is the existing exhaustive semantic-test fixture; it affects
only the in-range comparison constant. CUDA visibility was explicitly empty.
No GPU kernel, service, task, request, timing, or acceptance run was launched.

## Codegen result

| Metric | alias3 incumbent B1/B4 | owner-path v3 B1/B4 | warp32 v4 B1/B4 |
|---|---:|---:|---:|
| Registers/thread | 26 / 26 | 18 / 17 | 18 / 16 |
| Stack/local bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| Launch shared bytes | 0 / 0 | 8 / 8 | 0 / 0 |
| ELF static shared bytes | 0 / 0 | 1024 / 1024 | 0 / 0 |
| BAR / LDS / STS | 0 / 0 / 0 | 2 / 2 / 2 | 0 / 0 / 0 |
| Encoded SASS | 144 / 176 | 152 / 184 | 144 / 168 |
| Static non-control SASS | 132 / 159 | 134 / 162 | 129 / 149 |
| Static LDG / STG | 10 / 1 | 7 / 1 | 8 / 1 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

The split adds one static LDG opcode versus v3 because each warp-local half
has its own instruction. It still reads exactly 48 alias values in the single
owner program. In exchange, it removes the cross-warp shared-memory reduction,
all two barriers and shared loads/stores, eight encoded instructions at B1,
sixteen at B4, and one B4 register.

## Fixed32 scaling

The guard remains one launch per event. Its grid is `48 * B`, so fixed
physical-row and peer comparisons scale from 48 programs at B1 to 192 at B4.
Path validation scales from one owner program to four, while alias validation
stays one program and 48 values per event.

Relative to the retained alias3 incumbent, source-visible logical values before
compiler/cache effects fall from 2,976 to 1,937 at B1 (34.91%) and from 17,088
to 11,060 at B4 (35.28%). Static SASS has two fewer LDG instructions in both
specializations. Launch count and the single guard flag store are unchanged.

## Decision

Keep v4 behind the fail-closed route
`fixed32_triton_alias3_ownerpath_warp32_physical32_v4`. The work census records
two alias vector loads per event, one alias-validation program, and 48 alias
values. This is static codegen evidence only; real SWE-Verified byte gates and
B1/B4 full-step timing remain mandatory before any speed or acceptance claim.

The checked-in package contains reduced summaries and reproduction code only.
It excludes cubin, PTX, SASS, compiler caches, raw logs, task/model/request/
response/patch content, credentials, environment dumps, process IDs, and
container IDs.

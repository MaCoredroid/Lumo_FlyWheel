# Fixed32 FA2 qrow32 paired SM121a codegen rejection

Status: **reject the static-query qrow32 candidate; retain the direct-page
incumbent**.

Both sides were regenerated from pinned FA2 commit
`29210221863736a08f71a866459e368ad1ac4a95` and compiled host-only with the
same CUDA 13.0 SM121a command. The exact incumbent is the parent of the query
specialization, `adc96dd0e5e179f4e9693829544ab2683325451d`; the candidate source
commit is `d47b03f942defed2ef4b4f63053d5bb7fe8152df`.

## B4 qrow32 verdict

The static-query specialization does remove the intended control work:

- SASS instructions: 4,824 to 4,512, down 312 (6.468%).
- Kernel text: 77,184 to 72,192 bytes, down 4,992 (6.468%).
- Conditional/reconvergence work falls, while the ordered attention math and
  memory pipeline remain unchanged: 512 BF16 HMMAs, 132 FFMAs, 264 FMULs,
  176 `LDGSTS`, 288 `LDSM`, and 38 global stores on each side.
- Stack, local memory, spill loads, spill stores, and SASS calls remain zero.

It nevertheless fails resource admission. Register allocation rises from 254
to 255 registers per thread. That is the architectural per-thread ceiling and
removes the incumbent's last register of headroom. No GPU timing can override
this offline gate because the standing admission rule rejects a register
regression before runtime qualification. The candidate remains default-off,
byte-unqualified, timing-ineligible, and not authorized for production.

## B1 control

qrow32 is not a B1 route: its host admission requires B4, total query rows 128,
and query prefix `[0, 32, 64, 96, 128]`. B1 continues to use qrow16, so a paired
qrow16 control was compiled under the deployed physical32 constants instead
of mislabeling a qrow32 binary as B1 evidence.

The B1 control has no resource regression: both sides use 244 registers, zero
stack/local/spills, one barrier, and the same mandatory math and memory
pipeline. Current generic-header code is 16 instructions and 256 text bytes
smaller (5,456 to 5,440 instructions; 87,296 to 87,040 bytes). The four
compiler integer-division calls are unchanged. This control does not qualify
or time qrow16; it only proves that the qrow32 source work did not degrade the
deployed B1 kernel code generation.

No GPU, container, synthetic performance probe, real task, raw task data,
shared-object relink, byte gate, or timing run was used. Twenty-two focused
source tests pass, Python compilation passes, and `git diff --check` passes.

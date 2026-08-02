# Verification

- The packed decoder matches the established descriptorless source triple for
  every physical node from 0 through 31.
- The focused source suite passes 12 tests.
- Python compilation, Ruff, and `git diff --check` pass.
- B1 and B4 produce identical cubin, PTX, and SASS for each schedule after
  launch-only metadata is removed.
- Separate fresh Triton cache trees reproduce both schedules byte for byte.
- Fresh `nvdisasm` output matches each stored temporary disassembly.
- Both schedules have zero stack, local memory, spills, shared memory, and
  calls.

All compilation used `CUDA_VISIBLE_DEVICES=` and target `sm_121a`. No kernel,
service, Docker container, request, SWE task, or timing run was launched by the
offline audit.

# Verification

- The candidate launcher AST contains no `source_flat`, `.cpu()`, `.item()`,
  or `.tolist()` operation.
- The patcher supplies `_fr10_parent`, which is already a host Python list.
- The launcher accepts only a host `list`/`tuple` of exact Python `int` values
  and requires equality with all 32 entries of `FIXED32_PARENT`.
- Altered topology, shortened topology, tensor input, boolean input, and
  non-iterable input are rejected by focused tests.
- The packed source/kernel, gate, final-full-preseed, and ingress suites pass
  88 tests.
- Python compilation, focused Ruff checks, shell syntax, and
  `git diff --check` pass.
- The reproducible source manifest is bound to source commit
  `078bd0f23bfa8ecb4faae4a72f16553c0339a8a1`.

No GPU, Docker container, real task, synthetic probe, or timing run was
launched for this source-only checkpoint. A fresh real K64/root1 B1 byte gate
is mandatory.

# Fixed32 FA2 qrow32 CALL0 source checkpoint

Status: **source candidate complete; compiled CALL0 verification pending**.

The saved SM121a qrow32 SASS contains four calls to one compiler-generated
signed 64-bit division helper. All four originate from paged-KV row divmod in
`resolve_thread_kv_page_slice_offset`. The hidden exact4 B4 API gate already
requires `page_block_size == 1024`, and every active helper input is
nonnegative, so the qrow32 translation unit now opts into exact `>> 10` and
`& 1023` operations. Stock traits and qrow16 retain the original dynamic
division and remainder path.

The old positive-1024 path used an inline reciprocal-based divide; its four
SASS `CALL` instructions were signed fallback branches. The specialization is
therefore intended to remove both the executed inline divmod sequences and the
dead fallback call paths. No speedup or CALL0 result is claimed before a fresh
post-teardown compile and disassembly.

## Saved call map

The prior object has four call sites, all targeting `0x13830`:

- `0x1ae0`: initial final-block K/V page coordinate; the compiler shares the
  common divmod for the source calls at `flash_fwd_kernel.h:702-705`.
- `0x7ce0`: masking-loop K prefetch at `flash_fwd_kernel.h:999-1000`.
- `0xa130`: unmasked-loop V advance at `flash_fwd_kernel.h:1039-1040`.
- `0xef70`: unmasked-loop K prefetch at `flash_fwd_kernel.h:1066-1067`.

At each site, `c[0][0x4f0]` supplies `page_block_size`, the numerator is moved
through `R170:R171`, and the quotient returned in `R24:R25` is used to form the
remainder and page-table address. The target body at `0x13830-0x13cf0` is the
shared signed 64-bit division routine.

## Qualification boundary

Nineteen focused source tests pass, including opt-in/default-off behavior,
patch idempotence, stock/qrow16 isolation, exact shift/mask equivalence over
the qrow32 helper domain, and equality between the static page size and the
API gate. Python compilation and `git diff --check` also pass.

No NVCC, C++, link, disassembly, GPU execution, byte comparison, or performance
measurement was run for this checkpoint. After the live B4 pair tears down,
the candidate still requires a fresh SM121a compile, CALL0/resource admission,
the pinned GCC11.4 ABI relink, canonical real SWE-Verified exact4 Tail23 and
Hydra27 byte gates, and only then real-task timing.

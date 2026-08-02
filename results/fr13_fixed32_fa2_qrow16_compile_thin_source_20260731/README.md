# Fixed32 FA2 qrow16 compile-thin source

Status: source-complete, CPU-static-tested, production compile and GPU gate
pending. This artifact makes no timing, byte-parity, or acceptance claim.

## Why the first build is long

The real SWE-Verified B1 Nsight trace resolves tree attention to the BF16,
noncausal split-KV specialization with stock traits `256x64x64, 4 warps`.
Qrow16 needs only the matching `256x16x64, 1 warp` specialization in
`flash_fwd_split_hdim256_bf16_sm80.cu`.

The first source form changed the shared launch header. Ninja consequently
scheduled all 52 FA2 forward CUDA objects, and the required HD256 TU emitted 24
qrow runtime-switch variants. A read-only snapshot of that active compile found:

- unthinned candidate PTX: 291,529,691 bytes, 10,276,672 lines, 86 entries;
- exact-safe base embedded PTX: 22,640,506 bytes, 789,129 lines, 62 entries;
- active `ptxas`: running at 99.9 percent CPU with 5.3 GiB RSS;
- exact-safe target-TU Ninja duration: 168.493 seconds.

The 12.88x PTX expansion explains the long `ptxas` phase. It was CPU-active,
not blocked. Reducing unrelated object rebuilds cannot shorten that target-TU
critical path after it has started.

## Thin specialization

The new patch replaces only the target TU's explicit instantiation with an
explicit specialization of
`run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>`. The shared
`flash_fwd_launch_template.h` remains byte-identical to the exact-safe base.
The stock arm still calls the original `run_flash_splitkv_fwd` with
`256x64x64, 4 warps`.

The candidate arm directly takes one kernel address with the exact booleans
observed under Nsight: noncausal, nonlocal, no ALiBi, uneven MN, even K, no
softcap, no split, and no appended KV. Its guard also requires B1, BF16/HD256,
24 Q heads, four KV heads, 32 query rows, paged 1024-row KV, full-window tree
bias, `softcap == 0`, and `num_splits == 1`.

This is semantically the same candidate kernel selected from the earlier
24-variant switch tree. Grid construction, dynamic shared-memory attribute,
thread count, stream, launch check, and ordered full-K traversal are unchanged.

## Expected compile reduction

- candidate device instantiations: 24 to 1, a 95.83 percent reduction;
- total target-TU entries: 86 to 63, a 26.74 percent reduction;
- Ninja CUDA rebuilds from the exact-safe snapshot: 52 to 1;
- linear excess-PTX estimate: about 33.84 MB, 88.39 percent below the
  unthinned 291.53 MB PTX.

The PTX-byte estimate is explicitly unmeasured until the production compiler
runs. The exact entry count is source-derived and must be confirmed with the
candidate PTX or CUDA symbol dump.

## Verification

Static tests require idempotent TU patching, an unchanged shared launcher, the
exact eight kernel booleans, one candidate kernel address, the original stock
launcher call, no qrow combine path, and unchanged warp-local row mapping.

The next valid gate remains the same-process, same-SO live-paged raw-byte A/B
on one real SWE-Verified B1 task. Stock must be served and BF16 output plus FP32
LSE must have zero byte mismatches before any timing campaign.

# Fixed32 FA2 qrow16 hidden-TU ABI repair

Status: CPU build and strict ELF ABI gate pass. GPU byte parity and real-task
timing are still pending. This artifact makes no performance or acceptance
claim.

## Repair

The first compile-thin form replaced the stock HD256 BF16 explicit
instantiation with an explicit specialization. The production build exposed
three ABI changes: the stock dispatcher became strong, one weak nested launch
symbol changed, and C++ `getenv` added a new undefined libc symbol.

The repaired form keeps the stock source and object byte-identical. It creates
`flash_fwd_fr13_qrow16_hdim256_bf16_sm80.cu`, containing one hidden host
launcher and one exact qrow kernel. `flash_api.cpp` calls that hidden launcher
only when an exact B1 tree-bias tensor carries the private batch-stride sentinel
`0x46523133`. Because batch is exactly one, this stride is never used for an
address offset. The C++ guard fails closed on BF16, noncausal, B1, HD256,
24/4 heads, 32 query rows, paged 1024-row KV, full window, no ALiBi/append,
zero softcap, one split, and the paged split-kernel route.

Normal tensors cannot receive the sentinel from the patch unless the attested
production or live A/B path explicitly constructs the view. B2, B4, and every
ordinary call continue through the original dispatch body.

## Build evidence

- Exact-safe SO: `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`
- Candidate SO: `35ba18c9bab4b37362aa3b26441e8a58edfcd3d0a75692fda90fc131a0b3307c`
- Candidate size: 299,554,080 bytes
- Stock HD256 BF16 object, base and candidate:
  `fc31f75bf88bf318cd9530745734c2f7fee6d755b372fefdcf11d157f014f389`
- Qrow CUDA object:
  `87cb69fbbf25a7044ebc9dba9a02374a869755ebf2d49d20dd554dca2af72fe7`
- Qrow CUDA compile: 17.715 seconds using the pinned snapshot command
- GPU used for build/gates: no

The configured snapshot did not contain the newly globbed source. The build
therefore reused its exact `build.no-reconfigure.ninja` CUDA command with only
the source/object stem changed, compiled the changed API object, and appended
the new object to the exact recorded link command. A clean CMake configure
includes the new `flash_fwd_*.cu` source through the existing FA2 glob.

## Strict gates

Normalized `readelf --dyn-syms` records include type, binding, visibility, and
mangled versioned name. Candidate versus exact-safe results:

- defined dynamic records: 687 vs 687, byte-identical;
- undefined dynamic records: 169 vs 169, byte-identical;
- `DT_NEEDED`: 10 vs 10, byte-identical;
- `nm -D` defined names: 685 vs 685;
- `nm -D` undefined names: 168 vs 168;
- stock BF16/HD256/noncausal dispatcher: `WEAK DEFAULT`;
- qrow host launcher: `LOCAL`, absent from `.dynsym`;
- qrow CUDA main kernels: exactly one;
- qrow CUDA combine kernels: zero.

The required next gate is the same-process live-paged stock/qrow raw-byte A/B
on one real SWE-Verified B1 task. Only after BF16 output and FP32 LSE have zero
byte mismatches may the candidate enter the standing real-task timing set.

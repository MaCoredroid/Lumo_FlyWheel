# Fixed32 DFWD K64 FP8 M256 head

This artifact records the offline SM121a build and static audit of the
`k64_head_m256_byte_ab` diagnostic candidate for the fixed32 Tail23/Hydra27,
physical32, K64/root workload.

The exact K64 draft head is a block-FP8 GEMM with physical rows `M=1` for B1
or `M=4` for B4, `N=65536`, and `K=5120`. After CUTLASS swap-AB, both batches
have one scheduler-N tile. The stock collective uses a `128x32x128` tile and a
dynamic CLC scheduler. The candidate widens only scheduler M to
`256x32x128`, retains cluster `(1,1,1)`, a two-stage cooperative SM120
mainloop, FP32 accumulation, the stock epilogue, and ordered full-K
accumulation, and uses the audited one-N static scheduler.

The selector is default-off and diagnostic-only. An armed real-task call runs
stock first, runs the candidate into a separate tensor, compares every output
byte, logs the result, and always serves stock. There is no production
selector. The binary verifier pins the exact shared-object hash and admits only
the `k64_root` qualification profile.

## Static result

- Pinned vLLM and CUTLASS sources patched cleanly.
- The real `_C_stable_libtorch` blockwise translation unit compiled for
  `sm_121a`, the full shared object linked, and CPU-only `load_library` passed
  with CUDA hidden and uninitialized.
- FP16 and BF16 candidate kernels each use 168 registers/thread, 384
  threads/CTA, 1024 bytes static shared memory, 0 stack bytes, and 0 local
  bytes. The exact stock collective has the same resource envelope.
- Stock has 512 logical M128 output tiles and a 512-CTA dynamic launch, or 11
  nominal CTA waves over 48 GB10 SMs. The M256 candidate has 256 logical tiles;
  divisor balancing reduces the 48-CTA static base grid to 32 CTAs, one
  nominal wave, with exactly eight complete tiles per CTA. B1 and B4 share
  this geometry.
- Candidate SASS has 832 static instructions and 37 `BRA` instructions versus
  1176 and 75 for the exact stock collective. Candidate and stock contain no
  `LDL`, `STL`, or `CALL` instructions. These counts do not imply speedup.

These are static credentials only. No GPU runtime, Docker, synthetic timing,
probe timing, SWE-Verified task, B1/B4 correctness run, or performance
measurement was used. No acceptance or speed claim is made.

## Floor ledger

One FP8 K64 head call must read 335,544,320 weight bytes and 81,920 FP32 scale
bytes. Five calls therefore retain 1,678,131,200 mandatory bytes per event.
The established candidate full-step ledger is 30,989,326,208 mandatory bytes,
or 113.514015414 ms at 273 GB/s, with a 1.15x cap of 130.541117726 ms. This is
an optimistic mandatory-weight floor, not a measured end-to-end hardware
floor; nonweight work is excluded.

## Required live gates

1. Verify the exact binary and `k64_head_m256_byte_ab` selector under the
   `k64_root` profile.
2. Run a real SWE-Verified B1 byte comparison with the B1 arm. Require every
   admitted call to be byte-equal and stock-served.
3. Run the corresponding real SWE-Verified B4 byte comparison with the B4 arm.
4. Only after both byte gates pass, add a source-bound production selector,
   rebuild, and run clean exact4 B1 and B4 full-step timing. Do not time the
   diagnostic selector because stock/candidate execution and D2H comparison
   are intentionally intrusive.

`manifest.json` binds source and binary identity. `offline_audit.json` records
the build, launch, resource, SASS, and traffic ledgers.

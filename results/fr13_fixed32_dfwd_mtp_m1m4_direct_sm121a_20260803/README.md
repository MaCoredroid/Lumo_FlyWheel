# Fixed32 DFWD MTP M1/M4 direct scheduler

This artifact records an offline SM121a build and static audit of the
`mtp_m1m4_direct_byte_ab` diagnostic candidate for the fixed32 Tail23/Hydra27,
physical32, K64/root workload.

The candidate preserves the stock FP8 blockwise GEMM mainloop, epilogue, tile
`128x32x128`, stage-count selection, cooperative schedule, and full-K
accumulation order. It changes only complete-output-tile assignment for the
five admitted MTP projection shapes at physical GEMM rows `M=1` (B1) or `M=4`
(B4). Both rows remain one scheduler-N tile after swap-AB.

The selector is default-off and diagnostic-only. Armed real-task calls execute
stock first, execute the candidate into a separate tensor, compare every output
byte, log the result, and always serve stock. There is no production selector.
The binary verifier pins the exact shared-object hash and rejects any profile
other than `k64_root`.

## Result

- Pinned vLLM and CUTLASS sources patched cleanly.
- The real `_C_stable_libtorch` translation unit compiled for `sm_121a`.
- The shared object linked and loaded without using a GPU.
- BF16 and FP16 candidate kernels each use 168 registers/thread, 1024 bytes
  static shared memory, 0 stack bytes, and 0 local bytes.
- Candidate SASS contains 632 instructions versus 1176 for the exact stock
  swap-AB collective, with no LDL, STL, or CALL instructions in either dtype.
- Focused patcher, binary-verifier, and artifact tests pass: 78 passed.

These are static credentials only. No GPU runtime, Docker, synthetic timing,
probe timing, SWE-Verified task, acceptance campaign, or performance
measurement was run. No speedup is claimed.

## Corrected work ledger

Each speculative event has one initial MTP forward and four post-root graph
forwards. Each forward contains the five admitted projection GEMMs, so the
candidate covers 25 projection launches/event, not 20. Projection weights are
456,130,560 bytes/pass and 2,280,652,800 bytes/event. The complete MTP forward
ledger is 477,199,744 bytes/pass and 2,385,998,720 bytes/event.

The full DFWD floor remains 21.030922784 ms/event at 273 GB/s: 8.739922051 ms
for five complete MTP forwards plus 12.291000733 ms for five BF16 K64 heads.
The valid B1 Hydra DFWD is 36.813368134 ms/event, leaving 15.782445350 ms above
that floor.

Historical real-task attribution ranks the remaining work: five K64 heads
extrapolate to about 26.227 ms/event versus their 12.291 ms floor, while the
five-pass projection group extrapolates to about 10.643 ms versus its 8.354 ms
projection-weight floor. This candidate can address only the smaller projection
scheduler gap. The next high-impact experiment remains a K64 head GEMV kernel;
mapped-top3 traffic removal alone cannot close the head gap.

## Required live gates

1. Use only the fixed32 Tail23/Hydra27, physical32, K64/root configuration and
   real SWE-Verified tasks under the standing exact4 or exact16 rule.
2. Verify the pinned shared object with `fr13_cutlass_wave_binary.py` and the
   selector `mtp_m1m4_direct_byte_ab`; full-vocabulary verification must fail.
3. Run B1 with the B1 real-event arm and B4 with the B4 real-event arm. Require
   all admitted M=1/M=4 shape records to be byte-equal, with stock served.
4. Only after both batch byte gates pass, add a separate source-bound production
   selector, rebuild, and run clean exact4 B1/B4 timing. Do not time the
   diagnostic selector because its stock/candidate/D2H comparison is intrusive.

`offline_audit.json` carries the source, binary, resource, SASS, floor, and
historical-attribution ledger. `manifest.json` carries the qualification state
and the exact remaining gate route.

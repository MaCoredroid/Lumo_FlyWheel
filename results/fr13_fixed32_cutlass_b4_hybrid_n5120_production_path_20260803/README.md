# Fixed32 CUTLASS B4 Hybrid N5120 Production Path

This artifact records the host-ready, default-off production path for
`identity_hybrid_n5120_b4`. It does not claim live correctness, performance, or
production qualification.

The production credential is fail closed. It requires independent Tail23 and
Hydra27 byte-equality PASS records from the canonical four distinct
SWE-Verified tasks at batch size 4 and concurrency 4. Each arm must authenticate
all four task keys through the finalized engine ingress ledger, exercise the
five fixed projection shapes at 128 physical query rows, return stock output,
and report zero differing bytes. Both records, the installed candidate, and the
production attestation must bind to the same clean exact repository `HEAD`.

The exact4 timing runner starts only after that dual credential validates. Its
candidate arm selects `identity_hybrid_n5120_b4` directly, so no comparator is
present in timed execution. Timing remains a paired exact4 screen and is not the
formal statistical hardware-floor acceptance gate.

Host verification reused the integrated SM121a library already built from the
same pinned patch and generated dispatch digests. The library was independently
checked for byte identity, ELF linkage, CPU-side `torch.ops.load_library`, and
the four FP16/BF16 hybrid scheduler resource records. No binary, raw workload
content, request or response data, patch content, environment dump, process or
container identity, or execution log is published here.

Live Tail23 and Hydra27 byte gates, exact4 timing, and the later formal floor
gate require the target GPU host and were not run in this change.

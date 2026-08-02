# Fixed32 DFWD BF16 M32 source candidate

Status: **experimental, default off, not deployable until the live full-logit
byte gate passes**.

## Why this target

The real SWE-Verified B1 Nsight attribution contains 4,405 BF16 `gemvx`
instances over 881 complete events: exactly five draft heads per event. They
cost 26.227316 ms/event, or 5.2454632 ms for each unchanged 65536 x 5120 BF16
weight read. The five reads are 3,355,443,200 bytes, so their 273 GB/s
mandatory weight floor is 12.291000733 ms/event; observed effective weight
bandwidth is 127.94 GB/s.

The same trace measures the M32 verifier-head `nvjet` path at 12.153933 ms for
2,542,796,800 weight bytes, or 209.22 GB/s. Applying that measured efficiency
to one 671,088,640-byte draft slice projects 3.2076 ms/pass and 16.0382 ms/event,
about 10.1891 ms/event below `gemvx`.

## Candidate

`FR13_DRAFT_HEAD_M32=1` is accepted only for exact fixed32 B1 root64. It copies
the real `[1,5120]` BF16 hidden row into a persistent `[32,5120]` buffer, runs
`torch.mm(..., out=persistent_[32,65536])` against the unchanged contiguous
BF16 weight, and exposes row zero. Buffers are allocated before drafter graph
capture; steady state allocates nothing. The extra work is 21,474,836,480
FLOPs/pass and 107,374,182,400 FLOPs/event, while weight bytes remain unchanged.

This is intended to select the already measured SM121 M32 BF16 GEMM family.
Kernel identity still requires live Nsight confirmation.

## Exactness gate

`FR13_DRAFT_HEAD_M32_BYTE_AB=1` is a diagnostic mode and is mutually exclusive
with candidate serving. For every real root and captured loop head it runs the
M32 result and the existing BF16 `gemvx` result on the same hidden tensor,
compares their complete BF16 logits through `int16` views, accumulates mismatch
counts on device, and returns the reference logits. The next eager root reads
the cumulative counter and fails loud on any mismatch.

The candidate does not have a source-level bit-identity proof: cuBLAS may use a
different FP32 reduction tree for M32 GEMM. Zero mismatch on a short or random
probe is not acceptance. Required sequence:

1. Run the byte-A/B mode on a real SWE-Verified B1 task through hundreds of
   events; require zero full-logit mismatches across all five head positions.
2. Confirm exactly five M32 `nvjet` calls/event and no draft `gemvx` calls in a
   candidate-only real B1 Nsight window.
3. Run candidate-only Tail and Hydra exact4 with full wall TPS and acceptance;
   then exact16. Any token, top-k, task-resolution, or acceptance drift rejects
   the candidate even if timing improves.
4. B4 is deliberately not covered by this candidate. Measure its head dispatch
   first; the launcher fails loud if M32 is requested outside B1.

## Historical audit

- FIX-1 removed the duplicate draft-head read and is already present. It does
  not improve the remaining five causal heads.
- Root64/DVK changes the row set, not the BF16 kernel; the current exact run
  still executes five 65536-row BF16 heads.
- OPT-A changes scheduling for block-scaled FP8 verifier GEMMs. It cannot engage
  this unquantized BF16 `gemvx` dispatch.
- The prior FP8 draft-slice proposal changes draft logits and is excluded by
  the unchanged-logit requirement.
- A custom N=1 streaming kernel was not implemented. It has no measured GB10
  advantage over the proven M32 path and cannot reproduce the private cuBLAS
  `gemvx` accumulation tree by construction; it would carry the same byte-drift
  gate with substantially more codegen and tuning risk.

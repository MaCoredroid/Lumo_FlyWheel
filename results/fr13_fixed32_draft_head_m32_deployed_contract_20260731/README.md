# Fixed32 deployed-format BF16 draft-head M32 replacement

Status: **source-ready, default off, and not timing-qualified**. This work used
no GPU and ran no candidate, live, B4, or measurement workload. An unrelated
broad pytest unexpectedly invoked an existing Docker-based test; it was
terminated on detection, and both the container list and GPU compute-process
list were verified empty afterward. This artifact makes no performance,
byte-equality, task-resolution, or floor-acceptance claim.

## What the audit established

The deployed head is not a linear-layer object. Qwen3.5 constructs its untied
`lm_head` as vLLM `ParallelLMHead`, which inherits `VocabParallelEmbedding` and
owns `UnquantizedEmbeddingMethod`. The checkpoint tensor itself is BF16
`[248320,5120]`; `lm_head` is explicitly excluded from FP8 conversion.

The fixed32 root64 shim gathers the chosen vocabulary rows with
`index_select(...).contiguous()`. Therefore the live B1 operand contract is:

- reference GEMM/GEMV: `M=1`, `N=65536`, `K=5120`;
- hidden shape/stride: `[1,5120]`, `[5120,1]`, BF16;
- gathered weight shape/stride: `[65536,5120]`, `[5120,1]`, BF16;
- logical `weight.t()` stride: `[1,5120]`;
- logits shape/stride: `[1,65536]`, `[65536,1]`, BF16;
- five causal calls per event: one root head plus four post-root MTP heads.

The five-call census is independently present in the real SWE-Verified B1
Nsight artifact: 4,405 BF16 `gemvx` instances / 881 complete events = exactly
5. The root and all four loop heads use the same gathered weight and contract.

## Why the old candidate failed

The old M32 selector required the method class name
`UnquantizedLinearMethod`. The actual class is
`UnquantizedEmbeddingMethod`. The first real SWE-Verified event therefore
failed its aggregate representation guard before completing a draft event.
This was a selector-contract bug, not evidence that the deployed weight was
FP8, noncontiguous, or the wrong shape.

The replacement checks the actual class plus every shape, dtype, and stride.
It replicates the one hidden row into a persistent BF16 `[32,5120]` buffer,
runs `torch.mm(input_m32, weight.t(), out=output_m32)`, and consumes only output
row zero. Any contract error raises; production has no reference fallback.

## Credential boundary

`FR13_DRAFT_HEAD_M32_LIVE_AB=1` is a real-B1 diagnostic. It runs M32 and the
current reference on every one of the five real head positions, compares all
65,536 BF16 logits bitwise, accumulates the census on device, and serves only
the reference logits. The runtime writes its JSON only during the authoritative
final fixed32 flush, after the sole synchronization and complete-event
reconciliation. Its JSON PASS must bind the terminal flush nonce, generation,
producer, event-census hash, and boundary-snapshot hash; report an integral
number of complete five-call events; and report zero mismatches on
`astropy__astropy-12907`.

`FR13_DRAFT_HEAD_M32_PRODUCTION=1` is accepted only after the launcher issues
and re-verifies a source-bound pass sidecar. Sidecar issuance revalidates the
external final-flush and boundary files rather than trusting fields in the live
JSON alone. The patched runtime additionally requires the launcher-private
JSON alone. It also structurally validates and canonically rebuilds the B1
`fixed32_chat_traffic_audit.json`, including the authenticated proxy/engine
ledger, task trace, complete census interval, and clean agent/eval terminal
record, then binds the audit's raw SHA-256 into the sidecar. The patched runtime
additionally requires the launcher-private attestation and exact B1/root64
geometry. Its read-only engagement record is written only after a measured
CUDA-graph replay bound to one eager root selection, four captured loop
selections, and the exact B1 drafter graph signature. A graph first captured
during unmeasured warmup is allowed only when the later measured replay stays
on that same attested graph; the timing runner rejects missing or malformed
engagement.

The real gate command is:

```bash
FR13_GATE_QROW16=0 \
FR13_GATE_TAW_NATIVE=0 \
FR13_GATE_DRAFT_HEAD_PAD=0 \
FR13_GATE_DRAFT_HEAD_M32=1 \
FR13_GATE_BM8=0 \
FR13_GATE_GDN_BV=0 \
RUNROOT=output/<new-b1-gate-tag> \
TAG=<new-b1-gate-tag> \
FORKED_FA2_SO=<absolute-stock-fa2-path> \
bash scripts/fr13_run_b1_kernel_live_gate.sh
```

Only after that PASS, the paired real exact4 B1 timing runner is
`scripts/fr13_run_b1_draft_head_m32_timing.sh`. It runs the canonical four
SWE-Verified tasks sequentially at B1 for stock then candidate, records full
wall TPS and phase telemetry, validates production engagement, and labels the
result as a timing candidate rather than formal Tail/Hydra floor acceptance.
It requires `LIVE_PASS_JSON`, `LIVE_PASS_SHA256`, `LIVE_FINAL_FLUSH_JSON`, and
`LIVE_BOUNDARY_SNAPSHOT_JSON`, plus `LIVE_CHAT_TRAFFIC_AUDIT_JSON`, from the
same successful gate. Each of the four tasks must retain at least 64
pure-decode steps and at least 99% of those steps in the full-wall window. The
reducer recomputes wall time, SFWD/DFWD/CFWD time, acceptance, and sample counts
from raw per-task counters before it can emit a complete summary.

## Modeled ceiling, not a measurement

Historical real-B1 Nsight attribution measured the five reference `gemvx`
heads at 26.227316 ms/event. Reusing the measured effective bandwidth of a
different M32 verifier-head GEMM projects 16.038180 ms/event for this candidate,
or **10.189136 ms/event modeled recovery**. That number is a hypothesis only.
The smaller 65,536-row head can select a different private cuBLAS kernel, and
no M32 task timing exists yet.

The immutable five-head weight stream is 3,355,443,200 bytes/event. Its
weight-only floor at 273 GB/s is 12.291001 ms/event; including candidate input
and output traffic gives a 12.373821 ms/event lower bound. Even the projected
10.189136 ms recovery closes only 10.71% of the current 95.173072 ms wall gap
to the 1.15x cap. It is useful stack work, not a route to the cap by itself.

## Pinned evidence

- integrated base commit: `6bbafd5caee2d95081ec049faeaa2a2d1b4743a5`;
- deployed vLLM source: `fe9c3d6c5f66c873d196800384ed6880687b9e52`;
- model config SHA-256:
  `f78c412bfdec65a88c8aa2a031d39c2fda32e3377ae48a77f971bc40a4f095df`;
- model index SHA-256:
  `6d19a4e607604c1ac631f810a56e6084c892b4cb0251c530c6a24fc877f9fb4b`;
- candidate patcher SHA-256:
  `0ecd359c7ffb211f0212db3a83baabfff327c07286fd808ea99ac68d536798e2`.

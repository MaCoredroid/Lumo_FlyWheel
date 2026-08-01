# SFWD state-fusion real SWE-Verified B1 byte pass

This artifact records the completed authenticated real-task B1 byte diagnostic
launched on `2026-08-01`.

- Runtime source commit:
  `a83de22d2d32cee340f0447c03bd05761e5e8365`
- Post-run validator repair commit:
  `d86445697c867f8ef6f41c610111614727f1fbde`
- Task: `astropy__astropy-12907`
- Candidate: `fixed32_sfwd_state_fusion_v1`
- Batch: B1
- Physical rows per request: 32
- Draft vocabulary: full vocabulary (`root=0`, `K=0`)
- Run root:
  `output/fr13_b1_sfwd_state_fusion_live_gate_20260801T120458Z`

The real SWE-Verified task resolved cleanly. The orchestrator returned zero
after 522 seconds; the task verdict was `resolved` after 520.3 seconds.

The reference-first candidate completed 967 speculative events across all 48
GDN layers:

- Comparison records: 46,416
- Zero-diff records: 46,416
- Mismatching records: 0
- Bytes per record: 1,392,640
- Total compared bytes: 64,640,778,240
- Compared surfaces: `conv_out` and `commit_source_stage`
- Reference always served: true

The runtime lifecycle completed, including the eager KV16 equal-row path that
rejected the prior run. The final deterministic validator status is `pass`
with no errors and is bound to the captured PID 1 `--max-num-seqs 1` argv.

The runtime originally published the authenticated marker as mode `0400`.
After task completion, only its mode was normalized to the repaired `0444`
contract so the host validator could read it; marker bytes were unchanged and
remain bound by SHA256. No model traffic was replayed.

This is a one-task B1 correctness diagnostic. It is not B4 qualification,
timing evidence, production qualification, formal exact4/exact16 acceptance,
or a hardware-floor result. It contains no valid wall-TPS or confidence-bound
claim.

## External evidence bindings

- Comparator JSONL SHA256:
  `491d9466f68f14e3351957b2ff05a7f9d4e144ee124e8573336445d54a7c60d5`
- Live-pass JSON SHA256:
  `7ccfaf5cc907909b0646b752b94027e250b234a3b98bf461de61e6ae70f31782`
- Engine ingress ledger SHA256:
  `e785e62eb4a6319103d0a714ad562a9f528ec4000dc85ef82bcfce06bfd81668`
- Eager task bracket SHA256:
  `3d83ea7a2408509f2f1547740d2682025dad077b79023ec23689c77f5ab33af3`
- Process identity SHA256:
  `ba1b3abbaa623c4fdfe9fcd2c90eaf1a856e3577cfafe0d23e17941d0d08c93c`
- Authenticated marker SHA256:
  `04fe7f61a0e0bbd48bf28127385c481b85550b291535f3705511494ba24c8463`
- Arm runlog SHA256:
  `1c408b49960f9d85081afd53bdfd1de162a4bb88f7343cbc54ff253dc09de4e8`
- SWE orchestrator log SHA256:
  `748d32bf8d77a4709aae84b89e7b56a61b692c040dc6758ef9df83771e72fa42`

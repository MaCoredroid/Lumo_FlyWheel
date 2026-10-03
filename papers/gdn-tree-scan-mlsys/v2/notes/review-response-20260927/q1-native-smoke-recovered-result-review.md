# Recovered native instrumentation smoke: independent PASS

Run `q1-native-smoke-recovered-20260928T031726Z` passes its **one root-only native case × two repeats × one fresh model process** instrumentation contract. It produced two completed HTTP requests, two valid authenticated observation seals, and complete finite O0/O1/O2 artifacts. **Broader full-model qualification remains zero:** this run contains no tree candidate, accepted branch, non-root path, held-out task or workload comparison, and no cross-process replication.

## Exact evidence and reproducible audit

Run: `experiments/review-response-20260927/runs/q1-native-smoke/q1-native-smoke-recovered-20260928T031726Z/` relative to v2.

- Final receipt SHA-256: `b5f147cd494c249607c42eeea97f5c17037886dfdf809f1a00a5bc612d45cc66`.
- Launch binding: `97cba0b62176e2f62a91a768fb0c45f2d1e9198c713cdfbea2344d164f8207e2`.
- Approved gate snapshot: `678814649122bac8d4f36ce90c7acb7297de16d3eabee0e85e03cf72d9c6fdfc`.
- Driver verdict: `ffd1c2711b55372b095dda566af327f2cb7377054f9e426538e6246eb31bd574`.
- Boot attestation: `1f0630dd53e74ab3d02e7e01722552650f7fcb45db12be330f1d08d14ae4455c`.
- Independent audit reader: `audit_recovered_smoke.py` next to this note, SHA-256 `292cf9252a9f8a50e47cb85565b709aa5ff6d6a2d836831e45560a4e4f18e42d`.

I ran the reader locally with bundled Python/NumPy against the completed mirror. It verifies the actual bytes without importing the experiment's validator or using GPU/model code. All **38 non-object receipt members** match SHA-256 and length; membership is exact. Both case documents' internal canonical seals match. Every referenced object exists, and the object store has no missing or unreferenced members: **6,977 unique objects, 1,195,911,168 bytes**. All hashes and geometry-derived exact lengths pass. An independent exponent/finite check found no nonfinite values in **6,880 bf16 objects and 97 fp32 objects**.

All 17 gate/binding identities match, as do scope, run, job, engine configuration, command digest and actual launch argv/script. The actual container inspection names immutable image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. Hooks `1ac939405de01a4c22e5d73decb9994cfb3a3214fd838ed2a734b39110b2b8cf` and launcher `df5e0c1982f12060a4277bcd79d9762c371ed009c8a5ed47d3cfc7062903a721` match the accepted repair, with unchanged cleanup library `44ae0909759e3338fd0c3031c447f07523b2d47072add2404e78aea15cbd0524`.

## Runtime, observations and state coverage

The log contains one engine initialization, one model-weight load and two completion POSTs returning HTTP200. Lifecycle: start 03:19:13Z, healthy 03:24:24Z, stopped 03:25:01Z; RestartCount=0. Boot attestation has no problems: 48 explicit nonpacked GDN layers and 16 actual FA2 attention implementations, aligned cache mode, fp32 recurrent cache, no speculative config. Patched FA2 binary is `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`; interface is `dadab8aff63b7f608274834929c954335247366411382f390af92501361044a1`. Resolved groups/block identities are recorded. This remains the current pinned native NVFP4 model route, not a superseded Cat10 route.

The prefix's actual little-endian token file is 53,948 bytes = 13,487 tokens and independently hashes to `20321ed78102db81290634bab97cd0e97fd096343aeec7c2ab036d9605a6315c`. Both API requests report prompt_tokens= 13,487 and completion_tokens=3, with no HTTP error. The two requests and observation IDs are distinct.

Each repeat directly records consumed token 11352 at position 13,487 and token 25559 at position 13,488, using `runner.input_ids.gpu` on the embedding path and three-row position tensors of shape [3,1]. The field is observed execution input, not a copied expected-fixture value. The first repeat records the full prefix prefill; the second begins with 13,312 computed tokens and executes the remaining 175-token prefix segment, consistent with the native cache-reuse path. forced_count=3 includes the terminal forced token; the two required r/z inputs are explicitly observed.

For each repeat, O0 is captured before consuming r, with 13,487 materialized tokens; O1 is captured before consuming z, after r, with 13,488 materialized tokens. O2 is the full-vocabulary logit row from the z forward. Its recorded pre-forward count 13,488 is therefore consistent with the next-forward observation, not evidence that z was skipped.

Each O0/O1 contains the exact 48 GDN layer identities (all non-attention layers among 64 total) and exact 16 attention layer identities. GDN conv snapshots are bf16 [3,10240], recurrent snapshots fp32 [48,128,128], with 16 layers in each of groups 0/1/2. KV group 3 uses logical kernel blocks of 64 tokens, 4 KV heads and head dimension 256. Every attention layer has 210 complete logical blocks plus a 47-token O0 tail or 48-token O1 tail, all with valid, unique physical block IDs and exact serialized lengths. Logical-index ordering, per-layer KV digests, conv/recurrent aggregate digests and final state digests were independently recomputed. This establishes complete archived observations for these boundaries; actual restore/replay execution is outside this smoke.

## Repeat stability and raw next-forward logits

All 48 conv and all 48 recurrent states change from O0 to O1 in each repeat, showing the archived after-r boundary is not merely the before-r state copied under another label. Across repeats, however, every conv/recurrent object and every attention logical digest is identical at each corresponding boundary:

- O0 logical digest: `df8a55b6928e1bb6522e20f5d8d1b20740c58bfd9f1871cbe85da794dc91f34e`.
- O1 logical digest: `ecbf5d60b8eea9918566d31520819bae38f3e6d4ef0344f4de5116337727f601`.

Physical identities differ: recurrent rows 53/54/55 become 57/58/59, and all 16 attention layers have changed physical block mappings. Logical content remains exact despite that remapping.

The entire **248,320-entry** O2 vector is byte-identical across repeats, beyond merely sharing an argmax: raw fp32 serialization SHA-256 `3a791d3f61eea35286fc97a059b8b87549b2641c87a50531796cc5d8ecc1efd7`, length 993,280 bytes. The original logits are bf16, archived losslessly as fp32. Independent raw-vector reduction gives unique argmax **3274**, top1 **15.9375**, top2 **15.125**, margin **0.8125**, exact_tie_count=1. All recorded summary fields match. Natural-sampler trace after z also records 3274.

## Cleanup and next-gate boundary

Driver exit 0 and cleanup exit 0 agree with `COMPLETED_driver_rc=0_cleanup_rc=0`. Archived inspection has Running=false, Status=exited and ExitCode=0; the stop/removal records and FINALIZED marker agree, with no cleanup error. All receipt hashes still match after process exit. This audit used local evidence only, as requested; it did not perform a new live container query.

This pass is sufficient to retire the missing-input instrumentation failure for the exercised root-only route. Before a broader native-calibration gate, bind the predeclared case manifest and supported observation paths, source/image/binary hashes, process/repeat count and stop-on-invalid rules; confirm the exact intended non-root/cycle support is executable rather than inferred from this root-only result. Preserve separate calibration/held-out roles and require a fresh-process replication for any cross-process claim. Check CUDA allocatable memory against the unchanged requested reservation before spending another model boot; the prior startup failure remains part of the record.

The next approval can cover that separately reviewed native-calibration package. It should not automatically authorize a candidate full-model run: candidate continuation/state publication and its same-input next-forward comparison still require their own source/protocol review and gate. No numerical threshold change, performance conclusion, task-quality conclusion or broader qualification is implied. Both earlier failures remain preserved separately; unrelated workload attempts are not counted here.

No GPU/model launch, remote mutation, source edit, gate change or alteration of run evidence occurred in this audit.

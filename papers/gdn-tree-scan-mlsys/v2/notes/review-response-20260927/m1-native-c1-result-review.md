# Independent native M1-C collection and reduction result review

**PASS for the finite native-comparator baseline.** The two Lumo surfaces satisfy the predeclared reference-only validity requirements on both calibration seeds. This evidence supports the parent freezing native surface resolution before M1-Q. It does not qualify any Lumo/author candidate, establish full-model correctness, provide mechanism timing, or authorize a new run.

Run: `experiments/review-response-20260927/runs/m1-native-c1/m1-native-c1-20260928T212834Z`. Evidence/reproducers: `p0/monitor/review-response-20260927/m1-native-c1-result-independent-review-20260928/`. Raw tensors remain at their canonical remote paths; they were independently read, hash-checked and reduced there using CPU-only Torch 2.4.1, four threads and empty CUDA_VISIBLE_DEVICES. No GPU call or remote file mutation was made by the reviewer. Thirteen mirrored metadata/NPZ files match the corresponding remote hashes; a 302-entry remote file index preserves the larger evidence identities.

## Raw population and independent numerical verification

The authenticated pack reproduces all 96 operand/reference sets in the original accepted CPU receipt `0c6d5918fe2c982c7ae58dabf7e4bd3cb3251ef46a61dac21fd249b5de23cb36`. I independently compared each pack member's eight operand hashes and both C2 value hashes to that receipt, then checked the actual pack files and loaded tensor values. The collector source, native module/function, image, topology and gate dependencies match the reviewed identities.

There are exactly **96 layer-instance records × two raw repeats**, seeds 20260930/20260931 × 48 layers. Every record has the required BF16 `[28,48,128]` outputs, FP32 `[12,48,128,128]` state history and FP32 `[48,128,128]` alternate-sibling final state. All raw values are finite. Both repeats are byte-identical, and each raw state0 exactly equals its authenticated pristine S0. All file hashes and tensor hashes agree. Native-call counts are 310 per layer-instance, **29,760 total**.

I independently recomputed the per-cell RMS/max errors, C2 scales, frozen `1.1*error + 2^-24*scale + 2^-149` bounds, witness applicability, observable-difference and violation masks directly from the saved tensors. This auditor imports neither the baseline nor reducer implementation; sibling mappings come from the frozen topology. **All 4,416 layer-level saved arrays are bitwise equal to independent recomputation.** The two seed populations contain exactly **142,848 unique cells** (129,024 outputs and 13,824 publication states), with matching manifest digests and no omissions, extras or duplicates.

All **244,224 eligible structural witness evaluations** are detected. None is undetected or identically excluded. No-op and stale witnesses use the actual saved intermediate histories; publication-3 wrong-sibling state and applicable output sibling permutations have the declared scope. The largest observed native/C2 RMS error is about `7.1777e-5` for stored BF16 output and `1.2195e-6` for FP32 publication state. These are reference-baseline diagnostics, not candidate acceptance outcomes or speed results.

The CPU reducer receipt records `RESOLVED_CANDIDATE` for all four seed/surface records, source `27a3ac2b96d68713af4ece33828750461cf3b2ce51a7c1d6e321cde5e95214d4`, and 28.754 seconds. Its declared start invocation is CPU-only and bounded by 1,800 seconds; stdout/stderr are empty. The saved launch directory contains STARTED.json rather than a separate process-exit receipt; numerical completion is established by the sealed complete reduction and this independent raw audit. I do not infer an unrecorded parent process exit code.

## Single native launch and cleanup

The exact gate was approved at 21:28:34Z and consumed once at 21:28:57.903650Z by launcher PID 1653628 for the same output namespace. The marker is under `p0/monitor/review-response-20260927/GATE-M1-m1-native-c1-20260928T212834Z.json.consumed.json`; it is not a second campaign-root authority. The launch command evidence contains one create; four inspections authenticate the same image/name/nonce/CID. The container has restart count 0, started 21:28:58.044914524Z, exited 21:29:22.231649576Z with code 0, and was removed. Successful full-ID enumeration then proves its absence. CID: `c6352af7272f20104eb10d547e6d18f67a183e58516fca144ba0df76a9e7fd43`.

The owned collector reported 23.576 seconds and launcher 24.476 seconds. These are operational elapsed times, not mechanism throughput measurements. The in-process CUDA-free observation was 83,940,483,072B, above the unchanged 12 GiB floor. Peak Torch CUDA allocation/reservation was 13,246,464/23,068,672B; separately labelled Linux peak RSS was 1,029,230,592B. These are scoped observations, not a full device-memory bound. No full-model boot or memory recovery occurred in this stage.

## Evidence hashes

| Artifact | SHA-256 |
| --- | --- |
| Approved gate snapshot | `362f9952a5215aa281d3e8ce43952ef57f226a928f94fdf28271c9b78164a354` |
| Consumption marker | `66e21362f11e22a9cb4758904fc27805366143e9de6127f321a3d6f5e58b168b` |
| Owned launch receipt | `03bb4141e364ed1c52b246b68f67d24817a858badacd1860a0ab96fa4ae7d62c` |
| Native raw receipt | `b773400f4a4d2609e9231450bf53f756afdfd3f942bdc1fde9a7f4bbef63cf17` |
| Reference pack manifest | `1cf92788c78c083d7c14a6984c63bee200a9203df0a87e6b931035f9fb3f9452` |
| CPU reduction receipt | `499f1b9c637212c5273018334b46cc553ad0edfd276d5609871efbf7ac748831` |
| Independent raw audit | `58a9dfb2485ed732dee3809f2bad1eb877a3b9078168c3835b5c9b841d454aad` |
| Independent lifecycle/reference audit | `e54428c6fa13613d5b6310abc9993a26a6859783eec6f44248d50ef8a3381bee` |
| Local review seal | `673a916d3c2b66cbaf766bb6664cc7f7403360a4d8f3d0cc885058af14629963` |

The raw audit checked 10,228,222,703 bytes across indexed run evidence and pack members and took 24.200 seconds of CPU read/reduction work. It repeated no model inference or native kernel call. All prior source/failure evidence remains preserved. Native baseline collection is now evidenced; author/Lumo M1-Q qualification and complete-cycle timing remain pending separate prospective gates.

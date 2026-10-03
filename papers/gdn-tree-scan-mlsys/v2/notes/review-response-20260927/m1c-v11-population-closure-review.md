# M1-C v1.1 population closure and CPU admission review

**Disposition: approve the bounded source/package repair; technically ready for the parent to authorize baseline-only CPU M1-C calibration through the frozen caller.** The original population finding is closed. No remaining concrete blocker was found in this delta review. This note does not authorize or claim a calibration result, GPU execution, native Lumo C1 evidence, numerical qualification or M1 timing. The parent retains every gate.

## Authenticated final package

The final freeze was present on the single bounded remote inventory. Read-only SSH copied it and every listed payload into `p0/monitor/review-response-20260927/m1c-v11-reviewed-20260928T174134Z/repo/`. All **23/23 freeze-listed files** match their hashes and byte sizes. The freeze stayed identical across transfer and its declared canonical digest verifies. All **21 package-manifest payloads** and **6 local import-source bindings** also match the snapshot. `INDEPENDENT-SNAPSHOT.json` retains the receipt.

| Artifact | SHA-256 |
| --- | --- |
| `FREEZE-M1-C-BASELINE-v1.1.json` | `9f260d3989eb0dffb4a25973d85dcbcb96ec2cc4c4974a88b41f104ba73facdd` |
| `m1/M1-C-BASELINE-v1.1-MANIFEST.json` | `d580c5eb9f7215c1b6319cf960b8a1d632744cdc3b093fd25d491600b62192f2` |
| `tools/m1/m1_c_baseline_v1_1.py` | `00d1c8e5a593f2f1377e43aef2002f446a6ffb7c04ef0689b38cdbe163fd721b` |
| `identity/m1_c_baseline_v1_to_v1_1.diff` | `a6754ad8c0c9b8cf048ad4a40218fb075b71a3765106215a1f99effe4df58826` |
| `tools/tests/test_m1_c_baseline_v1_1.py` | `0f9b4e06032b138328f6818cbe7d94b77aaeee5aff0f706bc27a5fd3b72160d6` |
| supplied v1.1 test log | `d550f032c6b423a995821bbecfe54b8e2f6e46787c5a8a42c91db577e9165211` |

The supplied log reports **66 CPU tests passed**. Independently, **22 focused AST/stdlib controls passed**, recorded in `POPULATION-CLOSURE-CONTROLS.json`. These execute the actual population/disposition/CLI-domain functions with integer labels and the pinned manifest, not tensors. No Torch import, operand generation, calibration, model/GPU job, remote mutation or source/gate edit occurred.

All **25 accepted arithmetic function bodies and source hashes are byte-identical to reviewed v1**. The separate preparation, exact C1 BF16 output-store/FP32 persistent-state conventions, witnesses and independent C2p findings remain closed. Contract v4.2, manifest v1, constants and the unopened qualification seed are unchanged.

## Concrete closure

`cells_from_extents` and `population_record` (source lines 980–1015) count only arrays that exist. Missing surfaces contribute no cells; a declared schedule is separately recorded with `observed: false`. The record includes missing, extra and duplicate counts and accepts the sealed entry only when count and digest match exactly.

`completeness_disposition` (1025–1090) is the single outcome path. `FULL_CALIBRATION` requires exactly seeds **20260930 and 20260931**, all **48 layers**, and both actually evaluated surfaces for every manifest policy/seed, with `sealed_manifest_entry is True`. `None` and `False` both prevent resolution. A partial run yields only `PARTIAL_DIAGNOSTIC`; a full run with a validity failure yields `REPORT_ONLY`.

Focused controls covered the original one-layer counterexample, 47 layers, either missing seed, duplicate/extra seeds, null/false sealed-entry flags, an unobserved surface even with a forged complete flag, absent state arrays, missing nodes, extra/duplicate cells, and a complete positive population. The full positive has **71,424 cells per policy/input**. All incomplete cases are non-resolving. A valid full population plus an injected validity reason is correctly report-only.

The frozen `--full-calibration` caller (1093–1104, 1317–1354) hard-wires the two seeds and 48 layers and refuses simultaneous `--seeds`/`--layers` overrides. Controls confirmed those refusals and continued rejection of seeds **20260928/20260929**. `compute_receipt` now derives observed coverage from its evaluated arrays and copies outcomes exclusively from the centralized disposition. Lumo contributes its C2p/domain population only and receives no C1 or resolution outcome.

## Bounded execution requirements

Use the manifest's exact v1.1 executable through `--full-calibration`, the parent-owned authorization bound to its SHA above, both seeds and 48 layers, and a new write-once output directory. The authorization template is not itself approval. Keep the manifest's CPU invocation: `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=4`, with the existing CPU-capable Python/Torch environment. No GPU, container, author import, model server, memory reclaim or qualification seed is required.

The package estimates approximately **8–9 minutes** from small synthetic CPU runs. This is a scheduling estimate, not a measured calibration duration or performance result. Its actual peak RSS has not been established. No new RAM floor or memory-clearing requirement is justified by this source delta; the previous 12 GiB GPU-stage floor is not being changed or applied to this CPU job. Parent monitoring should retain the owned process's progress/exit and any `STARTED.json`/`FAILED.json` evidence if it stops. A timeout or partial receipt must not be converted into complete calibration evidence.

After execution, the parent still needs to verify full population, reference premises, witness results, function/source bindings and raw cell-file hashes before accepting any `RESOLVED_CANDIDATE`. Successful completion would characterize the frozen CPU comparators on this finite domain; it would not establish author-kernel correctness, actual native GPU nonregression, or complete-cycle speed.

# Independent M1-C baseline result review — 2026-09-28

**Disposition: accept the bounded CPU comparator-baseline evidence.** No material inconsistency remains in the inspected source bindings, population, saved per-cell metrics, witness reductions, or owned launch lifecycle. The six `RESOLVED_CANDIDATE` records support the parent considering the corresponding finite-domain comparator policies resolved. This review does not qualify an actual GPU implementation, provide complete-cycle timing, resolve native Lumo C1, or open any gate.

## Preserved evidence and source identity

Read-only remote evidence was copied into `p0/monitor/review-response-20260927/m1c-result-reviewed-20260928T180100Z/repo/`, retaining repository paths. All 28 files (36,729,608 bytes) match the identical remote SHA-256 inventories taken before and after transfer. Payload files are read-only. The snapshot includes 18 result files, four launch files, authority and consumption records, executable, owned launcher, freeze, and package manifest.

| Artifact | SHA-256 |
| --- | --- |
| `SNAPSHOT.json` | `2b2c2070af72f809a42d0cf331258e14fa08dcaad2ef0f1d9d0f9bc06c3224f1` |
| `INDEPENDENT-RESULT-AUDIT.json` | `8e8ba5783146769e2cfb0e11628b4dea3fcef2558e2db3f090924537d7406e0c` |
| Calibration receipt | `0c6d5918fe2c982c7ae58dabf7e4bd3cb3251ef46a61dac21fd249b5de23cb36` |
| Launch receipt | `7a0a1e2e56ea6d4dbeae484acff2339047b324d3ff53bc229ad3216e19d82e80` |
| Executable v1.1 | `00d1c8e5a593f2f1377e43aef2002f446a6ffb7c04ef0689b38cdbe163fd721b` |
| Freeze v1.1 | `9f260d3989eb0dffb4a25973d85dcbcb96ec2cc4c4974a88b41f104ba73facdd` |
| Package manifest v1.1 | `d580c5eb9f7215c1b6319cf960b8a1d632744cdc3b093fd25d491600b62192f2` |
| Owned launcher | `a3959efe8d9dadce4ef9a6e2f68ab1dfd41f4a2eb0d927f07072b8d436b836a2` |

The executable, all 25 sealed arithmetic functions, six import bindings, and C1/C2 declarations agree with the previously authenticated v1.1 package. The receipt records CPU-only Torch 2.4.1 (`torch_cuda_build: null`), four Torch threads, and the prescribed precision controls. Both prescribed seeds, 20260930 and 20260931, are present; neither qualification nor initialization seed was substituted.

## Population and numerical premises

All eight policy/seed population records contain the exact sealed 48-layer domain, observed output/state surfaces, matching regenerated cell digests, and no missing, extra, duplicate, or fallback cells. Each output array has shape `(48, 28, 48)`; each publication-state array has shape `(48, 3, 48)`. Across 16 NPZ files and 380 arrays, the run contains 428,544 comparator C1/C2 cells plus 142,848 Lumo C2-only cells, totaling 571,392. Every stored numeric metric is finite and nonnegative.

I loaded NPZs with NumPy and `allow_pickle=False`, without importing Torch or executing any recurrence. The stored bounds are bitwise identical to independent reconstruction of `1.1 * C1_error + 2^-24 * C2_metric + 2^-149`. Recomputed mandatory RMS/max violation masks equal the saved masks. Applicability was checked against the pinned topology, including the restricted wrong-sibling output/publication populations. All 732,672 eligible structural witness evaluations were detected, with zero undetected or vacuously excluded witnesses. The smallest diagnostic witness-to-bound ratio was about 18.11; this is an observation, not a newly imposed threshold. Sensitivity-flag totals match the raw arrays.

The six supported `RESOLVED_CANDIDATE` surfaces are TreeWY author-default outputs and durable state, Weaver author-default outputs and replay state, and Weaver aligned-local outputs and replay state. These are the declared CPU policy baselines, not executions or rankings of the authors' GPU kernels. Native Lumo C1 remains explicitly unavailable: `implemented_here=false`, `cpu_rewrite_substituted=false`, no C1 comparison or resolved outcome. Its four NPZs preserve C2 metrics only.

**Evidence boundary:** NPZs preserve per-cell reduced errors, metrics, bounds, witness flags, and sensitivity flags. Original prepared inputs and C1/C2 tensor values are bound by hashes in the receipt, not archived as full tensors. This audit verifies those bindings and the saved reductions, and relies on the accepted source and recorded hashes for original tensor arithmetic and repeat equality. It does not claim an independent numerical rerun from original tensors.

## Single owned launch and marker investigation

Authority was issued at 17:46:50Z, atomically consumed at 17:48:55Z, and remained valid through completion. The one recorded child PID/PGID 1420340 matches `PROCESS.json`, the launch receipt, and the consumed authority's intended invocation. It ran the exact frozen executable with `--full-calibration`, empty `CUDA_VISIBLE_DEVICES`, OMP threads four, the fresh output directory, and the owned 1,800-second timeout. Launch elapsed time was 462.125 seconds (scientific receipt: 460.475 seconds); return code 0, timeout false, no termination signals, process group gone, no launch error, and empty stderr. The launcher has one `Popen` and no retry path; its hashes cover all 18 result files.

The reported duplicate consumption markers were not reproduced in the bounded monitor/campaign inventories: exactly one M1-C consumed marker was found, SHA-256 `655f90e038daf4b260bc8c1d218e236209fa6a846141b2ca7a753739cddc976d`. The executable does not create a second consumption marker. Authority references in `STARTED.json` and the final receipt are provenance bindings, not separate executions. The preserved evidence supports one authorized launch and one child execution; this is not a claim about uninspected paths or unrelated runs.

No GPU, model, evaluator, workload, or calibration process was launched for this review. Scientific source, numerical constants, and all runtime gates were left unchanged. Existing GPU qualification, timing, and workload counters should remain unchanged.

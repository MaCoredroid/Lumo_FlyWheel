# Q1.2a reference-rebound retry: independent result review

27 September 2026. Read-only remote inspection and CPU-only in-memory tensor/reducer checks. No GPU launch, remote writes, source changes, or gate mutations by the reviewer.

## Narrow verdict

**Accept Q1.2a as complete native GDN characterization for the frozen synthetic calibration corpus.** Both fresh processes completed their 20 fixtures, and independent reduction and raw-witness checks agree with the saved results. This is sufficient input for drafting and reviewing the candidate-blind component numerical policy. It is not candidate qualification, a full-Q1 pass, or permission to launch C0 without that separate reviewed policy and executable gate.

Run: `runs/q1.2a/q12a-native-reref-20260927T212954Z` under the remote campaign directory `/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927`.

```text
87f46775b8973580718bd368b990e65b48e8ed36842ead3eeba099f274691fd8 FREEZE-Q1_2A.json v4
23bc48ca58c673c02ff038af0455d809d34bd2b5fa5650f42c3b35ffa9feb5f9 RUN-RECEIPT.json
84771344c6b4c3504e698f1f491cc1ce2b4879199954e8c47bcfe8c5d2aa63bf procA/result.json
8d4149d3e30259a4fb2e764415d79e943147529f24f5c648dab7a1810ce9ca62 procB/result.json
fe6818cbae8cfd80450ab22d9ab201555642e67629df3fd79b8347473e2b479a summary.json
2267f5197d9f2412a5227e1cf4d9b645ceec34f27b4a90ca18c4248ee0bf48d1 fixture_manifest.snapshot.json
8c89369a3b83a68dd044cf048a13df57e1a1f20bf6739055fbeb6d3cd35bb38d q1_execution_manifest.snapshot.json
```

## Execution and provenance

Process A ran from 21:30:37 to 21:30:55 UTC; B ran from 21:30:55 to 21:31:12 UTC. Both exited 0, followed by reducer exit 0 and `COMPLETED_reduce_rc=0`. These durations are scheduling receipts, not kernel or serving performance measurements. Distinct container hostnames (`17516482617b`, `fdf960be4513`) and attestation start times identify the separate processes.

Both attest the reviewed immutable image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`, Python 3.12.13, torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, and NVIDIA GB10 capability 12.1. The loaded native module SHA is `000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f`; the native function-source SHA is `fc0c2869092321bba91685d6e303629a1cf9e0b7e08d5cc6ba7fecf234318de7`. Runner and oracle hashes match v4. OMP/MKL and torch intra-op threads are 1 in both processes, matching the reference environment; inter-op threads are 20.

The launch gate snapshot binds this exact run ID and v4 hash to one native-only retry. All files listed in the launch receipt match their recorded SHA256. All launch-time tool hashes still match the actual source files. Both before/after GPU-contention receipts contain only their CSV header. The runner records no Lumo import, no engine boot, no timing claim, and no tolerance verdict.

I reconstructed the expected Cartesian product and checked it against the execution manifest: **960 distinct records**, with no missing or duplicate tuple, comprising **20 calibration fixtures × 3 variants × 2 processes × 8 repeats**. The result records provide every requested repeat hash. The per-token variant also records every one of its 12 token positions. These are comparison records for one operator geometry (16 K heads, 48 V heads, K=V=128), not 960 model/agent experiments, and the 48 value heads are not model layers. The 20 held-out evaluation fixtures were not executed.

## Independent reduction and raw checks

I imported the frozen `q1_2a_reduce.py` using `python3 -B` and called `reduce_run` directly on the launch snapshots, retaining its result only in memory. It returned 0 with empty malformed, determinism, and nonfinite findings. Serializing with the reducer's `indent=1` reproduced the saved `summary.json` **byte for byte**, at SHA `fe6818cb…`; no path normalization was needed on the remote host.

I then read all **120 retained raw `.pt` files** on the remote CPU with `map_location="cpu"`; no bulk file copies or remote outputs were created. The audit checked:

- **1,120 tensor hashes** against the result records: primary output/final state, every per-token output/state, padded output prefix/final state, for both processes and all 20 fixtures.
- Direct A/B equality of every retained tensor, including the complete padded-16 output. All match.
- Direct equality of per-token final state and padded final state to primary final state; padded output prefix and the full per-token output sequence to primary output. All match. These cross-shape equalities are observations of this corpus, not an extension of the predeclared integrity rule.
- Finite values in every retained native tensor. Recorded nonfinite counts for every repeat/token/variant are also zero; the reducer independently checks those records. All same-shape repeat hashes and A/B hash lists match. All recorded null-row checks remain true.
- **1,040 per-head metric-array checks** against the frozen reference final state and outputs: independently recomputed max-abs arrays match exactly, and RMS arrays match within CPU arithmetic agreement (`rel_tol=1e-12`, `abs_tol=1e-16`). This audit agreement allowance is only for recomputing already-recorded metrics; it is not a candidate numerical tolerance.
- Actual fixture-file hashes against the frozen manifest. All match. Both processes report zero reference final-state/output difference and exact per-token C2 stack hashes for every fixture. The earlier reference-only repair review independently established the unchanged inputs and exact fresh-CPU reference reproduction.

No discrepancies were found. The raw audit's native/reference arithmetic used saved float64 final-state/output references; it did not regenerate the oracle under the older host runtime and mistake its rounding for a failed new reference.

## Numerical findings and their meaning

The following are **native-versus-pure-C2 errors**, not C0 errors or thresholds. State is fp32; output is bf16. Each stratum contains two calibration fixtures.

| Stratum | Maximum final-state absolute error | Maximum output absolute error over all tokens |
| --- | ---: | ---: |
| ordinary-random | 1.700954923e-6 | 2.421783846e-4 |
| near-zero-key | 1.327152529e-6 | 2.419564173e-4 |
| exact-zero-key | 1.415203899e-6 | 2.440446080e-4 |
| decay-full-gate | 1.291576790e-7 | 1.198723463e-4 |
| decay-none-gate | 3.711279621e-7 | 2.433161987e-4 |
| saturated-beta | 1.429344183e-6 | 2.437548251e-4 |
| softplus-threshold | 1.459744977e-7 | 1.193802679e-4 |
| large-state-magnitude | 2.587453748e-3 | 2.430689445e-1 |
| coarse-grid-diagnostic | 2.132057405e-6 | 2.429202032e-4 |
| gqa-head-asymmetry | 1.025136666e-3 | 2.403553421e-1 |

The largest final-state error occurs in `large-state-magnitude_0`, head 22, where that head's C2 max magnitude is about 598.436 and RMS about 96.085. Its error RMS is about `3.93677e-4`. The largest output error, about 0.243069, occurs at token index 4/head 12 in the same fixture, whose corresponding reference output max magnitude is about 88.7431. Ordinary fixtures operate at far smaller scales. A single global `0.003` state or `0.25` output tolerance would erase this distinction and is not supported by the results.

The summary's state maximum is **final-state only**. For example, `large-state-magnitude_0` reaches `0.00285361943974749` at token index 9, then ends at `0.0025874537481058724`; `ordinary-random_0` reaches `1.8612843917242827e-6` before ending at `1.4587766437035299e-6`. A policy covering accepted prefixes must use the retained per-token errors/scales, not reuse final-state maxima as if they bounded every depth.

C2r remains a separate CPU fp32 operation-order diagnostic. For example, `large-state-magnitude_0` has native/C2r final-state max error `0.00054931640625` and C2r/C2 max error `0.0023433131231058724`; `ordinary-random_0` has the corresponding values `5.066394805908203e-7` and `1.2203580646019674e-6`. These do not make C2r exact GPU truth or justify replacing C2 with whichever comparator is closer. No candidate was measured.

## Recommendations before C0

1. **Freeze the policy in a separate reviewed artifact before candidate execution.** Bind it to these exact fixtures, source/runtime hashes, comparison surfaces, shapes and split. Use only native calibration data and C2; never candidate outcomes. This review does not substitute for reviewing the generated policy file and reducer implementation.
2. **Retain the already-proposed strict comparative rule, with actual storage precision.** For fp32 recurrent state, the proposed floor is `2^-24 × RMS(C2) + 2^-149`; the max-abs floor is `2^-24 × maxabs(C2) + 2^-149`. Combine the previously proposed `1.10 × native error + floor` relative-quality test with frozen per-head absolute caps. This is a deployment acceptance criterion, not a mathematical error theorem. Do not perturb an fp32 cache to bf16 to enlarge its budget. A bf16 output surface needs its own explicitly reviewed dtype-based rule; output rounding must never enlarge recurrent-state tolerance.
3. **Index caps by stratum, head, surface, and covered prefix depth.** Derive them from the archived per-token native/C2 errors and scales (including all 12 prefixes) or a specifically declared prefix envelope. Preserve the declared input/reference-scale coverage; out-of-range cases are UNCOVERED, not automatically granted a larger cap. There is no per-model-layer calibration here. Calibration contains two fixtures per stratum; eight repeats establish determinism, not 16 independent numerical samples.
4. **Keep structural checks exact.** Same-shape independent native replay, ring source bytes, path/slot/position/token identity, untouched state rows, and staging/padding bytes remain exact checks. This run's equality between native shapes strengthens their control role but does not establish candidate replay engagement, lease handling, or pointer safety.
5. **Power the designated negative controls using the frozen policy.** Wrong source path/sibling, wrong GQA head or state orientation, and stale/no-op publication must be rejected on the surfaces they target. Independently derive expected ring operands from raw fixtures plus topology; native replay of candidate-produced rings alone cannot identify wrong-branch gathering. Do not expand a cap after observing a candidate or a negative control fail. If a criterion admits a meaningful wrong-state control, revise the experiment/claim before opening C0, preserving the failed policy record.
6. **Keep the remaining scope explicit.** This corpus is synthetic, single-operator native GDN characterization. It does not validate current-route tree scan, the fixed32/two-level export, captured native per-layer replay, convolution rounding, FA2 KV remap, graph/cache lifecycle, pending-token/drafter mapping, or next-forward logits. Production `dt_bias` dtype remains a separately declared binding item. Full-model logits require their own independent native reference and reviewed calibration; they inherit no tolerance from this result.

The original failed attempt remains preserved: `q12a-native-20260927T211319Z/procA/result.json` still has SHA `ac1bba4cdb6caf1d6f682364f6ebe3d5065b654ec8c1a3c312f194b673576b7a`. It remains an incomplete failed reference-integrity attempt, not extra passing replication. The retry completes only the bounded characterization stage described above.

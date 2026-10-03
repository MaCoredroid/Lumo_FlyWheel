# Q1.2a first-attempt review and reference-only repair

27 September 2026. Reviewer performed read-only SSH inspection and CPU-only tensor comparison; no GPU execution, remote file writes or gate mutation.

## Verdict

**The first attempt remains failed/incomplete. Approve the proposed CPU-only reference-rebinding design, subject to a new frozen package review before any GPU rerun.** Preserve the original inputs, references and failed attempt. Do not loosen the exact reference-stack gate or any prospective numerical tolerance. No Lumo/C0 evidence has been measured or used to make this repair.

Attempt: `runs/q1.2a/q12a-native-20260927T211319Z`.

```text
ac1bba4cdb6caf1d6f682364f6ebe3d5065b654ec8c1a3c312f194b673576b7a procA/result.json
07256c21c8647e109dc2bdc8dd42f628b094e5ccf5b44da129773f68649f238e original fixture manifest snapshot
d4da018aa247177fecf118749533062179a08605a02a75df44e856609aacc6d3 original execution manifest snapshot
4a506fdddda96d87e8e6338a9e1b7bfc95d5b275c14c648a2ed04a8896ca98ea reviewed launch freeze v3
```

## Observed failure and cause

Process A ran on the pinned `ffa30d66…` image, torch `2.11.0+cu130`, with expected native module hash `000ab899…`, oracle hash `64594499…`, fp32 state and the reviewed input geometry. It completed 20 fixture records, but all 20 have `integrity_ok=false`. The **only false per-fixture integrity key is `per_token_reference_validated`**. Every corresponding host/container final-state/output difference is within the already-declared reference-only bound. All recorded native same-shape repetitions, null-row and nonfinite checks pass; all finite primary/per-token/padded shape relations in A are also true.

The launcher correctly retained `PROCESS_A_FAILED_rc=3`, stopped at `2026-09-27T21:14:15.049815+00:00`, and never created `procB`. No reducer or across-process comparison has run. A's `characterization_complete=true` describes completion of that process's loop only; it is not successful completion of Q1.2a. At most 480 A-side native operator records were executed, not 960 completed two-process records and not any qualified LumoTree case.

I independently read the raw saved tensor files for three representative fixtures and recomputed the unchanged float64 oracle on the host's torch 2.4.1 CPU, using the original recorded inputs. Every recomputed host state-stack hash and final state exactly matched its original frozen reference. The saved in-container stack has float64 dtype, is finite, and differs from that host stack as follows:

| Fixture | Maximum absolute host/container state-stack difference | Additional evidence |
| --- | --- | --- |
| `ordinary-random_0` | `3.3306690738754696e-16` | Per-token differences range from `2.22e-16` to `3.33e-16`. |
| `large-state-magnitude_0` | `3.410605131648481e-13` | Per-token differences are `2.27e-13` to `3.41e-13` at the deliberately enlarged state scale. |
| `decay-full-gate_0` | `1.1102230246251565e-16` | Final states are byte-identical, while several intermediate states differ. |

For all three, the independently hashed saved native per-token fp32 states and bf16 outputs match the hashes in `procA/result.json`. These are real retained witnesses, not just boolean summary flags.

This supports a **cross-runtime float64 oracle representation difference**: the original torch 2.4.1 CPU environment reproduces the old hashes, while the immutable image's torch 2.11.0 CPU reference has very small different rounding. The audit does not isolate PyTorch version from its CPU math libraries or other runtime details, so a claim that one specific library operation caused it would be premature. It is not evidence of a native GDN failure. It is also not grounds to waive the reference gate: the decay fixture demonstrates directly why a matching final state cannot establish matching intermediate states.

## Approved repair rule

1. **Preserve the original evidence.** Keep the original fixture files/manifests, freeze v3, A-side tensors/logs and exit-3 receipt immutable. Write repaired references into a new versioned fixture directory and link the failed attempt in the repair receipt. Do not relabel A as passed or reuse it as a completed two-process characterization.
2. **Rebind references from the same raw tensors, not from seeds.** Load the original `q`, `k`, `v`, `a`, `b`, `A_log`, `dt_bias`, and **`S0`**. Their dtype, shape and canonical contiguous data bytes must be unchanged. In particular, do not regenerate the warm initial state under the new runtime; S0 is part of the frozen input, even though its original construction used C2. Record per-tensor source/destination identity hashes, original fixture SHA and new fixture SHA.
3. **Use the unchanged pure C2 in the immutable image, CPU-only.** Launch the image without a GPU device request, compute CPU tensors only, and import only the independent torch oracle. Retain oracle equations, input dtypes, epsilon, threshold, output scale, GQA mapping and update order. Do not read native-result tensors, fit thresholds, select seeds or choose references by closeness to native/C0 outputs. Recompute final state, outputs and the exact per-token state-stack hash consistently in that runtime.
4. **Bind the numerical runtime.** Record image ID, Python/torch version, CPU library/build configuration, dtype, thread configuration and oracle source hash. Verify reference regeneration in fresh CPU processes before GPU work. Keep the exact stack-hash criterion; if reproducibility fails within the chosen runtime, investigate it instead of relaxing the gate. Differences from the preserved old reference are diagnostic provenance only.
5. **Keep all experimental conditions fixed.** The 20 calibration IDs, stratum membership, input tensors, repetitions, native variants and process count remain unchanged. Preserve calibration/evaluation separation. If evaluation references are also rebound for future use, do it solely through the same deterministic CPU operation and record the unchanged evaluation inputs; do not run or inspect candidate evaluation to choose them.
6. **Freeze and review the new package.** Update the fixture/execution manifests and launcher paths/provenance as necessary, retain old manifests, and rerun affected CPU integrity checks. Review the rebinding tool and its raw-input identity receipt. Open a new one-run native-only gate only for the resulting hashes, then rerun both A and B in fresh processes under a new run ID. Candidate numerical-policy approval still occurs only after that complete baseline characterization has been reviewed.

No candidate threshold, native tolerance, fixture difficulty or requested experimental scope changes under this repair. Reference rebinding is acceptable here because it corrects the oracle's execution-environment identity before any C0 measurement; it must not become a mechanism for making a measured candidate pass.

## Frozen repair review: v4 approved for a native-only retry

Reviewed the remote package frozen at `2026-09-27T21:23:16.399565+00:00`. **Approve one native-only Q1.2a retry against this exact freeze.** This approval does not qualify LumoTree, establish a candidate tolerance, or authorize timing, model inference, or the rest of Q1.

```text
87f46775b8973580718bd368b990e65b48e8ed36842ead3eeba099f274691fd8 FREEZE-Q1_2A.json v4
2267f5197d9f2412a5227e1cf4d9b645ceec34f27b4a90ca18c4248ee0bf48d1 fixtures/q1_2a_v2/manifest.json
8c89369a3b83a68dd044cf048a13df57e1a1f20bf6739055fbeb6d3cd35bb38d q1_execution_manifest.q1.2a.v2.json
c9ca7b09283994153db3efc71d127c4c8bab0be68e89d07277c8ceccf7e2db21 tools/q1_fixtures_reref.py
0cf1b564e10cb2767a6f237cf8eb9935d8db066f718f2b91ef8328a4808fb630 identity/reref/REREF-PROVENANCE.json
7f2ae326cfb897edbba7df85b2ef3bffb0797f416f73b278f9f68a928b04459b fixtures/q1_2a_v2/verification_R2.json
5f808b841a10ad9e2e3e324f60dca764fbd7f45230a86e4954c4f88bdbb501bf fixtures/q1_2a_v2/verification_R3.json
```

All file hashes declared by the v4 freeze match the actual remote files. I independently loaded every original/rebound fixture pair and checked all **320 raw input tensors** (eight tensors per fixture, including S0): dtype, shape, stride, contiguous data bytes and recorded tensor hashes are unchanged. All 40 original file hashes match the preserved original manifest; all 40 rebound file hashes match the new manifest. Seeds, splits, strata, T, warm-token count, shape/dtype metadata, scale, epsilon, softplus threshold and C2 specification are unchanged. The original failed result remains at SHA `ac1bba4c…`; none of its records becomes a passed calibration record through this repair.

The reviewed rebinding implementation loads original tensors with `map_location="cpu"` and recomputes only the pure C2 reference outputs, final state and full per-token state-stack hash (`q1_fixtures_reref.py:68–94`). It checks the source fixture against the old manifest before writing the versioned destination (`:130–152`). It neither imports the native implementation nor consumes native-result tensors. The oracle source remains exactly SHA `645944992d116115976eea5a925f1ebf9a23562d8d8a3ebd693116daa0ec5d82`; the fixture generator and reducer are also unchanged.

Fresh CPU receipts R2 and R3 each report **40/40 exact** output, final-state and full state-stack comparisons, plus original-input identity for all 40 fixtures. They record distinct container hostnames/timestamps and the same pinned runtime: image `ffa30d66…`, Python 3.12.13, torch 2.11.0+cu130, aarch64 CPU model `0xd87`, 20 CPUs, one torch intra-op thread, 20 inter-op threads, and `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`. I checked the actual receipt fields and the raw tensor identities, rather than relying only on `all_ok`. The recorded command receipt abbreviates the Docker command as a CPU-only run rather than retaining complete argv; CPU-only operation is supported by the tool's explicit CPU loads and pure torch oracle, with the omitted argv a provenance limitation rather than evidence of GPU use.

The runner change adds CPU/thread/environment attestation; the original strict exact per-token reference-stack criterion remains intact. The launcher now points to the rebound fixture and execution manifests, pins OMP/MKL threads for both native processes, and archives the repeat-verification receipts. The existing hash-bound gate and failure propagation remain intact. The frozen CPU log records **51 tests passed in 18.95 seconds**.

The execution manifest still contains exactly **960 unique records: 20 calibration fixtures × 3 native variants × 2 fresh processes × 8 repetitions**. The 20 evaluation fixtures have had their independent references rebound, but are not included in this calibration run. Start a new run and require both A and B plus the reducer to complete before interpreting characterization or considering a candidate policy. Same-shape or cross-process nondeterminism, nonfinite values, and reference-integrity failures remain blocking; finite cross-shape numerical differences remain characterization findings as already declared.

Reviewer performed no GPU launch, remote mutation, or gate update. No further fix is required for this narrowly scoped retry.

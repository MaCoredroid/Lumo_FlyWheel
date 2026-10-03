# Original M1-Q C0 result: independent saved-evidence audit

**Disposition: the original scorer's failures and uncovered Lumo disposition reproduce exactly.** No qualifying four-method comparison is established by this run. Lumo repeat 0 numerically passes its 71,424 evaluated cells, but repeat 1 failed before any call/capture; Lumo remains `UNCOVERED`. All six completed author-method cycles are finite and fail the unchanged, pre-C0-frozen per-cell criterion. Their two repeats are bitwise identical. This is a result for these pinned implementations and declared sequential comparison operators, not a general claim that Weaver or TreeWY is incorrect or that an agent degenerates.

Scope is the preserved original run `runs/m1-q-c0/m1-q-c0-20260928T221747Z`, original reduction `runs/m1-q-c0/reduction-original-20260928T222106Z`, and previously accepted held-out pack/seal. The reviewer used only saved files and CPU Torch 2.4.1 / NumPy 2.3.4 with `CUDA_VISIBLE_DEVICES=`. No candidate/reference generation, GPU operation, new comparison setting, threshold change or gate mutation occurred. Later lifecycle repairs and later runs are outside this review.

## Reproduced population and outcomes

Counts below apply separately to each indicated repeat. Outputs have 48 layers × 28 active nodes × 48 value heads = 64,512 cells; state has 48 layers × 3 publications × 48 heads = 6,912 cells.

| Implementation | Completed repeats | Failed output cells / repeat | Failed state cells / repeat | Original disposition |
|---|---:|---:|---:|---|
| Lumo fixed32/native replay | 0 only | 0 | 0 | `UNCOVERED`: second repeat absent |
| Weaver author preparation, TF32, `bf16_mode=none` | 0 and 1 | 53,745 | 1,154 | `FAIL` |
| Weaver with declared local FP32 preparation | 0 and 1 | 57,707 | 388 | `FAIL` |
| TreeWY author, `dot_bf16=True` | 0 and 1 | 37,528 | 6,912 | `FAIL` |

Seven complete cycles provide **499,968 repeated cell observations**; **71,424** expected Lumo repeat-1 observations are missing. All evaluated cells are finite. The 4,032 saved layer metric/mask arrays reproduce bitwise from raw tensors and the original sealed bounds. No repeat averaging or outcome selection was used. Every saved author output and all four state captures match its other repeat's raw digest.

The exact original Lumo repeat-1 error is `FR13_FIXED32_CONV_PREGATHER builder SSI group changed after preseed`. Its call log and state-capture list are empty. The failure occurs during the second executor's Backend boot/reset path, before measured cycle execution; it is not a numerical Lumo failure. The collector continued the declared serial author cycles, then correctly persisted `FAILED_PRESERVED` because collection was incomplete. The original scorer retained real author failures and demoted Lumo's numerical pass to `UNCOVERED`.

## Mapping checks and reference semantics

**No simple output-row or publication-order error was found in this bounded audit.** In addition to raw file authentication, the audit checked 1,632 recorded first-step input argument digests and all 336 complete-cycle initial-state equalities.

1. **Weaver order.** `m1_adapters_v2.py:241–299` passes the physical/level-order parent tree and preparation tensors to `tree_gdn_triton_verify`; the actual logs bind the physical parents, slot 1, original a/b/v/A_log/dt_bias, and the named BF16 or FP32 q/k storage. `m1_stage_collector_v3.py:245–249` captures the actual returned `o[0]` on step 0. The scorer selects the same 28 active physical rows as the frozen manifest.
2. **TreeWY order.** `m1_adapters_v3.py:68–86` gathers physical rows into the declared DFS ordering and applies its inverse to output. All 96 first-step layer outputs (48 × 2 repeats) pass a stronger byte check: taking the saved physical output back into DFS order exactly reproduces the input digest recorded at the actual inverse-remap call. The q/k/v/a/b argument digests at the author kernel also match the original operands gathered into that same DFS order. This rules out an omitted, doubled or wrong inverse-remap for these saved first outputs.
3. **State boundaries.** `m1_cycle_driver_v3.py:147–190` captures Weaver state after each replay, and TreeWY state after the next call's deferred commit, including the final flush. Both author repeats bind paths `[0]`, `[0,1,4,9,14]`, `[0,1,4,9,15]`. Actual Weaver leaf values are 0/14/15; actual TreeWY DFS leaf values are 0/4/14, across all 48 layer calls. All three state-capture digests equal the corresponding recorded cycle endpoints and chain from the preceding capture. The reference history is the uninterrupted 1/6/11-update sequence, not three independent restarts.
4. **Policy-specific C2.** `m1_c_baseline_v1_1.py:346–395` gives Weaver-default output BF16-normalized q/k with unrounded beta, and replay state BF16-normalized key **and BF16-rounded beta**. Aligned Weaver uses the declared FP32 preparation. TreeWY uses its own norm floor and unrounded gate/beta; it is not scored against Lumo's additive-epsilon policy. The raw reducer matches each named surface to that policy's archived float64 C2 and verifies its hash. The prior independent reference-seal review remains applicable.
5. **TreeWY normalization.** Actual pinned `tree_wy_triton.py:212–236` uses reciprocal multiplication by `1/max(sqrt(sum(x*x)),1e-12)` in FP32, with scale applied to q. C2's float64 `x/max(sqrt(sum(x*x)),1e-12)` is the intended mathematical policy (`m1_c_baseline_v1_1.py:364–365`); it is deliberately not instruction-identical FP32 arithmetic. The run is the original `dot_bf16=True` route, not an unimplemented normalization-aligned variant.

## Meaning of the numerical failures

The predeclared rule is the conjunction of per-cell RMS and maximum-error limits: `error(C0,C2) <= 1.10*error(C1,C2) + 2^-24*scale(C2) + 2^-149`. The original scorer/source was bound in the approved gate before C0 executed. None of the observed failures justifies changing it retrospectively.

The author C1 definitions explicitly describe independently written **sequential** software comparators, and explicitly disclaim reproduction of chunked/WY/Tensor Core accumulation or GPU transcendental rounding (`m1_c_baseline_v1_1.py:552–556`). Thus the result is failure of that declared non-regression envelope. It is not failure against a hardware-exact reference of the author's implementation.

A concrete TreeWY distinction is already visible without a new experiment: its DBF commit forms `delta = dot(bf16(weighted_vt)^T, bf16(stashed_key))` before adding decay (`tree_wy_triton.py:264–280`). The frozen C1 state comparator instead declares an FP32 residual/outer-product update, quantizing only reduced projection operands (`m1_c_baseline_v1_1.py:608–633`). Compact-state factorization and these different rounding locations are not equivalent in floating point. This is consistent with a substantial state-envelope failure, including root-only publication, but the present audit does **not** isolate a unique numerical cause. The bounds and original result stay intact.

Some observed errors are small in absolute terms despite failing a tight envelope. Aligned Weaver state RMS spans `6.51e-10`–`1.52e-7`; TreeWY state RMS spans `7.56e-6`–`4.13e-4`. TreeWY's per-cell state RMS relative to C2 RMS is `4.20e-5`–`2.81e-3`; all 2,304 cells at each of its three publications fail the original rule. Weaver-default state fails 74/484/596 cells at publications 1/2/3; aligned Weaver fails 12/143/233. These describe this fixed input population only and do not imply downstream token/task effects.

Weaver's pinned verifier uses TF32 compact-state dot products (`gdn_tree_triton.py:305–317,350–380`) while its replay is an FP32 elementwise recurrence (`chunk_tree_verify.py:894–906`). GPU norm/gate reductions can also differ from the sequential software comparator. The source supports those mechanisms as possible contributors; no causal decomposition or broader author correctness judgment has been established here. Preserve the raw results and qualify their comparator/precision settings in any paper discussion.

## Identity, lifetime and preserved evidence

- Original launch receipt SHA: `014bf1bfb81782bed88c02c9231fe7686b3cb52b487dbf708e950fb8cac7d8ea`.
- Original gate SHA: `3691f1b4921453691a4e88fbc667654c712c840cfa75a4de647e0339e40b518d`.
- Original reduction receipt SHA: `b45331eb41c3b56c8695636c982cdcf625b44d34a5683b3168d5848d675f83d4`.
- Pre-C0 scorer SHA: `4e2aa7dab16e385ddee610d85b69ef9e23f0ce0d0cdf51fc3538578117f736de`.
- Reduction authorization SHA: `be6042f5a058d27b1b480b993ddc633c2a91c514ff97014b890351b7317274f0`.
- Accepted reference seal remains `a2521810df84d9e42751f1b1d8e5777f38f885cc56c7171ada3332db479e04b8`; pack remains `5f6157fd525ad8b465ddcdbcbd0a54f2c04936cc551bf504da238fe4817cdba9`.

One owned container, CID `185a029be88762ec0247723d7104500cae3c3e1c44f3536629212d035e2238b8`, used the immutable image `ffa30d66…`, Torch 2.11.0+cu130 and Triton 3.6.0. The nonce/image/name checks and restart count zero agree throughout. Container and attach exit codes are 3; exact-CID removal succeeded and a successful full-ID enumeration proved absence. The one-use gate marker matches the original gate. Source hashes in seven completed method receipts resolve to the recorded bytes, including 33 source dependencies. TreeWY source is `c3dc8849…`, Weaver verifier `b151d4b2…`, and Weaver replay `1e141a8d…`. These are byte-bound local integrations with the disclosed loader shim, not full serving-stack comparisons.

The original launch lasted 75.329 seconds; the original CPU scorer completed with exit 0 and 51.114 seconds of reducer work. These are operational durations, not speed measurements. All original failed evidence is preserved. The reviewer authenticated 398 unique source/raw/evidence files totaling 12,170,126,800 bytes; CPU raw reduction took 14.63 seconds. All 72 copied result/launch metadata and NPZ artifacts match the remote hashes. No full raw-tensor bulk copy was needed.

Reviewer artifacts are in `p0/monitor/review-response-20260927/m1-q-c0-result-independent-review-20260928/`:

| Artifact | SHA-256 |
|---|---|
| `audit_raw_cpu.py` | `a28a3d34770b0ca5d5544b0107c6b9dbd9bc26805d6d0b3bd7d4e50954381220` |
| `RAW-AUDIT.json` | `88c3e9c99e841b6b2e6583b86f25371a287fb9a4ccf8abd8092c53a0ad4242ed` |
| `audit_metadata.py` | `fc7d5d61c63cb923b044b3aa4657d21297b751fa42706a4465e7c43d216ff8ba` |
| `METADATA-AUDIT.json` | `ba1bb3310f44eeac27bd345f71462857080bc83b074a7ae28e8f604d34a99dfc` |

The first metadata-auditor attempt mistakenly expected the native-reference nonce label; its `KeyError` and original script are retained. Correcting the reviewer lookup to the actual C0 label `lumo.m1.c0.nonce` produced the complete PASS audit. No producer source or artifact changed for that correction.

The smallest justified implementation repair remains the independently reviewed Lumo Backend lifetime fix. A later repetition must keep the original failure, inputs, reference policy and thresholds intact and remain separately bound. The six author results are reproducible failures of the original criterion; claims beyond that remain unsupported.

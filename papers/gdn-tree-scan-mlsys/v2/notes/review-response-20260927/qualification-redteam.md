# E1/E2 qualification red-team — 27 September 2026

Scope: independent source/test review for the approved adversarial-review response. No GPU jobs, remote mutations, runtime/source edits, commits, or manuscript edits were performed. This note is a proposed test protocol, not a qualification receipt. Paths below are repository-relative unless they start with `v2/`, meaning `papers/gdn-tree-scan-mlsys/v2/`.

## Findings that should change the immediate execution plan

1. **There is no reviewed current-route end-to-end state/next-forward oracle.** Existing component checks are useful but do not compose automatically into that claim. Run those cheaply, then build a diagnostic that actually calls the current fixed32 publication route and consumes the published boundary in the next target forward.
2. **Do not run the old Cat10 replay test as a current-route credential.** `tests/test_fr13_replay_gpu_byte_ab.py` uses the ten-node tuple, legacy node-state export, accepted-column initialization, and a non-`None` state buffer. Current `launch_tree_gdn_prepared` rejects a state buffer (`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:16790`). Its failure would not establish a new deployed defect, and its old results do not qualify Hydra27.
3. **Raw common operands do not currently mean identical mathematical inputs to the author methods.** TreeWY normalizes by `max(norm(x), 1e-12)`; Lumo uses `sqrt(sum(x*x)+1e-6)`. Weaver's serving path normalizes before its verifier. Resolve and report this difference before calling numerical disagreement a WY/compact-state defect.
4. **Charge actual commit policies.** TreeWY's fused call commits the preceding stash before verifying the next tree. Weaver's serving backend replays accepted operands; its existing BF16 benchmark does not time that replay. A verify-only comparison cannot support the proposed mechanism claim.
5. **Freeze the executed route from receipts and positive engagement, not HEAD or test filenames.** Optional single-launch GDN, all-parent acceptance and layer-batched commit tests cannot be promoted into credentials for the active two-level, per-layer native-replay route.

## 1. Route to freeze

The reviewed current production reference is NVFP4 Qwen3.8-27B, full vocabulary, Hydra27 over 32 physical rows, patched FA2 `gqa_pair_splitk`, two-level GDN path execution and one captured replay containing 48 native per-layer GDN calls. Hydra31 exists in source but is not a completed served replacement. September Cat10/Triton experiments are a separate lineage.

Source/receipt anchors:

- `scripts/fr13_fixed32_topology.py:62` onward defines the physical tree, Hydra27 validity and later Hydra31 profile. Hydra27 has 27 valid drafts, root, and four inactive physical rows. Every test must derive active paths from the Hydra27 authority, not make all 32 rows acceptable.
- `v2/results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/container_env.SANITIZED.txt`: `FR13_SUBTREE_PARALLEL=1`, `FR13_SCAN_ALIGN=0`, `BV=8`, `COMMITTER_LAYER_BATCH=0`, `GDN_SINGLE_LAUNCH_PRODUCTION=0`, `GDN_GQA_GROUP3_PRODUCTION=0`, and both TAW native-precompute selectors zero. Graphs, fused convolution, native commit and slot remap are enabled.
- Same receipt directory, `logs/fr13_fa2_qrow32_b1_production_engagement.json`: positive actual FA2 engagement, exact binary/source identity and sixteen target attention layers. Recollect this for the new boot; an environment flag is insufficient.
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:16632`, `launch_tree_gdn_prepared`; its path-launch block at `17270` uses `st["export"]`, first the root path and then terminal paths. This is two launches with real cut-state traffic.
- `_gdn_node_step` at `10409`: `SCAN_ALIGN=0` carries fp32 state, computes raw gates, and uses normalization with epsilon `1e-6`. Comments describing an older bf16 packed-decode oracle do not establish the dtype of the current model's state bank.
- `_fr13_fixed32_committer_graph_body` at `14551`: path capacity 16, per-layer rounded operand gather, neutral padding, and every native call writes to the request's column-zero row. The live alternative `layer_batch` must stay off for this qualification. `_fr13_fixed32_committer_replay` at `16127` and `launch_tree_gdn_replay_all_layers` at `16280` are the production-facing continuation entry points.
- Ring checks at `16800` require byte copies in the input dtypes. That guarantees stored inputs, not equality of differently scheduled scan/native arithmetic.
- `scripts/fr10_phase4_patch_vllm_tree_gdn.py:38708`, `_patch_gpu_model_runner_slot_reorder`, and accepted remap around `41768`: logical query order remains unchanged, while KV source slots and mask key columns must share the physical permutation. `scripts/fr13_patch_fa2_tree_bias.py:9430` is the mask-column side.

At review HEAD was `470a589d4948e274278b861ff2a85157409976cd`. The four source files below were clean and retained their audited hashes:

```text
d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8 src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
c382ef22a5a8d4e84209e6723f8eea21c3819543c6e942c62951385b5c1c003e scripts/fr10_phase4_patch_vllm_tree_gdn.py
c02703b4bc777e5edfa01623b446a289cba8f1023e31d8efd7fc8126666b38bc scripts/fr13_fixed32_topology.py
8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2 scripts/fr13_patch_fa2_tree_bias.py
```

The future run must additionally hash the **loaded patched modules**, imported native recurrence, model configuration/tokenizer/weights, FA2 `.so`, CUDA/Triton/PyTorch versions, compiler flags and launch environment. Hashing only the patcher misses its generated code and failed patch application.

## 2. Reusable inventory and exact limits

| Reuse | What it checks | Limit / needed extension |
| --- | --- | --- |
| `scripts/fr13_fixed32_semantics_test.py`, `test_compact_logical_reference_equivalence` (421), `test_invalid_node_poison_invariance` (478), `test_duplicate_token_masking_and_q_mix` (515), `test_tail_hydra_tail_cache_stability` (554) | Physical/logical sampler semantics, invalid candidates, duplicates, cache keys | CPU reference semantics, not the live acceptance implementation or full model. |
| `tests/test_fr14_gdn_schedule_contract_parity.py`, `test_the_planted_table_mirrors_the_serving_package_exactly` (131), `test_the_table_agrees_with_the_topology_authority` (148) | Generated runtime schedule and source authority agree, including parsed Python embedded in patch strings | Source contract checks do not establish executed geometry; compare live schedule census too. |
| `tests/test_fr13_fixed32_gdn_exact_io.py`, `test_fixed32_exact_io_is_byte_identical_to_dynamic_route` (78) | CUDA specialized fixed32 path I/O versus dynamic subtree route, ring/output bytes | Same implementation family, no independent sequential oracle; sets `tail6_fixed32`, `n_actual=32`. Reuse plumbing, explicitly arm Hydra27 and check all valid paths. |
| `tests/test_fr13_fixed32_gdn_path_bv_live_gate.py` (87,133,167,192) and kernel `fixed32_gdn_bv_live_capture_begin/end`, `_fr13_fixed32_gdn_bv_live_capture_register` (2730) | Useful pattern for retaining graph operands, refusing incumbent-versus-incumbent tests, restoring served bytes and gating first measured replay | Tests use controlled/mocked launches; candidate-width gate is not itself a sequential-state proof. Capture references alone are not immutable snapshots. |
| `tests/test_fr13_tree_conv_fused_byte_ab.py::test_fixed32_direct_leaf_matches_full_writeback_commit` (313) | Current physical32 ancestor-window selection | CPU fixture; only selected leaves. Extend to all valid accepted leaves, all layers and actual width/stride layout. Most other topology cases in this file are chain/Cat9/bushy/single, not Hydra27. |
| Same file `test_t5_full_pipeline_synthetic_gpu_byte_ab` (634), `test_t6_capture_payload_anchor_gpu_byte_ab` (660) | Convolution tap/cast/SiLU byte checks; captured operand anchor | Captured case silently **skips** without a payload and auto-searches old captures. Supply an explicit current-route payload and fail required qualification if absent. Its anchor is convolution arithmetic, not post-publication next-forward state. |
| `tests/test_fr13_fixed32_conv_commit_wiring.py` (65,109,173,255,417) | Direct fixed32 conv route, row guards, strides and preseed binding | AST/wiring checks need a GPU value oracle. |
| `tests/test_fr13_fixed32_kv_remap.py` (94,124,149) | Mixed-row KV mapping, fresh flat slots into drafter, stale-source negative | Exercises extracted functions with small tensors. Add physical spine-first source mapping and all sixteen attention caches from a real invocation. |
| `tests/test_fr13_fixed32_drafter_tree_ownership.py` (88,158,227,278,444) | Forward/replay identity, target16 versus MTP ownership, stale replay refusal | Does not compare real selected hidden-state rows or resulting next draft probabilities. |
| `tests/test_fr13_fixed32_prefill_owner_bootstrap.py` (211,264,326,383,469) | Long/mixed prefill owner establishment, async proposal/flush ordering | Runtime stubs; extend through actual request allocation/retirement and loaded graph replay. |
| `tests/test_fr13_fixed32_arctic_lifecycle.py` (83,217,434,647,711) | B1/B4 suffix context, owners, failures, same-id request replacement | Cache bookkeeping checks, not a full-model APC hit/miss oracle. |
| `tests/test_fr13_fixed32_task_boundary_snapshot.py` (332,580,628) | Counter/census reconciliation and malformed/partial evidence rejection | A valid receipt proves recorded engagement/accounting, not tensor correctness. |
| `tests/test_fr13_fixed32_taw_exact_commit_cuda.py` (264) | Full-vocab synthetic threshold cases and products, root/bonus counts, selected last row | `_run_pair` (130) explicitly exercises exact and **all-parent** candidates against torch. All-parent is off in current production. Reuse fixtures but call and attest the active dispatcher separately. |
| `results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe_result.json` | Pinned promoted fused draft-selection parity, ties and graph cases | Reuse unchanged binary evidence; require matching loaded `.so`. Not a state oracle. |
| `results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_credential.json` and bounds | Predeclared FA2 numerical/determinism scope | Synthetic Q/K/V at measured scales; random-projection argmax is **not model logits**. Do not transfer its bound to full-model continuation. |

Other historical patterns require adaptation: `scripts/fr13_committer_graph_varying.py` tests changing accept lengths/B but hardcodes 32 value heads and path capacity 12, whereas the current route uses 48 value heads and fixed16 staging. `scripts/fr13_native_committer_validate.py::packed_ref` (68) has a useful independent PyTorch recurrence and warns that native packed decode using null row zero can silently fail to accumulate. Its `main` prints a pass/fail string without nonzero exit on failure and covers only a tiny old fixture. Never accept process exit 0 as its scientific verdict.

## 3. E1 must separate three questions

### A. Does publication implement exactly the selected path?

Use **identical recorded per-layer operands and forced common accepted paths**. Start from cloned pre-tree state, not already-mutated banks. Independently construct ancestor lists from the topology, never from the tested accepted-path gather. The reference calls the loaded native update in path order and a separate high-precision recurrence. Inspect all 48 GDN running rows, convolution windows, sixteen attention caches, positions, emitted tokens, pending token and drafter hidden row. Check untouched request rows bytewise as well.

Publication-only comparison uses each arm's *own* computed candidate tensors to test gather/copy semantics exactly; full sequential comparison separately tests whether those candidate tensors are numerically equal. Otherwise a wrong index can be mislabeled rounding, or expected upstream numerical differences can be mislabeled a gather bug.

The no-draft-accepted case still materializes the root. `accepted_len` excludes that root in fixed32 commit metadata; replay begins with `[0]`. A correction/bonus token can already be emitted but is not materialized until its next target forward. Freeze an explicit list of **materialized token IDs** and **pending token IDs** for each fixture; do not derive the reference token count by copying the implementation's counter.

### B. Do candidate outputs and consumed state agree with the chosen reference?

Use two oracles:

- independent float64 recurrence on the **same rounded raw operands**, with explicit normalization epsilon, softplus threshold, sigmoid, GQA expansion, state orientation and output scaling;
- actual native sequential target execution from the same prefix and **forced token path**, with its real state dtype/rounding and attention backend recorded.

Report first divergence by layer, token/path depth and surface. Include max absolute error, per-head RMS, relative RMS with a fixed near-zero denominator floor, error distribution/ULPs, nonfinite counts and margins to the acceptance or greedy decision boundary. A single tensor-wide max scale can hide a corrupted small head. Comparing only a final argmax or accepting `torch.allclose` defaults is insufficient.

### C. Does the next step actually consume the published state?

After publishing each path, run the next target forward consuming the same pending token and position on both sides. Save full target logits, pre/post states and the selected next drafter input. Repeat multiple continuation cycles **without resetting the state each time**; isolated one-step replay cannot expose stale buffers or accumulated error. Do not use random linear projections as model logits.

Minimum structural matrix: root-only; every active Hydra27 root-to-node path (covers all valid lengths and off-spine branches); longest spine then root-only then a different branch; longest valid accept followed by short accept; pending correction and full acceptance with bonus; one and repeated graph replay; exact physical slot/page boundaries; cache cold/hit; request end/reuse and same-id replacement. B1 is the current deployment gate. B4 mixed owners/lengths is a separate component extension, not automatic qualification of concurrent agent serving.

## 4. Numerical gates and controls to freeze before timing

**Hard zero-tolerance surfaces:** parent/slot/path/token/position integers; logical validity; ring byte copies; gather/copy destination bytes relative to their source; untouched rows; generation/owner IDs; no nonfinite values on finite fixtures; no stale/double commit. Native replay from the same rounded operands through the same native operation order should be byte-identical to an independently prepared native replay. If comparing a different native kernel shape changes arithmetic, record it as a numerical comparison and retain a same-shape exact control; do not silently relax the structural gate.

**Baseline-only calibration for numerical surfaces:** there is no justified universal `atol=1e-2` for this full model in the reviewed evidence. Before candidate evaluation, freeze a disjoint calibration corpus of actual prefixes and adversarial finite states. Run at least eight repeated native executions within each of two fresh processes. Record exact determinism and separately record native error against float64 component oracles. Partition budgets by surface, layer/head and precision; lock reference-derived absolute scales and a near-zero floor. Calibration must never include the candidate errors being judged.

A defensible initial comparative rule for the component study is: on held-out cases, candidate RMS error to the high-precision oracle may not exceed `1.10 * native_RMS_error + roundoff_floor`, with a separately frozen per-head maximum-error cap and zero nonfinite disagreement. The multiplier is a **proposed policy**, borrowing the existing FA2 comparative strictness, not an already validated whole-model tolerance. Derive the floor and maximum cap from baseline-only calibration and declared dtypes, then hash the table. If the baseline itself has a large error or the cap cannot distinguish injected wrong-state controls, fix the reference/protocol; do not widen it. Publish raw errors even for failures. A mathematically correct but differently rounded compact method may fail this policy; describe that as failing this deployment's numerical criterion, not as an intrinsic mathematical defect.

For full-model next-forward logits, freeze a separate bounded-error policy from the native-only calibration; record greedy decision margins and full-distribution distances. Exact copied-state semantics can pass even when logits differ because the scan and native verifier round differently. A fixed seed does not imply identical stochastic samples across different RNG scheduling. Distribution preservation is an additional claim requiring a correctly matched sampler proof/test, not a consequence of a logits tolerance or no detected degeneration.

Required powered negatives:

- Replace one accepted leaf with a sibling whose data are distinct; oracle must detect conv/KV/hidden/recurrent mismatch.
- Permute physical KV slots but omit the corresponding mask-column or commit-source permutation; require the attention/publication tests to fail.
- Swap two request owners or two layer operand rings; check affected and untouched rows.
- Reuse an earlier generation's path/leaf metadata, fail a lease, or skip the last commit; state/next-forward checks must fail.
- Shrink accepted length without clearing trailing staging values; inactive values carry distinct **finite** sentinels. NaN poisoning is a separate unsupported-input stress case, since `0*NaN` can contaminate arithmetic even on mathematically masked paths.
- Materialize the pending token early or advance its position twice; check next-forward token/position and states.
- Intentionally bypass the candidate, returning the incumbent twice; engagement/identity checks must reject the test even if all tensor comparisons pass.

No required case may be silently skipped. Report skips as uncovered, and use nonzero exit for failures. Calibration, correctness and timing use separate output directories and receipts.

## 5. E2 adapters and accounting

All **14** vendored files still match `v2/experiments/author-code-20260926/SOURCE_MANIFEST.json` at this review. TreeWY is pinned to `sneha5gsm/vllm@b073ed6cacfa1cf5ae23111854729e662a72a6d1`; Weaver to `trymirai/sglang@aeac03f0d4c8789559411be95c5c127bdff24d1c`. Existing 39 TreeWY author tests are selected kernel/reference tests, not installed model tests. Existing Weaver nine numerical cases/two timings have no pass/fail numerical assertion and time verification only.

### Operand contracts

- Lumo `_gdn_node_step:10458`: epsilon inside `sqrt(sum(x*x)+1e-6)`; raw bf16 gates and raw q/k, fp32 carried state with `SCAN_ALIGN=0`.
- TreeWY `tree_wy_triton.py:212,233`, reference `_l2norm` at `tree_wy_ref.py:27`: `x/max(norm(x),1e-12)`; normalization is unconditional in the exported fused wrapper. `dot_bf16=True` (default) additionally casts normalized q/k, carried `s0`, weighted factors and output products to bf16 before dot products; fp32 state storage alone does not make it a fp32 computation. Test both default mode and `dot_bf16=False`, labeled independently.
- Weaver `gdn_backend.py::_fused_tree_verify_forward` (687): precomputes gating into stash, normalizes q/k with its serving l2norm helper, copies v into stash, invokes `tree_gdn_triton_verify(...use_qk_l2norm_in_kernel=False, precision="tf32")`. Its `bf16_mode` is environment-selectable unless explicitly fixed. The standalone BF16 benchmark first uses `F.normalize(...).to(bfloat16)` and therefore does not model Lumo's raw-operand boundary.
- Weaver `_chunk_tree_verify_forward` (760) uses different preparation/copies and a different verifier. Choose the fused production comparison explicitly; keep chunk mode as a distinct arm if added.

Use a shared raw-operand fixture with exact dtype, stride, logical parent and GQA mapping, plus independently computed post-normalization expectations. Preserve an **unchanged-author** result. If the TreeWY epsilon is aligned, save a minimal explicit arithmetic diff and label it a **local contract-aligned port**, not untouched author code. Supplying pre-normalized data does not solve the unconditional TreeWY re-normalization. Near-zero/zero-norm keys must be separate diagnostic strata; the normalization mismatch can dominate them. Author-default comparisons remain useful, but their differences combine numerical policy and method.

A Hydra27 fixture needs an invertible node-order adapter. TreeWY's wrapper documents DFS-preorder; Lumo's physical rows follow its own authority. Construct masks and reorder **all** q/k/v/gates, leaf IDs and outputs consistently, then inverse-map for comparison. Charge required per-step conversions, while separating genuinely static topology preparation. Invalid physical nodes must never become accepted ancestors. Preserve the same logical tree in compact and padded arms; do not compare 28 active rows against 32 physical rows without reporting the difference.

### Complete-cycle boundaries

- TreeWY `tree_wy_tree_commit_capture_triton` (751) commits the previous leaf from `vt/kk/gc` stash, then computes and stashes the new tree. `test_tree_capture_commits_branch_leaf_state` (138) already demonstrates a two-call check; sentinel reuse test (214) and graph replay with changed leaf (343) are useful lifecycle templates.
- Weaver `advance_ssm_states_after_verify` (816) calls `advance_ssm_states_along_accept_paths` (`chunk_tree_verify.py:925`), which replays normalized keys/gates into native state. The state kernel at 805 recovers ancestors from bitsets and supports optional prefix-cache snapshots. This is the serving commit contract; returning `LazyGDNVerifyState` from the standalone verifier is not equivalent to committing it.
- Lumo must charge ring export, cut-state handoff, accepted metadata/staging, and the captured per-layer native replay. Do not time just `_tree_gdn_path_kernel` and call it the full mechanism.

Measure a finite sequence of **M verification/acceptance cycles** from the same materialized initial state to the same materialized final state. Lumo/Weaver perform all M commits. TreeWY performs its initial no-op commit, M verifies and the final deferred commit. If the pinned author code has no commit-only flush, an extra author fused call safely exposes the last state but also performs unused verification: include that cost and label it an upper bound for finite-session flush, or implement/label a local commit-only adapter. Do not subtract an invented flush cost. Report both finite-cycle total and steady-state per-cycle latency, with endpoints explicit.

Reset state **and all stashes/metadata** outside each timed sequence; do not reset each step and thereby erase cumulative/lifecycle errors. Warmup must use separate disposable state. Use device-prepared common acceptance paths so selection is not accidentally host synchronization in one arm only. Either include equivalent selection in every arm or label the result recurrent verify-plus-publication with forced common acceptance. A GDN-only complete cycle is **not** a full-model verification cycle: convolution, attention, projection/logits, sampler, drafter and their publication costs belong to an additional serving-loop measurement.

Report device-event and synchronized host-wall intervals separately; compilation/graph capture separately from warmed execution. Interleave arm order, record thermal/clock/memory state, use fixed repetitions and full distributions, and never select the best topology/precision after seeing timings. Static allocation, actual tensor allocation/peak scratch, logical exported bytes and hardware traffic are different quantities. Record registers/shared memory/spills for **every launched kernel** on GB10; TreeWY's comment of 114 registers/zero spills is B200-specific evidence, not a GB10 measurement. Weaver caches A/QKD/Ainv workspaces (`gdn_tree_triton.py:430`) and allocates other outputs; naive peak deltas after warmup can hide persistent scratch.

## 6. Minimal feasible launch sequence

1. **No inference:** run current CPU topology/sampler/wiring/ownership/receipt tests listed above; freeze test selection, source hashes and required-not-skipped cases. Keep historical Cat10 tests outside the current gate. Verify author hashes and adapters with tiny known tensors.
2. **One boot-free GPU component job:** on the pinned CUDA image, run current physical32 I/O, current conv/value-gather tests, active acceptance products, and native-replay same-operands tests. Adapt hardcoded old committer fixtures to 48 value heads/fixed16. Require powered negatives, byte-level copied surfaces, all valid leaves and successive length changes. Run the unchanged author selected tests as environment sanity checks only if their image/source environment changes.
3. **One instrumented current server boot:** bind actual full graph/FA2/two-level path/native replay engagement. Capture a small predetermined recorded agent-prefix set spanning short, mid and long context and all layers. Hooks must snapshot pre-state and rounded operands at a real measured replay boundary; graph-construction tensor references or eager warmup captures are insufficient. Existing `FR13_DECODE_GDN_CAPTURE` injection at patcher 17885 is explicitly eager-only and cannot certify graph behavior by itself.
4. **Paired diagnostic continuation:** for every structural case, branch the identical boundary into current tree publication and independent native sequential execution, then consume the same pending token and continue multiple cycles. Lock baseline-only numerical tables before candidate verdicts. Required output is a per-case/per-layer tensor and next-forward manifest, not only a PASS line.
5. **E2 qualification before timing:** verify author and aligned adapters, all accepted lengths, controlled long-chain and branch controls, repeated state evolution, final flush, and normalization strata. Stop on structural failures. Preserve numerical losing cases.
6. **E2 timed job:** fixed Hydra27 primary geometry, predefined chain/branch sensitivity and accepted-length sweep; B1 primary, B4 optional component sensitivity. Current precision plus explicitly labeled author defaults. Record total recurrent cycle and stage cost, full scratch/traffic, codegen resources and endpoint state errors. Only then decide what mechanism merits a task-level campaign.

The launch sequence intentionally reuses tests without pretending their names or historical credentials qualify a new route. The main missing work is a small, source-bound **state publication plus next-forward harness**, not another broad uninstrumented task batch.

# M1 C0 failure triage and smallest next action

2026-09-29. Source/evidence-only review; no implementation changes, author-code execution, GPU, network, or new experiment. **No newly identifiable harness repair is established. The frozen failures stand.** The strongest source-supported explanation is a mismatch between the executed compact arithmetic and the declared sequential comparator envelope. Exact causal attribution, especially the small Weaver replay failures, is not established by source inspection alone.

Paths below are relative to this paper directory. Adapter/comparator paths abbreviated `tools/…` are under `experiments/review-response-20260927/`; pinned author paths are under `experiments/author-code-20260926/`.

## Existing evidence narrows the problem

Reused the independent original and repaired result audits rather than re-auditing unchanged implementation. All **33** method-receipt source bindings still match current bytes. Independently authenticated and read the six repeat-0 output/state NPZ reductions from `p0/monitor/review-response-20260927/m1-q-c0-repaired-result-independent-review-20260928/reduction/` against the repaired numerical receipt. Each file contains `rms`, `max`, `finite`, `pass`, `fail`, `uncovered`; dimensions are `[48 instances,28 active nodes,48 heads]` or `[48,3 publications,48]`.

| Policy | First-verification failures /64,512 | Root-node failures /2,304 | State failures at publications 1 / 2 / 3, each /2,304 |
|---|---:|---:|---:|
| Weaver author-default | 53,745 | 1,457 | 74 / 484 / 596 |
| Weaver aligned-local | 57,707 | 1,569 | 12 / 143 / 233 |
| TreeWY author-default | 37,528 | 979 | 2,304 / 2,304 / 2,304 |

All cells are finite and covered. Repeat equality was established in the prior raw audit. Root failures occur **before any accepted replay or prior deferred commitment**: fixing replay alone cannot qualify any arm. These counts can be reproduced by loading `cells.<policy>.repeat0.output.npz` and evaluating `fail[:,0,:].sum()`, and `cells.<policy>.repeat0.state.npz` with `fail.sum(axis=(0,2))`. Node-axis zero is physical root zero in the saved active-node list.

The prior audits independently checked actual q/k/v/gate identities, initial-state equality, TreeWY physical↔DFS output mapping, leaf IDs, uninterrupted 1/6/11-update history, and final deferred flush. Current `tools/m1/m1_adapters_v3.py:68–86` still performs the same gather/inverse mapping; `m1_cycle_driver_v3.py:147–190` still observes Weaver after replay and TreeWY after its next call/final flush. `m1_q_c0_collect_v1_1.py:138–178` loads hash-bound pristine operands, dispatches the pinned settings, and detects CPU input mutation. No evidence supports changing row order, skipping root publication, reinitializing between publications, or removing the final flush.

Use the **policy-specific sealed reductions** for qualification. The collector's `diagnostics_first_verification.json` labels its separate diagnostic reference as `q1_2b_fixtures.c2_node_refs`; that common diagnostic is not a substitute for the policy-specific C2 references in the admitted reduction.

## Source-supported causes and unresolved parts

**Weaver verification, both variants — concrete arithmetic mismatch, not an identified normalization omission.** `tools/m1/m1_adapters_v2.py:241–299` faithfully normalizes once and passes `use_qk_l2norm_in_kernel=False`, `precision="tf32"`, `bf16_mode="none"`; the default serving sequence matches pinned `weaver/gdn_backend.py:724–757`. In `weaver/gdn_tree_triton.py`, TF32 applies to Gram products (203–204), inverse construction (235–248), the initial-state projection and solve (305–317), and readout (370–380). C1 quantizes only sequential state/key and state/query projections, then performs sequential FP32 residual/outer-product updates (`m1_c_baseline_v1_1.py:186–197,286–303,552–601`). The compact implementation has additional rounding sites, even at the root (for example `inv @ acc`), so C1 is not a hardware emulation of it. `bf16_mode="none"` does not disable TF32. These are actual source differences, but their individual contributions have not been isolated.

**Weaver default replay — BF16 preparation is deliberate and already represented by its reference.** The default helper stores normalized keys in BF16; its gating helper explicitly casts sigmoid beta through the input `b.dtype` before the FP32 stash (`m1-author-dependencies/…/fla/l2norm.py:150–192`, `fused_gdn_gating.py:39–41`). Verifier K0 instead keeps beta FP32 (`weaver/gdn_tree_triton.py:142–149`). Both surface-specific seams are already declared in `m1_c_baseline_v1_1.py:346–395`; removing the beta cast would create a different local method, not repair a missed C2 convention. GPU normalization/reduction rounding can still move a stored BF16 value relative to independently normalized C1/C2. The current captures do not establish that as the unique cause.

**Weaver aligned-local replay — small rounding-level mismatch remains, separate from TF32 verification.** The local helper uses Torch FP32 `sum`/`rsqrt` and `log1p(exp(x))` (`tools/m1/m1_adapters.py:107–116`); the pinned replay uses `tl.exp`, a `tl.sum` projection, and FP32 elementwise updates (`weaver/chunk_tree_verify.py:886–906`). C1 explicitly uses ascending-index sequential FP32 reduction and separately evaluated gate arithmetic. Aligned-state RMS errors span `6.51e-10`–`1.52e-7`; the earliest root-state failure is instance 6/head 12, RMS `2.274139531953917e-9`, maximum `2.977167742890252e-8`. No TF32 dot occurs in this replay body. Changing only verifier precision cannot fix this surface. Also, the verifier recomputes gates from raw a/b while replay consumes prepared stashes; “aligned-local” does not mean those two paths share every gate instruction. There is no demonstrated missed or duplicate update.

**TreeWY — strongest direct design/precision mismatch.** The actual tree wrapper selects the mask-taking vhoist kernel (`treewy/tree_wy_triton.py:751–876`), so its relevant body is lines 175–334. Its declared norm floor differs from the other methods but is correctly represented in C2. With `dot_bf16=True`, the commit computes a dot of **BF16-rounded weighted residual factors and BF16-rounded stashed keys** (264–280); output also rounds compact `Co`/`vt` factors (313–315). The frozen sequential C1 deliberately leaves its residual/outer-product update FP32 (`m1_c_baseline_v1_1.py:608–633`). This difference already exists for root-only publication and is consistent with every state cell failing. It does not prove TreeWY's algorithm is incorrect. Changing the norm to Lumo's epsilon would neither repair this mismatch nor preserve the author policy.

## Minimum next action

1. **Do not rerun the full unchanged collection or change the frozen rule.** There is no source-proven harness bug to patch now. Keep all original failures and the repaired result. The current bounded numerical paper statement needs no additional experiment merely to remain valid.
2. If the parent pursues causal diagnosis, first prepare one separately named, untimed diagnostic for the existing first-verification/root-publication witnesses, retaining original fixed32 dimensions and operands. Instance **0/head 0** fails the root output in all three arms; default replay has an instance **0/head 5** witness; aligned replay needs instance **6/head 12**. Thus only instances 0 and 6 are needed initially, not a new corpus or workload. Capture actual prepared q/k/g/beta and intermediate compact factors before changing arithmetic. The current argument digests authenticate values but are not all readable intermediate tensors.
3. The smallest source-native precision probes, **if subsequently authorized**, are Weaver's existing `precision="ieee"` option and TreeWY's existing `dot_bf16=False` option. Keep the same original references/bounds for a diagnostic re-score, disclose that these are changed configurations, and do not count them as repaired original author-default results. Weaver K0 remains `tf32x3` even with `precision="ieee"` (line 140), and replay remains separate; neither toggle guarantees qualification. For replay, compare actual prep stashes first, then the one-update reduction/rounding path. Do not infer causality just from a changed final error.
4. Stop at the first attributable boundary. A diagnosed implementation repair or new precision configuration requires a separately bound full frozen-population qualification before any timing. If these witnesses instead reproduce the declared arithmetic differences without a contract violation, retain the numerical failure and stop searching for a harness “fix.” No threshold relaxation, repeat expansion, task change, or timing through failed gates is justified.

## Source and evidence bindings

| File | SHA-256 |
|---|---|
| Repaired numerical `runs/m1-q-c0/reduction-repair-20260928T223230Z/RECEIPT.json` | `22b5a831277f91c5f1b15f7a86f9cfa5ba0f198a3f0d1dfadd71c2ddaa3edaec` |
| `m1/M1-NUMERICAL-CONTRACT.v4.2.json` | `b98c3af2fd1d7cccf492cc10040a5847b82aefdc940c7570cb777b805ee4ee80` |
| `tools/m1/m1_q_c0_collect_v1_1.py` | `736ee7b49be82ebed4cbb37e10002a766461ff73cfda4b183446059bb50b8a08` |
| `tools/m1/m1_adapters_v3.py` | `c5c3b3b328f8f67d04ddb4d0a421da478a2d9e2c38958ef5496919f63372aeda` |
| `tools/m1/m1_adapters_v2.py` | `a5414791a08f1f526b38f5f86c73cd6cee56c739542be820f785d8a7bd3ffcce` |
| `tools/m1/m1_adapters.py` | `375a177b1da203a15b8deecc21f971a87a328d037e2f52913c41bd40f1444066` |
| `tools/m1/m1_c_baseline_v1_1.py` | `00d1c8e5a593f2f1377e43aef2002f446a6ffb7c04ef0689b38cdbe163fd721b` |
| `tools/m1/m1_cycle_driver_v3.py` | `4ad560d96844ad4fd6abe7bfc484f01ac0fbb16925c217c27d46c64801a29d75` |
| `weaver/gdn_tree_triton.py` | `b151d4b2ade0451a896e99f83e1656bcddd3d8ad279eab399e994c604e28a1ff` |
| `weaver/chunk_tree_verify.py` | `1e141a8d52af973abb06f8ce99380e7636528bb9e917646f6e060923f1392eec` |
| `treewy/tree_wy_triton.py` | `c3dc8849249b3eef336a77aff50f2b7963d9cc2567a6e8e926257aac04a1f4f7` |

The numerical receipt binds all six reviewed NPZ files individually. The repaired audit remains `notes/review-response-20260927/m1-q-c0-repaired-result-review.md` (`e2d9f8d6ab787e1a517d12985a6ba9fe1d11443414a2d1e840543c930521fe2e`); its raw identity and boundary checks were reused, not represented as newly rerun here.

# Q1 root KV reblocking — bounded independent review

Disposition: **PASS for the corrected root-harness source and bounded CPU mapping/audit scope**. No remaining concrete KV mapping blocker found. The initial source-level stride-validation and audit-dependency omissions were corrected during review. No GPU, model/agent, container, launch, or remote operation occurred; the failed root run `080536` is unchanged and remains zero valid cases. RNG review is separate.

## Byte mapping and transaction

`q1_candidate_kv_reblocking_v1.py:16–47` keeps export units and physical allocation units separate. With physical stride `[1048576,2097152,1024,256,1]` (BF16 elements), logical native chunk `i` starts at token `64*i`, page-table entry `floor(64*i/1024)`, offset `(64*i)%1024`, and length `min(64,extent-64*i)`. Plane `p` maps to byte address `base + 2*(p*1048576 + physical_page*2097152 + offset*1024)`, with length `n*2048` bytes. The full tensor is interleaved, while `kvc[p,page,offset:offset+n]` is contiguous. Since 64 divides1024, no chunk crosses a physical page. No reshape, allocation-layout change, serving flag change, or numerical conversion is introduced by this mapping.

The final mapper requires the complete interleaved stride, positive unique in-range physical page IDs, sufficient page-table coverage, native export64, source head/dtype match, exact logical indices/tail length and byte count. Hooks v1.3 independently compare actual block-table units with `kvc.shape[2]` before planning (`_hydrate`, line302).

Hydration v3 is the accepted v2 transaction with the new planner/view dispatch only. `plan_all:87–95` computes every selected GDN and attention destination byte span and rejects any collision before writes. `apply_all:163` authenticates all source objects before mutation; guards and runner/preseed identity are captured before the copy loop. Copy attempts begin at 179. It rereads **every** destination, compares object hashes, recomputes native-order per-layer K/V digests and full state digest, checks every registered guard, and checks runner identities after completion. Later failures remain fail-stop; no rollback or retry is introduced.

The tail guard begins at `extent % 1024` in the last physical page, not at the native64 tail remainder. Existing neighbor guards preserve positive unselected neighboring physical KV pages; selected pages are excluded from neighbor guards and handled by materialized writes plus the final tail suffix. Reserved KV page0 is not guarded by that neighbor loop (hydration line112). Null-row guarding applies to GDN. More distant pages and unrelated/MTP tensors remain outside the declared bounded protection. This is not a whole-pool check.

## Export and independent audit

`KR.snapshot:57–79` records actual `kernel_block_size=1024`, shape, stride, storage bounds and physical page map, separately from `export_chunk_size=64`. Every exported record includes physical page and offset. It hashes K then V for each logical64 chunk, including the partial final chunk; the aggregate order matches reference `_attention_kv` at 406–435 and hydration `native_order_digest` at 145–155. Actual physical geometry is never relabeled native64.

The candidate state auditor requires physical1024/head4/dim256 and export64, exact current interleaved stride, storage bounds, physical-page uniqueness, exact page/offset mapping for each logical chunk, all raw lengths/hashes and aggregate digest equality. Candidate raw audit calls this consumer for O0-natural/O0/O1, while native evidence still uses the unchanged native auditor. The different schema is explicit (`candidate-hooks.v1.3`).

Initial source review found that the first draft checked only positive plane/page strides and bounds, permitting an overlapping physical layout to be described without refusal. The parent tightened both the live mapper and offline consumer to the observed full stride. The retained final controls reject overlapping K/V plane and adjacent-page strides. The first attempted counterexample script ran after that correction and stopped on the new refusal; it is retained without claiming a pre-repair executed false pass.

The new state-auditor dependency initially did not appear in the raw audit's source hashes. It is now included at raw audit line162; the wrapper also binds raw audit, state auditor and native corpus in its reviewed dependency map. The job binds hydration v3 and mapper hashes; hooks verify their actual imported bytes. Generated diagnostic v3.2.2 selects patcher v1.3 in exactly the five predecessor locations. Read-only generator rendering matches its saved bytes; production statements/settings are not changed by the selector delta.

## Evidence and limits

Independent configured-Python stdlib controls: **25** (11 PASS, 14 expected refusals). Exact mapper functions were AST-extracted, with tensor metadata represented by CPU-only stubs. All nine extents `1,63,64,65,1023,1024,1025,2049,13487` cover reversed page-table order, both planes, exact per-token byte-address coverage/disjointness and tail-guard bounds. Full synthetic 48-GDN/16-attention records exercise the actual raw state auditor and reject overlapping strides, wrong offsets/pages/units and duplicate pages. These tests do not claim to execute Torch copies.

Parent-retained remote **CPU-only** test log reports three methods passing in0.789s, including real interleaved Torch copies/re-export and native-order digest equality across those nine extents, full tail/unselected-page byte comparisons, missing-object refusal before mutation, and post-copy tail corruption fail-stop. I read and retained the log and reviewed its source; I did not independently rerun those Torch tests. Configured local Python lacks Torch.

The wrapper passes local Bash syntax checking. The generated launcher hits the local old-Bash parser's unsupported, unchanged `[[ -v ... ]]` syntax; this is not treated as a new launcher defect or target-host syntax certification. Source generator equality and successor Python AST parsing pass. Validation limitations and the superseded first control draft are retained.

This review closes only the prospective root instrumentation's reblocking dependency. A source-bound, observed candidate run and its raw audit are still required; no valid observation or experiment result is inferred from CPU preparation.

## Exact reviewed source hashes

- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_kv_reblocking_v1.py` — `f3a5957f2ea8f934818500d1a21e1ba087e496460575fa71c3750f52a8f08213`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_hydration_v3.py` — `c311af48af6e65316ea1da0c471f57eefd0e87422a8b030733ce3bfb6efd3cc2`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_hydration_v2.py` — `136e1cf36992958a9e88f52efd3c3d16d69af780481533cb201f7c3b006349cb`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_hydration.py` — `92bf5bf0ca1e1c4ab58ddc3f41effad2b669d3b701f271d7f4488b84f05c8e0f`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_hooks_v1_3.py` — `16d962e9398501a87cd10333a1c5deef6cd162728b93f757733ace0b2bd9397d`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_state_audit_v1.py` — `2eec7075c0df0ff4292e7e98ea3fcba948274d9037c25fca7d1777fb3022ebba`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_raw_audit_v1_1.py` — `9112c467b1b9c7aad1f36f86a63c2509b15f2a052f92a8a056e0cf4cc9e0a963`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_job_v1_3.py` — `ab4e77c53e9c49148b0b7084fb023f1cc09886896acdfbef2726ab59492d1526`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_candidate_driver_v1_2.py` — `9aacf34325f05f169cf2197f5125fa7a85dbed0f9e6bc216e33e5577258aa2dd`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_patch_candidate_v1_3.py` — `5b3f4f4d66bb70a0df098079efd75eb25f9f88f988f812be136650a5fa87c15c`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/run_q1_candidate_stage1_v2_8.sh` — `a00631977cd7ebc87883fc5030d68982f2f0e8d8d5efef5cd4419acd141e3e36`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_make_diag_launcher_v3_2_2.py` — `4eaf80b1cedee4661836a4724fe0e47921cfa4306f74b5b5e8efd06a473259aa`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/generated/fr14_leg3_launch_nomiddleware.q1diag.v3_2_2.sh` — `33c6eadf09c877260f0ad719128d5bd40271beb7b29d274edd3825b48c0bf716`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests/test_q1_candidate_kv_reblocking_v1.py` — `873c65aa9446051c2fb557e6774c589984f1cec7f7ef1b16a5746bcf1779e448`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/test_log.q1_candidate_kv_reblocking_v1.attempt1.txt` — `2ed1bc46d995202c46639c3f9d69cbca97cfeb9fb2e9c7782db844145c3a33f7`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_reference_hooks_v2_1.py` — `1ac939405de01a4c22e5d73decb9994cfb3a3214fd838ed2a734b39110b2b8cf`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_native_corpus_v1.py` — `d4502b7f47ab1ac6ed8cd295aa5565726fad486c4a1ff1bda6a13fe24e4ea770`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/identity/launcher-diagnostic.v3_2_2-from-v3_2.diff` — `f94991af6f1938ea5de27a89d3ea8fa6e2347d61021231224f401cbc3482fda7`

Byte-for-byte review sources and controls are in `p0/monitor/review-response-20260927/root-kv-reblocking-independent-20260929T0840Z/`, sealed by `MANIFEST.json`.

# Native MTP history v1: bounded repair closure

Disposition: **PASS for the submitted prospective source repairs and evidence connection.** This is not a launch approval, native API equivalence result, numerical qualification, or approval of the separately versioned history-v2/same-input successor. The earlier failure review and controls remain preserved.

## Exact implementation snapshot

| Source | SHA-256 |
| --- | --- |
| `q1_native_mtp_owner_v2.py` | `77edbafbd9ff34813d1a866c5ab8c8c7c620da0c3ce1e532d4b41fa1e61cde7e` |
| `q1_native_mtp_program_v2.py` | `d3c60ba6cebb3cdb4fe07e1185cbd6e291e4ca7be5998177624a3a6b50df729c` |
| `q1_native_mtp_history_v1.py` | `d531fea5376ec263217c45636fb1661bfaf000792d9a18db8512c407adfa2d13` |
| `q1_reference_hooks_v2_3_mtp.py` | `94234ad678f995ec68c29299494ba9421e775a367b8281db20440b710ead0f8f` |
| `q1_patch_native_mtp_worker_v2.py` | `cbd48dbf3846821c39a77f258f614109e6ac9c68861d676d3cd42300cb9564b5` |

Copies, dependency hashes, tests and supplied logs are in `p0/monitor/review-response-20260927/native-mtp-history-v1-repair-review/`. Active implementation bytes were rechecked against these hashes when sealing this report.

## Closed findings

1. **Every real step protects the materialized prefix mapping.** History lines 68–70 compare generation, group, block size and every previously materialized block column using `last_lease`; line 107 updates that lease only after the native first pass. Independent exact-source `pre`/`lease` controls now reject the original ordinary-prefix remap, a changed fully materialized block, and a changed second partially materialized block, before setting the current step. Unchanged partial history, appending a newly needed block, and changing only an unmaterialized prior column remain accepted. This preserves the intended allocation semantics without allowing hidden history relocation.
2. **The saved top-k is computed with each actual declared k.** History lines 90–93 call `torch.topk` on the original B1 native tensor for k=2, 3 and 32, with `dim=-1, sorted=True`, before CPU/f32 conversion. The source labels top32 diagnostic-only; smallest-ID greedy/ties remain a separate definition. An independent AST execution recorded all three exact calls and original tensor identity. Parent-supplied Torch history controls compare each saved list to the same exact-k operator on tied logits. Those are CPU controls, not proof of cross-device tie equivalence.
3. **A failed real metadata builder cannot leave captured metadata usable.** Owner lines 180–185 always clear common metadata, binding and armed scheduler while setting failed and clearing ready, even if abort itself detects a scheduler mismatch. The failstop decorator now also clears the armed scheduler. The actual emitted builder handler calls abort then rethrows, and ends only while still armed. Independent execution of that emitted handler with actual owner methods preserves the successful control, propagates the injected post-capture failure while clearing all relevant state, and also clears state on an abort-owner mismatch. This closes the prior state-machine defect; the original report did not demonstrate an actual numeric false pass.

The patcher still applies five reference and seven ownership anchors to stock runner SHA `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0`. Independent source generation and parsing reproduce emitted SHA `57d370e249d5041eb6ec752175a7cc31426af87da1a5df34de5297a975870aae`.

## Added evidence and cache-salt path

Hooks lines 61–65 collect separately labeled MTP KV after native first passes at the prefix, before-Z and after-Z extents. The call uses the authenticated owner's singleton MTP layer/group and actual request table. The accepted `_attention_kv` implementation (base hook lines 390–439) checks declared dtype/head geometry, logical extent, finite materialized K/V and bounded physical blocks, then stores raw full-block/tail objects and their logical digest. An empty per-call `blocks_seen` prevents stale cross-boundary reuse. It does not mutate cache state. These are after-first-pass MTP boundaries, not a relabeling of target O0/O1/O2. The parent connected hook control verifies the three extents plus partial-prefill input, teacher-forced input, deferred checkpoint and terminal raw-O2 greedy versus server stop token; its KV writer is explicitly a fixture stub. No real MTP GPU KV capture has run in this review.

The newly retained exact-image renderer source SHA `862254b0b801b09c30a3073eab9bdd2f097a75b4b894d993cefd34ecb4aba0c5` fills the previously missing renderer middle edge: `render_cmpl_async` lines 914–940 applies prompt extras after tokenization; `_apply_prompt_extras` lines 569–579 updates the selected prompt; `_process_tokens_async` lines 746–770 carries its nonempty cache salt into the engine input; `process_for_engine_async` lines 868–888 returns that input. Together with previously checked completion request/preprocessing, engine input processor and first-block hashing sources, this supports the prospective per-observation salt path for this decoder-only token request. Independent exact-method controls retain two distinct nonempty salts and unchanged token lists, using a decoder-only target-selector stub; the imported `extract_target_prompt` implementation was not independently re-executed. Salt lineage does not prove runtime cache misses: initial `num_computed_tokens == 0` and contiguous complete history remain mandatory.

## Validation and limits

Independent standard-library controls: **11 named cases passed** (six prefix-map cases, three actual handler cases, exact-k dispatch, and renderer preservation with two salts). They execute extracted actual methods and the actual emitted error handler, without Torch, a model, Docker or GPU. `INDEPENDENT-CONTROLS.json` and `PATCH-CONTROL.json` preserve results.

Parent-performed supplied CPU logs: history attempt3 **6 methods PASS**, owner attempt2 **4 methods PASS**, hook attempt3 **1 connected method PASS**. Hook attempt2 remains preserved: its synthetic runner lacked `_q1_native_mtp_owner` after the new capture call was added. Parent corrected the fixture's owner/layer connection; production hook bytes did not change. These supplied Torch runs were inspected, not independently rerun here.

Program-v2 math and continuation extraction are unchanged. The separate same-input actual-native-API equivalence implementation, future raw record auditor/driver, launch dependency binding and runtime results remain outside this closure. No experiment or qualification count advances.

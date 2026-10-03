# Target-only versus joint native common-O0 route diagnostic

Bounded source/archived-metadata inspection, 2026-09-29. No inference, model/container/cache operation, unit tests, numerical criterion change, or new qualification result. The parent and another reviewer own the adverse raw-result/first-divergence analysis. This note does not independently re-reduce that result.

**No concrete harness defect was established in this route comparison.** The successful target-only result is not the same-input counterfactual for the joint result: all three selected target O0 digests changed. Preserve both outcomes; do not infer that adding MTP alone caused the difference or that unmaterialized padding caused it.

## Directly verified route differences

The frozen target-only plan is `fullmodel/native-common-o0-v2/CORPUS.json`; the joint plan is `fullmodel/native-joint-common-o0-v1/CORPUS.json`. Exact four archived `engine-config.json`, `engine_inspect.json`, and boot attestations were read, without tensor payload reads, into `p0/monitor/review-response-20260927/native-target-vs-joint-route-diagnostic/remote-metadata.json`.

The serve argument vectors match: same immutable image/model, B1, utilization0.6, seed0, max length131072, FA2, Triton GDN prefill, aligned nonpacked recurrence, float32SSM, mamba/logicalblock1024, batch-token limit4096, long-prefill threshold1024, APC and chunked prefill. After excluding run-specific `Q1_REF_*` fields, all four environments have identical canonical SHA `01ba850be8e5d1e81c2ee24b335545fc96d0f5d6ae3c82182580cc87ecec543f`. Boot attestations retain spec-off target scheduling, kernel block sizes[1024,1024,1024,64],48GDN/16targetattention. Joint group3 has17attention layers instead of16 because the independent MTP layer is registered there. There is no serving-parameter drift in this comparison.

The new worker does add real execution and allocations. `q1_patch_native_mtp_worker_v3.py:18–24,29–37` attaches/loads the MTP model, binds its allocated cache, profiles its dummy path, clones real common metadata, and captures target hidden rows. `q1_native_mtp_owner_v2.py:61–73` uses a copied outer config with a separate num_speculative_tokens2 configuration while retaining the target model/compilation registry; `:120–142` requires17backing tensors, exclusiveMTP storage and no occupied-span overlap with target caches. `:162–168` invokes native MTP dummy profiling. This changes allocation and kernel invocation history even though the target serve command is identical. It does not constitute evidence that a particular kernel tactic or scratch buffer caused divergence; those choices are not captured by the existing route receipts.

## O0 source confound is explicit

The target-only source mapping selects original aligned A/r0 independently for all84cases. The joint mapping uses three predetermined root-only source observations, one per prefix, for those same84paths. Prefix lengths/token hashes and roots match; source target values do not. Root-only target digests are:

| Prefix | Target-only target O0 | Joint target O0 |
|---|---|---|
|13487|6a933db92eb7e630a06f35fe04c5a2657e79a7854f81f84da99c3d8426b43625|a88b7b1aaa1cd777c620724a8b3cca8468458c1c5881f35b7dbc1f664cbd0c2b|
|28941|9cb94cfcf22bbf83f26352b0794ec5993af057797ba070fa650a77c5ae2f6c74|d1c1d66e010281ba537923e327dc6a82a28011aea258a315663e469895dd782b|
|60083|e81302d06331ca878e81afb3d1f4b819443cac6eb0a36f1b3a7e1407d4e14326|c9a7007e4e28818dadadfb124aec23fb1a370eed7ffaa855b8fdebbb497e805e|

These are direct manifest values, not inferred from sampled outputs. Internal joint O0 equality across A/B remains intact and its adverse continuation result remains applicable to that chosen joint source.

## Import/export and unmaterialized bytes

Both routes use the same frozen native target mapper `q1_native_common_o0_v2.py` and H4 primitive `q1_candidate_hydration_v4.py` (SHA b35f2e5d49e0f49827103db3b0c3ddecdec6ce38189883a7c4a4c13c79350fac). The joint path adds the MTP cache to one union transaction and verifies independent post-import target/MTP readbacks before history adoption (`q1_reference_hooks_joint_common_o0_v1.py:54–76`). No source hidden/scratch tensors are imported.

AST comparison confirms `_attention_kv`, base `on_pre_forward`, and `on_logits` are identical between target hooks v2.1 and the joint route's v2.2 target base. `_snapshot_state` differs only by using the already authenticated owner.target registry instead of collecting every global suffix; its remaining export operations are identical.

Exports bind all selected GDN conv/SSM values and materialized targetKV, including only the valid tail (`q1_reference_hooks_v2_1.py:386–432,435–475`). At prefix60083, that is51valid slots of a64-slot final block;13future slots are not imported or part of logicalO0. H4 explicitly guards the complementary tail bytes against import-time changes, rather than zeroing them (`q1_candidate_hydration_v4.py:109–174,206–228`). This is shared by both routes. It means full allocation bytes are not asserted equal; it is **not evidence they are incorrectly read**. The pinned FA2 backend takes the actual sequence length as `seqused_k` and passes it with blocktable and causal flag (`identity/native_source/vllm__v1__attention__backends__flash_attn.py:755–807`). No out-of-range tail read was demonstrated here.

The newly interleaved MTP path does not intentionally mutate target metadata: owner `capture_common` clones every tensor field (`q1_native_mtp_owner_v2.py:144–155`), and history clones prepared token/position/hidden tensors (`q1_native_mtp_history_v2.py:61–86`). Its checkpoint followup restores the exact two MTP scratch rows after the unchanged native-propose comparison (`:122–138`; `q1_native_mtp_same_input_v1.py:26–60`). These source protections rule out claiming a simple intended shared-metadata/cache write, but do not replace a live target-before/after-MTP mutation witness.

## Immediate diagnostic boundary

The other reviewer's reported representative finding is identical first-post-import target hidden/fullMTPscores, followed by differing next-row hidden. There is no immediately-after-first-row KV/state snapshot, so identical final hidden must not be promoted to identical intermediate targetKV/GDN or MTPKV. The existing first-differing final published layer is a localization clue, not necessarily the first arithmetic divergence.

The shortest justified next localization, if the parent chooses connected validation, is the already identified long-prefix failing path with per-layer state/KV/metadata identity around the first target row and its intervening MTP first pass. Check whether target cache bytes or target metadata change during that MTP-only interval before hypothesizing numerical tactics. No broad rerun, padding zeroing, criterion relaxation, or change of O0 source is justified by this source review alone.

Exact inspected source hashes are in `source-identities.json` beside the archived-metadata capture. Historical successful and adverse results remain unchanged.

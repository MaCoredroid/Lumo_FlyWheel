# FR10 Status

Updated: 2026-06-03 18:45 UTC

## Current Phase

- P1: CPU GDN tree-algebra parity proof passed. CPU recurrent rule is only the correctness oracle/gate, never the final kernel deliverable.
- P0: canonical cu130 `spines=1` baseline server booted on digest-pinned nightly with `kv_cache_dtype=auto`, `VLLM_BATCH_INVARIANT=1`, `--attention-backend FLASH_ATTN`, `--gdn-prefill-backend triton`, and working `POST /reset_prefix_cache`; greedy and temp=0.6 B4 reference streams captured.
- P2 active: GPU Triton tree kernel only, now validated against the digest-pinned cu130 GB10 production stack for Gate D/cost; host `.venv` remains CPU-only.
- Git workflow: active branch is `fr10-gdn-tree-kernel`; do not commit to main. Going forward, commit and push after every meaningful step. Commit messages must end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Read-In

- Read `FR10_KICKOFF.md` in full.
- Read `docs/reports/auto_research/fr10-gdn-stree-verifier-latest-stack-spec-20260603.md` in full.
- Read `docs/reports/auto_research/fr9-b4-temp06-options-closeout-20260601.md` in full.
- Read `docs/reports/auto_research/fr10-gdn-tree-algebra-grounding-20260603.md` in full after researcher steering.
- Future Gate C note received: `docs/reports/auto_research/fr10-lossless-mtp-tree-definitions-20260603.md` defines L1 verifier-state parity, L2 valid sequence-level tree rejection sampler with descendant rejection and no max/longest-accepted selector, and L3 greedy byte-exact via identical kernel. Not read yet because current work remains Phase 2 kernel.
- Read `docs/reports/auto_research/fr10-native-kernel-dtype-contract-20260603.md` in full.
- Stack unblock note recorded: official vLLM DGX Spark blog dated 2026-06-01 names `vllm/vllm-openai:cu130-nightly` as the CUDA-13 Spark compatibility track and says nightly tags must be pinned by digest after validation. Runtime flags in the blog match our target envelope: `--max-model-len 131072`, `--gpu-memory-utilization 0.85`, `--max-num-seqs 4`, CUDA graphs on by default.

## Commands Run

- `sed -n '1,240p' FR10_KICKOFF.md`
- `sed -n '1,220p' docs/reports/auto_research/fr10-gdn-stree-verifier-latest-stack-spec-20260603.md`
- `sed -n '221,415p' docs/reports/auto_research/fr10-gdn-stree-verifier-latest-stack-spec-20260603.md`
- `sed -n '1,220p' docs/reports/auto_research/fr9-b4-temp06-options-closeout-20260601.md`
- `sed -n '221,443p' docs/reports/auto_research/fr9-b4-temp06-options-closeout-20260601.md`
- `git status --short --branch`
- `ls -la`
- `rg --files tests scripts docs/reports/auto_research | rg 'gdn|selector|fr10|baseline|artifact|request_metrics|swe'`
- `sed -n '1,260p' docs/reports/auto_research/fr10-gdn-tree-algebra-grounding-20260603.md`

## Files Added / In Progress

- `scripts/fr10_p0_audit_baseline_artifacts.py`
- `scripts/fr10_gdn_tree_algebra_reference.py`
- `tests/test_fr10_gdn_tree_algebra.py`
- `output/fr10_p0_baseline_audit_20260603.json`
- `docs/reports/auto_research/fr10-gdn-tree-algebra-proof-20260603.md` (writing now)
- `output/fr10_phase2_container_stack_probe_20260603.json`
- `output/fr10_phase2_gpu_stack_probe_20260603.json`
- `scripts/fr10_phase2_triton_tree_gdn_microbench.py`
- `output/fr10_phase2_triton_tree_gdn_microbench_cpu_oracle_20260603.json`
- `output/fr10_phase2_branch_depth_cost_20260603.json`
- `output/fr10_phase2_native_gpu_single_spine_20260603.json`
- `output/fr10_phase2_native_gpu_compare_20260603.json`
- `docs/reports/auto_research/fr10-gdn-tree-kernel-microbench-20260603.md`
- `output/fr10_phase2_active_backend_probe_20260603.json`
- `output/fr10_phase2_backend_package_probe_20260603.json`
- `output/fr10_phase2_active_backend_deduction_20260603.json`
- `output/fr10_phase2_single_spine_null_oracles_smoke_20260603.json`
- `output/fr10_phase2_broken_audit_image_ldd_20260603.txt`
- `output/fr10_phase2_native_fla_dtype_smoke_20260603.json`
- `output/fr10_cu130_stack_probe_clean_20260603.json`
- `output/fr10_cu130_gdn_linear_attn_clean_20260603.py`
- `output/fr10_cu130_gdn_backend_context_construct_clean_20260603.json`
- `output/fr10_cu130_gate_d_production_gdn_single_spine_clean_20260603.json`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_boot_kvauto.log`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_models.json`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_container_inspect.txt`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_version.json`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_greedy_tokens.json`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_greedy_b1_b4_compare.json`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_temp06_b4_samples.json`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_temp06_logprobs.json`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_metrics_after_streams.prom`
- `output/fr10_p0_cu130_boot/fr10_cu130_p0_s1_baseline_summary.json`
- `docs/reports/auto_research/fr10-p0-cu130-batchinv-baseline-20260603.md`
- `output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_boot_batchinv.log`
- `output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_batchinv_greedy_tokens.json`
- `output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_batchinv_greedy_b1_b4_compare.json`
- `output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_batchinv_temp06_b4_samples.json`
- `output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_batchinv_temp06_logprobs.json`
- `output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_metrics_after_batchinv_streams.prom`
- `output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_batchinv_steptrace_window.jsonl`
- `output/fr10_cu130_gate_d_production_gdn_single_spine_batchinv_followup_clean_20260603.json`
- `scripts/fr10_canonical_state_commit_probe.py`
- `output/fr10_canonical_state_commit_probe_20260603.json`
- `scripts/fr10_real_dims_tree_vs_fla_cost.py`
- `output/fr10_real_dims_tree_vs_fla_cost_20260603.json`
- `scripts/fr10_tree_kernel_stage_profile.py`
- `output/fr10_tree_kernel_stage_profile_6n_20260603.json`
- `output/fr10_tree_kernel_stage_profile_8n_20260603.json`
- `output/fr10_tree_kernel_stage_profile_14n_20260603.json`
- `scripts/fr10_tiny_tree_acceptance_bound.py`
- `output/fr10_tiny_tree_acceptance_bound_20260603.json`

## Passed

- Read-in complete, including researcher grounding.
- Qwen3.6 config found at `/models/qwen3.6-27b-fp8/config.json`: 64 layers, 48 `linear_attention` GDN layers, `linear_num_key_heads=16`, `linear_num_value_heads=48`, `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`, `mamba_ssm_dtype=float32`.
- Tree-ancestry masked chunk solver smoke passed: 8-node synthetic tree had max state delta 0.0 and max logit delta 0.0 versus serial vLLM CPU recurrent oracle.
- P1 pytest passed after red-team greedy-gate correction: `.venv/bin/pytest -q tests/test_fr10_gdn_tree_algebra.py` -> `15 passed in 3.64s`.
- P1 negative controls covered: linear-mask leak fails parity; shared mutable parent state fails parity; greedy gate asserts per-node argmax identity plus `logit_delta <= 2e-5` with 256-wide synthetic logits and diagnostic margin only; longest-accepted hidden winner fails Gate C distribution shape.
- Red-team independently verified P1: packed-vs-serial fp32 parity `2.98e-8`; linear-mask leak diverges `0.0173` (~580000x tol) and shared-parent contamination `5.7e-4` (~28x tol), both non-vacuous and fail loudly.
- P0 audit found accepted baseline has sampled outcomes, per-event accept counters, engine-step latency, and nonzero per-task request metrics. Missing for full FR10 P0 without targeted follow-up: greedy token streams, CUDA graph/capture status, kernel-level Nsight traces, exact stack version record.
- Phase 2 container image exists locally: `lumo-vllm-audit:v0.22.0-cu129-min` (`sha256:af07ec6...`).
- Stack probe without GPU attachment: Python 3.12.3, PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA toolkit `13.1`, Triton `3.6.0`, vLLM `0.22.0`, FlashAttention `2.7.4.post1`; `cuda_available=false`, `nvidia-smi` unavailable because the container was not attached to the NVIDIA runtime.
- Phase 2 source map inspected: `chunk.py` wires `chunk_local_cumsum -> chunk_scaled_dot_kkt_fwd -> solve_tril -> recompute_w_u_fwd -> chunk_gated_delta_rule_fwd_h -> chunk_fwd_o`; KKT currently uses plain linear `o_t[i] > o_t[j]`; cumsum is linear; `chunk_delta_h.py` carries one chunk state and uses `use_cuda_graph` autotune setting.
- Phase 2 initial Triton microbench: 2-node shape passed CUDA graph replay on `NVIDIA GB10`: max output delta `4.8e-07`, max state delta `1.1e-05`, eager `13.4 us`, graph `12.5 us`.
- Phase 2 GPU stack with `--gpus all`: driver `590.48.01`, device `NVIDIA GB10`, Python 3.12.3, PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, torch CUDA `13.1`, CUDA toolkit `13.1.115`, Triton `3.6.0`, vLLM `0.22.0`, FlashAttention `2.7.4.post1`.
- Phase 2 all public node families passed GPU Triton tree kernel vs CPU serial oracle with CUDA graph replay enabled and bit-exact replay-vs-eager enforced. Worst CPU-oracle deltas: output `1.383e-06`, state `2.168e-05` on 14-node padded-to-16 shape.
- Phase 2 fixed-base branch-depth marginal table produced (`output/fr10_phase2_branch_depth_cost_20260603.json`): one extra branch row on the same `5->6 padded 8->8` base costs `11.90 us`, `6.29 us`, `9.23 us` marginal graph time at depths 0/1/2 in this initial unoptimized microbench; all rows graph-captured, graph-bit-exact, and CPU-oracle checked.
- Phase 2 native-GPU single-spine diagnostic table produced (`output/fr10_phase2_native_gpu_single_spine_20260603.json`): bf16 inputs + fp32 recurrent state, source-loaded native FLA chunk vs tree kernel. Full-spine output deltas `5.65e-05` to `9.50e-05`; final-state deltas `6.69e-04` to `9.43e-04`. Treat this as reduction-order drift evidence, not a final Gate D pass. All rows graph-bit-exact.
- Red-team graph/reporting fixes landed: graph replay is asserted with `torch.equal(replay, eager)` for out/state, `_tree_gdn_kernel` has no `@triton.autotune`, and oracle deltas now emit JSON `null` when the oracle did not run instead of misleading `0.0`. Smoke artifact: `output/fr10_phase2_single_spine_null_oracles_smoke_20260603.json`; full native single-spine artifact regenerated with null CPU oracle fields at `output/fr10_phase2_native_gpu_single_spine_20260603.json`.
- Active backend probe status: direct `ChunkGatedDeltaRule()` instantiation in `lumo-vllm-audit:v0.22.0-cu129-min` is blocked before constructor by installed `vllm._C` requiring missing `libcudart.so.12` while the image exposes CUDA 13.1 (`output/fr10_phase2_active_backend_probe_20260603.json`). Source-level deduction for this broken audit image is `triton` / native FLA only because sampled bundles have empty `kernel_selection`, no `additional_config.gdn_prefill_backend`, head dim 128 on GB10/CUDA13, and the package probe found no `flashinfer` or `nvidia-cutlass-dsl-libs-cu13` in the audit image (`output/fr10_phase2_active_backend_deduction_20260603.json`). Red-team infra finding accepted: this is an image artifact, not confirmed production backend identity.
- Sanity checks passed: `py_compile` for all FR10 scripts, JSON validation for Phase 2 stack/microbench/cost artifacts, and `.venv/bin/pytest -q tests/test_fr10_gdn_tree_algebra.py` -> `15 passed in 4.01s`.
- Latest validation after dtype/backend status updates: `python3 -m py_compile scripts/fr10_phase2_triton_tree_gdn_microbench.py scripts/fr10_gdn_tree_algebra_reference.py scripts/fr10_p0_audit_baseline_artifacts.py`; JSON load check for active-backend and single-spine artifacts; `.venv/bin/pytest -q tests/test_fr10_gdn_tree_algebra.py` -> `15 passed in 3.72s`.
- Native FLA dtype contract smoke: updated the Triton microbench so bf16 activation inputs store bf16 verifier output and fp32 recurrent state, matching `forward_native` / `chunk_gated_delta_rule` return dtypes. One-shape GPU smoke passed with `output_dtype=bfloat16`, `state_dtype=float32`, `graph_bit_exact=true`; artifact `output/fr10_phase2_native_fla_dtype_smoke_20260603.json`.
- STACK RESOLVED: `vllm/vllm-openai:cu130-nightly` is local and working on GB10. Registry digest: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`; local image ID: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`; created `2026-04-23T05:23:07.199020192Z`. Stack probe: vLLM `0.19.2rc1.dev134+gfe9c3d6c5`, torch `2.11.0+cu130`, CUDA `13.0`, Triton `3.6.0`, FlashInfer `0.6.8.post1`, `vllm._C` imports, `flashinfer.gdn_prefill.chunk_gated_delta_rule` exists, device `NVIDIA GB10`.
- cu130 actual GDN source re-anchored: production layer is `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py`, captured clean at `output/fr10_cu130_gdn_linear_attn_clean_20260603.py`. This is not the non-standard `/tmp/vllm-0.22-src/.../mamba/gdn/qwen_gdn_linear_attn.py` path.
- GB10 production GDN backend: Triton/FLA. cu130 `gdn_linear_attn.py` sets `supports_flashinfer = current_platform.is_cuda() and current_platform.is_device_capability(90)`. GB10 reports capability `(12, 1)`, `is_device_capability(90)=false`, `is_device_capability_family(120)=true`; therefore default `auto`, explicit `flashinfer`, and explicit `triton` all construct `ChunkGatedDeltaRule` with `_forward_method=forward_native`. FlashInfer GDN exists in the image but is Hopper-only for this layer and is not used on GB10.
- Exact-production Gate D single-spine table completed inside cu130-nightly with bf16 inputs, fp32 initial state, raw `g`, raw q/k passed to cu130 `ChunkGatedDeltaRule(... use_qk_l2norm_in_kernel=True)`, tree side normalized with the same vLLM `l2norm_fwd`, and production scale `1/sqrt(128)`. All public single-spine rows `{2,3,6,8,14}` used `production_gdn_forward_method=forward_native`, graph replay was bit-exact, max output delta was `6.103515625e-05`, and max final-state delta was `9.473264217376709e-04`. Artifact: `output/fr10_cu130_gate_d_production_gdn_single_spine_clean_20260603.json`.
- P0 cu130 baseline booted first without batch-invariant: first attempt with `--kv-cache-dtype fp8_e5m2` failed after memory cleanup with `ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints.` Root cause accepted from red-team/user: newer cu130 vLLM hard-errors where old vLLM 0.19 effectively fell back to `auto`; therefore `kv_cache_dtype=auto` is the lossless-equivalent E3/E5 behavior, not a deviation.
- Reset route root cause: cu130 source has `POST /reset_prefix_cache` in `vllm.entrypoints.serve.cache.api_router`, but `attach_router()` only mounts it when `VLLM_SERVER_DEV_MODE=1`. Without dev mode the API port `9950` returned 404. With `VLLM_SERVER_DEV_MODE=1`, `POST /reset_prefix_cache?reset_running_requests=false&reset_external=false` returns `200`.
- Batch-invariant root cause/fix: `VLLM_BATCH_INVARIANT=1` alone failed because cu130 resolves `attention_config.backend=None` too early. Relaunching with explicit `--attention-backend FLASH_ATTN` satisfied the batch-invariant guard while keeping GDN backend pinned to Triton/FLA. One retry failed from transient unreleased memory and was cleared by removing the exited container plus dropping page cache.
- P0 canonical cu130 launch now uses digest-pinned image `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776` (local ID `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`), `VLLM_BATCH_INVARIANT=1`, `VLLM_SERVER_DEV_MODE=1`, `--attention-backend FLASH_ATTN`, `--gdn-prefill-backend triton`, `--max-num-seqs 4`, `--gpu-memory-utilization 0.85`, `--max-model-len 131072`, MTP5 speculative config, and omitted `--kv-cache-dtype` so engine reports `kv_cache_dtype=auto`.
- P0 canonical boot evidence: health reached `200` at 17:34:14 UTC. Boot log confirms `Using Triton/FLA GDN prefill kernel`, `Using AttentionBackendEnum.FLASH_ATTN backend`, `kv_cache_dtype=auto`, CUDA graph capture completed for mixed prefill-decode `PIECEWISE=7` and decode `FULL=4`, graph pool memory `0.54 GiB`, GPU KV cache size `228,480 tokens`, and server version `0.19.2rc1.dev134+gfe9c3d6c5`.
- P0 canonical reference streams captured against live `http://127.0.0.1:9950` with reset working (`reset_prefix_cache_error=null` in all probes): greedy B1/B4 token artifact has 16 records and B1-vs-B4 exact match over 8 prompts; temp=0.6 B4 artifact has 64 samples with `batch_size=4`, `max_tokens=32`; temp=0.6 top-20 first-token logprob artifact has 16 records. Hashes: greedy `b8b1ec327f60e34073fcedf54c8dad402bee47264f650888f3e982176c2e9794`; B1/B4 compare `ebc6a1599ef7f27cf62db5243b00ee66ebfc0d9eeb233b4bdfd1dd8c6ec495c8`; temp0.6 B4 `7d5f0ab0f53b6fa7adab7bf650264d717b16bbb0ef2db39a8059a80fd521f113`; temp0.6 logprobs `06d80a8fe814154de0bd13c128cabeee363cc404ca0d1ab016049a9f33b73324`.
- P0 canonical token prefix for greedy prompt 0: `[271, 248068, 271, 248069, 271, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 13, 248044]`. This is the Gate B target for later tree-kernel greedy decode.
- P0 post-stream metrics recorded `spec_decode_num_drafts_total=398`, `spec_decode_num_draft_tokens_total=1990`, `spec_decode_num_accepted_tokens_total=1206`, positions `{0:273,1:266,2:253,3:238,4:176}`. Bounded live steptrace window around B4 temp=0.6 load: 30 rows over `44.53515648841858 s`; deltas `gen=306`, `prompt=192`, `iter_sum=498`, `iter_cnt=37`, `acc=217`, `draft=375`, `drafts=75`, `dec_sum=16.833757460815832`, `pre_sum=5.048701603198424`; mean step wall time `1.203652878065367 s`, tokens/step `13.45945945945946`, accepted/draft-token `0.5786666666666667`.
- P0 side-channel caveat: the unmodified cu130 OpenAI server exposes aggregate spec counters via `/metrics`. The old `/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl` file was stale during this run and is not a live P0 source for this stock container. True per-event accept counters still require the instrumented side-channel in a patched serving stack.
- RED-TEAM independently verified P0 after commit: re-ran the greedy probe against the live cu130 server and reproduced all 16 greedy token-id streams byte-exact versus the committed baseline (`0/16` mismatches). P0 is done; this is the stable Gate B target.
- Exact-production Gate D rerun completed inside the same digest-pinned cu130-nightly stack: command used `--capture --production-gdn --production-scale --input-dtype bf16 --single-spine-table`. All rows `{2,3,6,8,14}` resolved `production_gdn_forward_method=forward_native`, used bf16 inputs/fp32 initial state/raw `g`/`use_qk_l2norm_in_kernel=True`, and graph replay was bit-exact. Production max output delta was `6.103515625e-05` for all rows; max final-state delta was `0.0009473264217376709`; per-row final-state deltas `{2:0.0008978471159934998, 3:0.000874541699886322, 6:0.000655151903629303, 8:0.0009473264217376709, 14:0.0007828027009963989}`. Tree graph times `{2:12.556us, 3:43.164us, 6:308.687us, 8:352.262us, 14:1041.471us}`. Artifact: `output/fr10_cu130_gate_d_production_gdn_single_spine_batchinv_followup_clean_20260603.json`.
- Exact-production Gate D interpretation: single-spine tree mask equals the linear causal mask, so this proves apples-to-apples production algebra within bf16 roundoff for logits/output. The remaining `<1e-3` recurrent-state drift is reduction-order drift and confirms canonical native state commit is mandatory before any lossless claim.
- cu130 native decode commit primitive located: `vllm/model_executor/layers/mamba/gdn_linear_attn.py` imports and calls `fused_recurrent_gated_delta_rule_packed_decode` from `vllm/model_executor/layers/fla/ops/fused_recurrent.py`. The production decode call passes `mixed_qkv`, `a`, `b`, `A_log`, `dt_bias`, `scale=self.head_k_dim**-0.5`, `initial_state=ssm_state`, `out=core_attn_out[:num_actual_tokens].unsqueeze(1)`, `ssm_state_indices`, and `use_qk_l2norm_in_kernel=True`.
- Canonical-state-commit probe implemented in `scripts/fr10_canonical_state_commit_probe.py` and run inside cu130-nightly. The probe constructs production-shaped bf16 packed `mixed_qkv`, bf16 `a/b`, fp32 `A_log/dt_bias`, fp32 state bank, valid `ssm_state_indices=1`, and replays a 5-token accepted path through `fused_recurrent_gated_delta_rule_packed_decode`. Results: packed replay output bit-exact `true`, packed replay state bit-exact `true`, replay max output/state deltas `0.0/0.0`, one-token CUDA graph replay bit-exact `true`, packed decode vs sequence recurrent diagnostic max output/state deltas `0.0/2.9802322387695312e-08`. This closes the state-commit design point: tree verifier logits drive accept/reject, tree state is scratch, accepted path state is committed only by native packed decode.
- Canonical-state-commit bandwidth framing: per accepted token native commit reads `3,145,728` state bytes and writes `3,145,728` state bytes (`48*128*128*4` each). A 5-token accepted path reads `15,728,640` bytes and writes `15,728,640` bytes. This commit cost is the correctness-preserving price for byte-exact native recurrent state after tree verification.
- RED-TEAM independently verified the canonical state commit probe inside cu130 on GB10 and reproduced the exact committed numbers: packed-vs-sequence output `0.0`, state `2.9802322e-08`, packed replay bit-exact, and one-token graph replay bit-exact.
- Component proof status: P1 proves tree ancestry algebra vs serial oracle (`~3e-8`), Gate D proves single-spine tree verify matches cu130 production FLA `forward_native` output to one bf16 quantum (`6.103515625e-05`) with `<9.48e-4` scratch-state drift, and canonical commit proves accepted-path state can be persisted through native packed decode to fp32 roundoff / bit-exact replay. This proves the losslessness architecture components, not the integrated serving loop.
- Remaining decisive losslessness gate: END-TO-END Gate B is not yet proven. We have not integrated the tree kernel into speculative decode and shown its greedy output stream equals the canonical P0 baseline token-for-token (`fr10_cu130_p0_s1_batchinv_greedy_tokens.json`, sha256 `b8b1ec327f60e34073fcedf54c8dad402bee47264f650888f3e982176c2e9794`). Do not claim full losslessness until this live integrated Gate B passes.
- Route A before Phase 4: build an offline end-to-end check over the P0 baseline prompts by capturing native per-layer GDN inputs/states for accepted MTP draft positions, running tree-kernel verify plus canonical native commit on those exact tensors/sequences, and confirming accepted greedy tokens plus committed states reproduce native. Route B after A: wire Phase 4 at `mamba/gdn_linear_attn.py` plus the MTP tree draft, rerun the live greedy probe, and require token-for-token equality with P0 baseline sha `b8b1ec32...`.
- COST-GATE active: real-dims speed is a red flag. cu130 FLA chunk is flat at about `135us` for `2..14` tokens. The standalone tree kernel beats FLA only for tiny trees (`2 nodes 12.325us`, `3 nodes 45.339us`) and is not competitive for larger public trees (`6 nodes 306.008us`, `8 nodes 340.857us`, `14 nodes 996.084us`). Current dense masked solve scales with padded node count (`N_PAD^2`) and cannot be treated as a viable speed path until profiling shows tree sparsity can cut it toward `O(N*depth)`.
- COST-GATE correction: fixed-base marginal rows `5->6 padded 8->8` produced negative medians (`-13.939us`, `-13.005us`, `-11.282us`) only because base and extended trees run in the same padded bucket and the delta is timing/codegen noise. Do not report marginal leaf cost as free. Reliable per-node bandwidth framing remains `3,145,728` fp32 state bytes read + `3,145,728` written per Qwen3.6 GDN verifier node.
- COST-GATE profile completed for `6/8/14` nodes. Graph medians: full `{6:303.243us, 8:334.785us, 14:995.777us}`; dense triangular solve variants `{6:160.921us, 8:160.861us, 14:636.507us}`; state/output-only variants `{6:221.282us, 8:285.975us, 14:352.529us}`. Ancestor sparsity exists (`14` nodes has only `36/120` strict lower-tri pairs and `50/256` visible square pairs), but simple sparse-pair scaling still does not plausibly bring `8/14` below the `~135us` FLA flat cost. Do not pour effort into big-tree optimization unless a more radical STree-style accumulated-state recurrence removes the padded dense solve/state traversal.
- COST-GATE decision: narrow the speed case to the tiny-tree niche (`<=4` nodes) unless new evidence changes the profile. Next step is to evaluate whether MTP-1/2 plus a tiny suffix/tree has enough acceptance to beat the P0 linear MTP-5 spines=1 baseline (`accept/draft=0.5787`, `13.46 tok/step`).
- COST-GATE tiny-tree screen completed from P0 counters. Depth-4 upper-bound from P0 accepted-position counters gives `2.588` accepted tokens/draft, `3.588` sequence tokens/draft, and projected `11.983 tok/step` at unchanged step time versus P0 `13.459 tok/step`; it would need `10.973%` step-latency reduction to match P0. Replacing `135us` FLA with a `45.339us` tiny tree across all `48` GDN layers saves only `4.304 ms/step`, about `0.358%` of the measured `1.203653s` P0 step. The `<=4` tiny-tree niche is therefore not competitive under observed P0 spine acceptance.
- COST-GATE outcome: current standalone tree-kernel speed case does not clear. Component correctness/losslessness architecture remains proven, but speed now requires either a materially different STree-style accumulated-state kernel that removes the dense padded solve/state traversal, or real branch-sampler evidence that acceptance rises far beyond the P0 spine-position counters. Do not proceed into large-tree vLLM integration as a speed deliverable without one of those.

## GPU Kernel Plan After P1 Gate

- Fork from vLLM 0.22 FLA ops under `/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/fla/ops/`.
- Surgical points: `solve_tril.py` linear triangular mask -> tree-ancestry mask; `chunk_scaled_dot_kkt.py` KKT interaction -> ancestry-masked interaction; `cumsum.py` linear cumsum -> ancestor-path `g` accumulation; `chunk_delta_h.py` and `chunk.py` chunk state recurrence -> per-node tree state.
- Static padded descriptor families: `{2,3,6,8,14}` nodes. Fail closed on unwarmed shapes.
- CUDA graph design constraints from first kernel: no allocation during capture, preallocated descriptors/buffers, warmed shapes only, graph replay parity against eager.
- Record CUDA, driver, PyTorch, Triton, vLLM, FlashAttention, and model revision in every GPU artifact.

## Blocked

- None for P1 CPU gate.
- P0 targeted follow-up for greedy token streams, temp=0.6 B4 streams, logprob support, CUDA graph capture status, reset route, batch-invariant guard, and exact stack version record is complete on the cu130 stack. Full SWE campaign rerun remains separate if requested; current P0 was the targeted baseline/freeze artifact path.
- Phase 2 red-team findings in progress:
  1. `graph_ok=true` now requires bit-exact graph replay output/state vs eager, not only capture success. All rerun public shapes reported `graph_bit_exact=true`.
  2. Exact-production Gate D target is now cu130-nightly `ChunkGatedDeltaRule.forward_native` / `fla_chunk_gated_delta_rule` on GB10, not FlashInfer. Single-spine output matches within one bf16 quantum; recurrent final state still differs at `<1e-3`, so canonical state commit remains required for byte-exact losslessness.
  3. Branch-depth cost table rebuilt with one fixed base tree (`5->6 padded 8->8`) for depths 0/1/2.
- Remaining Phase 2 implementation work: move standalone deterministic tree kernel into a reproducible vLLM FLA fork/patch and keep production precision (`bf16` inputs, fp32 recurrent state).
- Speed side status: current standalone tree-kernel path is blocked by cost gate. Next meaningful work is either (a) research/prototype a true accumulated-state sparse tree recurrence that changes the cost model, or (b) continue Route A/Gate B offline correctness capture separately while treating speed as unresolved.
- User dtype directive: do not approximate dtype flow. First detect active production backend by instantiating `ChunkGatedDeltaRule()` under E3/E5 serving config and reading `.gdn_prefill_backend`; then match that exact wrapper. For FlashInfer: reuse vLLM `l2norm_fwd`, q/k/v bf16, `g=torch.exp(g.float())` outside kernel, `beta.float()`, `initial_state.float()`, fp32 accumulation, bf16 output, fp32 final state, scale `1/sqrt(128)`.
- Active backend name for the real cu130 GB10 stack: `triton` / native FLA (`forward_native`). The broken audit image is no longer the production reference.
- Matched-bf16 single-spine native-vs-tree deltas (`6.103515625e-05` output, `<9.48e-4` state) are now treated as reduction-order state drift, not an automatic losslessness pass. State-commit implementation rule: use tree-kernel logits only for accept/reject verification; after acceptance, re-run the accepted short path through `fused_recurrent_gated_delta_rule_packed_decode` and commit that native state, discarding the tree kernel's approximate recurrent state. End-to-end Gate B remains native greedy token stream == tree-verifier greedy token stream on real prompts.
- Initial Phase 2 kernel is standalone deterministic microbench code, not yet a vLLM FLA op fork/integration. Next after red-team fixes: move this algebra into a reproducible vLLM FLA fork/patch under the `chunk.py` pipeline and pin/disable relevant autotune paths.
- Red-team Phase 2/3 flag: pin/disable autotune for the tree kernel and control `chunk_delta_h.py use_cuda_graph` autotune behavior. Determinism is a losslessness prerequisite. Confirmed `_tree_gdn_kernel` has no `@triton.autotune`; keep it that way.
- Future Gate C sampler must be sequence-level and must never use a max-over-branches / longest-accepted hidden winner. MTP draft quality affects acceptance rate only, not correctness, if sampler is valid.

## Final-Stack Blocker

- RESOLVED for Gate D target: `vllm/vllm-openai:cu130-nightly` is now the working production reference stack. The old `lumo-vllm-audit:v0.22.0-cu129-min` remains broken and should not be used as production reference. Artifact for the broken image remains `output/fr10_phase2_broken_audit_image_ldd_20260603.txt`.
- Remaining stack work: Phase 4 vLLM integration still needs to be run on this same stack. P0 cu130 `spines=1` targeted baseline is booted and captured with `kv_cache_dtype=auto`.
- Kernel/algebra work is not blocked: continue pure Triton work against `fla_chunk_gated_delta_rule` / `forward_native` as the guaranteed-available backend target; match its dtype/cast points exactly; report single-spine roundoff vs that exact wrapper as diagnostic; implement canonical native decode state commit so tree verify state drift cannot become committed-state drift.
- Route assessment: adding a CUDA 12 compatibility `libcudart.so.12` to the broken audit image is the quickest smoke experiment, but it is not the fastest reliable production route because the image still lacks `flashinfer` and may expose mixed cu12 `_C` with CUDA 13 Torch/runtime semantics. Faster viable route is to build a fresh 0.22 serving image from `nvcr.io/nvidia/pytorch:26.01-py3` per spec §6, updating `docker/Dockerfile.nvidia-vllm` from its current vLLM 0.19 defaults to a pinned vLLM 0.22 source/wheel built against the image CUDA, then install/verify the selected GDN backend packages and boot Qwen3.6 FP8/MTP before any final losslessness claim.

## Stack Bootstrap: cu130 Nightly

- Candidate image validated: `vllm/vllm-openai:cu130-nightly`, registry digest `sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`, local image ID `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. Promote this digest-pinned image to exact-production Gate D target, P0 cu130 `spines=1` baseline stack, and Phase 4 integration target.
- Fallback if nightly lacks GDN or sm121 support: build from source with `CUDA_HOME=/usr/local/cuda-13`, `TORCH_CUDA_ARCH_LIST=12.1`, `VLLM_TARGET_DEVICE=cuda`, and Triton `>=3.5`, using the DGX Spark setup references from user research. Do not claim final Gate D or lossless-vs-production until one of these stacks boots Qwen3.6 FP8/MTP and exposes a working GDN backend.
- Audit-image backend correction superseded: native FLA is also the real cu130 GB10 production GDN backend, but for a different reason: cu130 source only enables FlashInfer GDN on Hopper sm90, while GB10 is sm121/family120.

## Questions For Researcher

- None yet.

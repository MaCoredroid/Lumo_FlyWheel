You are an autonomous kernel-research agent for iteration 004 of round qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z.

# Your one job
Propose ONE mutation to the local CUTLASS source workspace at /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/cutlass_source_workspace (mounted over /opt/vllm-source during controller validation) that is faster than the current
best on the workload, AND passes the parity gate at /home/mark/shared/lumoFlyWheel/benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/fp8_gemm_v1.yaml.
For this GB10 run, "faster" means a material increase in warm decode throughput
on this machine. Recent live runs are around 7.5 generated tokens/s; mutations
that only move a tiny subcomponent without a plausible path to raising that
end-to-end number are not worth submitting.
Aim for step-change performance, not local polishing: the search ambition is
over 100% warm decode throughput improvement. The 20% speed gate is the
minimum spend-control threshold. Think out of the box when evidence supports
it, including writing a whole new CUTLASS kernel, dispatch path, or scale/shape
specialization, as long as the hypothesis is one reviewable mutation with a
local compile/preflight story and the parity contract is preserved.
Use the Karpathy/autoresearch lesson correctly: the loop is a same-machine,
fixed-budget keep/discard search. Tiny one-line nudges are fine only when the
local evidence says that exact knob is the bottleneck. Otherwise prefer a larger
coherent mutation that changes an actual performance mechanism enough to be
measurable, while keeping the diff reviewable and the parity contract intact.
After repeated byte-traffic blocks, do not stop at "schedule edits cannot clear
20%" unless you have also considered a broader CUTLASS-backed byte mechanism:
for example caller-level fusion, persistent/reuse staging, paired-projection
reuse, or a new specialized CUTLASS route that reduces launches or B-weight
streaming while preserving the public dtype/layout/scale/parity contract. If
such a mechanism is impossible, name the exact contract or source boundary that
prevents it.

# Hardware context (MATTERS for what mutations are worth proposing)
This kernel runs on a **DGX Spark GB10**. Treat it as bandwidth-bound:
128 GB LPDDR5x unified memory at roughly 273 GB/s, not an HBM3e server GPU.
Mutations that reduce memory traffic or improve cache reuse are more likely
to matter than compute-only micro-optimizations.
Do not treat this like H100/H200: GB10 has a unified LPDDR memory pool and
the known workload profile is decode-heavy.

# L0c Strategy Brief

- Round: `qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z`
- Kernel target: `fp8_gemm`
- HLD source: `docs/HLD-Serving-Backend-AutoResearch-v0_2-L0KernelPlan.md`
- Prior rejection ledger: `prior_mutations_rejected.tsv` (20 rows)
- Prior measured-trial memory: `prior_research_memory.tsv` (40 rows)
- Current-round measured-trial memory: `research_memory.tsv` and `research_memory.md`.

## Bottleneck Thesis

- Treat the canary as decode-dominant and bandwidth-sensitive on GB10 LPDDR5x.
- Prefer changes that reduce memory traffic, improve cache locality, or narrow one load/store behavior at a time.
- Do not trade numerical order or target-specific state semantics for speed; parity dominates throughput.
- P3a: `output/p3a_roofline_probe_20260429T193758Z/p3a_roofline_probe.json` probe_count=1, wall=18.302s, decode_share=0.977, gen_tok_s=6.994, gpu_util_mean=86.4%.
- P3a decision: no target-specific P3a decision recorded for FP8 GEMM; use the FFN GEMM pivot brief as the governing context.
- P3a limitation: no full Nsight kernel-category split; keep mutations conservative.

## GB10 CUTLASS Timing Breakdown

- Timing source: `output/p3a_agent_flow_roofline_20260501T192001Z/p3a_agent_flow_roofline_full10_summary.json`. These are CUDA-event elapsed times around vLLM call sites, not mutually exclusive low-level kernel self-time.
- P3a timing: gatedattn_attention_with_kv_read share=67.6%, self_time_ms=43683.2, ms_per_requested_output_token=273.020.
- P3a timing: ffn_linear share=20.0%, self_time_ms=12895.5, ms_per_requested_output_token=80.597.
- P3a timing: deltanet_projection_linear share=7.0%, self_time_ms=4544.2, ms_per_requested_output_token=28.401.
- P3a timing: gatedattn_projection_linear share=2.0%, self_time_ms=1315.3, ms_per_requested_output_token=8.221.
- P3a timing: deltanet_core share=1.8%, self_time_ms=1139.3, ms_per_requested_output_token=7.121.
- P3a timing: norm share=0.9%, self_time_ms=554.8, ms_per_requested_output_token=3.468.
- CUTLASS-relevant baseline: `ffn_linear` is the current FP8 GEMM/CUTLASS proxy at 20.0% leaf share; schedule/shape/scale mutations must explain how they reduce this slice or why a narrower sub-slice is still worth measuring.
- If a proposed mutation cannot plausibly affect `ffn_linear`/CUTLASS timing or a measured long-tail request path, do not submit it.
- Before proposing a CUTLASS patch, the authoring agent must state which timing component it expects to reduce and why the changed dispatch/shape/scale/schedule should affect that component on GB10.
- Candidate analysis must contain a structured compute/bandwidth accounting block, not only prose: representative M/N/K shape(s), FLOPs, estimated bytes moved, arithmetic intensity, GB10 roofline/ceiling comparison, current `ffn_linear` ms/token proxy, expected changed bytes/FLOPs/overhead, and expected end-to-end tok/s delta. The 273 GB/s LPDDR roofline and 10.1 tok/s full-model stream ceiling are context numbers, not proof of achieved memory bandwidth.
- Candidate analysis must also contain a low-level evidence block: exact source file/symbol being changed, proof that the live warm shape dispatch hits that path, A/B/scale/output/epilogue byte split with an explicit statement about whether B-weight bytes change, and a before-mutation observation from warm-diagnostic, source-level experiment, microbench, profiler, or targeted compile/preflight. If the evidence does not support at least a 20% end-to-end warm decode lift, write BLOCKED.md instead of submitting another low-upside patch.
- Think out of the box when the evidence supports it. The search ambition is over 100% warm decode throughput improvement; the 20% speed gate is only the minimum spend-control threshold. A candidate may write a whole new CUTLASS kernel, dispatch route, or scale/shape specialization if it has a credible local compile/preflight path and a parity-preserving hypothesis.
- After repeated byte-traffic BLOCKED rows, do not repeat the same local-schedule impossibility argument unless you have first checked a broader CUTLASS-backed byte mechanism: caller-level fusion, persistent/reuse staging, paired-projection reuse, launch reduction, or a new specialized CUTLASS route that reduces B-weight streaming or repeated memory traffic while preserving public dtype/layout/scale/parity semantics. If blocked, cite the exact source or operator contract that prevents that mechanism.
- Treat this section as the pre-change CUTLASS timing baseline. If a cheap CUTLASS-internal timing/proxy is available, record it before and after patching; if not, explicitly say no low-level CUTLASS sub-kernel timing is available and use `ffn_linear` as the controller-owned proxy.
- After writing the patch, the authoring agent must compare the patch against this same breakdown in its final message. The controller owns the expensive after-change vLLM measurement.

## Forbidden Mutation Families

- Do not change the GEMM call signature, tensor layout contract, dtype contract, or scale semantics.
- Do not change public operator signatures or unguarded architecture dispatch behavior.
- Do not edit fixture capture, replay, parity, controller, or measurement code.
- Do not retry a mutation hash listed in `mutations_rejected.tsv` or `prior_mutations_rejected.tsv`.
- If an older `BLOCKED.md` suggestion conflicts with this forbidden list, the forbidden list wins.

## Prior-Art Memory Contract

- Follow AutoTVM/Ansor/TVM MetaSchedule/OpenTuner style memory: keep workload keys, compact config/schedule traces, feature tags, git diff excerpts, build/parity/measurement outcomes, failure class, and next-search implications. Use memory to bias search away from repeats, not as a blind syntax ban.
- Follow the Karpathy/autoresearch fixed-budget lesson: on this same GB10 machine, score is the measured objective under a fixed controller window. After many tiny schedule/tile nudges have failed, prefer a larger coherent mechanism when the evidence supports it. A candidate may coordinate related dispatch, schedule/stage/tile, scale-placement, or caller-launch edits, but it must still be one reviewable hypothesis with a plausible path to raising warm decode tok/s.
- Every candidate must be materially different from poor prior rows in `research_memory.tsv` or must explain why a new dispatch, shape, scale, or schedule fact changes the expected outcome.
- Before mutating, classify the patch surface and expected affected path: dispatch predicate, schedule tile/CTA shape, scale placement, workspace/SM-count behavior, memory traffic, cache locality, occupancy, register pressure, or instruction count.
- Treat `patch_diff`, `failure_class`, `mutation_features`, `schedule_trace`, and `search_bias` as the trial database fields: a new patch must change a feature/trace that can plausibly change the measured outcome and must account for whether the old failure was correctness, build, preflight, or performance.

## Ranked Likely-Safe Targets

1. Use prior parity failures to avoid wrapper-only dispatch, reshape-only, or padding-only mutations that do not change a defensible CUTLASS behavior.
2. Investigate actual CUTLASS dispatch, scale layout, GEMM problem shape, and schedule/source constraints before proposing a patch.
   For synthetic FP8 GEMM rounds, keep the Triton FP8 GEMM call boundary as the analogous dispatch/shape contract.
3. Prefer mutations that are justified by a cheap local diagnostic or primary-source finding, then preserve scale semantics and output dtype behavior exactly.
4. If the staged source cannot reach the needed compiled CUTLASS schedule, write BLOCKED.md with the missing C++/rebuild surface instead of submitting another cosmetic Python patch.

## Prior Measured-Trial Memory

- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 001: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation objective=; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 002: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation objective=; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 003: unknown_or_mixed / unclassified_low_level_mechanism => agent_no_patch objective=; next=agent must submit mutation.patch or BLOCKED.md with a concrete source-surface reason
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 001: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation objective=; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 002: cutlass_cxx_dispatch_or_schedule / unclassified_low_level_mechanism => discard objective=0.041480; next=avoid repeating this surface/path unless a new dispatch, shape, scale, or schedule fact changes the hypothesis
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 003: cutlass_sm120_dispatch / schedule_tile_or_cta_shape => parity_generation_speed_below_baseline objective=; next=do not spend controller validation on adjacent candidates unless preflight analysis explains how they clear the post-parity speed threshold
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 001: cutlass_sm120_dispatch / schedule_tile_or_cta_shape => agent_blocked_compile_preflight objective=; next=agent must fix compile/preflight locally before controller validation
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 002: unknown_or_mixed / scale_placement_or_quant_semantics => safety_routes_cutlass_round_to_non_cutlass_backend objective=; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 003: cutlass_sm120_dispatch / schedule_tile_or_cta_shape => discard objective=0.042228; next=avoid repeating this surface/path unless a new dispatch, shape, scale, or schedule fact changes the hypothesis
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 005: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_compile_preflight objective=; next=agent must fix compile/preflight locally before controller validation

## Prior Rejections Carried Forward

Rows with `agent_exit_*` or `agent_spawn_failed` are historical context unless their `source_ref` points to a parity artifact.
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 001: agent_no_patch; note: # Candidate 001 Blocked
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 002: agent_no_patch; note: # Candidate 002 Blocked
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 003: agent_no_patch; note: # Candidate 003 Blocked
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 001: agent_no_patch; note: # Candidate 001 Blocked
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 003: parity_generation_speed_below_baseline; note: iteration rejected by controller's parity/generation speed gate
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 001: agent_blocked_compile_preflight; note: # Candidate 001 Blocked
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 002: safety_routes_cutlass_round_to_non_cutlass_backend
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 005: agent_no_patch; note: # Candidate 005 Blocked

Use this brief as direction, not proof. The controller parity gate is canonical.

## Auto-Refreshed Candidate Memory

This block is regenerated from candidate artifacts before each authoring spawn. Treat it as the canonical patch_diff/failure_class table for previous attempts in this round.

# L0c Research Memory

This file is the agent-readable memory index for the L0c mutation loop.

## Prior-Art Alignment

- AutoTVM keeps measurement inputs/results and uses history-best records.
- Ansor and TVM MetaSchedule retain schedule traces plus measured run times, then use them to guide evolutionary search.
- OpenTuner shares a common results database across search techniques.
- Therefore this loop records measured and failed trials with surface tags and next-search implications instead of relying only on prose warnings or hard bans.

## Schema

- `workload_key`: workload/kernel identity for cross-round reuse, mirroring AutoTVM/MetaSchedule workload records.
- `surface`: mutated subsystem, such as CUTLASS SM120 dispatch, C++ schedule source, Python wrapper, or hardware-info path.
- `changed_region`: diff header paths or the closest available source region.
- `expected_affected_path`: low-level mechanism claimed by the patch.
- `mutation_features`: compact knob/config features such as CTA shapes, M guards, scale changes, or SM120 guard changes.
- `schedule_trace`: a short schedule/config trace for comparing trials without reading full patches.
- `patch_diff`: compact sanitized git diff excerpt from `mutation.patch`, enough to see the changed code shape.
- `controller_gate`: preflight, compile, parity, or measurement gate that produced the outcome.
- `outcome`: measured keep/discard, parity failure, compile failure, duplicate, or other terminal status.
- `failure_class`: success, performance, correctness, build, preflight_safety, duplicate, authoring, or context.
- `measurement_policy`: whether the row reached controller vLLM measurement, parity-only, or author compile preflight.
- `next_implication`: how future agents should use the row.
- `search_bias`: explicit budget/search treatment for adjacent proposals.

## Current Round Rows

- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z 001: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation, objective=; failure_class=authoring; gate=agent_blocked_no_mutation; features=shape=_64,_128,_128,m_guard=M<=256,scale_related; patch_diff=- `M=1,N=34816,K=5120`: total `178.376 MB`; B weights `178.258 MB`; arithmetic intensity `1.999 FLOP/B`.\n- `M=1,N=5120,K=17408`: total `89.179 MB`; B weights `89.129 MB`; arithmetic intensity `1.999 FLOP/B`.\n- Qwen3.5 already fuses obvious paired projections: `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba` are packed in `Qwen3_5ForCausalLMBase.packed_modules_mapping`.\n- FFN down projection cannot reuse the gate/up B stream because it consumes a nonlinear activation and a distinct B matrix. Fusing it would require a new FFN operator signature, not a drop-in `cutlass_scaled_mm` mutation.\n- Persistent staging across tokens is not exposed by the per-GEMM public op and would not fit the model's full B-weight working set.\n- Prior same-machine trials already rejected the compile-clean adjacent families: workspace/hardware-info caching, zero-workspace, M1 branch/padding, fixed ...[truncated]; bias=context_only; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z 002: unknown_or_mixed / scale_placement_or_quant_semantics => agent_blocked_no_mutation, objective=; failure_class=authoring; gate=agent_blocked_no_mutation; features=shape=_64,_128,_128,shape=_1,_1,_1,m_guard=M<=256,scale_related; patch_diff=- `M=1,N=34816,K=5120`: total `178.376 MB`; B weights `178.258 MB`; arithmetic intensity `1.999 FLOP/B`.\n- `M=1,N=5120,K=17408`: total `89.179 MB`; B weights `89.129 MB`; arithmetic intensity `1.999 FLOP/B`.; bias=context_only; next=read artifacts before proposing adjacent mutations

## Prior Rows

- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 001: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation, objective=; failure_class=authoring; gate=agent_blocked_no_mutation; features=m_guard=M<=256,scale_related; patch_diff=- `M=1,N=34816,K=5120`: total estimated bytes `178.376 MB`, B weights `178.258 MB`.\n- `M=1,N=5120,K=17408`: total estimated bytes `89.179 MB`, B weights `89.129 MB`.; bias=context_only; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 002: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation, objective=; failure_class=authoring; gate=agent_blocked_no_mutation; features=scale_related; patch_diff=- `M=1,N=34816,K=5120`: total estimated bytes `178.376 MB`, B weights `178.258 MB`.\n- `M=1,N=5120,K=17408`: total estimated bytes `89.179 MB`, B weights `89.129 MB`.; bias=context_only; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T204655Z 003: unknown_or_mixed / unclassified_low_level_mechanism => agent_no_patch, objective=; failure_class=authoring; gate=agent_no_patch; features=unclassified_low_level_mechanism; patch_diff=; bias=context_only; next=agent must submit mutation.patch or BLOCKED.md with a concrete source-surface reason
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 001: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation, objective=; failure_class=authoring; gate=agent_blocked_no_mutation; features=scale_related; patch_diff=- `M=1,N=34816,K=5120`: about `178.376` MB moved, with `178.258` MB from B weights.\n- `M=1,N=5120,K=17408`: about `89.179` MB moved, with `89.129` MB from B weights.; bias=context_only; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 002: cutlass_cxx_dispatch_or_schedule / unclassified_low_level_mechanism => discard, objective=0.041480; failure_class=performance; gate=ran_passed_with_tier4_downstream_logit_diagnostic; features=shape=int,int,int,int; patch_diff=--- cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh\n+++ cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh\n@@ -6,6 +6,8 @@\n+#include <vector>\n+\n@@ -24,12 +26,50 @@\n+struct CutlassWorkspaceCacheEntry {\n+  int device_id;\n+  cudaStream_t stream;\n+  torch::Tensor buffer;\n+};\n+\n+static inline void* get_cutlass_workspace_ptr(torch::Device device,\n+                                              cudaStream_t stream,\n+                                              size_t workspace_size) {\n+  if (workspace_size == 0) {\n+    return nullptr;\n+  }\n+\n+  int device_id = static_cast<int>(device.index());\n+  if (device_id < 0) {\n+    device_id = 0;\n+  }\n+\n+  thread_local std::vector<CutlassWorkspaceCacheEntry> workspace_cache;\n+  for (auto& entry : workspace_cache) {\n+    if (entry.device_...[truncated]; bias=deprioritize_adjacent_without_new_evidence; next=avoid repeating this surface/path unless a new dispatch, shape, scale, or schedule fact changes the hypothesis
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T185949Z 003: cutlass_sm120_dispatch / schedule_tile_or_cta_shape => parity_generation_speed_below_baseline, objective=; failure_class=performance; gate=parity_generation_speed_below_baseline; features=shape=_128,_64,_128,shape=_1,_1,_1,m_guard=M==1,m_guard=M<=256,enable_sm120_family,swap_ab,scale_related; patch_diff=--- cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh 2026-05-03 17:28:52.000000000 +0000\n+++ cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh\n@@ -24,8 +24,10 @@\n-          class EpilogueScheduler, class MainloopScheduler>\n+          class EpilogueScheduler, class MainloopScheduler,\n+          bool swap_ab_ = false>\n+  static constexpr bool swap_ab = swap_ab_;\n@@ -53,9 +55,13 @@\n-  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<\n+  using ScaleConfig = conditional_t<swap_ab,\n+      cutlass::detail::Sm120BlockwiseScaleConfig<\n-        cute::UMMA::Major::MN, cute::UMMA::Major::K>;\n+        cute::UMMA::Major::K, cute::UMMA::Major::MN>,\n+      cutlass::detail::Sm120BlockwiseScaleConfig<\n+        ScaleGranularityM, ScaleGranularityN, S...[truncated]; bias=deprioritize_until_speed_gate_hypothesis_changes; next=do not spend controller validation on adjacent candidates unless preflight analysis explains how they clear the post-parity speed threshold
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 001: cutlass_sm120_dispatch / schedule_tile_or_cta_shape => agent_blocked_compile_preflight, objective=; failure_class=build; gate=agent_blocked_compile_preflight; features=shape=_64,_128,_128,shape=_64,_128,_256,m_guard=M<=256,scale_related; patch_diff=- Switch the M64 blockwise schedule from `KernelTmaWarpSpecializedBlockwisePingpongSm120` to CUTLASS auto/cooperative variants.\n- Increase the M64 blockwise K tile from `Shape<_64, _128, _128>` to `Shape<_64, _128, _256>`.; bias=repair_compile_surface_before_resubmit; next=agent must fix compile/preflight locally before controller validation
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 002: unknown_or_mixed / scale_placement_or_quant_semantics => safety_routes_cutlass_round_to_non_cutlass_backend, objective=; failure_class=preflight_safety; gate=safety_routes_cutlass_round_to_non_cutlass_backend; features=m_guard=M<=16,scale_related; patch_diff=--- cutlass_source_workspace/vllm-source/vllm/model_executor/layers/quantization/utils/fp8_utils.py 2026-05-03 17:28:52.000000000 +0000\n+++ cutlass_source_workspace/vllm-source/vllm/model_executor/layers/quantization/utils/fp8_utils.py 2026-05-05 09:18:11.035164244 +0000\n@@ -370,6 +370,14 @@\n+        self.uses_cutlass_blockscale = (\n+            cutlass_block_fp8_supported and not use_aiter_and_is_supported\n+        )\n+        capability = current_platform.get_device_capability()\n+        capability_int = -1 if capability is None else capability.to_int()\n+        self.use_triton_gb10_decode_path = (\n+            self.uses_cutlass_blockscale and capability_int == 121\n+        )\n@@ -391,6 +399,16 @@\n+        self.triton_decode_input_quant_op = (\n+            QuantFP8(\n+                False,\n+                self.act_quant_group_shape,\n+                column_major_scales=F...[truncated]; bias=context_only; next=read artifacts before proposing adjacent mutations
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 003: cutlass_sm120_dispatch / schedule_tile_or_cta_shape => discard, objective=0.042228; failure_class=performance; gate=ran_passed_with_tier4_downstream_logit_diagnostic; features=shape=int,int,int,int,enable_sm120_family,KernelHardwareInfo,sm_count,scale_related; patch_diff=--- cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh\n+++ cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh\n@@ -30,13 +30,30 @@\n+static inline cutlass::KernelHardwareInfo get_cached_kernel_hardware_info(\n+    torch::Device device) {\n+  int device_id = static_cast<int>(device.index());\n+  thread_local int cached_device_id = -1;\n+  thread_local int cached_sm_count = 0;\n+\n+  if (cached_device_id != device_id || cached_sm_count <= 0) {\n+    cached_device_id = device_id;\n+    cached_sm_count =\n+        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(\n+            device_id);\n+  }\n+\n+  return cutlass::KernelHardwareInfo{device_id, cached_sm_count};\n+}\n+\n-  cutlass::KernelHardwareInfo hw_info;\n+  cutlass::KernelHardwareInfo hw_info =\n+      get_cached_kernel_hardware_i...[truncated]; bias=deprioritize_adjacent_without_new_evidence; next=avoid repeating this surface/path unless a new dispatch, shape, scale, or schedule fact changes the hypothesis
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 005: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_compile_preflight, objective=; failure_class=build; gate=agent_blocked_compile_preflight; features=shape=_64,_128,_128,shape=_64,_64,_128,m_guard=M<=256,scale_related; patch_diff=- Add a guarded `Shape<_64,_64,_128>` small-N config for `M <= 256 && N <= 8192`. This would double CTA count while keeping B-weight bytes, dtype, layout, public signature, output dtype, and scale granularity unchanged, but the SM120 blockwise instantiation failed to compile.\n- Add an optional tile-scheduler template parameter and route `M <= 256 && N <= 8192` through `cutlass::gemm::StreamKScheduler` with the existing legal `Shape<_64,_128,_128>` tile. This preserved all scale/data semantics and changed only the launched CUTLASS scheduler for the low-CTA shapes, but the SM120 blockwise `GemmUniversal` instantiation failed to compile.; bias=repair_compile_surface_before_resubmit; next=agent must fix compile/preflight locally before controller validation
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 006: cutlass_sm120_dispatch / schedule_tile_or_cta_shape => discard, objective=0.045165; failure_class=performance; gate=ran_passed_with_tier4_downstream_logit_diagnostic; features=shape=_64,_32,EpilogueTile,scale_related; patch_diff=--- cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh\n+++ cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh\n@@ -24,7 +24,8 @@ using namespace cute;\n-          class EpilogueScheduler, class MainloopScheduler>\n+          class EpilogueScheduler, class MainloopScheduler,\n+          class EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto>\n@@ -72,7 +73,7 @@ struct cutlass_3x_gemm_fp8_blockwise {\n-      cutlass::epilogue::collective::EpilogueTileAuto,\n+      EpilogueTile,\n@@ -134,7 +135,7 @@ struct sm120_blockwise_fp8_config_M64 {\n-      EpilogueSchedule, KernelSchedule>;\n+      EpilogueSchedule, KernelSchedule, Shape<_64, _32>>;; bias=deprioritize_adjacent_without_new_evidence; next=avoid repeating this surface/path unless a new dispatch, shape, scale, or schedule fact changes the hypothesis
- qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T083523Z 007: unknown_or_mixed / schedule_tile_or_cta_shape => agent_blocked_no_mutation, objective=; failure_class=authoring; gate=agent_blocked_no_mutation; features=shape=_64,_128,_128,m_guard=M<=256,scale_related; patch_diff=- M1,N5120,K5120: 0.0813 ms event, 0.0836 ms wall.\n- M4,N5120,K5120: 0.0704 ms event, 0.0712 ms wall.\n- M1,N5120,K17408: 0.4141 ms event, 0.4154 ms wall.\n- M1,N34816,K5120: 0.7840 ms event, 0.7848 ms wall.; bias=context_only; next=read artifacts before proposing adjacent mutations

Use the context already in this brief and the local rejection history, then
do a short targeted research pass before choosing the mutation. Use Codex's
online research/search tools when available, and prefer primary docs/source:
vLLM source/docs, NVIDIA CUTLASS docs, CUDA docs, and local container source
over generic advice. Keep this bounded: extract the dispatch, scale, shape, or
schedule fact that changes your mutation choice, record the source/fact in your
notes or final message, then move on.

# Mutation size guidance
The previous round history is dominated by small single-knob CUTLASS changes
that did not move warm decode throughput materially. Your candidate may be
larger than those attempts if it remains one coherent mechanism. Good larger
mutations can coordinate 2-4 related edits across dispatch predicates, SM120
schedule/stage/tile choices, scale placement, or caller launch behavior. Bad
larger mutations are grab bags, signature/layout/scale semantic changes, or
unreviewable rewrites. Before patching, state the single mechanism and why its
combined edits should move the same-machine warm throughput objective.

# Hard rules
- Edit ONLY files under /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/cutlass_source_workspace. No parity fixture, controller, tests, or files outside this staged vLLM source workspace. C++ CUTLASS schedule/dispatch files are in scope. Python/model/runtime files inside the workspace are also in scope only when the diff directly changes how the existing CUTLASS FP8 GEMM path is reached, shaped, scaled, or fused, and the candidate_analysis.md proves that `CutlassFP8ScaledMMLinearKernel` still handles the affected FP8 GEMM calls.
- Do not change the kernel's input/output signature.
- CUTLASS dispatch, shape, scale-placement, schedule-source, and CUTLASS-backed caller/model/runtime shape-lift edits are in scope inside the staged vLLM/CUTLASS source tree, but preserve the public GEMM signature, tensor layout, output dtype, scale semantics, and CUTLASS FP8 backend identity exactly.
- Read mutations_rejected.tsv. Mutations identical to a prior rejection
  by patch hash are immediately rejected without re-running. Read the
  rejection reasons (first_diverging_probe, tolerance_overshoot) and
  propose something genuinely different.
- Your mutation MUST pass parity. Latency is irrelevant if parity fails.
- Before writing a patch, state the expected speed mechanism in your
  own notes and make the diff implement that mechanism directly.
- Avoid reshape-only, variable-cache-only, and cosmetic control-flow
  mutations; they have failed parity and are too small to be useful.
- Do not replace `process_weights_after_loading` wholesale. If you
  touch it, preserve CUTLASS weight transposition, fused scale
  conversion, static input-scale handling, and AZP adjustment.
- Prefer exact guarded changes that affect actual CUTLASS dispatch,
  shape, scale, or schedule behavior: backend selection predicates,
  quantization scale shape/placement, FP8 input grouping/padding,
  or calls into alternative CUTLASS-supported scaled-mm paths.
- Do not self-block merely because the best mechanism is not a C++
  schedule tweak. CUTLASS-backed Python/model/runtime mutations are
  valid when they preserve the CUTLASS FP8 backend and change the
  GEMM problem shape, call grouping, launch count, activation/scale
  placement, or fused caller path enough to plausibly clear the speed gate.
- In particular, investigate mechanisms that amortize compulsory B-weight
  streaming by making CUTLASS verify or compute more useful tokens per
  target forward, or by fusing adjacent activation/quant/caller work into
  the CUTLASS-backed path without changing public dtype/layout/scale semantics.
- This round is CUTLASS-only. Do not route warm decode to Triton,
  DeepGEMM, FlashInfer, cuBLAS, AITER, or another non-CUTLASS backend.
  A patch that calls `w8a8_triton_block_scaled_mm_func` or otherwise
  bypasses `CutlassFP8ScaledMMLinearKernel` is a hard preflight reject.
- If your patch cannot plausibly change the launched CUTLASS op,
  scale tensors, or GEMM problem shape, write BLOCKED.md instead
  of submitting another Python-wrapper cosmetic mutation.

# Parity contract
- Tier 3 GEMM-output tolerance: rtol=0.002 / atol=0.002
- Tier 4 downstream-logit guard tolerance: rtol=0.001 / atol=0.001
- Controller validation restarts vLLM with the patched CUTLASS source workspace mounted in-place, then runs Tier 3/Tier 4 parity before measurement.

# Reading prior-iteration history
The canonical per-iteration record is `candidates/<NNN>/parity_check.json`
(written by the round controller's authoritative re-run). `BLOCKED.md`,
when present in a candidate dir, means the controller REJECTED that
mutation — its content is the controller's reason, NOT the agent's
prior commentary.

For this FP8 GEMM CUTLASS source workspace, do NOT run apply-and-test from the agent.
Full vLLM restart is the expensive Tier 4 path, so the controller must inspect
mutation.patch first and decide whether the patch is safety-rejected or admitted
to Tier 4. Your job is proposal plus cheap local patch/compile validation only.
You own authoring-time compile failures: if the patch does not compile in the
cheap preflight surface, revise mutation.patch and rerun the cheap checks before
submitting. Do not leave obvious compile failures for the controller.

# Recent winning diffs (positive memory)
The following diffs were accepted in recent L0c rounds with measured improvement.
Use them for orientation about mutation shapes that have worked. They are not
requirements.

No prior winning L0c diffs found for this kernel target.

# Controller-enforced preflight patterns
The round controller checks your patch before apply-and-test. Tier-1 patterns
are soft-demoted into canary admission; Tier-2 patterns are hard-rejected.
This is enforcement, not preference.

Tier-1 soft-demote:
- None configured.

Tier-2 hard-reject:
- `safety_mutates_parity_fixture_code` — patch touches parity fixture builders or parity-check implementation
- `safety_mutates_l0c_measurement_controller` — patch touches the L0c controller or measurement-recording implementation
- `safety_mutates_l0c_tests` — patch touches `tests/test_l0c_*.py`
- `safety_mutates_rejection_or_filter_writer` — patch touches rejection-ledger or filter-hit-review writer code

# Procedure
1. Read /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/cutlass_source_workspace, /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/cutlass_source_workspace/README_L0C_CUTLASS.md, strategy_brief.md, prior_research_memory.tsv,
   research_memory.tsv, research_memory.md, prior_mutations_rejected.tsv,
   mutations_rejected.tsv, results.tsv (best_so_far).
   For prior iters' parity status, prefer `candidates/<NNN>/parity_check.json`.
2. Before editing, you MUST run a cheap warm-request diagnostic or read the
   existing cheap warm-request diagnostic against the
   already-running live server, then use its concrete fields in your analysis:
     cd /home/mark/shared/lumoFlyWheel && /home/mark/shared/lumoFlyWheel/.venv/bin/lumoserve auto-research warm-diagnostic --round-id qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z --iteration 004 --round-root /home/mark/shared/lumoFlyWheel/output/auto_research --phase pre_mutation --warmup-requests 1 --request-count 2 --max-output-tokens 64
   This measures the current warm live stack; it does not apply your patch.
   Read `warm_pre_mutation.json` and quote the relevant
   `aggregate_consumption.step_consumption`, `aggregate_consumption.gb10_reference`,
   and `per_step_consumption` fields for per-step token/time/cache consumption,
   bottleneck_hint, and GB10 bandwidth roofline context. If the file already
   exists, read it before deciding whether to rerun the command. If the live
   endpoint is down, write /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/candidates/004/warm_diagnostic_skipped.json with
   the command, error, and reason, then use the last baseline/candidate
   measurement traces instead.
3. Before editing, write /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/candidates/004/candidate_analysis.md with the
   baseline timing breakdown and compute/bandwidth breakdown you are using.
   This is a required cheap preflight artifact, not optional commentary. Include:
   - the current warm decode rate in generated tok/s and ms/generated token,
   - current round measurement rows or the nearest baseline/candidate traces,
   - the GB10 bandwidth fact and the bytes/token roofline context from the warm
     diagnostic or measurement trace,
   - which CUTLASS time component you believe is limiting decode, including the
     CUTLASS/FFN timing component and the `ffn_linear` proxy from
     strategy_brief.md when no lower-level CUTLASS timer is available,
   - an estimated FLOP or arithmetic-intensity sanity check for the GEMM shape
     class you are changing, even if approximate,
   - the exact mechanism by which your mutation should lift the observed warm
     decode rate materially above the recent ~7.5 generated tok/s level.
   Also include a compact structured table or bullet block with these exact
   fields: representative shape(s) as M/N/K, FLOPs per token or per GEMM,
   estimated bytes moved, arithmetic intensity, GB10 roofline/ceiling comparison,
   current `ffn_linear` ms/token proxy, expected changed bytes/FLOPs/overhead,
   and the expected end-to-end tok/s delta if the hypothesis is right.
   Include a separate 7.5 tok/s breakdown line that computes observed/implied
   effective bandwidth in GB/s, percent of the 273 GB/s GB10 ceiling,
   the 10.1 tok/s full-model FP8 stream ceiling, `ffn_linear` share of
   ms/token, and non-FFN residual ms/token. This can be approximate, but it
   must be numeric and it must explain whether the patch attacks bandwidth
   traffic, launch/schedule overhead, or another residual. Treat the GB10
   roofline as context, not proof of achieved memory bandwidth, unless you
   have a profiler measurement.
   Report `ffn_linear` share of ms/token and non-FFN residual ms/token
   explicitly, since those two numbers are the controller's current proxy for
   the CUTLASS-vs-rest split.
   Also include a low-level evidence table or bullet block with these exact
   fields: source file/symbol, live-shape dispatch-hit proof, before-mutation
   observation, byte-component split for A/B weights/scales/output/epilogue,
   whether B-weight bytes change, and why the expected lift is at least 20%
   end-to-end. If you cannot defend the 20% lift from those facts, write
   BLOCKED.md instead of mutation.patch.
   The controller will also run a cheap post-parity generation speed gate:
   if the patched runtime does not exceed `warm_pre_mutation.json` decode
   tok/s by at least 20%, it is discarded before the expensive paired
   measurement windows.
   If you can cheaply produce a CUTLASS-internal timing/proxy, include
   before-change and after-change values; otherwise state that only the
   `ffn_linear` proxy is available and do not invent lower-level CUTLASS times.
   After writing the patch, only run another warm request diagnostic if your
   mutation can be exercised without a vLLM restart. For compiled CUTLASS
   changes, say the post-patch warm request is controller-owned.
4. Before editing, run cheap local diagnostics or source-level experiments that
   test the exact dispatch/shape/scale/schedule assumption behind your idea.
   Examples: inspect registered op schemas, grep the mounted vLLM/CUTLASS source,
   compile the touched Python/C++ file if applicable, or run a tiny non-vLLM
   import/shape probe. Do not start vLLM and do not run apply-and-test.
   For FP8 CUTLASS source changes, prefer the shape-level CUTLASS microbench
   when the live-shape hypothesis depends on in-kernel schedule/shape behavior:
     cd /home/mark/shared/lumoFlyWheel && /home/mark/shared/lumoFlyWheel/.venv/bin/lumoserve auto-research cutlass-microbench --workspace-source /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/cutlass_source_workspace/vllm-source --shape 1x34816x5120 --shape 1x5120x17408 --shape 4x34816x5120 --shape 4x5120x17408 --warmup-iters 3 --benchmark-iters 10 --compile-jobs 1 --output-path /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/candidates/004/cutlass_microbench_pre.json
   Save the output as /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/candidates/004/cutlass_microbench_pre.json and cite
   event_ms_mean, estimated_effective_bandwidth_gb_s, and arithmetic_intensity
   in candidate_analysis.md. If the microbench is unavailable, explain why.
5. Do a short online/source research pass only if it can inform the mutation:
   use Codex online research/search tools when available, prefer primary
   docs/source, and record the specific source-derived fact in your
   notes/transcript. If online tools are unavailable, state that and use local
   vLLM/CUTLASS/CUDA source instead.
6. Write your proposal to /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/candidates/004/mutation.patch.
   Generate the patch with a real diff tool; do not hand-write hunk counts.
   The patch must apply with:
     cd /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z && patch --dry-run -p0 < /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/candidates/004/mutation.patch
7. Run the cheap patch apply check from step 6. If `patch --dry-run`
   fails, fix or regenerate mutation.patch and run the dry-run again.
8. Run local syntax/compile checks on any changed Python files, for example:
   cd /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/cutlass_source_workspace/vllm-source/vllm/model_executor/kernels/linear/scaled_mm && python3 -m py_compile $(find . -name '*.py' -print)
9. If you changed C++/CUDA, run a compile-level preflight on a temporary copy.
   Start with metadata mode while iterating, then run targeted mode before final submit
   when the change touches compiled CUTLASS files. Targeted mode builds only
   the CUTLASS FP8/SM120 objects inferred from mutation.patch:
   cd /home/mark/shared/lumoFlyWheel && /home/mark/shared/lumoFlyWheel/.venv/bin/lumoserve auto-research preflight-patch \
     --kernel-target fp8_gemm --patch-path /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/candidates/004/mutation.patch \
     --workspace-source /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260505T213552Z/cutlass_source_workspace/vllm-source --compile-mode targeted --compile-jobs 1
   If this compile/preflight exits nonzero, read the JSON `reason`,
   `compile_preflight.output_tail`, `matching_rule`, `code_snippet`,
   and `evidence_snippet`, then revise mutation.patch and rerun the cheap checks.
   Also read `speed_gate_preflight`: it reports the controller's post-parity
   generation-speed threshold from `warm_pre_mutation.json`. If your patch cannot
   plausibly beat `required_decode_tokens_per_s`, write BLOCKED.md instead of
   spending controller validation on a baseline-speed candidate.
   `speed_gate_analysis_preflight` is enforced here: if candidate_analysis.md
   forecasts less than the required generated tok/s threshold, preflight fails
   and you must revise the hypothesis or write BLOCKED.md.
   A compiled-file mutation is not ready to submit until this authoring-side
   targeted compile preflight passes or explicitly reports no CUTLASS C++ targets;
   fix compile errors in the patch before exiting.
10. For Python-only mutations, run the same command with `--compile-mode python`.
11. Do not run `auto-research apply-and-test` for this FP8 GEMM CUTLASS source target.
   The controller owns canary admission, parity, and measurement after you exit.
12. Exit 0 after either mutation.patch passes the cheap checks or BLOCKED.md explains
   why no patch can pass preflight. Do not exit nonzero for a cheap-check failure;
   nonzero exit is reserved for agent/tool infrastructure failure.

# What you do NOT do
- You do not call finalize-round. Python does that.
- You do not run measurement directly. The CLI does that.
- You do not write any file except mutation.patch, candidate_analysis.md,
  BLOCKED.md, cutlass_microbench_pre.json, and the warm diagnostic
  artifact/skipped note from step 2.

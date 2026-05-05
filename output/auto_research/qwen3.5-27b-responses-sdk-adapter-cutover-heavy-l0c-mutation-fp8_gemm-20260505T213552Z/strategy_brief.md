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

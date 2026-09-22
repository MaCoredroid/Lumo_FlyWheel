# LumoFlyWheel v2: authorized bounded numerical experiments

## Task and authority

The user requested P0 first, followed by a new interactive Claude session on the DGX Spark using Fable, xhigh effort, and `--dangerously-skip-permissions`. P0 is complete without model inference. Implement and execute the bounded experiment scope below in this new worktree. This is an execution handoff, not a request for a plan alone. Begin E7a; progress to E2/E7b and E1 when the stated gates pass. Record failures and stop dependent work when a necessary gate fails. No extra approval is needed for the authorized bounded work.

Working directory: `/home/mark/lumo-paper-v2-20260921`; branch `codex/paper-v2-pilot-20260921`; base commit `984f613d7885d9bf7898866827552b68a3199d0c`. This is a sparse checkout containing source, scripts, tests, patches, csrc, docker, configuration, papers, docs, research, and root documents. The v2 paper and P0 artifacts were transferred from the user's Mac and are initially untracked. Keep new code/results here. Expand sparse paths narrowly if needed. Never check out the enormous historical output tree wholesale.

The existing `/home/mark/shared/lumoFlyWheel` checkout is on an older branch and has two modified evidence files. Preserve it and all existing tmux sessions. Its archived tensors and the model/binary paths below may be read or mounted read-only. Do not reset/clean/switch that checkout, write into its captures, kill unrelated processes, or alter shared model weights. No external messages, uploads, publication, push, new quantization campaign, or broad task-quality campaign is requested.

## Read first

1. `papers/gdn-tree-scan-mlsys/v2/p0/P0-REPORT.md` and JSON manifests beside it.
2. `papers/gdn-tree-scan-mlsys/v2/review-experiments.md` for exact E7a, E2/E7b, E1 scope and decision rules.
3. `papers/gdn-tree-scan-mlsys/v2/notes/numerical-comparison-decision.md` and `notes/closest-work-reading.md`.
4. Relevant current implementation and qualification documents. Existing pause notes are superseded by this user's bounded experiment authorization; numerical and artifact qualification checks remain requirements.

Literature primary sources: https://arxiv.org/html/2608.01651v1 (Bole) and https://arxiv.org/html/2608.20961v1 (TreeWY). Verify the equations and precision choices when implementing. Call our variants local reimplementations of mechanisms, not results from the authors' systems. Mathematical equivalence is distinct from finite-precision native agreement. The old WY defect was partly repaired; the archive does not establish a universal defect in TreeWY, Bole, or WY. Diffuse discrepancies also affect sequential routes.

## P0 findings to retain

- Reconstructing each July H3 patcher at its own recorded commit against the launchers' pinned image yields two target-logit constraint calls; the current constraint insertion yields one. Target temperature divisions compound, but top-k/top-p also repeats and bonus paths differ. Do not relabel the entire old run as ordinary temperature 0.36.
- These are source reconstructions. Original historical loaded FA2/JIT hashes and model-file identities remain unknown. Present-day hashes cannot recover those identities.
- H3's 42.74/39.95/32.85 tokens/s are measured pure-decode event accounting, not full campaign throughput. Keep physical-step and request-event denominators distinct. H4 is a different campaign and cannot be added to H3.
- All 66 current safetensors shards match recorded download hashes. Pin fresh experiments to the full current manifest and record actual loaded paths.
- At P0, GPU utilization was zero and Docker empty, but only about 52 GiB system memory was available and swap was already occupied. Recheck before full-model work. Do not blindly reuse the old 0.6 memory-utilization configuration. Report resource limits instead of evicting other users' processes.

## Available runtime and data

- NVIDIA GB10, aarch64, driver `590.48.01`.
- Cached image: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`.
- Image packages: vLLM `0.19.2rc1.dev134+gfe9c3d6c5.cu130`, torch `2.11.0+cu130`, Triton `3.6.0`. Use image Python for tensor/CUDA work. The Mac Python path in paper.config.yaml is for the Mac document workflow, not the remote runtime.
- Model: `/home/mark/shared/models/qwen3.6-27b-fp8` (also `/models/qwen3.6-27b-fp8`). Weight metadata revision `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`; local config has FP8 adaptations. All file hashes are in `p0/model-weight-hashes.json` and `remote-readiness.json`.
- Available FA2: `/home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so`.
- FA2 SHA-256: `97fa2519739b3f976debb8377f8829cf3a167b410d1770bb42db390f8c5c0ae1`. CPU-only torch library load passed. This is not live GPU dispatch/numerical qualification. Set an explicit absolute `FORKED_FA2_SO` where the launcher requires it; the sparse worktree has no inherited output binary.
- `scripts/fr13_run_b34_fa2_qrow32_gqa_pair_live_gate.sh` is a relevant production-path qualification entrypoint. Inspect its current requirements before running. Do not disable refusal checks or fabricate credentials; an old pass at a different revision is insufficient.
- `p0/capture-inventory.json` lists available historical tensor files under the old repo. Start with a small relevant ladder, inspect keys/shapes safely, and establish whether pre-state/QKV/gates/topology are actually present. Available captures are not automatically complete operands or current-checkpoint evidence. Mount read-only; use `torch.load(..., weights_only=True)` when compatible. Do not load all 16 GB of the decisive archive indiscriminately.

Useful existing code includes `scripts/fr12_wy_tree_recurrence_check.py`, `scripts/fr12_wy_tree_kernel_probe.py`, `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`, and existing FR13 state/KV/conv/graph/sampler fixtures. Inspect the corrected state-write precision path; do not use the known pre-fix bug as the principal WY baseline.

## Execute in this order

1. **E7a implementation and diagnostic pilot.** Compare our sequential scan/replay, ancestor-masked triangular solve with compact reconstruction, and Bole finite-Neumann with compact reconstruction on identical rounded operands. First establish algebra against a high-precision serial oracle. Separately compare pinned native speculative update and one-token recurrent decode; do not merge references or tolerances. Include actual B1/B4 and depths 1/5/11 where supported, chain and fixed branching topology, padding/sibling reorder controls, and identical accepted prefixes. Compare verifier factors/output, then commit using identical reference factors, then each method's own combined output/state. Record error, decay ranges, scratch memory, and separate GPU verification/commit timing. Synthetic inputs may establish algebra but must be labeled synthetic; historical operands remain historical and do not count as fresh confirmation prefixes.
2. **Freeze eight diagnostic prefixes and 32 separate confirmation prefixes.** Select diagnostic inputs using native-reference behavior before reading candidate outputs, including margin/context strata. If the old captures lack required operands or provenance, use them only for debugging; collect fresh operands after model runtime/memory qualification. Use the eight-prefix pilot to measure cost/variance and freeze candidate selection, tolerances, unacceptable decision changes, timing precision target, and analysis/resampling unit before confirmation. Save IDs/seeds/manifests. These are bounded diagnostics, not a powered quality or distribution-equivalence study.
3. **E2/E7b shared continuation harness.** Reuse existing fixtures and map actual missing coverage. Isolate verify-only, commit-only, and combined substitutions with teacher-forced inputs. Measure per-layer residuals, final logits/margins/probabilities/argmax, actual sampler acceptance, and next state/logits at horizons 1/8/32. Check zero/full acceptance, siblings, pending correction token, recurrent/conv/attention KV, warm/cold/evicted/reused cache, B1/B4, graph route, and negative controls. A bounded failing diagnostic may localize error; it does not qualify deployment. Preserve known B1 cumulative-sum nondeterminism limits.
4. **E1 frozen-stack timing after qualification.** Native MTP-5, native MTP-11, sequential tree x B1/B4 x three paired independent boots = 18 cells. Balance order and share frozen inputs/context strata. If one reconstruction route qualifies, retain sequential as a control and add it as a fourth arm = 24 cells. If neither qualifies, the verified sequential baseline still proceeds. Preserve all failures/exclusions. Report physical steps, events, event-normalized costs, committed output, TPOT, and aggregate output over an explicit interval. No full-service claim from pure-decode accounting.
5. **E3 only if promoting unfinished optimizations.** Four B4 arms (baseline/scan/accept-walk/both), four matched blocks = 16 cells; correctness first. Otherwise defer. E4 broad serving, E5 broad quality, E6 NVFP4, and E7c authors-system ports remain deferred.

## Outputs and stopping conditions

Create `papers/gdn-tree-scan-mlsys/v2/experiments/STATUS.md` immediately with execution state and next step; update it at stage boundaries. Store raw artifacts under a new uniquely named output directory in this worktree. Keep manifests with source diff/hash, actual extension/JIT identities, loaded sampling source, image, flags, model files, inputs, seeds, graph route, timing boundaries, and exclusions. Keep instrumentation separate from timed runs. Record source revisions and meaningful verification for changes. Update the paper only with evidence-backed results, retaining failed mechanisms and limitations.

Start by acknowledging the handoff and inspecting the smallest reusable operand captures plus corrected local WY code. Do not launch a long full-model serving campaign as the first step. If access, memory, runtime qualification, or a provider error prevents progress, write the exact blocker and work completed into STATUS.md. Do not silently switch model or expand experimental scope.

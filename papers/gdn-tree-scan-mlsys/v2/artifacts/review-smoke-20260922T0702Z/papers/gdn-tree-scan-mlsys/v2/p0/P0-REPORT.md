# P0 audit: provenance, sampling, measurement, and experiment readiness

Audit date: 2026-09-21. Host: `mark@100.103.10.122` (`gx10-edb9`, NVIDIA GB10). **No model inference was performed for P0.** Source reconstruction, file hashing, cached-image inspection, and CPU-only library checks are distinct from live GPU qualification.

## Decision

**P0 is complete as an audit with explicit unresolved historical identity gaps. The bounded E7a captured-input arithmetic pilot can begin.** Historical gaps do not need to be invented away before collecting new measurements. Full-model E2/E7b and E1 require fresh post-boot artifact/flag records, the repository's actual qualification gates, and adequate memory headroom. They are not pre-certified by this audit.

Use the isolated worktree `/home/mark/lumo-paper-v2-20260921`, branch `codex/paper-v2-pilot-20260921`, starting at `984f613d7885d9bf7898866827552b68a3199d0c`. The pre-existing main working directory `/home/mark/shared/lumoFlyWheel` is at the older `b5bc84e49` branch with two modified historical evidence files. It was not reset, cleaned, or switched. Its shared captures are inputs, not a directory for new experiment outputs.

## 1. July H3 configuration and the sampling finding

| Arm | Recorded source revision | Requested sampling | Reconstructed target constraint passes |
| --- | --- | --- | --- |
| Native MTP-5 | `103cedccd75b3149c87b465c9201416abff7533a` | temperature 0.6, top-p 0.95, top-k 20 | 2 |
| Native MTP-11 | `2fc0faf624eb44dc1bfd43cf48e915c936b72804` | same | 2 |
| Tail6 tree | `224c5db735dae01bd5a6b4d1dd1599db31640175` | same | 2 |

The run records show B4/max-num-seqs 4, prefix caching enabled, BF16 compute, FP8 quantization, float32 recurrent-state cache, and the Qwen3.6-27B-FP8 model directory. Native arms used FLASH_ATTN, while the tree used TREE_ATTN and the recorded 21-node descriptor. KV remap and slot reorder were enabled in all three recorded environments. The tree's suffix proposal source was `merged`, so the depth-matched native chain is not a pure topology ablation. Native source revisions differ; these were not three arms of one immutable source/build snapshot.

The launcher at each recorded revision defaults to the same pinned image, which is still cached on the Spark. Its stock rejection sampler already applies constraints. Executing each historical sampler text-rewrite function against that image's stock source produces a second application in `RejectionSampler.forward`. This is a **source reconstruction**, not recovery of the removed historical containers or proof that no unrecorded override occurred.

For those target rows, the temperature divisors compound to 0.6 × 0.6 = 0.36. Both passes also include top-k/top-p filtering. Bonus and self-logit paths have separate processing; therefore the entire run must **not** be relabeled as an ordinary, uniformly sampled temperature-0.36 run. The correct label is: *requested temperature 0.6; duplicated target constraints indicated by source reconstruction; original patched-runtime identity unavailable*.

The selected revision's exact constraint-insertion block detects the stock pass and injects no second pass (`stock_apply=True`, `legacy_double=False`, `injected_extra_apply=False`). The resulting target path has one constraint call. This current check exercises that exact insertion block, not all modern patcher/deployment prerequisites. A fresh boot must confirm the complete loaded path before sampling-dependent measurements.

Evidence: `historical-run-manifests.json`, `stock-image/sampling-source.json`, the four reconstructed source files, and `../scripts/audit_p0.py`. The manifests hash the run's source ID, environment, boot log, proxy settings, and measurement JSON. Secret environment values are not copied into the manifests.

## 2. Actual available software and model identity

- Image digest: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`.
- Local image ID: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`; ARM64.
- Installed image versions: vLLM `0.19.2rc1.dev134+gfe9c3d6c5.cu130`, PyTorch `2.11.0+cu130`, Triton `3.6.0`.
- Host GPU driver: `590.48.01`.
- Available forked FA2 binary: 301,219,928 bytes; SHA-256 `97fa2519739b3f976debb8377f8829cf3a167b410d1770bb42db390f8c5c0ae1`. Exact path and modification time are in `remote-readiness.json`. Its present hash is not a recovered July runtime hash.
- Current model resolves to `/home/mark/shared/models/qwen3.6-27b-fp8`. All **66 safetensors shards**, totaling 30,866,866,928 bytes, match their recorded Hugging Face LFS hashes. Download metadata records revision `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`.
- Current config, tokenizer, tokenizer config, generation config, and index are SHA-256 pinned separately. The local config includes FP8 adaptations; do not describe the entire directory as an untouched upstream snapshot merely because the weights match.

Full records: `remote-readiness.json`, `model-weight-hashes.json`, and `fa2-load-check.json`. The FA2 check is only an ABI/library-load check without GPU exposure; it does not qualify numerical behavior or kernel dispatch.

**Historical limitations:** July boot records use `revision=None` and a mutable model path. They do not preserve each loaded FA2/JIT binary digest or a full model-file manifest. Today's hashes cannot establish July file identity. Preserve these as unknown and retain bounded archived observations; do not fabricate a historical immutable configuration.

## 3. Measurement definitions reconciled

- The paper's H3 42.74 / 39.95 / 32.85 tokens/s are committed tokens per request verification event divided by measured wall time per event. They are not total service output divided by full campaign elapsed time.
- Wall intervals join consecutive pure-decode steps, break at mixed/prefill steps, and cap idle gaps. They include step overhead within that measurement window. The interval policy must remain explicit.
- GPU forward time uses the matched draft/event denominator from the timed pure-decode steps, not the global speculative counter that also counts mixed steps.
- Physical steps and request events differ at B4. Component times must be converted using each arm's measured events/step before adding them. Recomputed totals agree with the archived JSON; committed/event equals accepted/event plus the bonus token in this instrument.
- Drafter and committer spans have the archived per-task-overlap caveat. A GPU-only subtotal is not full-service throughput, and H4's physical-step milliseconds cannot be subtracted directly from H3 event milliseconds.

## 4. Disposition of paper evidence

| Evidence | Decision | Scope retained |
| --- | --- | --- |
| H1 KV-remap fixture and served-script checks | KEEP with existing bounds | Recorded repair witness; nominal sampling and historical binary uncertainty remain; no broad quality guarantee |
| H2 shared-spine layout checks | KEEP with existing bounds | Recorded numerical/greedy witness in its tested configuration; no universal byte-equivalence claim |
| H3 three-arm July timing | RELABEL | Preserve arithmetic and observed costs; disclose duplicate constraints, different source revisions, and unresolved loaded-binary/model history |
| H4 later paired attention optimization | KEEP separate | Its own physical-step paired campaign; not a measured additive improvement to H3 or an already-qualified current build |
| H5 task-diversity analysis | KEEP | Dataset/repeated-arm limitations; not an independent new quality experiment |
| H6 WY/diffuse chronology | KEEP documentary status | Failure, repair, and later qualification; available tensor captures do not automatically upgrade this to current reproduction |

## 5. Readiness and bounded handoff

At inspection the GPU was idle and Docker had no running containers. System memory was about 117 GiB total with **52 GiB available** and about 7 GiB swap already occupied. Idle GPU utilization does not establish enough memory for the old 0.6-utilization serving configuration. Do not kill unrelated processes or blindly launch that configuration. Small captured-tensor arithmetic is suitable to start; a fresh full-model pilot must choose and record a configuration with adequate headroom or report the resource limitation.

Historical captures are available under the old repository, including the WY pre/post-fix ladders and the per-layer numerical diagnostics. `capture-inventory.json` records locations and sizes. Use these read-only and preserve their historical status. Establish operand/shape/precision lineage before using them; collect new confirmation prefixes under the pinned current configuration.

The August closeout explicitly requires current-revision qualification before the optimized production FA2 path is served. Source availability or an old credential is not a passing gate. New source or binary changes require the corresponding recheck. Do not forge credentials, disable refusal checks, or silently replace the selected reference to manufacture a pass.

Start E7a with the eight-prefix diagnostic pilot and stage-isolated arithmetic. Follow the final experiment document for the separate 32-prefix confirmation, E2/E7b continuation validation, and E1's 18/24-cell design. Large agent-quality campaigns, NVFP4 expansion, full serving sweeps, and authors-system ports remain deferred.

Claude Code `2.1.269` is installed and authenticated. A one-word Fable API health prompt received an Anthropic safeguard rejection tagged `cyber`; that failed probe remains recorded. The actual interactive task was subsequently accepted: tmux `lumo-v2-20260921` shows Fable 5.1 with xhigh effort and bypass permissions enabled, and Claude acknowledged the handoff and began reading it. Session ID: `c8955ff9-f3ae-4551-869b-1a98b38ca2c3`. This verifies the handoff, not completion of any model experiment. The observed state is recorded in `handoff-status.json`.

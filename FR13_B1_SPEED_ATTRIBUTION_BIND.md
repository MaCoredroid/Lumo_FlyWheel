# FR13 B=1 Speed Attribution Bind — per-kernel, node-trace, gate-passed

Date: 2026-06-12 UTC. Workflow `wf_c3b79cf7-3e7` (fr13-b1-speed-first), all syntheses
adversarial-verify **holds=True**. Raw: `research/fr13_workflows/speed_recon_*`,
`speed_gpu_kernel_attrib_*`, `speed_synthesis_fixplan_*` (+ INDEX.md rows).

## Verdict

The B=1 tree per-forward tax (chain5 0.3034 vs native 0.2182 s/fwd = 1.39x, 85.2 ms/fwd
by /metrics) is **~96% one bug, MEASURED**: the caterpillar drafter computes the
full-vocab bf16 lm-head logits TWICE per drafter step.

- `scripts/fr10_phase4_patch_vllm_tree_gdn.py:7288-7291` (root) and `:7401-7406` (loop)
  call `compute_logits` for top-2 packing AND `self._greedy_sample`; live vLLM
  `eagle.py:385-389` makes `_greedy_sample` RECOMPUTE `compute_logits`.
- Measured: +5.45 cuBLAS `internal::gemvx` bf16 calls/draft × 15.05 ms/call =
  **+81.94 ms/draft of the +92.25 ms/draft GPU-busy delta** (chain 356.28 vs native
  264.03 ms/draft; `output/fr13_b1_kernel_attrib/delta_per_draft.csv`,
  `window_summary.json`).
- Each call reads the lm-head weight: vocab **248,320** × 5120 bf16 = **2.543 GB** →
  ~169 GB/s effective = 62% of GB10's 273 GB/s (verify-corrected; earlier 1.556 GB /
  103 GB/s narrative was wrong, story unchanged).
- chain5 (spine-only) computes the explicit step logits and **never uses them** — pure
  waste; cat9 uses them but redundantly with `_greedy_sample`'s recompute.

## Demoted hypotheses (measured)

- **Committer blocking DtoH** (315.7 ms/draft block, 91.7% main-thread) = WAIT LOCATION
  where the slow GPU pipeline is absorbed, not the cause; true serialization residual
  ~4.7 ms/event.
- **Graph-node inflation** (28,902 extra creation events, 3.01x; localized to the
  main-model verify graph's Python conv emulation, drafter graph clusters identical):
  wall impact bounded ≤10 ms by the no-tree-GDN 1.347x discriminator. NOTE (census
  verify): the 43,270/14,368 counts are CAPTURE-TIME creations, not per-forward
  executions; the ratio logic holds, per-forward framing is inference.
- **Tree-GDN kernels net +1.7 ms** (`_tree_gdn_replay_kernel` 3.36 ≈ native GDN
  fused_sigmoid_gating 3.36 — a wash), TREE_ATTN ≈+0.3, model fp8 GEMMs 0.0 delta,
  eager-op storm ≈+8 (second-order target).
- **NATIVE pays the same lm-head class: 85.15 ms/draft = 39% of its forward** — joint
  lm-head work (fp8 / split-N) is the post-fix sub-native lever for BOTH arms (user
  stretch goal: strictly faster than native).

## Capture-method fix (why all prior kernel tables were empty)

GB10+CUDA13 hardware trace engine drops ALL kernel records at delayed-session stop
("incomplete CUPTI events dropped"; flush-interval and CuptiUseRawGpuTimestamps did NOT
fix it — the latter no longer exists in nsys 2026.2.1). **Fix: `--trace=cuda,cuda-sw`**
(software CUPTI kernel records; 364,628 chain rows, zero drops). Wired as
`LUMO_NSYS_TRACE` (+ `LUMO_NSYS_FLUSH_MS`, `LUMO_NSYS_CONFIG_DIRECTIVES`) in both
launchers, test-covered (committed dd45c3c1 + e28699cc).

## FIX-1 (next live step) + binding lossless gate

Diff (~6-10 lines, drafter-only): reuse the already-computed logits tensor —
`draft_token_ids = _fr10_step_logits.argmax(dim=-1)` (root analog for `_fr10_logits`),
delete the verified-unused root topk, skip the loop topk ONLY in spine-only mode (cat9
needs top-2 packing), branch-guard `self.use_local_argmax_reduction`. Flag-gated
`FR13_DRAFTER_SINGLE_LOGITS` default ON (OFF = exact legacy path, the A/B instrument).

Gate (5 steps, verified complete): (0) fresh PRE-change temp-0.6 capture cat9+chain5
(greedy refs already banked: `output/fr13_b1_current_gate/`,
`output/fr13_b1_chain_speed_discriminator/`); (1) within-boot same-seed repeat
byte-identity (class 8); (2) post-vs-pre served-stream byte-identity greedy AND t0.6,
both arms — this IS the proof that logits-call-1 == logits-call-2; (3) accept/event
EXACTLY unchanged; class-11 caveat: same-seed cross-boot near-tie flips exist (2.8824
vs 3.1875 chain boots) — on any mismatch, FIRST re-boot pre-change and compare
pre-vs-pre; (4) regular-decode == pristine (one no-spec probe); (5) speed verdict on
metrics-clean run, basis `decode_seconds/spec_drafts`; engagement assert ≈ 5 fewer
gemvx/draft (~11.1 → ~6) via one short `LUMO_NSYS_TRACE=cuda,cuda-sw,nvtx` capture.
Expected landing: chain5 → ~0.222-0.235 s/fwd (1.02-1.08x native); cat9 similar
(doubling is topology-independent).

## Implementation cautions (from holds=True verifies)

- `:1283 _fr10_path0_x` gather is NOT dead — consumed under `FR10_METRICS=1`
  (`:1481,:1625-1642`). Move into the gated branches; never delete (the
  diagnostics-off byte gate cannot catch that breakage).
- `:1192 _fr10_weight_f` truly dead; `:3159` self-copy verified genuinely redundant
  (`launch_tree_gdn_prepared` writes into `out=` and returns it).
- Graph-launch CPU absolutes from the node-mode capture are ~10x inflated by profiling;
  use clean-capture 0.71 vs 0.29 ms/draft as the unprofiled basis.
- Conv-fusion node-savings mapping (35% vs 75-80% of the delta) unproven — confirm by
  node-count measurement when that fix lands (wall ≤10 ms regardless).

## Fix queue after FIX-1 (re-rank against post-fix residual)

2) eager-op storm (+8 GPU ms/draft) / committer single packed DtoH; 3) conv emulation
fusion (graph-launch CPU + nodes); 4) trivials (delete :3159 self-copy, hoist
:1173/:1178 aranges, gated-move :1283); then the joint lm-head sub-native lever.

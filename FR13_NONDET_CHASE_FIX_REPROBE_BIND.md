# FR13 nondet chase: fix + re-probe BIND (2026-06-10)

## Root cause (pinned by audit + bisection, all three confirmed in source)

1. **Committer global-RNG seed** — `_lumo_tree_canonical_multidraft_sample`
   seeded its numpy rejection rng from the GLOBAL torch CUDA RNG
   (`torch.randint` without `generator=`), once per rejection_sample call,
   consumed sequentially across requests in batch order. Same-seed reruns see
   a different global offset (every prior tree event advances it) -> within-
   boot lcp=1 at temp 0.6 with NO logits change. Per-request seeds
   (sampling_metadata.generators) were bypassed entirely.
2. **Slot-keyed persistent accepted-path/lens buffers** — allocated once at
   metadata-builder init, written by the committers keyed by batch position,
   never reset per request. First tree-verify forward of a slot's new
   occupant consumed the previous occupant's path/lens (wrong state-bank
   remap + wrong h0 column -> different logits for identical input);
   persistent-batch swap-removal reorders handed rows across requests.
3. **In-place overlapping-permutation race** — `_linear_remap_rows_kernel`
   parallelized over path columns while spine paths overlap their
   destinations (src cols [1..L] -> dst cols [0..L-1]): program k-1 reads the
   bank row program k concurrently overwrites, on BOTH ssm and conv state,
   for every accepted_len >= 2. Race by construction (also a correctness
   bug); dynamically excluded at B=1 occupancy only.

## Fix (commit cc008587 on main, pushed; flag-gated, all default ON)

- `FR13_TREE_PER_REQ_GEN=1` — committer rng seeded per request from
  `sampling_metadata.generators[req_i]` (stock vLLM mechanism); legacy global
  rng with `=0`. Distribution-preserving (only the stream source moves).
- `FR13_TREE_REQKEY=1` — committers publish accepted paths/lens keyed by
  request id; new `gpu_model_runner` pre-forward rewrite installs the CURRENT
  spec-row occupants' own values (zeros for a first spec step) into the
  persistent device buffers each step; prunes finished requests. No-op
  semantically under stable occupancy beyond the first event.
- `FR13_TREE_REMAP_SEQ=1` — race-free gather-then-scatter remap kernel
  (`_linear_remap_rows_gather_kernel`): one program owns all path columns for
  an element block; all source slices in registers before any store.
- Launcher forwards all three; CPU behavioral tests in
  `tests/test_fr13_nondet_chase_fixes.py` (13 pass: RNG isolation from the
  global stream, flag-off legacy equivalence, req-keyed publish, first-event
  zeroing, prune, flag-off inertness, pristine-source patch application).

## Re-probe (ONE boot, FULL capture, BI env VLLM_BATCH_INVARIANT=1 +
## FR13_BI_TREE_ATTN=1, GPU_UTIL=0.82, FR10_METRICS=0, fixes ON,
## enforce_eager=False, FA2 fork sha 97fa2519... matches campaign)

Probe: `scripts/fr12_deliverable_swe4_probe.py`, temp 0.6 / top-p 0.95 /
seed 1313 / max-tokens 64 / warmup. Artifacts:
`output/fr13_nondet_chase_fixprobe/` (probe JSONs, compare JSONs, logs/).

| matrix | pre-fix (boot2, same methodology) | post-fix |
|---|---|---|
| B=1 t06 x2 same-seed | 0/3 identical (lcp 1/11/1) | **bit-identical 64/64** |
| B=4 t06 x2 same-seed | 0/4 identical (lcp 23/1/18/31) | **2/4 identical** (misses lcp 11, 31) |
| B=4 greedy x2 | 0/4 (lcp 1/2/21/10) | 0/4 (lcp 5/2/3/18) |

- The campaign signature (within-boot lcp=1 token-1 divergence) is GONE.
- B=1 same-seed determinism at temp 0.6: ACHIEVED (was the literal key
  deduction of the chase).
- B=4: improved but NOT deterministic. Greedy 0/4 with t06 2/4 says the
  residual channel is SMALL logit-level perturbation under different
  arrival-order/batch composition (argmax flips on any top-2 sign change;
  the t06 per-request rng draws are now fixed, so sticky distributions
  survive small perturbations). This is the batch-composition class the BI
  env does not cover on the tree stack (prefill chunking/interleaving of the
  4 SWE prompts; GDN triton prefill + tree kernels are outside vLLM's BI
  allowlist). Native (FLASH_ATTN, no tree) was 3/4-exact on the same matrix.

## accept/event (engagement asserted: 251 sampled + 205 greedy committed
## tree events; spec_decode drafts_total=428, accepted_tokens_total=839)

- sampled (canonical_multidraft): mean accepted_len 1.948 (committed/event
  2.948), 251 events.
- greedy (path-LCP-max): mean accepted_len 1.834, 205 events.
- Campaign basis ~1.9-2.1 -> UNMOVED. Determinism fixes do not close the
  acceptance gap to native 3.076: the acceptance deficit is a separate open
  front, consistent with the conv prior-window root
  ([project_fr13_conv_priorwindow_root]: conv1d_out wrong bank-row/cols at
  num_accepted>1, fixable wiring at fr10_phase4_patch_vllm_tree_gdn.py
  conv-read path) which this chase did not touch.

## Decisive next probes (not run; single-boot budget consumed)

1. Residual B=4 channel: B=4 greedy x2 with PINNED slot assignment
   (staggered sequential submission so arrival order is identical) — if it
   converges, the residual is pure arrival-order batch composition (extend
   BI coverage to the GDN prefill/tree kernels); if not, bisect per-op at
   B=4 occupancy.
2. Acceptance: land the conv prior-window read-col fix and re-measure
   accept/event against the 3.076 floor.

## Teardown
Container removed, `recover_host_memory()` run, 105G available, docker
empty.

# vLLM Upstreaming — Execution Plan (v1, 2026-08-24)

> Companion to `RFC_hybrid_tree_lossless_spec_decode.md`. All file:line references are
> against vLLM HEAD `23ab0cfdb` (2026-08-24), verified by the five-checker recheck.
> Our pin was `fe9c3d6c5` (0.19.2rc1.dev134); 4,332 commits behind — every anchor here
> was re-verified at HEAD, not carried from the pin.

## Objectives (what this plan is optimizing)

1. **vLLM**: land the recurrent-state tree-verification discipline while that surface
   is still forming; smaller correctness fixes as standalone value.
2. **Resume**: merged PRs in vllm-project/vllm + a shepherded RFC — front-loaded in
   Tier 1–2, does not require Tier 3.
3. **Paper**: arXiv v1 is independent of upstreaming; "under RFC discussion" is a
   bonus sentence, not a dependency.
4. **Constraints**: one GB10, metered credit → Tier 1 is writing/review-dominated,
   ~zero GPU. Mark personally reviews every line before anything leaves the fork
   (vLLM AI-disclosure policy: no pure-agent PRs).

## Tier structure and gates

| tier | contents | calendar | gate to next |
|---|---|---|---|
| 1 | 4 small PRs + 1 issue + RFC filed + 3 decision spikes | 4–8 wk (review-latency-bound) | a shepherd/DRI engages the RFC, or a maintainer signals appetite |
| 2 | RFC phase 0 (state interfaces) + phase 1 (tree mask + equivalence gate) | 2–3 mo | phase 0–1 merged; co-review relationship with GDN kernel authors working |
| 3 | RFC phases 2–3 (proposer composition; walk + lossless commit) | 6–12 mo | decided only at gate 2, never earlier |

---

## Tier 1, detailed

### PR-1 `[Bugfix]` Uniform-decode dispatch guard (V1 model runner)  — *lead PR*

- **Bug**: V1's cudagraph uniform-decode dispatch can select a decode-shaped graph
  while a prompt chunk is in the batch; on hybrid-GDN models this leaves stale
  `spec_state_indices` → `EngineDeadError`. Live at HEAD: `gpu_model_runner.py:3983–4002`
  is byte-identical to our pin (git -L: one commit ever).
- **Fix shape**: `has_prefill = (num_computed_tokens_cpu[:n] < num_prompt_tokens[:n]).any()`
  — no new plumbing. ~30–80 LOC + tests.
- **The argument is pre-won**: #51865 (Nick Hill et al., 08-11) fixed the identical bug
  in Model Runner V2 with the docstring "a prompt chunk can have a decode batch's
  shape" (`worker/utils.py:659–673`). Our PR = "port the accepted fix to V1, where
  hybrid models still run" (hybrids are excluded from the V2 default,
  `config/vllm.py:736–738`). Cite #51865/#51917; the two authors are natural reviewers.
- **Tests**: extend the existing CPU-only suite at
  `tests/v1/worker/test_gpu_model_runner.py:1588–1646`; attach the GDN reproducer.
- **Pre-empt in the PR body**: "why not just use MRV2?" → hybrid excluded from V2
  default; MRV2 documented experimental.
- **Estimate**: 0.5–1.5 wk. Mark review budget: ~2 h.

### PR-2 `[Bugfix][Quantization]` Quant-config for Qwen3-Next lm_head (C-residual)

- Upstream landed our Unit C's design in May (#42124/#42546, NVIDIA). Residual: 32 of
  139 `ParallelLMHead(...)` sites still omit `quant_config`; the two that matter to us:
  `qwen3_next.py:765`, `qwen3_next_mtp.py:205` (both already have `self.quant_config`
  in scope — two one-line changes).
- **Tests**: clone the pattern from upstream's own `test_nemotron_h_quantization.py`.
- **Estimate**: 0.5 wk. Mark review: ~30 min.

### Issue-1 + PR-3 `[Bugfix]` head_dtype × ModelOpt interaction bug — *our newest finding*

- `logits_processor.py:144–148` requires `UnquantizedEmbeddingMethod`, but post-#42124
  any ModelOpt config — even one excluding lm_head — returns `UnquantizedLinearMethod`
  (`modelopt.py:186`, `:2421`). Fires with `head_dtype != dtype` (#48390) or any pooling
  runner. A genuine interaction bug between two of their own merged PRs.
- **Play**: file the issue with the analysis (costs nothing, pure credibility), attach
  the ~5-line fix as PR-3.
- **Estimate**: 0.5 wk combined. Mark review: ~30 min.

### PR-4 `[Core]` Plural `proposal_methods` in SpeculatorsConfig (F-PR1)

- `configs/speculators/base.py:117–127` truncates to `proposal_methods[0]` with the
  comment "Currently we only support one proposal method" — the schema is already
  plural on disk; vLLM discards the rest. Teach the config to accept >1 with
  validation (no runner changes). This opens the drafter-composition conversation
  (RFC phase 2) at a spot upstream has annotated as a known limitation.
- **Estimate**: 0.5–1 wk. Mark review: ~1 h.

### PR-5 `[Kernel]` Tree visibility mask via FA4 `mask_mod` (E-a) — *the RFC's demo*

- FA4's `mask_mod` (`flash_attn_interface.py:212–216`, two shipped exemplars at
  `flash_attn.py:1427, 1535`) expresses a tree visibility mask in ~150–300 LOC of
  Python/CuteDSL — no vllm-flash-attention fork PR. Validate against a chain-shaped
  tree (mask == causal ⇒ outputs identical): that equivalence test IS the deliverable.
- **Caveat, stated in the PR**: FA4 gates to capability families 9/10/11 — sm_121
  (GB10) stays on FA2, so this is upstream-value now, GB10-value later. It is also
  the concrete exhibit for RFC phase 1.
- **Estimate**: 2–4 wk. Mark review: ~3 h. *May slip to early Tier 2 if review
  bandwidth is tight — it is the only Tier-1 item over 1.5 wk.*

### RFC filing (parallel, week 2)

1. Mark approves the draft (this branch) →
2. file as issue on vllm-project/vllm via the RFC template →
3. post to `#contributors` on vLLM Slack →
4. CC: the spec-decode area owner (author of the #42121 tree removal — engage
   directly, our design answers his deletion rationale), and the NVIDIA authors of
   #51674 / #52539 / #51855 (active on our exact surface; collision risk today,
   co-reviewers tomorrow — **contact them BEFORE filing**, see spike 3).

### Decision spikes (≤2 days each, parallel, week 1)

| spike | question | gates |
|---|---|---|
| S1 V1-vs-MRV2 | which runner do phases 0+ target? (36/68 of our patch functions touch `gpu_model_runner.py`, which upstream is migrating off; new spec-decode is V2-only, cache drafters V1-only) | RFC phase plan wording; F-PR2; A/B translation strategy |
| S2 GB10 kernel selection | does the tuned-JSON path (Triton, 7th of 8 in dispatch) even run on sm_121, or does CUTLASS sm120-blockwise win? | G2 go/no-go (currently gated) |
| S3 NVIDIA contact | open the line to #51674/#52539/#51855 authors re: trees on GDN | RFC CC list; G4 framing (now "generalize the RecoverSSM hook", not "cdiv fix") |

### Struck / gated (do not work these)

- **G3** (tree-attn cudagraph): struck — target deleted upstream (#42121).
- **D top-3 half**: no consumer at HEAD. D-argmax (3–5 wk) waits for Tier 2; its
  tie-break blocker is dissolved by #52816 (in-tree top-k documents *our* order:
  score desc, index asc).
- **G2**: gated on S2. **G4**: gated on S3 conversation.
- **E-b** (FA2 tree-bias fork for GB10): private deployment artifact, never upstream.
- **`fr14_patch_nvfp4_lmhead.py`**: dead at HEAD (0/3 anchors). Retire in our repo;
  do not rebase.

## Tier 2 sketch (activated only at gate 1)

- **Phase 0**: tree-capable interfaces on the recurrent spec-state context
  (`MambaSpecDecodeGPUContext` + the `mamba_cache_mode=align` path, which already
  overlaps our commit design): per-node parent indexing, carry-slot budget declaration,
  accepted-path replay hook. All no-ops for chains. Interface + tests only.
- **Phase 1**: the E-a mask generalized + the bit-level equivalence gate
  (chain-shaped tree == native, greedy byte-exact).
- Both target the substrate chosen in S1 (expected: MRV2).

## Budgets and cadence

- **Mark's hours, Tier 1**: ~8–12 h total review + ~2–3 h/wk on the RFC thread while
  it is active. Everything else is drafted here and reviewed on fork-internal PRs.
- **GPU**: ~0 for Tier 1 (S2 spike needs ~1–2 h of serve time; PR-1's reproducer can
  be captured from an existing corpse).
- **Cadence rule**: never more than 2 upstream PRs in flight at once (review-quality
  over throughput; their reviewers respond in 2–3 days, ping at 7).
- **Process invariants**: DCO sign-off on every commit (`git commit -s`); AI-assist
  disclosure in every PR body; internal fork-PR review by Mark before anything goes
  to vllm-project; no PR carries campaign machinery.

## Risk register

| risk | likelihood | mitigation |
|---|---|---|
| RFC gets no shepherd | real | sunk cost = 1–2 wk of writing; Tier-1 PRs + paper unaffected; retry after more PRs merge |
| NVIDIA ships trees first | moderate | S3 contact converts collision → co-review; arXiv timestamp protects the ideas |
| upstream churn invalidates an in-flight PR | proven (we watched 4,332 commits do it) | small PRs, short-lived branches, re-verify anchors at submission day |
| review latency stalls momentum | high | pipeline: while PR-n is in review, PR-n+1 is in internal review; RFC thread runs in parallel |
| Mark review bandwidth | the true scarce resource | every PR sized ≤2 h of his review except PR-5 (~3 h); PR-5 may slip to Tier 2 |

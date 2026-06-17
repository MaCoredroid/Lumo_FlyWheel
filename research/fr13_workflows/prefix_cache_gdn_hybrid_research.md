# Prefix caching for Qwen3-Next GDN-hybrid: why it's off, how to enable, and a safe lossless gate

> Scope: Qwen3-Next-27B fp8 GDN-hybrid, served on the installed vLLM build **`0.19.2rc1.dev134+gfe9c3d6c5`** (PR [#40092](https://github.com/vllm-project/vllm/pull/40092), late-Apr-2026 main; verified live: `docker exec fr13-bigdenom-cat555_b1 python3 -c "import vllm; print(vllm.__version__)"` → `0.19.2rc1.dev134+gfe9c3d6c5`), on a GB10 (unified mem, decode HBM-bound). Lever: prefix caching is OFF, so each codex+SWE-Verified turn re-prefills its append-only ~11–14k context (OBSERVED `prefix_cache_queries_total=0`, `prompt_tokens_cached_total=0`). This doc is design + verdict only — no GPU op was run.

---

## 1. WHY it's off today (the exact vLLM mechanism + correctness reason)

**It is a SOFT DEFAULT, not a hard raise.** Prefix caching for this hybrid is disabled by configuration default, not refused by a guard. Two independent mechanisms make "off" the current state:

- **The model class does not advertise mamba-prefix-caching support.** `Qwen3NextForCausalLM` bases are `(HasInnerState, SupportsLoRA, SupportsPP, QwenNextMixtureOfExperts, IsHybrid)` — it does **not** declare `SupportsMambaPrefixCaching` (grep count = 0). With prefix caching *disabled*, `config.py` then forces the mode to `none` (verified live):
  ```python
  # vllm/model_executor/models/config.py:375-378
  else:  # not enable_prefix_caching
      if cache_config.mamba_cache_mode != "none":
          cache_config.mamba_cache_mode = "none"
          logger.warning("Mamba cache mode is set to 'none' when prefix caching is disabled")
  ```
  So today `mamba_cache_mode == "none"` → no checkpointing → every turn re-prefills.

- **The only *hard* raise is for `all` mode, which is unrelated to "off".** Qwen3-Next refuses `all`-mode caching outright (verified live at `qwen3_next.py:1161-1165`):
  ```python
  if cache_config.mamba_cache_mode == "all":
      raise NotImplementedError(
          "Qwen3Next currently does not support 'all' prefix caching, "
          "please use '--mamba-cache-mode=align' instead")
  ```
  This is a guard against the *wrong* mode, not the reason caching is off. The actual "off" is the `none` default above.

**The correctness reason caching is non-trivial here (KV-block vs recurrent-state mismatch).** Automatic prefix caching reuses **attention KV blocks**, which are block-addressable and position-stable. The GDN/mamba layers instead carry a **single per-sequence recurrent SSM state + a width-W rolling conv1d window** (`MambaSpec`), which is **not block-addressable** — it is one rolling buffer that only exists *after* the forward sweep reaches a given position. A prefix-cache hit skips re-running that sweep, so the recurrent + conv state at the prefix boundary must be *reconstructed from a checkpoint* rather than recomputed. If that checkpoint is even 1 bf16-ULP off the fresh-prefill state, the error compounds across the ~48 GDN layers (gate `1/rms` amplification, deep full-attn) until argmax flips — the same diffuse-accumulation carrier we already characterized internally (`reference_diffuse_gdn_accumulation_explained`). Research framing: Marconi, [arXiv 2411.19379](https://arxiv.org/html/2411.19379v1), on why SSM-state prefix caching is hard. This is exactly why vLLM keeps hybrid APC behind a non-default, "experimental"-logged mode.

---

## 2. HOW it could be enabled (state of the art + concrete path for our version)

**State of the art (merged into mainline, and into OUR build).** Our pin `gfe9c3d6c5` is downstream of the entire mamba-APC line. Tracking issue [#26201](https://github.com/vllm-project/vllm/issues/26201) enumerates:

| PR / Issue | What | Status | In our build? |
|---|---|---|---|
| [#25752](https://github.com/vllm-project/vllm/pull/25752) | Mamba2 APC (parent) | merged 2025-10-04 | yes |
| [#30877](https://github.com/vllm-project/vllm/pull/30877) | "Mamba Prefix Caching with **align** mode" — explicitly covers **GDN** + ShortConv + LinearAttention + Mamba1/2, "without modifications to the underlying kernel code"; validated on Qwen3-Next-80B-A3B (~2× throughput) | merged 2026-01-23 | yes |
| [#33705](https://github.com/vllm-project/vllm/pull/33705) | "Enable spec decoding in mamba cache **align** mode" (removes the temporary spec-disable from #30877, re-enables `test_mamba_prefix_cache.py`) | merged 2026-02-13 | yes |
| [#26807](https://github.com/vllm-project/vllm/pull/26807) | GatedDeltaNet **all**-mode APC | **OPEN** | n/a (Qwen3-Next rejects `all` anyway) |
| [#42406](https://github.com/vllm-project/vllm/pull/42406) / [#42792](https://github.com/vllm-project/vllm/pull/42792) | Model-Runner-V2 copy-free align prefix cache (+ spec variant) | **OPEN**, marginal gain (3.09 vs 2.90 req/s) | not merged — not a near-term lever |

So **no upgrade and no cherry-pick is required**: the GDN align-mode path and the spec+align re-enable are already present in `gfe9c3d6c5`.

**The minimal enable path for our pinned version:**
- Pass **`--enable-prefix-caching`**. Because we also run spec-decode, `config.py` *auto-forces* `align` (verified live, `config.py:326-360`):
  ```python
  # config.py:326-337
  if cache_config.enable_prefix_caching:
      if cache_config.mamba_cache_mode == "none":
          if (model_config.supports_mamba_prefix_caching
                  and vllm_config.speculative_config is not None):
              cache_config.mamba_cache_mode = "align"   # ← our case (with spec)
          else:
              cache_config.mamba_cache_mode = ("all" if model_config.supports_mamba_prefix_caching else "align")
      if cache_config.mamba_cache_mode == "all" and not model_config.supports_mamba_prefix_caching:
          cache_config.mamba_cache_mode = "align"        # ← also lands here (no SupportsMambaPrefixCaching)
      if cache_config.mamba_cache_mode == "align":
          assert vllm_config.scheduler_config.enable_chunked_prefill, \
              "Chunked prefill is required for mamba cache mode 'align'."
  ```
  Two things follow. First, Qwen3-Next reaches `align` by *both* branches (it has neither `SupportsMambaPrefixCaching` nor spec-clean `all`-support), so `align` is the **only** reachable mode — you cannot accidentally select `all`. Second, **`align` hard-asserts `enable_chunked_prefill`** — so turning on prefix caching turns on chunked prefill, a **second behavioral change** vs our current deploy that must be controlled in the A/B (see §3, §4).

**Documented caveats baked into the installed source (all verified live unless noted):**
- **Experimental, logged as such**: `config.py` emits *"Prefix caching in Mamba cache 'align' mode … Its support for Mamba layers is experimental. Please report any issues you may observe."*
- **`mamba_block_size` must be a multiple of 8** to align the `causal_conv1d` kernel (`cache.py:108-111`, verified: *"Value must be a multiple of 8 to align with causal_conv1d kernel."*). Defaults to `block_size` when unset (`config.py:373-374`). The Mamba2 parent ([#25752](https://github.com/vllm-project/vllm/pull/25752)) additionally noted multiple-of-256 (chunk granularity) for `all`-mode — informative but not binding for us.
- **`block_size ≤ max_num_batched_tokens`** asserted in align mode (`config/vllm.py:1865-1879`).
- **Silent 0%-hit efficiency trap**: align mode keeps only the checkpoint at the **last block boundary**; reuse happens only when `floor((prompt_len-1)/block_size)*block_size ≤ shared_prefix_len`, else hits silently drop to 0% ([#45238](https://github.com/vllm-project/vllm/issues/45238): 52/64→0/64 hits, TTFT 433→905 ms; [#40696](https://github.com/vllm-project/vllm/issues/40696): prompts `< block_size 528` get 0% reuse; [#36697](https://github.com/vllm-project/vllm/issues/36697): block_size can even exceed `max_num_batched_tokens`). Our append-only ~11–14k context is long enough to clear typical floors, **but block_size / TP must be tuned and the hit-rate must be measured, not assumed** — this is the cache-hit guard in §4.
- **Cold-vs-warm dtype divergence**: [#26807](https://github.com/vllm-project/vllm/pull/26807) reports bf16 cache produces *slightly different* output cold vs cache-hit for the same prompt; **float32 cache dtype resolves it**. Our build exposes `--mamba-cache-dtype` / `--mamba-ssm-cache-dtype` (default `auto`, `cache.py:112-128`) to force fp32 state cache if our lossless gate needs it. (all-mode finding; treat as a hypothesis for align-mode under our gate, not a fact.)

**The one OPEN red flag that makes this "measure, don't assume":** [#43559](https://github.com/vllm-project/vllm/issues/43559) — observed on vLLM 0.21.0 — a **~20% accuracy drop when `--enable-prefix-caching` is combined with MTP speculative decoding on a Qwen3.6 GDN-hybrid**, i.e. *our exact feature combination*. Each feature alone is clean; only the **APC+MTP-spec** combination regresses. Status OPEN/"in progress"; candidate fix PR [#45477](https://github.com/vllm-project/vllm/pull/45477) referenced, no confirmed fix. Our pin (`gfe9c3d6c5`, late-Apr-2026) is **behind** the 0.21.0 build where this was seen and behind #45477 — so the bug may be present, absent, or different in our exact commit. **This is unresolved until measured on our build.**

---

## 3. Interaction with OUR FR13 tree spec-decode stack

Enabling APC touches **three** FR13 assumptions, two of which are already-proven flip carriers. All citations are to our patcher and verified live.

**(A) GDN recurrent-state base assumes fresh prefill this turn — the most load-bearing collision.** Our depth-position remap computes:
- recurrent-state base = `max(0, int(num_computed_tokens_cpu[req_idx]) - 1)`
- MRoPE/depth base = `int(num_computed_tokens_cpu[req_idx])`
- logged contract (`fr10_phase4_patch_vllm_tree_gdn.py:9846`, verified): `"base_contract": "state=num_computed_tokens_cpu-1,mrope=num_computed_tokens_cpu"` (state base read at `:9742/:9746`).

With APC ON, `num_computed_tokens` **jumps by the cached-prefix length on the first scheduled step** (cached blocks count as already-computed) *without the GDN/conv forward having been run this turn*. The `-1` state base then indexes a **block-boundary checkpoint**, which must be byte-exact to the fresh-prefill state or every downstream GDN layer diverges. **Must re-verify** that the checkpoint our base now points to is the byte-exact state, not a `wip`-rounded one (`mamba_attn.py:223-237`, chunk-alignment rounding labelled "wip").

**(B) Conv prior-window committed-path snapshot has no checkpoint-reconstruction path — already a proven carrier.** `FR13_CONV_COMMITTED_PATH` (baked ON, `:825/:1042/:2123/:2215-2229`) snapshots the prior conv window from the accepted leaf's bank row **before** the in-place remap mutates the bank, indexed by `spec_state_indices_tensor` (the per-sequence GDN bank) — **not** by any attention block table (`:2100-2123`, `:1556`). The conv prior-window is a single per-seq rolling buffer; there is no block-addressable conv state. This is precisely the FR13 flip carrier already root-caused in `project_fr13_conv_priorwindow_root` (conv1d_out diverged **18.375** at `num_accepted>1` when the wrong bank-row/cols were read). With APC, the conv width-W window at the boundary must be the **exact bytes adjacent to the prefix boundary**; if align-mode reconstruction supplies a checkpoint-derived window, `causal_conv1d_update` reads a different window. **Must re-verify** conv1d_out row-0 = 0.0 under APC, the same decisive seam from the conv-priorwindow closeout.

**(C) Committer `num_computed_tokens` / `seq_lens` / block-table ownership under spec.** The mamba backend disables block-table updates when spec is active: `supports_update_block_table = True` *"Will be disabled if speculative decoding is used"* (`mamba_attn.py:82`, verified) and the `all`-mode buffer carries *"Speculative decoding not supported with prefix caching, so keep shape consistent with prefill buffer"* (`mamba_attn.py:114`, verified). align mode adds `num_accepted_tokens`-driven `postprocess_mamba` state copies (per Agent A/B; `gpu_model_runner.py` region). **Must re-verify** that our committed-path bank-row gather reads the correct per-sequence rows when the cache manager owns block assignment, and that our depth-remap assertion "spec rows have exactly `_fr10_tree_n` scheduled tokens" survives chunked-prefill splitting of a partially-cached prefix (our assert region near `:9748`).

**Upstream precedent that this combination is freshly wired and was crash-prone:** vLLM [#39809](https://github.com/vllm-project/vllm/issues/39809) — mamba-prefix-cache + MTP-spec crashed at startup (buffer-size mismatch, token-vs-request slicing, `selective_state_update` illegal memory access) and recurrent/conv reconstruction at boundaries got **no special handling when spec tokens are rejected**. #33705 re-enabled the combination, but #43559 shows it is **not yet proven safe**.

---

## 4. SAFE lossless measurement design (reuse our infra — do NOT reinvent)

A **same-boot, same-prompt A/B**: arm A = prefix-cache **ON** (align + chunked-prefill), arm B = prefix-cache **OFF** (current deploy). Same-boot only — never cross-boot (`feedback_no_cross_boot_byte_gate`: fresh B=1 forks at the autotune floor). Run both arms in ONE boot; reset between arms.

**Compare-target rule (binding).** Ground truth is the **NO-spec RECURRENT decode oracle** — `scripts/fr13_recurrent_decode_oracle.py`, which loads the model once offline with **no `speculative_config`** → pure no-spec recurrent path, and is intrinsically cache-OFF (it *forces* `mamba_cache_mode=none` because Qwen3-Next lacks `SupportsMambaPrefixCaching`; docstring L11). It fail-louds class-9 if zero recurrent decode calls fired (*"recurrent decode path did NOT fire — vacuous oracle"*). **Cache-ON is lossless iff its flip-vs-this-cache-OFF-oracle matches native E5's flip-vs-this-oracle within the native floor.** Never a serial-torch ref / chunked-prefill / fallback / backend-NAME proxy (`feedback_fr13_lossless_compare_target`). Depth-match: cat9 = depth-5 → **E5**; a depth-3 arm → E3 (`feedback_depth_matched_accept_compare`).

**Gate 0 — cache-HIT guard (makes the whole test non-vacuous; run FIRST).** Scrape `/metrics` and require **all three** to be `> 0`: `prefix_cache_queries_total`, prefix-cache `hits`, `prompt_tokens_cached_total` (currently **all 0**, OBSERVED). `fr13_gold_margin_probe.py:189-203` (`_metrics_excerpt`) already scrapes `/metrics` — extend it. If the cache silently 0-hits ([#45238](https://github.com/vllm-project/vllm/issues/45238)/[#40696](https://github.com/vllm-project/vllm/issues/40696) trap), tune `mamba_block_size` (multiple of 8) / TP against the [#45238](https://github.com/vllm-project/vllm/issues/45238) boundary formula **before** trusting any lossless verdict — a 0-hit arm A is identical to arm B and passes vacuously.

**Gate 1 — greedy byte-identity (in-process same-boot).** temp=0, top_p=1, seed=1313, `fr10_decode_mode=tree_mtp`. Capture both arms with `fr13_gold_margin_probe.py capture` (calls `/reset_prefix_cache` at L115 for the cache-OFF arm; warm the cache-ON arm with a prior identical request), assert `served_token_ids` are **byte-identical** (use the `within_boot_det_rep1_eq_rep2` determinism check). **int-view equality, never atol** (`feedback_fr13_lossless_compare_target`).

**Gate 2 — per-token argmax clear-margin flip-rate (binding statistical gate).** `scripts/fr13_b1_lossless_prescore.sh` with an added `cat9_cacheON` arm: N=40 turns → `fr13_swe_stream_to_oracle_src.py` → `fr13_recur_rescore_in_container.sh` (SEED=1313 TOPK=20 THRESH=1.0). The cache-ON **clear-margin** flip-rate (per-token argmax vs the no-spec recurrent oracle, THRESH=1.0 nat) must be **statistically within the native E5 floor** (Wilson-CI overlap; cache-ON CI not above native). A 1-ULP state-reconstruction error compounds across ~48 GDN layers → flip-rate above floor. This is the instrument that scalar metrics are blind to (`reference_scalar_metric_per_token_blindspot`). Per-fork drill-down with `fr13_gold_margin_probe.py reduce --tree cacheON.json --native cacheOFF.json --threshold 1.0`: a **clear-margin** fork = the cached state reconstruction changed the argmax = NOT safe; a **near-tie** fork = within floor.

**Gate 3 — temp-0.6 distributional drift (catches sub-argmax shift Gate 2 misses).** `scripts/fr13_temp06_drift_estimate.py` + `fr13_temp06_drift_gate_workflow.js`: per-position bag-TV between the cache-ON verify dist and the no-spec recurrent dist, vs the **multi-seed native floor** `BAG_TV_FLOOR` (p95 of C(N,2) native-vs-native draws — **upgrade the 0.0593 single-draw constant to N=6–8 p95** before hard-thresholding, per the script's own caveat). cache-ON bag-TV must be **within** the native temp-0.6 floor. This catches a 1-ULP error that shifts the *sampled* token while leaving the top token unchanged.

**Bug-class traps to avoid (FR13 banked instruments):**
- **Vacuous cache** → Gate 0 (`/metrics` all `> 0`); the silent-0% align trap is real ([#45238](https://github.com/vllm-project/vllm/issues/45238)) and makes arm A == arm B.
- **Cross-boot floor** → same-boot only; cross-boot fresh forks at the autotune floor are NOT a behavior change (`feedback_no_cross_boot_byte_gate`).
- **int-view-not-atol** → Gate 1 is exact byte equality; never grant per-stage tolerances (`feedback_math_correct_vs_bitexact`).
- **Confound: chunked-prefill is a second variable.** align mode forces `enable_chunked_prefill` (`config.py:358-360`). Run a **control arm** = cache-OFF + chunked-prefill-ON so any divergence is attributed to the cache reconstruction, not to chunked-prefill splitting the tree-depth-position logic (our `_fr10_tree_n`-scheduled-tokens assert near `:9748` could be violated by a partially-cached chunked step).
- **Scalar blindspot** → the binding instrument is the per-token argmax probe (Gate 2), not accept/event or pass-rate (`reference_scalar_metric_per_token_blindspot`).

---

## 5. Verdict + cost

**YELLOW.** The enable *path exists and is already in our build* (no upgrade, no cherry-pick: `--enable-prefix-caching` → auto `align` + chunked-prefill; #30877/#33705 present). The deploy upside is large (eliminate ~11–14k re-prefill per codex turn; #30877 saw ~2× throughput on the 80B sibling). But it is **not GREEN**: there is a direct, *open, unresolved* upstream correctness regression for **exactly our APC+MTP-spec combination** ([#43559](https://github.com/vllm-project/vllm/issues/43559), ~20% accuracy drop, no confirmed fix, observed on a build *ahead* of ours), and our FR13 stack has **two already-proven flip carriers** (conv prior-window, GDN recurrent-state base) whose fresh-prefill assumption APC violates by construction (§3). It is **not RED**: nothing in our pinned source hard-blocks the align path, the spec+align plumbing is wired (`mamba_attn.py` spec branches, `gdn_attn.py` `num_accepted_tokens` path), and our existing instruments fully cover proving or disproving losslessness without reinvention.

**Single biggest risk:** the align-mode block-boundary checkpoint for the GDN recurrent + conv state is **not byte-exact** to the fresh-prefill state (vLLM's own [#26807](https://github.com/vllm-project/vllm/pull/26807) bf16 cold-vs-warm divergence + the [#43559](https://github.com/vllm-project/vllm/issues/43559) APC+MTP regression both point here). A sub-ULP checkpoint error compounds across ~48 GDN layers into argmax flips above the native floor — i.e. **lossy**. Mitigation lever to try if Gate 2/3 fail: `--mamba-ssm-cache-dtype float32` (per #26807), at a unified-memory cost on the GB10 budget.

**Sequencing:** Gate 0 (cache-hit, block_size tune) **before** any lossless claim — a 0-hit arm passes vacuously and wastes the GPU slot.

**Rough GPU cost of the measurement:** one same-boot serve (align + chunked-prefill) + 3 arms (cache-OFF, cache-OFF+chunked control, cache-ON), each N=40 turns through the existing `fr13_b1_lossless_prescore.sh` battery + the temp-0.6 drift estimate (banked top-20 rescore is CPU, no extra GPU). On the order of **a single B=1 serve-boot's worth of GPU time (~30 min to a couple hours)**, comparable to one `fr13_b1_lossless_prescore` run — cheap enough to schedule **when the speed sweep frees the serialized GPU**, not worth pre-empting it. Net: a plausibly-cheap, *potentially* lossless path that is **measurement-gated, not assumption-gated** — measure on our exact `gfe9c3d6c5` commit before believing #43559 applies or doesn't.
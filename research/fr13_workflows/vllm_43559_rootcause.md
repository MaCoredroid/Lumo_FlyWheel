# vLLM #43559 (APC + MTP-spec GDN accuracy drop) — root-cause analysis + draft contribution

Workflow `wa8xh6j5z` (7 agents, source-verified) + adversarial red-team, 2026-06-19. This is the
RED-TEAM-CORRECTED version: the raw synthesis overclaimed on three points (flagged below); the
corrected position + a hedged draft comment are what we'd actually consider contributing. **Nothing
posted to GitHub.** Branch fr13-prefix-cache.

## Verified facts (live GitHub, 2026-06-19)
- **#43559 OPEN**, label `bug`, created 2026-05-25, last activity 2026-06-09, 9 comments. Title:
  *"Accuracy drops ~20% when --enable-prefix-caching is used together with MTP speculative decoding
  (Qwen3.6 35B-A3B)."* Reporter used a **Qwen3.6 35B-A3B MoE SFT** checkpoint + **forced cache hits
  via a shared system prefix**. → same GDN-hybrid+MTP *architecture family / feature combo* as ours,
  **NOT our literal Qwen3-Next-27B weights** (correction to earlier "our exact combo" framing).
- **All fix PRs OPEN/unmerged:** #43650 (drop-final-mamba-block), #45477 (keep-chunks-aligned),
  #45614 (clamp EAGLE hit length), #40738 (offset-based GDN conv/SSM ngram fix), #26807 (fp32 SSM).
- **Only #45473 MERGED (2026-06-16):** DS-layout Mamba tail-copy for MTP align. Does NOT close
  #43559; does NOT touch the **SD** conv path our build uses. (Red-team: it is "DS layout", NOT
  "NVFP4 Nemotron" — that qualifier in the raw synthesis was unsupported; struck.)
- **Source verified verbatim at our pin gfe9c3d6c5** (the strongest finding): in
  `vllm/model_executor/layers/mamba/mamba_utils.py`, `get_conv_copy_spec` reconstructs the conv
  window by a **position-shift** `state[src_block_id, offset:]` (offset = num_accepted−1) from the
  *base speculative* block, while `get_temporal_copy_spec` takes a **whole-row snapshot** of the
  *accepted* block. This asymmetry is real and is the precise (b) fix target.

## Two-carrier decomposition (with red-team scoping)
**(a) Block-misalignment cache POISONING — the thread's actual story. Fixable scheduler WIRING. APC-induced.**
For a prompt < 2× mamba block size, the EAGLE prune zeroes `last_cache_position`, the align branch
in `scheduler.py:_mamba_block_aligned_split` is skipped, and under **concurrent** prefills a chunk
ends unaligned → mid-block state hashed as a boundary snapshot → later requests lose ~a block of
context → runaway gens / stray `</think>` / malformed tool calls. Fires only under
**short-prompt + concurrency + APC**; absent with APC off. #45477 (end-position rounding) is the
right fix; #43650/#45614 are complementary surfaces. **This plausibly explains 100% of the
thread's 20%** (failure shape + zack041's "retains the final page" localization + temp-0.1 forced-
shared-prefix eval = classic context-loss signature).

**(b) Conv-window reconstruction non-invertibility — the `get_conv_copy_spec` offset-shift. OUR proven carrier (tree-spec); a LATENT hypothesis for #43559's native+CUDA case.**
conv1d+SiLU is non-invertible, so shifting position within an already-advanced (post-all-drafts)
window does not reconstruct the true K−1 history after step k. **Our PROVEN evidence
(`project_fr13_conv_priorwindow_root`, dual-verified):** at a num_accepted=6 boundary, GDN L0,
`conv1d_out` diverged 18.375 while the recurrent `h0` SSM state was BYTE-EXACT (~1e-7) → carrier is
the conv window, not the SSM state. **Scope (red-team, load-bearing):** this was measured on **our
cat9 TREE-spec committer** (branched, num_accepted=6), NOT native MTP+APC (linear chain). #43559 is
native MTP. So (b)'s presence/magnitude on their stack is **unestablished**.

### CRITICAL correction (red-team) — SGLang #25587 is NPU-specific
The raw synthesis cited SGLang #25587 as proof that carrier (b) is "spec-decode-intrinsic, present
with or without APC" on CUDA. **Wrong.** #25587 is *"Hybrid-GDN MTP not lossless **on Ascend NPU**"*
and its own body says **"On NVIDIA the two are identical; on NPU they diverge"** — i.e. NVIDIA's
`causal_conv1d_update` **already implements the snapshot path**. So on our CUDA stack the conv
*kernel* likely snapshots correctly; the open question is only whether the **APC cache-boundary
copy-spec** (`get_conv_copy_spec` offset-shift) reintroduces the defect at num_accepted>1. We cannot
claim (b) is intrinsic on CUDA. This is why the **native-E5+APC arm** of our A/B is the decisive
experiment: it tests whether #43559's carrier reproduces with native MTP + APC on CUDA.

## Does #45477 fully fix #43559? Likely yes for the thread's repro; (b) is a separate latent risk.
#45477 fixes admission (which chunk-end gets hashed); it does not touch the `offset:` conv copy. If
the 20% is all (a), #45477 closes it. Our **long-prompt config dodges (a)'s trigger entirely**
(prompts ≫ 2× block) — so #45477 wouldn't change our exposure; the only thing that could bite us is
(b), which the rescore will measure.

## Concrete fix direction (validates the user's "reference SGLang" steer — for carrier (b))
Function: `get_conv_copy_spec` (mamba_utils.py, present at gfe9c3d6c5). Fix = **snapshot-not-
reconstruct** (SGLang/NVIDIA #10335):
- **Cheap interim (our `FR13_CONV_COMMITTED_PATH` already ships this):** conv copy reads the **full
  accepted-node row** (`block_ids[cur_block_idx + num_accepted − 1]`, mirroring the temporal copy)
  instead of `offset:`-slicing the base row → drove our multi-accept `conv1d_out` boundary to 0.
- **Full upstream-correct variant (kernel+buffer change):** verify-forward `causal_conv1d_update`
  saves the K−1 window per draft step into an intermediate buffer; scatter the step-(k−1) snapshot
  back on acceptance. Bit-exact-to-non-spec by construction; also fixes the DS layout (currently
  `NotImplementedError` at offset>0). Conv window stays **full/native precision** (tiny). The fp32
  SSM cache (#26807) and SGLang's int8 store are SEPARATE axes — do not conflate.

## What we can uniquely contribute (non-redundant)
1. **Source-anchored fix pointer** — the `get_conv_copy_spec` offset-shift vs `get_temporal_copy_spec`
   whole-row asymmetry + the #10335 snapshot framing. Verified accurate. **Highest-value, postable.**
2. **A rigorous per-token lossless instrument** the thread lacks (all their data is downstream
   benchmark deltas): our 4-arm same-boot A/B vs a no-spec RECURRENT oracle within the native E5
   floor — once the rescore lands.
3. **Our conv-vs-SSM decomposition** — scoped to our tree-spec path, offered as a hypothesis for
   their stack, not a measured contributor.

## Red-team verdict: NEEDS-EDITS before postable. Required edits applied to the draft:
1. Re-scope #25587 to Ascend NPU; note NVIDIA already snapshots → (b) is a latent CUDA risk, not
   confirmed-present. (DONE above.)
2. Strike the timothysu/fp8-KV "last comment" — UNVERIFIED (comment thread wouldn't load
   unauthenticated). Do not attribute.
3. Scope every (b) evidence sentence to "our TREE spec-decode committer"; state #43559 is native MTP.
4. Steelman (a)-only: allow that (a) may explain the full 20%; fp32-SSM-off (#26807) is an
   independent sufficient cause to rule in/out.
5. **HOLD the "we don't see the 20%" sentence** until the 4-arm rescore lands — Gate-0's healthy
   acceptance is a scalar feasibility/non-collapse signal (exactly the
   [reference_scalar_metric_per_token_blindspot] class), AND our config dodges (a)'s trigger by
   construction, so "we don't see it" is near-tautological. Not a counter-repro.

## Binding TODO before any post
- [ ] The 4-arm lossless rescore number (arm A cat9+APC + **native-E5+APC**) vs the recurrent oracle
      within the E5 floor — IN PROGRESS (arm A serving now). The native-E5+APC arm specifically
      settles whether (b) reproduces on native MTP + CUDA.
- [ ] Re-read #43559 comments with authenticated `gh` to verify/attribute the fp8-KV clue.
- [ ] Confirm we're willing to disclose our checkpoint/config publicly.

Sources: vllm-project/vllm #43559 #43650 #45477 #45614 #45473 #40738 #26807; sgl-project/sglang
#25587 #10335; our `project_fr13_conv_priorwindow_root`, `FR13_CONV_COMMITTED_PATH`,
`sglang_mamba_radix_cache_design.md`, `apc_gate0_verdict.md`. Full raw synthesis + red-team:
workflow wa8xh6j5z output.

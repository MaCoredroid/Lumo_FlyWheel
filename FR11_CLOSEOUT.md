# FR11 Closeout: no-copy GDN tree verify resolution

Date: 2026-06-05

## Verdict

The FR10 no-copy no-go was reopened correctly by `FR10_NOCOPY_RESOLUTION.md`: the literature did not prove an impossibility theorem, and two decisive engineering probes had not yet been run. FR11 has now run those probes and the acceptance A/B. The no-copy decision is **RE-VALIDATED ON FIRM EVIDENCE**.

No-copy GDN tree verify stays **BANKED** on this stack. On the primary spec-only basis, the no-copy tree remains at `0.87535953978907` accepted/draft-event versus same-harness native MTP-5 at `3.076171875`, a gap of `2.20081233521093`. On the clearly labeled total/event basis, the tree is `1944 / 1043 = 1.8638542665388303` total/event versus native `2048 / 512 = 4.0`, a gap of `2.1361457334611697`. The deficit is not a single cheap seam; it is the diffuse fundamental tree-on-GDN divergence observed in the live verifier. The forward route is copy-recurrent multi-spine, where spine A is native MTP-5 by construction, or `spines=1` native MTP-5 as the clean lossless default.

Keep `FR11_TREE_CONV_NATIVE_BF16_TAPS` defaulted to `0`. The flag is a useful diagnostic and numerics knob, but it did not recover acceptance.

## Evidence

FR10/FR11 established the large acceptance gap on a consistent basis:
- Native reference is the same-harness same-8-prompts run `output/fr10_native_mtp5_same8_20260604T210257Z/quick_native_mtp5_same8.json`.
- Primary basis is spec-only `accepted_per_draft_event`: no-copy tree baseline `0.87535953978907` vs native MTP-5 `3.076171875`, gap `2.20081233521093`.
- Optional total/event basis is separately labeled: no-copy tree baseline `1.8638542665388303` vs native MTP-5 `4.0`, gap `2.1361457334611697`.
- The verdict is unchanged and stronger after the basis correction: the deficit is about `2.2` on either consistent basis.

FR11 closes the three concrete engineering candidates:

1. **GDN scan algebra is not the cause.**
   - FR10's real-tensor validation exonerated the non-diagonal GDN tree scan to `7.45e-9` output parity.
   - Recurrent-step, softplus-g, beta, scale, and l2norm seams were already at numerical floor.
   - This keeps the hard scan kernel banked; do not rework it as an acceptance fix.

2. **Wrong initial recurrent/conv state is excluded.**
   - Probe beta compared event-0 tree handoff against native MTP-5 next-read state.
   - h0 tree accepted row vs native next-read h0: bit-exact, `max_abs=0.0`.
   - Conv decisive prior window tree vs native: bit-exact, `max_abs=0.0`.
   - Verdict: `MATCH_WRONG_INITIAL_STATE_EXCLUDED`.
   - This excludes the #40738/#39273 wrong-initial-state class for the measured live spine.

3. **Conv tap-dtype seam is real, but not acceptance-limiting.**
   - Probe alpha extended through the real layer-0 `post_attention_layernorm -> MLP gate/up/down`.
   - The seam can produce a post-MLP delta of `0.015625`, so the post-attention-only alpha verdict was correctly retracted.
   - The focused high-output check is caveated: at the captured `~1.85` peak the delta was `0.0`, across captured `abs>=1.75` elements max was `0.0078125`, and the whole-tensor `0.015625` occurred at native `1.2578125`.
   - Acceptance A/B at B4, `temperature=0.6`, `top_p=0.95`, metrics off, tree engagement asserted:
     - `FR11_TREE_CONV_NATIVE_BF16_TAPS=0`: spec-only `accepted_per_draft_event=0.87535953978907`.
     - `FR11_TREE_CONV_NATIVE_BF16_TAPS=1`: spec-only `accepted_per_draft_event=0.7597340930674265`.
     - Single-run flag delta: `-0.11562544672164354`, within plausible temp-0.6 sampling noise.
   - The robust point is not the sign of that delta; it is that neither run is remotely close to native `3.076171875` on the same spec-only `accepted_per_draft_event` basis. The native gap is about `2.2`, roughly `19x` the absolute flag delta. On total/event the baseline gap is also about `2.14`. A greedy temp-0 A/B could tighten the sign but is not worth the GPU cost.

Therefore the conv tap-dtype seam is a precision-floor contributor, not the acceptance-limiting cause.

## Literature Cross-Check

`FR10_PAPER_NOGO_RESEARCH.md` remains the correct literature stance:

- There is **no theoretical no-go** in the cited papers. No impossibility theorem, no superset-only theorem, and no paper-supported claim that shared state necessarily degrades path0.
- STree is itself a no-copy / single-shared-state tree method and refutes the broad "shared-state degrades path0" argument.
- But STree's validated exactness is for diagonal Mamba2/S6-style transitions, not the gated delta rule's non-diagonal rank-1 update.
- No paper supplies a validated no-copy recipe for Qwen3.6's gated delta rule.

So the closeout is not a mathematical impossibility claim. It is an engineering closeout on this stack: the scan is banked, the state-handoff bug route is excluded, the conv seam does not recover acceptance, and the residual deficit is diffuse live tree-on-GDN divergence.

## Decision

No-copy GDN tree verify is closed for FR10/FR11.

The path forward is:
- **Copy-recurrent multi-spine**: each candidate spine owns isolated recurrent state; spine A is the native MTP-5 chain by construction, so the lossless baseline is preserved while extra spines compete as candidates.
- **`spines=1` native MTP-5**: the clean lossless default until multi-spine isolation passes its own gates.

Do not spend more GPU on no-copy dead ends unless new evidence appears that contradicts one of the three excluded candidates above. Do not retry M-RoPE broadcast, TREE_ATTN, big-tree speed optimization, or byte-exact-vs-MTP5 as a goal. Do not spend a greedy conv A/B just to tighten the sign of a single-run temp-0.6 negative delta; the recovery gap is already too large.

## Primary Artifacts

- FR11 results and numbers: `FR11_RESULTS.md`.
- Reopened no-copy resolution: `FR10_NOCOPY_RESOLUTION.md`.
- Literature review: `FR10_PAPER_NOGO_RESEARCH.md`.
- Original closeout context: `FR10_CLOSEOUT.md`.
- Probe beta script: `scripts/fr11_probe_beta_event0_handoff.py`.
- Probe alpha replay: `output/fr10_nocopy_resolve/gpu_conv_seam_replay.py`.
- Conv tap flag implementation: `src/lumo_flywheel_serving/fr10_tree_conv.py` and `scripts/fr10_phase4_patch_vllm_tree_gdn.py`.

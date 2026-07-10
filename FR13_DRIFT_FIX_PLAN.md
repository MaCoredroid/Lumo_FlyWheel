# FR13 — garble fix plan (drift-fix design, 2026-07-10)

From design workflow wf_c7d5ed49 (map existing infra + design fix+gate), then verified
against the study docs. Supersedes the naive "rms-clamp first" idea.

## The reframe (KEY, verified 2026-07-10)
The live-pinned garble (`from_geodetic`->`from_geodentic`, cat8 task 13398) is a
MANIFESTATION of the campaign's central hard problem, NOT a separable quick fix. Verified
against FR13_AMPLIFICATION_LEVERS.md (header: "SUPERSEDED AS THE PRIMARY LEVER") +
FR13_E5_VS_CAT9_SPINE_DRIFT.md (study wsvy4vn5k):
- The ~1.166x/layer amplification (gate 1/rms + deep full-attn) is SHARED with native E5
  (GDN 1.075x/layer, full-attn 1.236x/layer). Reducing it (fp32 / rms-clamp / residual
  re-anchor) lowers BOTH arms equally -> does NOT differentially close the cat9-vs-E5 gap
  -> WILL NOT fix the garble. => DO NOT build the amplification levers as the garble fix.
- The real carrier = CO-RESIDENCY M-DEPENDENCE at the L0 GDN birth-amplitude. Existence
  proof: `chain5` (cat8's exact kernels, spine alone at M=5, no co-resident branches)
  de-cascades to 2 <= native 3. So our tree kernels CAN spine at the floor when M is low.
- The baked pad-block (LUMO_FB_KERNEL_ROWS=1) made in_proj_ba M-invariant, yet garble
  persists -> a DIFFERENT sub-op still carries M-dependence: conv1d_update (receptive-field
  entanglement) OR fused_sigmoid_gating recurrent scan (state-feed). UNLOCALIZED.

## Sanctioned levers (source-side, NOT amplification)
1. SPINE M-INVARIANCE: localize the surviving M-dependent sub-op, then make it M-invariant
   (extend the pad-block idea to conv/scan). Localize via FR13_GDN_SUBOP_MAB.
2. TREE-RESHAPE: narrower/shallower tree -> lower co-residency M -> less drift (chain5 proof;
   project_fr13_tree_reshape_unifying_lever). Tradeoff: fewer draft tokens = lower accept = speed.

## Gate (from the design synthesis, still valid)
PRIMARY (cross-boot-immune): teacher-forced oracle-flip WRONG-ACCEPT gate. For each committed
near-neighbor token, teacher-force cat8 itself POST /v1/completions {prompt: prompt_ids+
served[:i], max_tokens:1, temperature:1.0, logprobs:20} (real prefill dist, reset_prefix_cache
first — dodges the -12.422 spec placeholder). WRONG-ACCEPT iff oracle argmax==correct AND
margin(correct-committed)>1.0 nat. Count over G1 low-margin prompts (fr13_garble_gate PROMPTS,
which DO elicit; geodetic is high-margin, 0 elicit). Glue over fr13_oracle_stream_teacher_force.py.
CROSS-CHECK (per-layer ladder): fr13_node5_ladder_drive.py + fr13_ladder_table.py (tree-verify
vs no-spec-clean final-logit max_abs + argmax-flip, within-boot).

## Next step = LOCALIZE (cheap, sanctioned, gap #3 of the mapper)
Boot cat8 dedicated (scripts/fr13_cat8_dedicated_server.sh EXTRA_FLAGS="FR13_GDN_SUBOP_MAB=1";
patcher auto-writes sidecar /logs/fr13_gdn_subop_mab.flag). Run scripts/fr13_gdn_subop_mab_drive.py
(2 passes, asserts within_boot_det) -> reduce scripts/fr13_gdn_subop_table.py. Read M10-vs-M5 raw
max_abs per sub-op (conv1d_update, fused_sigmoid_gating). FIRST sub-op with M10!=M5 on the deep
spine row = the carrier birthplace -> that determines the M-invariance fix.
KNOWN FRAGILITY (mapper gap #2): the drive needs the right pinned capture event
(output/fr13_commit_argmax/tree_capture.json + prompts_swe4.json); class-9 vacuous risk if it
lands on the wrong event -> assert engagement (EXPECT_TREE_N=10, within_boot_det) before trusting.

## Do NOT
- Build rms-clamp/fp32/re-anchor as the garble fix (superseded, shared with E5).
- Edit the sampler (reward-hack).
- Trust garble RATE (boot-noise) or wa_capture top_logprobs=1 (-12.422 placeholder).

## Localization attempt (2026-07-10) — MAB tooling BLOCKED

Ran the sanctioned FR13_GDN_SUBOP_MAB localization on a dedicated cat9 server. Two blockers:
1. GRAPH mode (default FULL_AND_PIECEWISE) -> hook is EAGER-ONLY (patcher:1924 returns during
   capture) -> VACUOUS (0 records, no engage-fail markers). Fix: ENFORCE_EAGER=1.
2. EAGER mode -> hook ENGAGED (FR13_SUBOP_STAGE=engaged tree_n=10 deep_row=8 num_spec=1) but the
   reduced-M (M5/M1) sub-op re-run hit `Triton Error [CUDA]: device-side assert triggered` ->
   arm-fail -> crashed the EngineCore (server died, GPU recovered clean). The re-run code
   (patcher:2072-2160) already carries a HISTORY of device-assert fixes ("FR13_SUBOP_MAB
   device-assert fix", "FR13_SUBOP_AB_CRASHED_PIVOT_CHAIN3 FIX") and STILL asserts on this
   cat9-eager boot -> the tool is unreliable for this config; a further fix needs
   CUDA_LAUNCH_BLOCKING=1 + Triton index analysis = a rabbit hole into the diagnostic internals,
   not the target.

STATUS: the carrier (conv1d receptive-field vs recurrent-scan state-feed) remains UNLOCALIZED by
the MAB. Mapper-2's HYPOTHESIS (well-supported by FR13_E5_VS_CAT9_SPINE_DRIFT.md: drift born at L0
GDN compute; reference_gdn_verify_sequential_dispatch: verify uses fused_sigmoid_gating) = the
recurrent-scan state-feed at num_accepted>1 (chunk-vs-recurrent realization).

## THE FORK (needs a call; speed matters, user asked)
The garble fix = the campaign's CORE M-invariance problem, two sanctioned paths:
  A. SPINE M-INVARIANCE (speed-preserving, sanctioned PRIMARY): make the surviving M-dependent
     sub-op M-invariant (extend the pad-block idea to the scan/conv). Blocked on localization
     (MAB crashes) OR proceed on the recurrent-scan hypothesis (risk: could be conv). Deep build.
  B. TREE-RESHAPE (sidesteps localization): narrower/shallower tree -> lower co-residency M ->
     less drift. chain5 (M=5 spine-alone) de-cascades to 2<=native3 = existence proof. SPEED COST:
     fewer draft tokens = lower accept rate. Testable via live-SWE (slow ~90min/task) + garble-watch.
Alternate diagnostic to try instead of the crashing MAB: FR13_REPLAY_DURABLE_AB (mapper-1 "PRIME
lead") compares our replay GDN recurrent state vs the native sequential kernel over the accepted
chain (directly tests the scan-state-feed hypothesis) — also eager-only, may also be fragile.
RECOMMENDATION: get a user call on the speed tradeoff (A preserves speed but is deep + tooling-blocked;
B is simpler but costs speed). All fix-validation is slow live-SWE (fast instruments compromised).

## DECISION (user 2026-07-10) — PATH A, with a hard no-HBM-tax rule
Direction LOCKED:
1. FIX THE TOOLING — repair the FR13_GDN_SUBOP_MAB device-assert (and/or find a cheaper
   localization) so we can actually localize the surviving M-dependent sub-op.
2. LOCALIZE cheaply — which ops need M-invariance (conv1d_update vs fused_sigmoid_gating scan
   vs anything else surviving the pad-block), with a MEASURED SPEED COST for each candidate fix.
3. HARD RULE — NO HBM TAX. Any M-invariance fix must be COMPUTE-ONLY (no extra HBM traffic /
   no added copies), because GB10 is memory-bandwidth-bound (273 GB/s) — compute is ~free,
   bandwidth is the ceiling. The pad-block is the template (pads a GEMM = more compute, zero
   extra HBM). Reject any fix that adds HBM copies/reads even if numerically correct.
4. APPLY — build the compute-only M-invariance fix for each localized op (default-OFF flag),
   measure speed, verify the garble drops (teacher-forced oracle-flip gate + live-SWE 13398).
This is the sanctioned speed-preserving path (spine M-invariance), NOT tree-reshape (speed cost)
and NOT amplification levers (superseded).

## LOCALIZATION RESOLVED (2026-07-10) — carrier = conv1d prior-window (skip the broken MAB)
The per-layer ladder in FR13_E5_VS_CAT9_SPINE_DRIFT.md ALREADY localized it (don't need the crashing MAB):
enter L0 byte-exact -> pre_conv(in_proj)=0.0 (pad-block fixed in_proj_ba) -> FIRST-nonzero sub-op =
`conv1d_out` = 0.000977 (1 bf16-ULP) at num_accepted>1. Named prime carrier = the conv1d PRIOR-WINDOW /
state-bank column geometry (project_fr13_conv_priorwindow_root: "conv1d_out wrong bank-row at num_accepted>1;
OPEN"). Mechanism = the deep SPINE row's depthwise-conv receptive window is contaminated by CO-RESIDENT
BRANCH rows instead of its true spine ancestry (M-dependent). Fix target = fr10_phase4_patch_vllm_tree_gdn.py
~:797-818 (conv source-index wiring). Existing PARTIAL infra aimed here: FR13_CONV_COMMITTED_PATH (baked ON,
snapshots committed-path conv rows) + FR13_TREE_CONV_FUSED (source-by-width, gather_committed_path_conv_prior)
— but garble PERSISTS -> the source-by-width wiring is INCOMPLETE (some conv window index still co-resident-keyed
at num_accepted>1). This is M-DEPENDENT + the fix is a RE-INDEX/WIRING correction = COMPUTE-FREE, NO HBM ->
passes the hard no-HBM-tax rule cleanly.

REVISED PLAN (autonomous): (a) MAP the conv prior-window path — what FR13_CONV_COMMITTED_PATH/TREE_CONV_FUSED
do, WHICH index is still co-resident-keyed at num_accepted>1, why garble persists. (b) DESIGN the compute-only
re-index fix (key the deep spine row's conv window to the spine/path0 ancestry, not the flat co-resident
layout). default-OFF flag, byte-identical off. (c) BUILD + measure speed (expect ~0 tax) + verify garble drops
(teacher-forced oracle-flip gate + live-SWE 13398 fr13_garble_watch). NO HBM copies.

## FIX DESIGN (2026-07-10, wf_9b24f818) — REALIZATION REPLICA, not a re-index
Re-index premise REFUTED: the deep-spine conv window is ALREADY spine-keyed (fr10_tree_conv_source_indices
= parent-chain ancestry, :3371; flat/co-resident source is diagnostic-only). The residual conv1d_out=9.77e-4
(1 bf16-ULP) from byte-exact pre_conv=0.0 is a KERNEL REALIZATION diff: OUR tree-conv (fused_tree_conv_taps_acc
bf16-tap MAC @fr13_tree_conv_fused.py:236-251 + triton_ex2_silu_bf16 @fr13_ex2_silu.py:17-23) produces a
different bf16 than native causal_conv1d_update on IDENTICAL input. Only num_accepted>=3 (deep-accept) carries
enough ULP to flip an argmax.

FIX = FR13_CONV_EX2_REPLICA (default OFF, sidecar for spawn worker): (a) align tap-MAC accumulation ORDER to
native's column order (native oldest..newest; reverse the width<=4 loop if it differs); (b) make
triton_ex2_silu_bf16 a bit-exact ex2.approx replica of native's silu (mul -x,log2(e) -> ex2.approx.f32 -> 1+ ->
div -> cvt.rn.bf16). KEEP bf16 taps (fp32-taps REGRESSES L0 to 0.0625). PURE COMPUTE, ZERO HBM (same kernel
launch, loop reorder) -> passes no-HBM rule. FR12_NATIVE_SPINE oracle (copies spine rows) = validation-only, BANNED as ship.

VALIDATION (cheap-first): (1) engagement assert (worker self-report, replica fired>0). (2) OFFLINE per-layer
ladder BOOT-FREE: capture spine-only conv inputs (pre_conv+conv_state+weights+bias) for clean layers L0/4/8/12/24/36/44,
iterate replica vs native causal_conv1d_update on IDENTICAL inputs until conv1d_out==0.0 RAW/int-view (NOT atol)
every clean layer. (3) teacher-forced oracle-flip gate vs FR12_NATIVE_SPINE. (4) live-SWE 13398 fr13_garble_watch
same-boot A/B. (5) speed derived_tps_gpu ON vs OFF ~0.
KEY RISK (#3): the L0->L63 GROWTH is dominated by the DIFFUSE GDN recurrent SCAN state-feed (reference_gdn_verify_sequential_dispatch),
which the conv fix does NOT touch. If perfect conv alignment does NOT collapse the flips -> route to tree-reshape,
NOT another per-op patch. So conv-replica is NECESSARY-maybe-not-SUFFICIENT. Design detail: FR13_CONV_FIX_DESIGN.md.

## Native kernel located + build plan (2026-07-10)
Native `_causal_conv1d_update_kernel` (Triton, readable) at
/tmp/lumo_vllm_main_audit/vllm-v0.22.0/.../mamba/ops/causal_conv1d.py (host copy; LIVE served =
v0.19.2rc1.dev134 — match THAT via the offline harness since bit-exact is codegen/SASS-sensitive, class-10).
Structure: loads width taps explicitly col0(oldest)..col3(newest) (width-4), then a fused fp32 MAC + ex2.approx
silu + bf16 store in ONE Triton program. Our replica target = fused_tree_conv_taps_acc (fr13_tree_conv_fused.py:236-251,
current order (((bias+p0)+p1)+p2)+p3, bf16 products cast f32) + triton_ex2_silu_bf16 (fr13_ex2_silu.py).
§6 OPTIMISM: conv1d_out is FIRST-NONZERO, GDN scan is M-invariant-as-op (downstream), so a bit-exact conv replica
should CASCADE the whole L0-GDN to 0 (feedback_fr12_subkernel_zero_gate) -> plausibly SUFFICIENT.

NEXT (autonomous build):
1. OFFLINE BIT-MATCH HARNESS (the cheap loop, NO model boot): a script run INSIDE the live vLLM image
   (docker run lumo-flywheel-vllm...v0.19, no server) that imports native causal_conv1d_update + our
   fused_tree_conv_taps_acc+triton_ex2_silu_bf16, feeds IDENTICAL random bf16 window/weights/bias/conv_state,
   compares int-view (RAW 0.0, NOT atol). Iterate our MAC order + silu until bit-exact vs native. Random inputs
   expose the order/rounding diff (no captured real inputs needed for MAC-order matching).
2. Wire FR13_CONV_EX2_REPLICA (default OFF, sidecar for spawn worker, byte-identical off) selecting the matched
   replica MAC/silu in fr13_tree_conv_fused.py + fr13_ex2_silu.py.
3. GPU GATES: engagement assert; per-layer ladder conv1d_out->0 within-boot; teacher-forced oracle-flip vs
   FR12_NATIVE_SPINE; live-SWE 13398 fr13_garble_watch same-boot A/B; speed derived_tps_gpu ~0.
HARD RULE reaffirmed: compute-only, NO HBM copy (loop reorder / same kernel launch). Do NOT promote taps to fp32
(regresses L0 to 0.0625). If conv1d_out->0 but e2e flips persist -> scan/tree-reshape, NOT another per-op patch.

## CONV-REPLICA FIX REFUTED (2026-07-10, VERIFIED first-hand) — carrier is recurrent conv-STATE content
Offline bit-match harness (scripts/fr13_conv_replica_offline_gate.py, run INSIDE the live image, no boot)
PROVES our tree-conv (fused_tree_conv_taps_acc + triton_ex2_silu_bf16) is ALREADY BIT-EXACT to native
causal_conv1d_update: 0 int-view mismatches across 36 configs + spec-decode num_accepted>=3 + D1
offset-engaged + D2 byte-preserving window-build. Only fp32-products (MODE2) diverge = the known 0.0625
regression -> bf16 products are CORRECT. So the design doc's "bf16-tap MAC realization" attribution
(FR13_CONV_FIX_DESIGN.md §2b) is WRONG; DO NOT build FR13_CONV_EX2_REPLICA (dead).

=> the production conv1d_out=9.77e-4 (from byte-exact pre_conv=0.0 + bit-exact conv + byte-preserving
window-build) can ONLY come from the conv-STATE BANK CONTENT being 1-ULP different = a RECURRENT carry.
Since the conv writes bit-exactly, the taint ORIGIN is UPSTREAM of the decode conv: prefill state-seed
(chunked/tree prefill conv path) OR the tree-verify committed-state write-back — NOT a decode-op numerics
fix and likely NOT compute-only/no-HBM. The per-token ladder sees conv1d_out as "first-nonzero" only
because it READS a pre-tainted state; the true first-nonzero is in a PRIOR forward's state write.

CONSEQUENCE for the no-HBM rule: the cheap compute-only conv fix is EXHAUSTED. The remaining carrier
(recurrent state-seed) has no identified compute-only/no-HBM fix -> cost-gate juncture.

BONUS (free, unrelated to garble): the harness's V2 = a single fused Triton conv kernel (MAC+silu) is
bit-exact and collapses the current pytorch-mul+cast+3adds+separate-silu into ONE kernel = a device-node/
speed win, no correctness justification needed. Un-wired; available if wanted.

## NEW FRONT (user 2026-07-10): REPRODUCER first, then LOCALIZER
Foundation check (user challenge): the "misspell == 1-ULP spine flip" link is UNPROVEN, and the observed
13398 garble is a MIX: from_geodentic ×9 (near-neighbor), from_geodec ×2 (trunc), AND from_geode+hundreds-of-"ode"
(DEGENERATE REPETITION loop @ trace line 835 tool_use — verified real, not artifact). The repetition does NOT
fit a 1-ULP flip. No stable cheap reproducer exists (geodetic=0-elicit, G1=boot-noisy, wa_capture=placeholder,
live-SWE=90min stochastic). PLAN: (1) build a GOOD reproducer = deterministic capture-REPLAY: reconstruct the
exact garbling context (messages from the qwen_trace, OR pair-dump prompt_token_ids+seed) and replay on cat8-tree
vs no-spec, same boot/seed -> stable, cheap, faithful reproduction + differential. (2) GOOD localizer = at the
garbling token, in-process compare tree-verify vs no-spec commit/logits -> CLASSIFY each garble: (a) spec wrong-accept
(tree commits what no-spec rejects), (b) numeric argmax flip, (c) drafter repetition rubber-stamped. Only THEN fix.
Pair-dump infra exists (proxy_pair_dumps/pair_*.json in other runs; NOT saved for cat8_nofix_g5).

## JUDGMENT (2026-07-10): MISSPELL and RUNAWAY are TWO issues (related, distinct mechanisms)
- MISSPELL (from_geodentic ~13%): single-token near-neighbor wrong-accept = spec-decode drafter proposes
  near-neighbor + tree-verify rubber-stamps (the 1-ULP-drift candidate; native clean -> spec-specific). PRIORITY.
- RUNAWAY (from_geode+hundreds "ode", 1x): degenerate repetition attractor = drafter proposes a repeat + tree
  accepts (self-high-prob once started); a decoding pathology AMPLIFIED by spec, NOT a single argmax flip.
  Track SEPARATELY; likely a different fix (repetition in drafter/verify-accept), not M-invariance.
Localizer classifies each garble occurrence -> (a) wrong-accept near-neighbor, (b) numeric drift-flip, (c) drafter-repetition.

## REPRODUCER design (within-boot; garble is boot-autotune-dependent):
- Pair-dumps capture the FULL request (instructions=system + input=conversation, Responses-API) but NO seed/token_ids;
  cat8_nofix_g5 didn't save them. So: fresh cat8 boot + run 13398 (or shorter) with pair-dumps ON + live garble-watch
  on the streaming trace -> capture a garbling turn's request WITHIN the boot.
- CHEAP replay: extract the PREFIX ending right before "from_geode", generate ~10 tokens on cat8 (fixed seed) ->
  does it re-emit from_geodentic / geodeode? If deterministic -> cheap reproducer (capture once, replay cheap).
- LOCALIZER (same boot): at the garble token, teacher-force (max_tokens=1, top_logprobs=20, reset_prefix_cache) the
  prefix -> the no-spec prefill dist (avoids the -12.422 placeholder); compare to what the tree committed -> classify.

## LOCALIZER VERDICT (2026-07-10, VERIFIED) — misspell is a MIX; damaging class IS a spec wrong-accept
scripts/fr13_garble_localizer.py (measurement-only, teacher-force /v1/completions max_tokens=1 top_logprobs=20
temp=1.0 on exact return_token_ids prefixes, 0 placeholder hits, determinism 6/6, within-boot). Reproducer = G1
(this boot 40% garble = 75 garbles/180 gens). Verdict over 75:
- WRONG_ACCEPT 39 (52%): TRUNCATION/dropped-char (applied_entry_index->applied_entry_idx 21/21; input_wcs_header->
  input_ws_header; world_coordinate_offset_values->trunc). The no-spec model gives the CORRECT spelling ~0.9999 and
  the committed garble ~1e-6 (BELOW top-20), yet the tree committed it. = GENUINE spec-decode wrong-accept (proven).
- MODEL_TAIL 31 (41%): PLURAL/camelCase (expected_row_count->expected_rows_count; camelCase). The model itself is
  ~50/50 (p 0.37 each) -> would garble at temp 0.6 WITHOUT spec-decode. NOT spec-specific. (Still NameErrors, but
  a spec fix won't touch it.)
- OTHER 5: genuinely ambiguous (trailing-s).
CONCLUSION: the wrong-accept hypothesis HOLDS for the truncation class (the NameError-causing one). Fix locus =
the tree spec-decode COMMIT/ACCEPT path for that class. OPEN NEXT: WHY does the tree accept a token the model gives
~1e-6? Two candidates: (a) tree-verify forward DRIFT-inflates the garble's target prob (but conv bit-exact + amp dead
+ no compute-only fix found), OR (b) the tree COMMIT path is LENIENT (greedy-argmax / LCP / top-k membership) rather
than proper rejection sampling with the TRUE target. Determine via: read the tree commit code (_patch_rejection_sampler_
tree_lcp ~8008, _patch_rejection_sampler_gpu_committer ~18781) + measure the tree-verify accept-time target prob for
a garble with CORRECT node alignment (the final nail wa_capture missed). NOTE memory: sampler kernel proven
distribution-equivalent (offline parity 22/22) — so if commit uses THAT sampler with the true target it'd reject ->
leans toward drift OR a DIFFERENT commit path than the tested sampler.

## COMMIT PATH = GREEDY-ARGMAX (2026-07-10) -> reframes the wrong-accept away from 1-ULP drift
The tree committer (_lumo_tree_path_lcp_max_greedy_sample, patch ~18784; FR13_COMMIT_ARGMAX_GATE @8050-8073)
commits the ARGMAX of the tree-verify logits ROW for each committed position (greedy, LCP path). So a wrong-accept
(model true dist: correct p=0.9999 vs committed garble p=1e-6, a 12-NAT gap) means the tree-verify logits ROW's
argmax literally IS the garble token. A 1-ULP diffuse drift CANNOT flip a 12-nat-gap argmax -> the "1-ULP drift"
framing is WRONG for the wrong-accept class. Two candidates, distinguished by FR13_COMMIT_ARGMAX_GATE:
  ch1 = COMMITTER ROW-MAPPING BUG: the greedy committer indexes the WRONG tree-node's logits row -> commits that
        node's token (a near-neighbor drafted at a different position). LEGIT FIX. Fits: native clean (no tree),
        conv bit-exact (numerics fine), 12-nat gap (whole different row).
  ch2 = verify-forward argmax at THAT row is grossly wrong (not 1-ULP).
DECISIVE NEXT EXPERIMENT: boot cat8 EAGER + FR13_COMMIT_ARGMAX_GATE=1, run G1 reproducer, capture the gate jsonl
(/logs/fr13_commit_argmax_gate.jsonl): for the wrong-accept garbles, is served_token != argmax-of-indexed-row
(ch1 row-mapping mismatch) OR is the verify-forward argmax itself the garble (ch2)? ch1 mismatch -> row-mapping fix.
This is the localizer->fix bridge. Eager-only gate; within-boot; reproducer = G1 (fr13_garble_gate).

## MECHANISM LOCALIZED (2026-07-10): wrong-accept = LCP COMMIT-PATH, a leaf winning the tie-break
The greedy committer _lumo_tree_path_lcp_max_greedy_sample (@8851; scoring @9206-9272) commits the MAX-LCP path:
per root-to-leaf path, LCP = longest prefix where drafts[node]==parent_targets[node] (draft matches per-node
target argmax); tie-break = plain `lcp > best_lcp` (earliest leaf). Committed row = drafts[best_path[:lcp]] + a
bonus target token at the boundary. So the wrong-accept _idx is committed because a LEAF/SIBLING path's LCP WON
(pulling in its garbled draft) — the KNOWN "lcp boundary shifts under co-residency / a leaf wins on sub-1-nat
margin" issue (code comments @8080-8086, partial "deterministic rank-2" mitigation, incomplete). = spec-decode
COMMIT-PATH bug, NOT numeric drift. Existing lever FR13_FORCE_SPINE_COMMIT (@9258) commits the spine path ONLY.
DECISIVE TEST: run the reproducer with FR13_FORCE_SPINE_COMMIT=1 -> garble VANISHES => confirmed alt-path-winning
(fix = spine-preferring / margin-gated LCP tie-break, likely compute-only); garble PERSISTS => the spine node's
own target argmax is drifted (forward). CAVEAT: force-spine may cut accept rate (rejects valid alt accepts) ->
it's the diagnostic; the ship fix = a corrected tie-break (prefer spine on ties, require a margin for alt paths).
PLUMBING: FR13_* diagnostic flags (gate, force_spine) don't reach the forward worker pid176 via env-copy (2 vacuous
gate boots); need the SIDECAR mechanism (mirror _fr13_gdn_subop_mab_enabled @1776-1794) for whichever flag we test.

## PLUMBING (2026-07-10): ALL env-copy fails to reach the forward worker -> SIDECAR required
Confirmed: neither VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=FR13_ NOR VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=<flag>
gets FR13_FORCE_SPINE_COMMIT / FR13_COMMIT_ARGMAX_GATE to EngineCore forward worker pid176 (86GB GPU). The MAB
worked ONLY via its sidecar (_fr13_write_subop_mab_sidecar @18948 writes /logs flag at pid-1; resolver
_fr13_gdn_subop_mab_enabled @1776 reads env-then-sidecar). So ANY committer diagnostic/fix flag needs the SAME
sidecar. FR13_FORCE_SPINE_COMMIT is read PER-CALL at TWO injected sites (@8919, @10433). SHIP FIX (spine-preferring
LCP tie-break @9232) will ALSO need OFF-gating via a sidecar flag OR be a baked code change (baked=in-worker, no
propagation issue, but not byte-identical-OFF).

## COURSE-CORRECTION (2026-07-10, agent red-team): greedy-LCP was the WRONG committer
The greedy-LCP committer (_lumo_tree_path_lcp_max_greedy_sample) + FR13_FORCE_SPINE_COMMIT are TEMP-0 ONLY
(all_greedy path). Production + the localizer baseline run at TEMP 0.6 -> not all_greedy -> dispatch @11556->11590
to the SAMPLED multidraft committer _lumo_tree_canonical_multidraft_sample (@10419) -> fr13_device_multidraft_commit
= SpecInfer residual-mix accept (accept ~ min(1,p/q_mix), residual fallback; @10458). So the baseline wrong-accepts
(applied_entry_idx 21/21) came from the SAMPLED committer; FORCE_SPINE hard-raises on it (can't test) and my
greedy-LCP "leaf wins the tie-break" hypothesis targets an off-production committer. Agent proved this behaviorally
(FORCE_SPINE raised from pid176 at temp0.6) + built a clean byte-identical-OFF sidecar (in worktree
agent-a9f91400978bbd3fc, reusable pattern) — but FORCE_SPINE itself is now moot for production.

CORRECTED FIX LOCUS = the SAMPLED committer accept rule (@10419-10684, _canonical_accept_prob @10655). Since the
device multidraft sampler was PROVEN distribution-equivalent (offline parity 22/22), a wrong-accept there means the
INPUTS are off: the tree-verify p_target for the garble is high enough vs q_mix to accept, despite the true model
giving 1e-6. NEXT: capture the accept-time p_target (tree-verify) + q_mix for a garbled draft token in the SAMPLED
committer (its own _canonical_accept_prob dump / LUMO_TREE_SAMPLER_DEBUG_LOG, with the sidecar pattern to reach
pid176) -> p_target INFLATED = forward drift (back to the hard problem); p_target LOW but committed = a residual-mix
accept-logic bug (fixable in the committer). This is the wa_capture goal done RIGHT (in-process accept-prob, correct
node alignment, not surfaced logprobs). Do NOT re-run the greedy FORCE_SPINE test for production.

## CORRECTED DIAGNOSTIC (2026-07-10): the sampled committer already records accept-time p_target
_lumo_tree_canonical_multidraft_sample builds step_trace_rows (@10680) with, per accept step:
target_prob_at_draft_token_ids (= tree-verify p_target for each drafted token @10672), canonical_accept_prob
(=min(1,p_target/q_mix) @10655), selected_token_id, accepted. Gated by LUMO_TREE_SAMPLER_DEBUG_LOG (dump write).
This IS the accept-time p_target capture done RIGHT (in-process, correct node alignment, the SAMPLED committer,
NOT surfaced logprobs). NEXT EXPERIMENT: boot cat8 (temp 0.6) + LUMO_TREE_SAMPLER_DEBUG_LOG=<path> (check if LUMO_
reaches the worker via docker -e like LUMO_FB did, else sidecar) + run the reproducer -> for a committed garble
token (e.g. applied_entry_idx's _idx), read its target_prob_at_draft_token: INFLATED (>>1e-6) => forward drift
(the tree-verify gives the garble high prob -> back to the hard drift problem, but now with a clean measurement);
~1e-6 but accepted => multidraft ACCEPT-LOGIC bug (q_mix mis-weight / residual resample committing a low-p token)
= a FIXABLE committer bug. Likely eager-only (dump syncs/.item()s). This is the fork that ends the investigation.

## *** [OVERTURNED 2026-07-10 — DO NOT USE. See FR13_GARBLE_COMMITTER_CLEARED.md + FR13_GARBLE_DRIFT_BINDING_PROVEN.md] ***
## The "ACCEPT-LOGIC bug" below is WRONG (stale). The 1.99e-6 was target_RAW_prob_draft (RAW, PRE-constraint, temp-1.0)
## read at a NON-COMMIT gather node. The committer is PROVEN to commit each token at EXACTLY its POST-constraint
## target_prob_draft (offline gate 22/22), so a genuinely-1.99e-6 token CANNOT commit at ~8-13% — mathematically. A
## commit_trace instrument (fr13_device_multidraft_kernel.py) measured the committer's OWN input AT the actual commit
## node: committed_prob = 0.0809 (IN the top-p nucleus) for expected_row_count->expected_rows_count, while no-spec masks
## it at ~1e-6. => the garble IS forward drift (tree-verify inflates the near-neighbor into the nucleus; the correct
## committer commits it). The committer is NOT the bug. [Kept below for history only.]
## *** DECIDING RESULT (2026-07-10): NOT forward drift — it's a multidraft ACCEPT-LOGIC bug ***
LUMO_TREE_SAMPLER_DEBUG_LOG works (reaches the forward worker unlike FR13_ flags; all_greedy=False confirms the
sampled path). tree_logit_gather rows carry target_raw_prob_draft = the tree-verify's prob for each drafted token.
MEASURED on the EXACT garble: _idx -> _index (applied_entry_idx): tree-verify target_raw_prob_draft = 1.99e-06 (x3,
identical). no-spec model (localizer teacher-force) = ~1e-6. => the tree-verify forward is NOT drifted; it correctly
scores the garble ~2e-6 (== no-spec). Aggregate: 1740 near-neighbor garble drafts, median tree-verify prob 0.001,
truncations ~1e-10 ('ied'->'plied' 4e-10); the only HIGH(>0.1) are genuinely-ambiguous ('_list'->'list' 0.5 =
model-tail). So the WRONG-ACCEPT is the multidraft committer (fr13_device_multidraft_commit / _lumo_tree_canonical_
multidraft_sample, accept~min(1,p_target/q_mix)+residual) COMMITTING a draft it itself scores ~2e-6 at ~13% instead
of 2e-6 = ACCEPT-LOGIC / q_mix / residual bug, spec-decode-specific, in OUR committer. NOT numeric drift.

CONSEQUENCE: the ENTIRE drift/M-invariance chase (conv bit-exact, amplification superseded, pad-block) was the WRONG
problem for this garble. Fix locus = the multidraft accept computation @10419-10684 + fr13_device_multidraft_commit
@10566 + the q_mix computation. TENSION w/ offline parity gate (22/22 device==host): either the host reference ALSO
over-commits (algorithm issue) OR the q inputs (q_mix/q_draft for the near-neighbor node) are mis-computed
(min(1, 2e-6/tiny_q)=1 -> always accept, residual not correcting). NEXT: read the accept + q_mix -> find why a
2e-6-p_target draft commits at ~13%. Fix = correct the accept/q_mix (compute-only). NOT a reward-hack: committing a
token at ~6500x its true (and tree-verify-agreed) prob is a genuine correctness bug. Data: output/fr13_dbg/dbg_snapshot.jsonl.

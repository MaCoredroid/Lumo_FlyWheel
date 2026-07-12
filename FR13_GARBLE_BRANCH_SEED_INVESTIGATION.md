# FR13 garble — BRANCH-node seed investigation (2026-07-12, frontier re-open)

> **GOAL (user, firm): cat6/cat8 BRANCHED trees garble-free. chain5/spine-only/tree-reshape is an
> ANTI-SOLUTION and is OFF THE TABLE — removing branches deletes the deliverable (branches = accept =
> speed). Any past "chain5 kills garble" result is used ONLY as a diagnostic that branch co-residency
> is the carrier; the fix must KEEP branches and correct their forward compute-only.**

User reopened the garble drive (the 2026-07-10 "accept within-floor" close is REFERENCE, not a
wall): native GDN with **real batching** has 0% garble, so our machinery has a specific, findable
defect that accepts an impossible token. This doc breaks down the numbers and localizes it.

## The numbers (all measured, our instruments)

### The 16 localized garbles (output/fr13_dbg/commit_trace/localizer.jsonl) split 2 / 14
- **Class B (14/16) = "impossible token accepted in verify":** the TRUE (teacher-forced) forward
  gives the committed garble a **median e^-12.4 (~1/240,000)** prob — down to **1.6e-8 at rank 15** —
  with the CORRECT token at ~0.9999. Our tree committed it anyway.
- Class A (2/16): genuine argmax flips (forward really preferred the garble; margin_nats < 0).

### The 64-layer drift ladder (output/fr13_node5_ladder/per_layer_maxabs.json) — why it's accepted
- Input embedding into layer 0: **bit-identical (0.0)**.
- Layer-0 GDN (linear_attention) output: **0.012 L2** — the seed enters HERE, from the recurrent
  state, NOT the token.
- Amplifies **×~1.166/layer** (full-attn layers dominate at ×1.2–1.6) → **178 L2 at layer 63** =
  **~14,800×**. That logit kick lifts an e^-17 token into the accept nucleus.

## Who we compare against, and why native-batched is immune
- **no-spec decode** = ground truth (gives the near-neighbor ~1e-6). bf16 sequential.
- **native MTP-5 (E5)** = native's own linear spec-decode; ~0 garble; the RIGHT reference for our
  SPINE (same kernel family).
- **branch-path oracle** (fr12_branch_path_oracle_probe.py = SpecInfer/STree path-rerun) = the RIGHT
  reference for a BRANCH node (native MTP has no branches to compare against).
- **Native-batched immune because:** independent requests each carry their OWN exact recurrent
  state, and the near-neighbor is NEVER offered as a candidate. Cross-request coupling = ~1 ULP GEMM,
  never accept-flipping (empirical: native+cache+realB8 Running=8 = 0.00%). Our tree does two things
  native never does: OFFERS the near-neighbor (a branch) AND seeds that branch's forward wrong.

## Localization — the seed is on BRANCH nodes (spine is clean)
- Spine conv prior-window was FIXED (3a9039cc: call2 conv1d_out 18.375→0.0) and chain5 (spine-only)
  kills the garble — so the SPINE path is clean.
- node5 = (0,1) = a BRANCH node still shows the 0.012 layer-0 drift.
- Every compute-only lever that attacked the SHARED/spine path is refuted: geometry (scan proven
  bit-exact both geometries, AND reverting warps reintroduces the spill = HBM tax), fp32-conv
  precision (≤2× vs 5.71× needed), M-invariance, committer (abandoned as a class). See
  FR13_GARBLE_FIX_DECISION.md, FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND.md.
- **UN-REFUTED, compute-only lever:** branch-node forward CORRECTNESS vs the path-rerun oracle,
  never tested on the current (post-conv-fix, stateless-baked) build.

## Reconciliation with the spine M-invariance work (FR13_WIDTH_CARRIER_INPROJ_BA_BIND.md)
The "M-invariance NO-GO" framing is CORRECTED. Spine M-invariance was FOUND and is BAKED:
- The +17 leaf co-residency carrier = the **bf16 in_proj_ba GEMM** (M-keyed: bf16 cuBLASLt Split-K
  differs M=10 vs M=5). FIX = pad in_proj_ba to fixed M (`LUMO_FB_PROJ_PAD_ROWS=16`, baked). Flips 26→18.
- The bind doc's refutations (qkvz/o_proj/conv M-invariant, state byte-exact) were all **SPINE-row**
  checks. So the residual ~18 flips = **BRANCH-node seeds the spine fixes never covered.** in_proj_ba
  pad covers all rows, so branches are ALSO in_proj_ba-M-invariant ⇒ the branch seed is a
  BRANCH-SPECIFIC op: the conv prior-window along the branch PATH, or the branch state read.

## Infra-rot blocker (2026-07-12): the branch-path oracle capture is DEAD on the stateless build
`fr13_branch_seed_localize.sh` v1 CRASHED-by-design: `FR10_TREE_GDN_CAPTURE_PAYLOAD` (the oracle's
input source) raises a fail-loud RuntimeError under `FR13_REPLAY_ROUTE=1` (patcher :4316) — it fed the
DELETED scan scratch. `REPLAY_ROUTE=0` would allocate an EMPTY scratch (scan store deleted) ⇒ garbage.
So `fr12_branch_path_oracle_probe.py` cannot be fed as-is. Killed the boot (concrete refutation).
- `FR12_SUBKERNEL_CAPTURE` DOES fire on the replay path (patcher :4336 tree, :5383 native) and taps
  per-node stages (input_hidden, pre_conv, conv1d_out, h0_state_in, gdn_scan_out).
- PATH FORWARD to re-enable branch localization: a CPU reducer that reconstructs each node's root→node
  path from the captured `pre_conv` (post-in_proj inputs, available on replay) and replays the native
  recurrent update — no patcher surgery. OR go straight at the branch fix: read the replay kernel's
  per-node conv/state handling, propose a compute-only branch-path fix, gate end-to-end on the temp-0.6
  garble gate (the ultimate test; doesn't need the oracle).

## 2026-07-12 HONEST STATUS: localization instruments are dead-ended; pivot to empirical fix-search
Four GPU boots trying to localize the branch sub-op ALL failed — recorded honestly, not rationalized:
1. payload/branch-oracle capture → fail-loud RuntimeError (rotted: fed the deleted scan scratch).
2. CAPTURE_ONLY warmup → 0 events (warmup too shallow to fire L0 capture).
3. MAB + FULL graph → engaged (sidecar) but host-syncs mis-run under graph capture + boot overran
   the 720s health window.
4. MAB + ENFORCE_EAGER → booted healthy (rc=0) BUT 0 MAB events + **18/18 syntax-errors** (0%
   undefined-rate is a MIRAGE — unparseable output). The MAB re-runs scan arms (M5/M1) DURING live
   decode with SIDE EFFECTS that corrupt the real forward. MAB is not viable for live localization.
**Decision: stop fighting the MAB/oracle instruments.** The garble is well-characterized (branch
co-residency forward drift, spine clean). Pivot to EMPIRICAL fix-search against the WORKING plain
garble gate (fr13_flag_garble_gate.sh, baseline 9.62%, native 0%, reliable): arm a compute-only
candidate, measure the garble rate, keep it if it drops — no MAB needed. Candidates: amplification-
reduction levers (fp32 state accumulation / rms-clamp / re-anchor, [[project_fr13_amplification_levers_queued]]).
in_proj_ba pad is B>=2-only (patcher :6077 `_nspec > 1`) and GEMM rows are independent, so it is NOT
the B=1 branch source; that lead is weak — not chasing it further without evidence.

## 2026-07-12 cycle: mis-diagnosis corrected + ship baselines + candidate vetting
- **CORRECTED my own error:** the MAB-eager run's "18/18 syntax errors" was NOT corrupted output — it
  was 18/18 `<HTTP 500>` (requests failed when MAB fires live). Normal syntax-error rate = 0/72
  (cat8_cacheon, cat8_bake, native all 0). MAB fails requests live; not viable — but "corrupts output"
  was wrong; "fails the requests" is right.
- **SHIP-CONFIG BASELINES (clean, syntax 0, the numbers the fix must beat):**
  cat8+cache=5.96% (one boot; 9.86–9.92% other boots — BOOT-VARIABLE), cat6+cache=9.91%, native=0.00%.
  So the fix needs same-boot A/B (or a drive-to-~0) since boot-variance (6–10%) can swamp a partial fix.
- **Candidates VETTED & REFUTED this cycle:** SSM-cache-dtype — already float32 (launcher:225), matches
  native, not the seed. (Adds to the refuted list: geometry, fp32-conv, committer, M-inv-spine-done.)
- **Amplification levers doc is SUPERSEDED (2026-06-15)** and not armable — the targeted-fp32 lever
  (fp32 at the RMSNorm gate + the ~4 deep full-attn hotspots where small act → large 1/rms multiplies
  drift) needs IMPLEMENTATION.
- **SOLE un-refuted compute-only direction:** targeted-fp32 at the amplification hotspots. It is
  UN-TESTED for garble (2026-07-10 tested fp32-CONV, NOT fp32-RMSNorm-gate / full-attn-hotspots) — so
  not refuted, just unbuilt. NEXT: implement (default-OFF flag, targeted not whole-model = no HBM tax),
  gate same-boot A/B on cat8+cache garble vs native-0%, then live SWE. Build carefully; don't rush.

## 2026-07-12 red-team corrections (two of my own leads weakened)
- **Conv prior-window is NOT the branch defect.** fr13_tree_conv_fused.py builds each node's conv
  window from ITS OWN root→node path (build_tree_conv_window_source_indices :98-104: last `width` rows
  of `[prior] ++ [width-1+path_node for path_node in path]`; state write-back :119-159 gathers per
  path). Branch conv is path-aware by construction ⇒ not the seed.
- **The node5-ladder 0.012 "branch drift" is CONFOUNDED.** node5=(0,1) is a BRANCH candidate; the
  ladder's "clean" reference is the full-context re-run of the COMMITTED sequence. A branch that was
  NOT the committed continuation legitimately differs from the committed clean — that 0.012 is a
  reference artifact (branch-vs-committed), not a proven branch bug. Same reference-confound the BV
  doc flagged.
- **What survives: chain5 (spine-only) kills the garble = the STRONG evidence** the garble requires
  branches. But WHY is now open between (a) branch nodes seeded wrong (needs the path-rerun oracle to
  confirm — rotted) vs (b) a residual M-dependent op still perturbing accept at branch commits
  (in_proj_ba fixed, but 18 flips remain). Conv ruled out; scan bit-exact; in_proj_ba padded ⇒ the
  live suspects are the branch STATE (h0 read/realization) and any un-audited M-keyed op on the
  branch commit path.
- **Only unconfounded next step:** rebuild the branch-path oracle on the WORKING FR12_SUBKERNEL_CAPTURE
  (CPU reducer: reconstruct root→node path from captured per-node pre_conv, replay native recurrent
  update, compare to captured branch scan_out). That is the clean branch-vs-OWN-path drift measurement.

## 2026-07-12 SCAN_ALIGN REFUTED (user's "it's a bias" reframe — scan-body tested)
FR13_SCAN_ALIGN=1 (match native's 3 scan-body BIAS seams: l2norm div-vs-rsqrt, beta bf16-roundtrip,
K1 carried-state bf16-roundtrip = the depth-growth recurrent carrier) applied to the served REPLAY
path via _gdn_node_step. Clean N=216 run, ENGAGED (worker FR13_SCAN_ALIGN=1), syntax 5/216:
**undefined-name-rate = 12.89%** vs default no-cache 9.62% = NOT a reduction (slightly worse).
=> The scan-body per-token PRECISION bias is NOT the dominant seed. (First boot failed on pure
720s health-timeout — SCAN_ALIGN=1 forces kernel recompile+graph re-capture; fixed via BOOT_TIMEOUT.)
NOT baking SCAN_ALIGN (refuted; keep fp32-carry default).

**Every NAMEABLE bias/seed candidate is now refuted:** scan-body precision (SCAN_ALIGN, empirical),
conv (already native-bf16-matched), geometry/BV (inert+harmful), in_proj_ba (B>=2-only, spine-fixed),
SSM-cache-dtype (already fp32), fp32-conv (asymmetric), committer (proven correct). The user's
insight holds (it's a systematic bias, ~492x amplified) but the remaining biased op is the
STRUCTURAL co-residency (branch present vs absent), which is UN-localized — every per-op localizer
(payload-oracle, MAB) is rotted or side-effectful on the current build (5 failed attempts).

**Datapoint:** cat6+cache=9.91% ~= cat8+cache=5.96-9.92% => garble does NOT scale with branch COUNT;
it's the PRESENCE of any branch co-residency (saturated), consistent with a structural (not additive)
co-residency seed.

**NEXT (honest):** the co-residency op needs a WORKING localizer. Options: (a) minimal read-only
eager branch-vs-native L0 probe (capture served M10 stages read-only + compute native L0 offline from
captured inputs — no MAB re-run side effects), or (b) dedicated rebuild of the branch-path oracle for
the stateless build. NOT another quick flag test — the nameable flags are exhausted.

## 2026-07-12 read-only FR12_SUBKERNEL_CAPTURE (6th attempt) — INFRA ALIVE, wrong events/stages
Read-only eager FR12_SUBKERNEL_CAPTURE (no MAB re-run, no rotted payload) FIRED on the current build:
42MB subk.pt, eager, NO 500 — the capture infra is alive (vs 5 prior total failures). BUT:
- Captured shape (2048,5120) = PREFILL forwards, not the 9-row tree decode. LIMIT=12 consumed all
  prefills before any spec-decode event.
- Stages = input_hidden/gate_out/o_proj_out (OUTER) only. MISSING the inner GDN stages
  (pre_conv/conv1d_out/gdn_scan_out/h0_state_in) needed to localize — those only populate on the
  tree-decode path (patcher:4336-4351), which wasn't reached.
NEXT (concrete): FR12_SUBKERNEL_CAPTURE_SKIP past the prefills (each garble-gate prompt = 1 prefill
then N tree decodes) to land on spec-decode events (num_spec_decodes>1, ~9 rows), and confirm the
inner-stage capture (h0_state_in @4351, conv1d_out) wasn't dropped in the stateless cleanup. Then the
CPU reducer (replay native fused_sigmoid_gating_delta_rule_update from h0_state_in+conv1d_out+weights
vs captured gdn_scan_out) gives the first non-confounded branch-vs-native drift number. NOT thrash —
the infra works; it's a SKIP/stage-selection fix, done fresh next cycle.

## 2026-07-12 ROOT-CAUSE of the capture wall: host-sync mid-forward crashes live decode
NUM_TOKENS=9 correctly targeted the tree decode: captured pre_conv (9,10240) with tree metadata
(tree_parent, spec_query_start_loc) — format + filter PROVEN right. BUT stages stop at
input_hidden+pre_conv; conv1d_out/h0_state_in/gdn_scan_out MISSING and score=syntax-errors 6/6 (all
500s). The .pt host-sync SAVE mid-forward crashes the request right after pre_conv. This is the SAME
failure mode as the MAB re-runs => **any host-syncing capture (FR12 .pt save OR MAB re-run) crashes
live decode on the stateless build.** That is the true root cause of all 6-7 localizer stalls — not a
missing flag. The Jun-6 captures that worked used a CONTROLLED single-forward driver
(fr13_node5_ladder_drive style), not the live concurrent garble gate.

**PATH FORWARD (instrument rebuild, pick one):**
(a) Controlled single-forward capture: drive ONE request through a garbling prompt, capture the ONE
    tree-decode L0 payload cleanly (no concurrent stream), like node5_ladder_drive. Lowest risk.
(b) Defer the FR12 save to forward-END (accumulate all stages GPU-side, single save after o_proj) so
    no mid-forward host sync — patcher surgery, higher blast radius.
(c) Device-side capture (copy to a pre-alloced device buffer per stage, D2H once at the end).
=> This is a focused instrument-rebuild task, NOT tail-of-session flag tweaking. The garble
characterization is complete (branch co-residency, ~492x amp, seed=branch h0-init, scan-math refuted);
the block is purely the capture instrument.

## 2026-07-12 ISOLATION VERDICT: eager CLEAN, FR12 capture is the crash (root cause proven)
Clean eager cat8 gate (NO capture, NO MAB): n=36, undefined-name-rate=9.32%, **syntax-errors=0/36**.
=> EAGER IS CLEAN (garbles normally at 9.32% with the classic near-neighbor identifiers like
runningBalanceaccumulator). So the host-syncing CAPTURE (FR12 .pt / MAB re-runs), NOT eager, crashes
the decode — root cause PROVEN, not assumed. Bonus: clean same-boot cat8 eager baseline = 9.32%.

REFINED crash locus: the warmup capture completed all 3 outer stages (input/gate/o_proj) fine; the
tree-decode capture broke right after pre_conv. So the TREE-PATH stage captures are the crash:
conv1d_out @3903 (create=True, reads mixed_qkv_spec + starts a NEW payload) and/or h0_state_in @4349.
The saved payload had input_hidden+pre_conv = the payload conv1d_out's create=True flushed before the
crash. Likely cause: the conv1d_out capture's create=True payload-boundary + host-sync during the
tree replay loop, OR a tensor changed by the stateless cleanup.

REBUILD (scoped patcher surgery, next focused cycle): fix the FR12 tree-path capture — either make it
NON-syncing (accumulate stages device-side, single D2H at forward-end) or fix the create=True boundary
so all inner stages land in one payload. Then eager (proven clean) + fixed-FR12 = complete branch
capture => the CPU native-replay reducer => the branch-vs-native sub-op drift. Eager is NOT the
blocker (proven); the FR12 tree-path capture is.

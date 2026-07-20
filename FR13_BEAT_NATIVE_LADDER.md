# FR13 BEAT-NATIVE LADDER — the bar is a WIN, not parity (2026-07-18)

Framing (user directive): we control the WHOLE pipeline — drafter, verify kernel, committer, tree shape,
scheduler. Stock native MTP-5 is the same model served naively; the deliverable must BEAT it, not tie it.

Reference (B=4 speed-gate basis: per-forward GPU + decode-bracketed accept, qwen-code nudge-free):
- native MTP-5: fullstep 27.9 tok/s, tps_gpu 75.96, per_req 5.49, accept 3.415, step ~158ms
- tail6 today:  fullstep 18.5,      tps_gpu 56.9,  per_req ~5.06, accept 4.317, step ~289ms

## Rungs (compounding; every rung gated: same-session A/B, matched pf/eff-concurrency, lossless, OFF=byte-identical)

| # | rung | mechanism | expected fullstep | status |
|---|------|-----------|-------------------|--------|
| 0 | native-committer bake | committer replay 99→75ms (linear fused path) | ~20 (+8% vs today) | cng16 GATE IN FLIGHT (CFWD 75.1ms @1650 spans, 4/16 tasks) |
| 1 | **ASYNC-SCHEDULING bake** | `--async-scheduling` overlaps host schedule/prepare/sample N+1 with GPU forward N (hides ~250ms host stall) | +14% cross-run (as1 n=16: fullstep 21.1 vs b7 18.5; accept 4.953; per_req 5.03 measured) | **PROMISING, NOT YET VALIDATED** — task #40's A/B lost its baseline arm (reaped at arm-2 start, 0 fatal; the "5.9>5.49" in its commit was the HYPOTHESIS, not measured). Async arm itself clean 16/16 on GDN-hybrid. => the async pair in the combined campaign is REQUIRED (same-session confirm + lossless gate), then bake |
| 2 | PIGGYBACK | committer →~16ms (fold accepted-path advance into next forward's fused scan) | ~27.3 alone; **~31 with async (+11% vs native)** | bundle scouts running (woi1w5mxi); seams 0/3/1a landed |
| 3 | verify tree-tax trim | 88→~70ms/draft (tree-scan cost toward native's 66) | ~34 (+23%) | after rung 2 proves |
| 4 | drafter CUDA-graph capture (LEVER 5) | drafter ~100→~50ms (collapse sequential M=1 launches); hardest invariant (in_proj_ba M-dep) ALREADY SOLVED by FR13_SLOT_REORDER | ~40+ (+45%) | design exists (plan L5); de-risked, not started |
| 5 | accept levers (numerator ×) | tail6realloc (zero-node d6 realloc, prepped); tail6-pb hybrid (K=8 chain + rare-overflow replay, P(accept>7)≈0.27); MTP-d6 seam (cost-gated) | multiplies all above | queued behind 2 |
| 6 | (different axis) CONC oversubscription | effective concurrency 1.3→~4 at max_num_seqs=4 (HBM amortization) | 2–3× aggregate tasks/hour, NOT per-stream | deployment economics; one sweep, queued |

## Sequencing (bundled, per user directive)
- NOW (GPU busy with cng16): apply the piggyback BUNDLE (all seams, flag-gated) + wire cat9_pb + bake
  --async-scheduling into the serve-variant arms for the validation campaign.
- WHEN cng16 lands: ONE combined campaign = {cat9_pb-ON, cat9-OFF} (piggyback gates) + {tail6+async, tail6}
  (clean same-session async confirm) — 4 arms; short-subset first for engagement/lossless, 16-task for winners.
- THEN rungs 3→4→5 in order, each on the proven predecessor.

Honest guards: async's accept 4.953 is trajectory-bound (cross-run) — only the same-session delta counts;
native+async should also be measured eventually (the fair endgame bar); piggyback lossless is within-floor
(1.19e-7 state-carry) → trajectory gate, not byte gate.

## RESOLVE GATE (user-mandated, 2026-07-18): verdict pass-count ~8/16-ish; drifting below = issue signal
Measured (subset_b4_sixteen, WALL=1800): native 6/16 passed (2 wall-truncated, 5 tests_failed = completed
attempts); tail6 b7 3/16 (12 TRUNCATED mid-work); async as1 1/16 (9 truncated, 6 ended-text); cng16 interim
1/8 (5 truncated). TRACE CLASSIFICATION: zero give-up texts, zero garble — truncated traces end with clean
mid-investigation tool calls. => the tree's resolve deficit is WALL-CENSORING (34% slower => agent gets
fewer turns in the fixed 30min wall => truncated => empty_patch), NOT token-quality degradation.
IMPLICATIONS: (1) the speed deficit ALREADY costs ~2x resolves at deployment-faithful walls — speed converts
directly to resolutions; (2) resolve-recovery = the cleanest end-to-end ladder gate: as rungs land, tree
truncations must convert to attempts/passes toward native's band (~6-8/16 on this subset); failure to
recover once speed is fixed => THEN suspect behavioral/token issues; (3) per the no-wall-on-gates policy,
LOSSLESS gates treat wall-tripped as NA (right-censored) — but the fixed-wall resolve count is the honest
DEPLOYMENT metric and is now reported per arm alongside accept/CFWD/tps. WATCH: async's 6 ended-with-text
(vs tail6's 1) — final text without applied patch; classify during the async lossless gate.

## RESOLVE GATE CORRECTION (user, 2026-07-18): resolve is measured WALL-FREE (WALL=0) — consistent with the
## established speed-gate policy (no AGENT_WALL_S on gates; trace-inactivity watchdog = hang protection)
My WALL=1800 launches today (cng16, p1g1, del1/2, via1) were a METHODOLOGY DRIFT from the no-wall gate
policy; the driver's 1800 default is deployment-faithful but right-censors gate signal. Consequences:
- ALL wall-censored resolve numbers above (native 6/16, tail6 3/16, async 1/16) are NA for the RESOLVE GATE
  (they measure speed×wall, not quality). The user's ~8/16 band is the WALL-FREE basis.
- Wall-free, resolve = pure behavioral/quality parity gate (agent runs to natural completion; retries ~2x);
  speed shows up separately as wall-clock/task + the per-forward GPU metrics. Truncation-conversion applies
  only to walled DEPLOYMENT reporting (kept as a secondary deployment-faithful view, clearly labeled).
- FUTURE GATE CAMPAIGNS: WALL=0 (driver emits empty AGENT_WALL_S). The combined ladder campaign runs WALL=0;
  its tail6_base arm gives the wall-free tree resolve baseline vs the ~8/16 band.
- cng16 (in flight, WALL=1800): its accept/CFWD reads are wall-independent (decode-bracketed + per-call
  span timers) => still valid for the FR13_COMMITTER_NATIVE bake; its resolve = NA (wall-censored). The
  wall-free resolve read for the native committer rides the combined campaign (post-bake arms).

## R5 REFRAMED (2026-07-18, from the cat9-vs-deliverable discussion): geometry re-optimization under pb costs
cat9_pb = the MECHANISM-PROOF vehicle only (locked baseline; depth-5 accept fits the K=8 chain with 100%
coverage; 18 streams fits the n_pad=32/BV=8 wall). The ~99ms replay being eliminated is geometry-INDEPENDENT
(same 48-kernel machinery for every tree) => the CFWD collapse proven on cat9 transfers to the family.
DELIVERABLE geometry = re-decided AFTER the mechanical+V2.5 gates, by re-running the geometry optimization
UNDER THE PIGGYBACK COST MODEL — the old conclusions (cat8-near-optimal, tail6 break-even 0.138 accept/node,
depth-lever-dead) were all derived at replay-era committer cost (~99ms); at ~16ms the calculus shifts
(depth cheaper, branch break-even lower). Candidates: tail6-pb HYBRID (K=8 chain + fallback replay on
accept>6 overflow, ~30% of steps => ~70% of the collapse, keeps accept 4.317) vs cat9-family widened under
the new break-even vs a re-swept shape. This is a MEASUREMENT (same-session sweep), not an assumption.

## R5 SWEEP SPEC (user directive 2026-07-18): the (n, x, branching) two-proposer sweep under the pb cost model
The deliverable-geometry decision = re-run the TWO-PROPOSER sweep (FR13_TAIL6_IMPROVEMENT_PLAN.md framing:
MTP head depth n; arctic tail length x; branch width/placement w_over/w_tail/tail_bd) with the calibrated
survival model (fr13_tail_config_sweep.py, calibrated at tail6; re-calibrate uplifts from the b7-era
measured conditionals), but with the COST side replaced by the PIGGYBACK-ERA model:
  tps(n,x,b) = committed(n,x,b) / step_ms(n,x,b)
  step_ms    = drafter(n MTP forwards + arctic host) + verify(node_count, depth) + committer_pb
NEW CONSTRAINTS the replay-era sweep did not have:
  1. CHAIN BUDGET: piggyback consumes 8 of the 32 n_pad slots => tree budget 24 nodes (was 32) for
     full-coverage chains; deeper trees (max committed > 6, i.e. n+x > 5) OVERFLOW the K=8 chain =>
     committer_pb becomes the HYBRID BLEND: P(accept<=6)*~16ms + P(accept>6)*replay(~70ms w/ native
     committer baked). The overflow probability comes from the measured per-depth survival profile.
  2. committer_pb replaces the flat ~99ms; the branch break-even (accept/node) drops accordingly —
     re-derive it from the measured cat9pb-vs-cat9f CFWD before sweeping.
  3. Depth is cheaper (no replay-depth cost) but verify still scales with nodes; the K=8 slots also
     pay verify cost (identity rows are cheap but not free) — measure the 18-stream vs 10-stream
     s_per_fwd delta in the pbmech pair and feed it in.
Sweep output = the deliverable (n, x, b) shape; then ONE same-session confirm campaign (winner vs tail6-pb
hybrid vs native) on subset_b4_sixteen, WALL=0, wall-free resolve gate. Do not hand-pick a shape without
the sweep (feedback: swap MTP-n / tail-x / branching with the cost model).

## MEASURED-TPS ALIGNMENT GATE (user directive 2026-07-19): derived must align with measured
User: "We must have a measured TPS to align with derived TPS, otherwise we would be wrong if
there's extra overhead outside three components."
- `derived_tps_fullstep_gpu` = committed/(verify+drafter+committer GPU) is a compute-basis
  UPPER BOUND — blind to host glue (input prep, sampler, chain packer, bookkeeping, scheduler
  gap). A cross-arm verdict on the derived number alone is invalid if the arms carry different
  out-of-component overhead (piggyback moves work INTO exactly those buckets: packer, repair
  scatters, interleave markers).
- MEASURED twin (FR13_STEP_WALL, wired 2026-07-19 into the sfwd sidecar + snapshot synthesis +
  reducer): start-to-start wall deltas between CONSECUTIVE pure-decode steps (chain broken on
  mixed/prefill steps; deltas > FR13_STEP_WALL_CAP_S (default 1.5s) rejected so agent think-time
  idle cannot pollute). Fields: `measured_tps_fullstep_wall`, `wall_s_per_event`,
  `overhead_other_ms_per_event` (= wall − 3 components), `fullstep_alignment_ratio`
  (= derived/measured).
- GATE (applies to EVERY speed verdict from now on, incl. the pb CFWD-collapse verdict and all
  R-rung A/Bs): quote derived AND measured; if alignment ratio drifts far from ~1, the verdict
  MUST be made on the measured number and the residual bucket investigated per-arm.

## R2 PIGGYBACK: BAKED (user decision 2026-07-19, golden-rule basis)
pbm1 clean pair (WALL=0, async on, 16-task): cat9pb resolve **9/16** (band 8-9/16; the earlier
0/16 read was a reader bug — eval_report.json verdict is the TOP-LEVEL "passed" field),
accept 3.385 == cat9 baseline 3.397 (ACCEPT-NEUTRAL), committer 70.7 -> ~10-12 ms/span (5.7x),
measured_tps_fullstep_wall 27.82 (vs bar 27.9) at the weakest-accept proof shape. ~6h live
agentic decode, zero guard fires. User: bake on the golden rule; V2.5 restored-vs-oracle moves
to POST-bake verification (explicit user override of the pre-bake ship rule).
- cat9_pb is the reference arm for R3/R4 A/Bs from here.
- TAIL6 CAVEAT (user-caught confound): ALL historical tail6 resolve numbers are WALL-suspect —
  cng16 (3/16) ran WALL=1800 (env snapshot), right-censoring ~90-min tasks. rg1 (in flight,
  WALL=0) is the FIRST honest tail6 resolve read; the deliverable-shape decision (tail6 port vs
  sweep shapes vs cat9pb-as-is) waits on it + the rg2 async twin (R1).

## SEQUENCING LOCK (user directive 2026-07-19): R1→R2→R3→R4 to COMPLETION before ANY sweep work.
The (n,x,branching) sweep + geometry ports (tail6-x8, 30-node shapes, BV re-tile) ALL belong to R5
and are PARKED until R1-R4 are done. The wall-free sweep recalibration (lad2 per-position) stands as
R5 prep only — no sweep-driven campaigns before R4 closes.
- R1: rg1 (tail6 wall-free, IN FLIGHT) -> rg2 (async twin) -> decision+bake; native wall-free pair
  (fr13_native_nowall_seq.sh) re-anchors the bar in the same window.
- R2: BAKED.
- R3 verify tree-tax trim: DESIGN NOW (CPU, while the resolve matrix holds the GPU) -> flag-gated
  impl -> same-session A/B after R1 arms finish.
- R4 drafter CUDA-graph capture: after R3 (design de-risked by FR13_SLOT_REORDER).

## MEASURED-TPS IS THE COMPARISON BASIS (user directive 2026-07-19, strengthens the alignment gate):
ALL cross-arm comparisons and rung verdicts use `measured_tps_fullstep_wall` (FR13_STEP_WALL).
`derived_tps_fullstep_gpu` is DIAGNOSTIC ONLY (component decomposition; known-invalid under async
where component spans overlap — pbm1: derived 18.3 vs measured 27.8). Never rank arms on derived.

## CACHE-ON FROM THE NATIVE PAIR ONWARD (user directive 2026-07-19)
rg1/rg2 finish cache-OFF (their own matched pair). Every subsequent arm — the native wall-free
bar pair (nativemtp5apc kind), R3/R4 A/Bs, R5 confirms — runs APC cache ON: (a) validates the
spec stack under the ship config (GOAL = spec+cache lossless), (b) prefix reuse speeds campaigns.
Tree cache-ON arms use the solved APC stack (HIT_RECURRENT_SUFFIX + overshoot fix +
ZERO_MAMBA_ON_ALLOC/COPY_SRC_FIX baked); a tail6/cat9_pb cache kind gets wired when its first
arm is scheduled. Cross-pair comparisons stay within-cache-config; the final bar compare =
tree-cache-ON vs native-cache-ON.

## rg1 VERDICT (2026-07-19, tail6 wall-free no-async cache-OFF, 16 tasks): tail6 REHABILITATED
resolve 9/16 (band met; walled-era 3/16 = censoring artifact). accept bracketed 4.892 (+0.58 pure
wall effect over walled 4.31). measured_tps_fullstep_wall 30.26 — ABOVE the old native bar 27.9
with committer still at 85.5ms (pb un-ported) and no async. eff-conc 1.63, pf 0.50. Residual
async-accept question: 4.892 vs lad2 5.469 — rg2 (async twin, SAME code) running, decides R1.
pb-port-on-tail6 projection at rg1's own numbers: ~40 wall TPS pre-R3/R4. Reducer note:
overhead_other_ms needs eff-conc normalization (true non-component overhead ~+4ms/event here).

## DRIVE PLAN (owned, 2026-07-20) — every GPU window pre-assigned
GPU queue (fires in order as windows free; loop ticks execute):
1. rg2 tail6+async (RUNNING) -> R1 verdict: accept attribution (vs rg1 4.892 same-code /
   lad2 5.469 same-config), resolve band, measured-wall delta. Bake decision per standing rule.
2. tail6_pb MECHANICAL (fr13_tail6pb_mech_seq.sh, 4-task): first boot of the ported pb on the
   deliverable shape — needles(29), 0 fatal, CFWD collapse vs 85.5ms. PASS -> full-subset
   tail6_pb resolve run becomes the R2-on-deliverable bake candidate (~40 TPS projection).
3. native cache-ON pair + native11 depth-match (fr13_native_nowall_seq.sh, 16-task):
   re-anchored fair bar + the depth/branch/mixed-drafter accept decomposition.
4. COMPOSITION PROBE (fr13_tail6_pureprobe_seq.sh): tail6 sync + chunked-prefill OFF —
   decides trajectory-feedback vs composition-numerics as the accept-why (pure-fraction data:
   async 5.5% pure vs sync 12.7% REFUTED the pure-shape-consistency story).
5. V2.5 restored-vs-oracle on the pb arm (post-bake formal check) + R3 profiler one-shot
   (FR13_R3_VERIFY_TRIM_DESIGN.md step 1) in the same window; then R3 impl -> R4.
Reducer upgraded: overhead_other_ms_per_event_norm (eff-conc-normalized residual) in every
deploy record. WHY-ledger so far: wall censoring (+0.58 sync), async assoc. lift (+0.64
matched, d6-concentrated, 10/12 tasks), pure-shape story refuted, trajectory-feedback leading.

## R1 CLOSED (2026-07-20, rg2c mid-run attribution): ASYNC BAKES
The accept triangle resolved: lad2(old code,async)=5.469; rg1(gate-bug,sync)=4.892;
rg2(gate-bug,async)~4.80; rg2c(gate-FIXED,async)=5.36@10.9k RESTORED. The "async accept lift"
was my CPU-placeholder gate suppressing the two middle arms (stale-root conditioning); with the
gate corrected, async ~= accept-neutral-or-better + overlap speed => --async-scheduling BAKES.
rg2c closed mid-run (attribution complete); ALL-ON arm supersedes.

## ALL-ON BASELINE ATTEMPT (user directive): tail6_pb + APC cache + async, 16 tasks
fr13_allon_seq.sh. All-flags-forward playbook; boot needles fail fast; peel order
async->cache->pb. On-band resolve => NEW BASELINE (ship config) and all follow-ups
(R3 profile/impl, R4, native re-anchor, sweeps) run FROM this config.

## ALL-ON BASELINE = LIVE-VALIDATED (allon5, 2026-07-20)
allon5 (tail6_pb + APC cache-ON + async, WALL=0, 16 tasks) COMPLETED CLEAN — the tail6_pb
PORT is validated end-to-end on the deliverable shape:
- **Resolve 9/16** (ON-BAND: == R2 cat9pb bake 9/16 == rg1 tail6 9/16). 7 failed.
- **All 3 novel mechanisms engaged LIVE, 0 fatals / 0 raises**:
  - `RESHAPE_WIDE engaged: depth=11 nodes=21` on the DE-ROOTED base subtree
    `[(0,),(1,),(2,),(0,0),(0,1),(0,2),...]` — the wide-on-base fix (97a8b71d1) worked;
    chain-prefix stripped exactly.
  - `extended drafter engaged: 29 cols (8 chain + 21 base)` — packer fix worked.
  - `committer GDN replay DROPPED` + overflow-fallback fired once (deep-tail accept>6
    handled by the len-0-identity + C-INT-2 catch-up path; no crash).
- accept_per_event **4.417** (committed 5.417), engaged 29/29 tok_per_draft.
- measured_tps_fullstep_wall **23.67** @ prefill_frac **0.616** (HIGH), eff_conc 2.07.

SPEED NOT YET A VERDICT (basis unmatched): the baked cat9pb 27.82 was measured CACHE-OFF
async; allon5's 23.67 is CACHE-ON async at pf 0.616. The gap conflates (a) deeper-tree
drafter+committer host cost (drafter 103 ms/step, committer 53.7 ms/step over 29 cols) with
(b) the cache-ON chunked-prefill regime (high pf). Per BASIS rule "same-campaign matched
pf/eff-conc only", a MATCHED cache-ON A/B is required before ranking tail6_pb vs cat9pb.

NEXT (matched speed gate): one cache-ON async campaign, same 16 tasks, arms = cat9pb +
tail6_pb + nativemtp11apc (depth-matched native). Isolates tree-shape from regime and
answers both "beat our baked baseline?" and "beat native at matched depth?" in one matched run.

## ACCEPT REGRESSION under piggyback+cache (allon5 4.417 vs rg2c 5.36) — ISOLATING
User flagged: the accept>5 restoration (stale-root fix, FR13_REPLAY_ROUTE=1) — is it
in the ship config? VERIFIED against container_env.txt:
- rg2c = plain tail6, **pb OFF, cache OFF**, async, fix ON -> accept **5.36** (mid-run).
- allon5 = tail6_pb, **pb ON, cache ON**, async, fix ON -> accept **4.417**.
The fix is engaged in BOTH; 5.36 was NEVER demonstrated *with* piggyback. The -0.94 drop
is the two vars rg2c lacked (pb + cache), NOT a missing fix.
OPEN + sharp: is the stale-root fix EFFECTIVE under pb? It keys on `_fr13_all_neg`
(scheduled spec list all-placeholder = async first-spec signature). Under pb the packer
fills the 8 chain cols; if that defeats the all-neg signature, pb rows fall back to the
buggy CPU-gate. Cannot prove statically (sched list vs packer tensor are different buffers).
Per-pos (allon5 vs rg1): deep-tail conditional 0.81-0.91 vs 0.87-0.94 AND a sharper
head->tail boundary drop (pos4->5 0.58 vs 0.68) -- consistent with a state-quality
regression, not a missing suffix mechanism (the tail IS drafted+accepted).
mab KILLED (both arms pb+cache = can't isolate). Running acciso 2x2 completion:
  arm1 tail6_pb cache-OFF (bisect: vs rg2c=isolate pb, vs allon5=isolate cache)
  arm2 plain tail6 cache-ON (isolate cache on pb-free shape)
Decisive: arm1 ~5.3 => pb innocent, cache carries; arm1 ~4.4 => pb carries, fix neutered => localize+force root-repair on pb rows.

### STATIC TRACE (2026-07-20, arm1 booting): stale-root fix is NOT neutered by pb
Traced the host spec_token_ids path in the live container source:
- repair reads `scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens` (runner:1631).
- scheduler:557 sets `scheduled_spec_decode_tokens[req]=request.spec_token_ids` (prev-step draft ids).
- tree_mtp arms take the NATIVE async staging (scheduler:1750); `select_path0_spec_tokens` is
  NAIVE_MTP-only, does NOT touch tail6/tail6_pb.
- Under async the drafter output is the GPU `_draft_token_ids` tensor (runner:1699,1989) scattered
  into input_ids at the NEXT forward (1976-1994); the HOST spec_token_ids stages `-1` placeholders
  regardless of pb. The pb packer fills the GPU tensor, NOT the host list. No pb code writes host
  spec_token_ids (grep: only NAIVE_MTP select_path0 + native assigns).
=> `_fr13_all_neg` (all-placeholder) fires IDENTICALLY for tail6_pb (29 slots) and plain tail6 (21).
   The stale-root TOKEN repair engages under pb. "Fix neutered by pb" hypothesis REFUTED by code.

REFINED suspect (survives the trace): the tail6-SPECIFIC OVERFLOW FALLBACK (accept>6 -> len-0
identity chain + C-INT-2 catch-up replay root_node=8) -- BRAND-NEW code, FIRST exercised in allon5.
- Fires ~19.5% of events (allon5 pos6 accepts 5170/26483). cat9pb NEVER overflows (accept 3.4<<6),
  which is why R2 measured pb ACCEPT-NEUTRAL (3.385==3.397) -- that test never covered this path.
- Mechanism: on overflow the chain does NOT re-derive col-0; the catch-up replay must restore it.
  If catch-up != chain re-derivation byte-wise, overflow rows get a degraded col-0 GDN state ->
  lower NEXT-step deep draft quality -> the deep-tail conditional drop seen in allon5 (0.81-0.91 vs
  rg1 0.87-0.94). Consistent with the per-pos signature.
ARM1 (tail6_pb cache-OFF) discriminates: ~5.3 => pb/overflow innocent, CACHE is the carrier;
~4.4 => pb OVERFLOW is the carrier => localize catch-up-vs-chain col-0 discrepancy (my code).

## COMPONENT-COST TARGETS vs NATIVE (user directive 2026-07-20: match native best-effort on all 3)
Native MTP-5 (nt1): drafter 93.1ms | committer 7.2ms | verify 0.181 s/fwd | accept 3.415.
tail6_pb (allon5):  drafter 103.4  | committer 53.7   | verify 0.413 (conc 2.07; ~0.20/seq) | accept 4.417.
GAPS: committer +46ms = THE gap (piggyback halved 99.4->53.7; rest = host DtoH loop ~36ms +
GDN-commit 17.8 vs 7.2 ~11ms). drafter +10ms (R4). verify near-native per-seq.
LEVERS:
- committer host loop -> FR13_GPU_COMMITTER=1 + FR13_COMMITTER_SYNCKILL=1 (G2 device-resident +
  side-stream, byte-identical by construction; built fr13_gpu_committer_kernel.py; DEFAULT-OFF,
  NEVER live-validated). Gate = byte A/B (class-10) + s/fwd vs native. Drops ~36ms -> ~17.8ms.
- drafter -> R4 CUDA-graph capture (~103 host-bound).
- accept 4.4->5.4 -> overflow-fallback fix (col-0 workflow running).
NOTE overflow catch-up replay (~19.5% steps x ~70ms per-layer replay) lands in overhead_other
(~13.7ms/event), NOT the committer span -> fixing overflow ALSO cuts wall overhead, separate from
the GPU-committer lever. Two distinct committer/overhead workstreams.

## CORRECTIONS + STATE (2026-07-20)
1. ACCEPT ATTRIBUTION CONFIRMED (arm1 killed early per user, decisive): tail6_pb cache-OFF
   accept 4.364 @4261 drafts == allon5 cache-ON 4.417 (cache delta ~0.05, negligible), both
   ~0.9 below rg2c plain-tail6 5.36. => PIGGYBACK is the accept carrier, NOT cache. Overflow
   fallback = the specific suspect (col-0 workflow localizing). 2x2 answered; plain-tail6
   cache-ON arm skipped (confirmatory only).
2. COMMITTER LEVER CORRECTION (my earlier GPU_COMMITTER claim was WRONG): FR13_GPU_COMMITTER is
   the GREEDY LCP committer, OFF the temp-0.6 path. The deployed temp-0.6 committer is
   fr13_device_multidraft_commit (FR13_DEVICE_MULTIDRAFT, BAKED on-device). So the 53.7ms CFWD
   is NOT a host compute loop GPU_COMMITTER would fix; it is multidraft-kernel + result-DtoH +
   verify-wait. RIGHT lever = decompose via FR13_MULTIDRAFT_GPU_TIMER first (measure before
   claiming). commdecomp arm (tail6_pb ship, cd1) running to split it.
   Committer strategy TBD by cd1: kernel-heavy => optimize multidraft kernel; residual-heavy =>
   async/overlap the result DtoH.
3. Native component TARGETS stand: drafter 93.1 (tail6_pb 103, +10 = R4), committer 7.2
   (tail6_pb 53.7 = THE gap, strategy pending cd1), verify near-native per-seq.

## ACCEPT REGRESSION = LARGELY ARTIFACT (2026-07-20, task-matched + workflow-verified)
Two independent results collapse the "piggyback ate 0.8 accept" narrative:
1. Workflow wgrmh5q1l (read-only, adversarial): ALL 9 col-0 divergence candidates in the
   overflow catch-up REFUTED. No single-step col-0 bug; the catch-up restores col-0 byte-correctly.
2. Task-matched accept (allon5 PB vs rg1 no-PB, 16 tasks each): median delta **+0.008**
   (NEUTRAL); 13/16 tasks allon5 ties-or-beats. The mean -0.274 is 3 tasks ONLY: 14539(-1.24),
   14598(-1.36), 14995(-2.05) -- allon5 collapses to ~3.6-3.9 there, rg1 holds 4.9-5.7.
=> The "5.4->4.4" gap was rg2c MID-run (10.9k drafts = early high-accept tasks) vs allon5 FULL
   run (incl the 3 collapse tasks). NOT a uniform pb bug. Almost implemented a wrong col-0 fix;
   the workflow + task-match caught it.
REAL OPEN = the 3-task end-of-run collapse. Candidates (all NEW, not col-0):
  (a) CACHE-ON late-run degrade (allon5 cache-ON, rg1 cache-OFF) -- ship-config risk.
  (b) OVERFLOW ULP-ACCUMULATION over thousands of steps on deep tasks (workflow tested per-step
      byte-neutrality, NOT accumulation; diffuse-GDN precedent).
  (c) async trajectory divergence on those tasks.
  Attribution needs a cache-OFF FULL run (arm1 was killed at 3 tasks) or a per-task overflow-rate
  probe on 14539/14598/14995. DEFER behind the committer lever (cd1).
COMMITTER remains the confirmed native-matching lever (53.7 vs 7.2); cd1 decomposing.

## COMMITTER DECOMPOSITION (ac1 cache-OFF, 900 spans) — the strategy
FR13_MULTIDRAFT (rejection kernel):  4.4 ms  -> near-native, NOT the cost.
FR13_COMMIT_FULL (kernel+assembly+GDN publish): 23.6 ms -> assembly+publish ~19.2 ms.
CFWD (full _sample dispatch, allon5): 53.7 ms -> DtoH+verify-wait ~30 ms (CFWD - commit_full).
=> The 53.7ms committer is PIPELINE overhead (host assembly/publish ~19 + DtoH/wait ~30),
   NOT compute (kernel 4.4). Matches why native committer = 7.2 (cheap linear commit, minimal
   DtoH). FIX = async-overlap the assembly/publish + result-DtoH off the critical path (side-
   stream materialization + CUDA event; the AsyncGPUModelRunnerOutput shape). Kernel opt is dead
   (already 4.4ms). Impl = author the async committer materialization; validate byte-identity +
   CFWD collapse. NOT FR13_GPU_COMMITTER (greedy, off temp-0.6).

## COMMITTER FIX DESIGN (localized, ready to implement when GPU frees)
Root of the ~50ms: AsyncGPUModelRunnerOutput (gpu_model_runner.py:6308, used under
--async-scheduling) defers ONLY the final sampled_token_ids. The TREE committer's own DtoH +
output-row assembly (accepted_tree_rows, committed tokens, GDN path/len publishes) runs
SYNCHRONOUSLY before that boundary, inside _lumo_tree_canonical_multidraft_sample (patcher
:9618). Native's cheap output rides the async stream; the tree committer's does NOT.
FIX = device-resident multidraft committer + side-stream materialization (the SYNCKILL concept
applied to the TEMP-0.6 rejection committer, NOT the dead greedy FR13_GPU_COMMITTER): keep the
walk outputs as DEVICE tensors, materialize host copies on a side stream + CUDA event, and let
the GDN path/len publishes (needed pre-next-forward) stay device-keyed. Target: 53.7 -> ~24ms
(commit_full), then chip the 19ms assembly. Author flag-gated + byte-identity gate + CFWD collapse.
BLOCKED on GPU (acctrl running for the accept-collapse attribution -- the user's priority).

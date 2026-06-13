# FR13 cat10 (root-sibling) B=1 Gate — DRAFT bind (no commit; monitor commits)

Date: 2026-06-13 UTC. HEAD 813cb9fd. Subagent GPU run, serialized (one container at a
time), recover_host_memory + free-g check before each boot. Artifacts: `output/fr13_cat10/`.

## What was tested
cat10 = the 10-node caterpillar that adds the MISSING depth-0 (root) top-2 sibling node
`(1,)` to cat9, flag `FR13_CAT10_ROOT_SIBLING=1`. Goal (user 2026-06-13): rescue the d0
rejects (62% of greedy rejects were at step-0/d0 with no root-branch rescue) -> more
accept/event WITHOUT making it slower.

4 arms booted (serialized, FLASH_ATTN native / TREE_ATTN tree, qwen3.6-27b-fp8, B=1,
greedy seed 1313, pinned `output/fr13_acceptance_ladder/prompts_swe4.json`):
- **oracle** = no-spec recurrent oracle (`FR12_NO_SPECULATIVE_CONFIG=1`, FLASH_ATTN,
  speculative_config=None) — the CLEAN lossless reference, teacher-forced per served pos.
- **cat9** = control (`FR13_CAT10_ROOT_SIBLING=0`, all other FIX-1/2/3/A ON), num_spec 9.
- **cat10** = `FR13_CAT10_ROOT_SIBLING=1`, num_spec 10.
- **native_e5** = `fr10_launch_speed_server.sh` SPEC_CONFIG num_speculative_tokens=5,
  FLASH_ATTN (depth-matched linear MTP-5 baseline).

## ENGAGEMENT (class 9) — PASS
- cat10 booted with `num_speculative_tokens=10`, tree includes `(1,)` root sibling,
  `FR13_CAT10_ROOT_SIBLING=1` in container env. Drafter log (engagement proof):
  `FR13_CAT10_ROOT_SIBLING drafter topology: cat10=True (env=1) is_cat10=True is_cat9=False
  is_spine_only=False num_spec=10`. draft-toks/event measured = **10.0** (non-vacuous).
  No "caterpillar drafter disengaged" raise.
- cat9: num_spec 9, flag 0, draft-toks/event = 9.0. native_e5: num_spec 5, draft-toks/event = 5.0.

## WITHIN-BOOT DETERMINISM (class 8) — PASS, all arms
rep1 == rep2 byte-identical served streams on ALL 4 prompts for cat9, cat10, native_e5,
and oracle (greedy). 4/4 each.

## AXIS 1 — LOSSLESS (binding per-token argmax-vs-CLEAN-oracle, thr=1.0 nat)
fr13_verify_bisect_probe classify, CLEAN ref = the no-spec recurrent oracle (teacher-forced
max_tokens=1 per served position on byte-identical served prefix).

| arm | clear-margin flips | positions | flip rate | per-prompt |
|---|---:|---:|---:|---|
| cat9 (control)  | **22** | 482 | 0.0456 | [6,6,4,6] |
| cat10           | **22** | 457 | 0.0481 | [2,6,8,6] |

- **cat10 does NOT improve lossless and does NOT regress it meaningfully.** Same absolute
  flip count (22). Per-position rate marginally HIGHER (4.81% vs 4.56%) only because cat10's
  prompt-0 greedy stream is shorter (73 vs 98 tokens => fewer positions).
- The root sibling DID redistribute flips: prompt 0 dropped 6->2 (real p0 help), but
  prompt 2 rose 4->8. Net flat. The known deep-row committer non-argmax defect persists.
- cat9 reproduced the banked **22** exactly (per-prompt [6,6,4,6], both known deep-row flips
  present) -> oracle reference validated.
- (native_e5 vs oracle = 95 flips is cross-config trajectory drift: native diverged from the
  oracle greedy path at ~pos 6, so downstream teacher-forced positions are off-trajectory.
  NOT a native losslessness floor; do not read it as a flip baseline.)

## AXIS 2 — ACCEPT (d0 rescue + total)
**Whole-window accept/event is TRAJECTORY-CONFOUNDED** (each arm generated a different
greedy stream; served_lens differ): cat9 [98,128,128,128], cat10 [73,128,128,128],
native_e5 [128,128,86,51] (p2/p3 hit a natural EOS early this boot). Per the gold-bind,
whole-window accept is a DRAW, never a superset verdict. Reported as raw counters (class 12):

| arm | accept/event | per-depth accept RATE d0 / d1 / d2 / d3 / d4 |
|---|---:|---|
| cat9       | 3.1983 | 0.871 / 0.828 / 0.638 / 0.483 / 0.379 |
| cat10      | 2.9316 | **0.906** / 0.726 / 0.564 / 0.419 / 0.316 |
| native_e5  | 1.3631 | 0.554 / 0.357 / 0.274 / 0.107 / 0.071 |

- **d0 RESCUE = REAL but small: d0 accept rate 0.8707 -> 0.9060 (delta +0.0353).** The root
  sibling does rescue some step-0/d0 rejects (the structural intent worked at d0).
- **BUT total accept/event DROPPED: 3.198 -> 2.932 (-0.27).** The d0 gain is MORE than offset
  by drops at every deeper spine position (d1 0.828->0.726, d2 0.638->0.564, d3 0.483->0.419,
  d4 0.379->0.316). The extra root-sibling verify slot dilutes deeper acceptance (the
  committer takes the shorter root-sibling path or the tree's co-residency shifts), so the
  net per-event yield falls.
- native_e5 this boot = 1.36 (trajectory-confounded; p2/p3 EOS'd early). The Jun-10 native
  ladder boot on identical seed/prompts gave 3.154 (full 128-tok streams) — the cross-boot
  spread is the documented native self-floor / trajectory sensitivity, NOT comparable as a
  fixed superset bar here.

## AXIS 3 — SPEED
| arm | s/fwd (decode_s/drafts) | warm decode TPS | per-req decode TPS median | verify rows |
|---|---:|---:|---:|---:|
| cat9       | 0.2258 | 18.40 | 16.01 | 9 |
| cat10      | 0.2287 | 17.08 | 15.17 | 10 |
| native_e5  | 0.2171 | 10.78 |  9.72 | 5 |

- cat10 s/fwd = **1.013x cat9** (+0.0029 s = ~2.9 ms/fwd) — the predicted +1 lm-head verify
  row tax (cat10 10 rows vs cat9 9). As expected (~+15 ms/row claim is an upper bound; here
  ~3 ms at B=1).
- **NET TPS REGRESSED:** warm decode TPS 18.40 -> 17.08, per-req median 16.01 -> 15.17. The
  +1-row cost was NOT offset by the d0 accept gain because total accept/event FELL.

## VERDICT — no_help (root sibling does not net-help; mild speed cost, lossless flat)
- Lossless: FLAT (22 == 22 flips; rate 4.81% vs 4.56%, within band). NOT regressed, NOT improved.
- Accept: d0 rescue REAL (+0.0353 d0 rate) but **total accept/event DROPPED 3.198 -> 2.932**
  because deeper-position acceptance fell across d1-d4. The root branch helps d0 but costs more
  downstream.
- Speed: cat10 +1.3% s/fwd (the +1 row), net TPS regressed (no accept gain to offset).
- **cat10 is NOT a net win on any axis.** The d0-rescue hypothesis is partially validated
  (d0 rate did rise) but the extra root-sibling row dilutes the spine's deeper acceptance more
  than the d0 gain recovers, and the binding lossless flip count is unchanged. RECOMMEND keep
  cat9 as the default; cat10 flag stays OFF.

## DRAWS / caveats (class 12)
- Whole-window accept/event and native_e5's absolute number are trajectory-confounded draws,
  not gate criteria. The per-depth RATE and the within-arm d0 delta are the fair reads.
- The 22-vs-22 flip equality and the d0 +0.0353 are the load-bearing measured facts.
- All boots full-health, det-clean, recover_host_memory clean between each.

## KERNEL POLICY (user 2026-06-13): replay route ALWAYS ON; WY parked
Precise (two DIFFERENT kernels — see FR13_KERNEL_STATUS.md): (1) the **replay route**
(`_tree_gdn_replay_kernel` fr10_gdn_tree_kernel.py:546 + the verify scan with
store_node_states=False, store_node_states = not _fr13_replay_route_on :4115) is **default-ON**
(FR13_REPLAY_ROUTE=1) and STAYS ON — it is the shipped path, and FIX-3 conv-fusion + FIX-2
eager-pack REQUIRE it (:827, :5464). (2) The literal **WY kernel** (`_tree_gdn_wy_kernel`,
whole-tree Gram/UT-solve, never bit-exact) is **NOT on HEAD** (grep 0 on HEAD, 2 on
fr13-wy-archive) = already parked. So cat9/cat10 + the 22 flips + the −28 are the SHIPPED-PATH
(replay-ON) numbers, and the chase runs ON THE SHIPPED PATH — NO replay-off baseline (that would
entangle conv-fusion/eager-pack into a 3-flag change). The node-7 per-sub-op ladder localizes
in-place whether the replay-scan / conv-fusion is the flip carrier (first nonzero sub-op).

## VERDICT FOR THE PRIORITY: drift-fix FIRST, topology later
cat10 confirms the re-prioritization — the root sibling is the wrong lever (dilutes the spine
more than it rescues d0; lossless flat at 22). The real ceiling = (A) the 22-flip lossless defect
(VERIFIER side: committer/verify-forward, node-7 ladder + committer-row gate; INCLUDE the
gate-blindspot analysis — why the prior SCALAR gate missed a ~4.8% per-token flip, 30d749a4) +
(B) the d1-d4 dilution (DRAFTER co-residency, spine-only-drafter A/B). Chase BOTH on the shipped
(replay-ON) path. cat10 flag stays OFF; cat9 default.

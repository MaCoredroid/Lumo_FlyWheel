# FR13 S1/S2/S3 Discriminators — Bind (2026-06-10)

Findings bind only — **NO pass/fail close**; verdicts on the three acceptance-ladder seams are for the user.

## Header — flag state + seeds (rider-complete)

**Substrate**: HEAD `4d45be27` (S1 fix: committer bonus row = accepted leaf self-target), on top of the cc008587
determinism fixes (`FR13_TREE_PER_REQ_GEN`/`FR13_TREE_REQKEY`/`FR13_TREE_REMAP_SEQ` = 1, defaults ON). B=1 strictly
sequential, 4 pinned SWE prompts (`output/fr13_acceptance_ladder/prompts_swe4.json`, byte-identical reuse),
max_tokens=128, probe = `scripts/fr10_quick_decode_tps_probe.py` samples_per_prompt=1 warmup=0.
**Sampling**: greedy = temp 0.0 / top_p 1.0 / seed 1313; t06 = temp 0.6 / top_p 0.95 / seed 1313 (seed 1313 on BOTH
regimes this run; the ladder reference used 1313/1717).
**Run dir**: `output/fr13_s1s2s3_discriminate` (`run_header.json` = authoritative per-boot flag state). Serial boots,
teardown + host-memory recovery between, `docker ps` empty before each.

| boot | launcher | topology (TREE env) | backend | BI | key flags |
|---|---|---|---|---|---|
| boot1 tree caterpillar | `fr13_launch_forked_fa2_tree_server.sh` | 9-node caterpillar (spine [0,1,3,5,7], alt leaves [2,4,6,8]) | TREE_ATTN forked-FA2 + tree GDN | 1 | `FR13_TREE_BONUS_SELF=1`, `FR13_BI_TREE_ATTN=1`, `FR10_METRICS=1`, GPU_UTIL 0.82, fp32 logit capture NUM_TOKENS=10 LIMIT=600 |
| boot1c tree chain-only | same | 5-node chain (spine only, child_drafts all 1) | same | 1 | same flags, capture NUM_TOKENS=6 LIMIT=400 |
| boot2 native BI=1 | `fr10_launch_speed_server.sh` | linear MTP-5 (`{"method":"qwen3_5_mtp","num_speculative_tokens":5}`) | FLASH_ATTN naive_mtp | 1 | `FR10_ENABLE_TREE_GDN=0`, `FR10_METRICS=1`, GPU_UTIL 0.86 (see problems), capture NUM_TOKENS=6 LIMIT=600 |

References reused: `output/fr13_acceptance_ladder/native_greedy` (FLASH_ATTN naive MTP-5 **BI=0**, greedy 1313) and
pre-S1-fix `tree_greedy` (BI=1). Resume note: boot1/boot1c artifacts reused from the spend-limit-interrupted run
(same HEAD 4d45be27 substrate); boot2 re-run fresh.

Topology/node ids (committer BFS): spine = [0,1,3,5,7]; alts (2,4,6,8) are leaves; leaf order [2,4,6,7,8] ⇒ leaf-order
path_idx 0 = [0,2], spine = path_idx 3.

---

## D1 — S1 RE-GATE: HEALED (fix live, discriminating evidence)

Tree caterpillar boot, `FR13_TREE_BONUS_SELF=1` (artifact `s1_regate.json`):

- 182 events walked (183 committer rows, 0 leftover), 61 full-accept events, bonus_sources = {reject_parent_target 121, tree_self_target 61}.
- **21 [0,2]-winner events — ALL serve `st[2]`** (the accepted leaf's own self-target row); **18 of 21 are
  discriminating** (legacy node-8 `native_bonus_token` differs from `st[2]`); `bonus_violations = []`.
- **superset_violations = 0** with the TRUE spine path index (spine_path_idx=3) — the diagnostics half of the fix is
  live too (`path0_lcp` now reports the spine's lcp).
- **Prompt-3's S1 fork HEALED**: the [0,2]-winner at gen_pos 12 now serves 3418 == native's token; match_len vs
  `native_greedy` 14 → 31.
- Served-stream forks vs native BI=0 reference: pre-fix 16/24/21/14 → post-fix **16/11/25/31**. p0's fork at pos 16 is
  the S2 corrupt event (below). p1's fork moved EARLIER (24→11, tree 12182 vs native 26622) — an S3-class drafter
  flip, not S1.
- accept/event 1.819 (182 ev; 331/182) vs pre-fix tree 2.082 vs native BI=0 3.154. **Not like-for-like**: the fix
  changes served trajectories from early events, so this delta mixes trajectory change with acceptance (see
  progression section).

## D2 — S2 EPISODIC VERIFY CORRUPTION: REPRODUCED + CAPTURED, SURVIVES BI EQUALIZATION

The event-7 class recurs **deterministically on the post-fix boot** at prompt-0 gen_pos 16 (tree event 6):

- Root argmax tree 369 vs native 3051 at **native margin 7.25** (far beyond near-tie); d1 row also flips (279 vs
  23748, margin 3.5).
- Magnitude: event mean|d| 2.79 (vs native BI=0) / 3.32 (vs native BI=1) against lockstep baseline median
  0.405/0.466 (**~7x**); max|d| 15.9 / 15.75.
- **fp32 logits banked on BOTH arms**: `tree_greedy/logs/tree_final_logits.call6.pt` +
  `native_bi1_greedy/logs/native_final_logits.call5.pt` (per-request alignment maps strict-score 1.0 at greedy).
- **Trigger pattern** (`trigger_context` in `s2_tree_vs_nativeBI1_greedy.json`): previous event accepted_len=2,
  winner_path [0,1,4], bonus_source reject_parent_target — i.e., **follows an acc=2 BRANCH commit**, NOT
  [0,2]-winner-adjacent and not S1 residue.
- **Survives BI equalization**: present vs both native BI=0 and native BI=1 ⇒ tree-arm verify-forward corruption, not
  BI noise. Consistent with the live multi-event conv-prior-window residual (3a9039cc verified single-event only).
- t06: 21 lockstep pairs, 0 corrupt events (max event mean|d| 1.83) — episodic, not pervasive.
- Conv/window diagnostic-env re-boot NOT run this session (deferred; the corruption is deterministic and captured —
  next GPU tick can replay with state capture).

Artifacts: `s2_tree_vs_nativeBI0_greedy.json`, `s2_tree_vs_nativeBI1_greedy.json`, `s2_tree_vs_nativeBI1_t06.json`.

## D3 — S3 CHAIN-ONLY DISCRIMINATOR: m1 PARTIAL

Chain-only topology (5-node spine, no alts) vs the caterpillar, lockstep identical-committed-prefix drafter
comparison (identical-draft-event rate; per-depth flips in the artifacts):

| tree arm | vs native BI=0 ref | vs native BI=1 |
|---|---|---|
| chain-only (boot1c) | **0.533** (8/15; d0 3/15 … d4 6/15) | **0.400** (6/15; d4 8/15) |
| caterpillar (boot1) | 0.348 (8/23; d0 9/23 … d4 15/23) | 0.182 (4/22; d0 8/22, d4 17/22) |

- Chain is consistently and substantially closer to native ⇒ **alt co-residency (m1) is REAL**.
- But chain is **NOT token-identical** ⇒ removing alts does not eliminate the flips ⇒ **m1 is PARTIAL**, residual
  remains.
- Chain first-flips at gen_pos 1 on prompts 1/2/3 (full-depth, d0) occur **before any alt has ever been processed
  and before any accept/reject commit** (the first rollout consumes prefill state only) — these flips cannot be m1
  and are not post-commit rebuild (m3) either; they bound an arm-level forward-drift share (the S2-class baseline
  tree-vs-native event mean|d| 0.43–0.47 flipping near-ties, the Method-A "tree-specific non-BI-covered channel").
- Caterpillar top1/top2-swap signature persists vs BI=1 native: 5 of 10 d≥1 first-flips have the ALT holding native's
  token (then dead-ending, since alts are leaves); chain has none by construction.

## D4 — S3 BI-EQUALIZED BOOT: m2 REFUTED

Native BI=1 boot (FLASH_ATTN naive MTP-5): greedy accept/event 3.047 (127 ev), t06 2.843 (134 ev).

- BI-equalized lockstep identical-rates **DROPPED** (caterpillar 0.348→0.182; chain 0.533→0.400) — equalizing the BI
  flag does not recover lockstep identity. **m2 (BI asymmetry as the flip mechanism) is REFUTED.**
- Control native(BI=1) vs native(BI=0): greedy served streams fork at p0 exact-128 / p1 pos15 / p2 pos25 / p3 pos71;
  drafter identical-rate 0.714 (40/56; d0 3/56). The BI flag flips near-ties in the native arm itself — that is the
  cross-boot floor — and the tree-vs-native flip rates (d0 8/22 caterpillar) sit far beyond it.
- Consequence for any future lossless gate: **the native reference is BI-sensitive across boots; pin the BI flag
  state on BOTH arms.**

## S3 MECHANISM CALL

**m1 (alt co-residency): PARTIAL-CONFIRMED — but the locus is the VERIFY side, not the drafter rollout.**
**m2 (BI asymmetry): REFUTED.**
**Residual: m3 (post-commit state rebuild/handoff) and/or the S2-class tree-arm forward drift** (which D3's
gen_pos-1 flips show is nonzero independent of alts and commits).

CPU code-read precision (changes where the m1 fix goes): the active drafter is the
`FR10_CATERPILLAR_NATIVE_SPINE_TOP2` overlay (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:5719-5860`), which is
**alt-free by construction** — each depth step feeds back only `_fr10_spine_tokens[-1]` (1 row per request per
depth); leaves are `topk(logits,2)[:,1]` of the same step logits and never enter `input_ids`, the forward, or any
recurrent state. The hypothesized "2-row depth batches" do not exist in this code path (they belong to stock
`propose_tree`, which is not engaged). The chain-vs-caterpillar boots therefore differ only in the **target-side
per-event pipeline**: the 9-row vs 5-row tree verify forward and its post-commit state advance — that is where the
alt co-residency that D3 measures must live.

### m1 fix direction (described, NOT implemented)

Make the committed-path state alt-invariant in the verify/state-advance path:

1. **Conv prior-window bank** (`fr10_phase4_patch_vllm_tree_gdn.py` ~797-818: `_fr10_conv_read_cols` /
   `_fr10_prior_conv_bank_rows` gather over `spec_state_indices_tensor` indexed by `accepted_len-1`): with alts
   present the winner can be a branch path ([0,2], [0,1,4]) and the bank row/col selected for the NEXT event's prior
   conv window can hold a window containing alt/off-path tokens. This is the same seam family as S2 / the
   conv-prior-window residual (3a9039cc verified single-event only; FR13_DECISIVE_TEST gate = conv1d_out row0 → 0.0).
   Fix = commit a conv window rebuilt from committed-path tokens only (per accepted path, not per batch-row column
   arithmetic that is only spine-valid).
2. **GDN recurrent checkpoint handoff (h0)**: the next event must consume the checkpoint of the accepted node's
   path; per-path checkpoint selection with no accumulator shared across branches (STree's diagonal-only shortcut is
   insufficient for this; per-path WY/UT — see the branch-oracle losslessness reference). Verify byte-equality of the
   handed-off state against a chain run on forced-identical commits.
3. **Drafter-layer KV/conv slots for rejected positions**: the 9 draft positions (vs 5 in chain) write drafter-layer
   cache rows; assert rejected/alt rows are overwritten or invisible at the next rollout (slot-mapping audit, not a
   kernel change).

Decisive A/B (cheap, one boot): caterpillar topology with commit forced to the spine path (alts verified but never
winning) — if next-event drafter spine tokens then match the chain boot token-for-token, the contamination is
entirely in the branch-commit state advance (seams 1-2); if not, alt rows corrupt state even without branch commits
(seam 1 batch-shape / seam 3).

## ACCEPT/EVENT PROGRESSION + REMAINING GAP DECOMPOSITION

| stage | accept/event | basis |
|---|---|---|
| pre-fix (pre-cc008587 default path) | 2.024 | B=4 captured gate reference |
| post-cc008587 (determinism fixes, B=1 ladder) | 2.082 | 159 ev, `fr13_acceptance_ladder` tree_greedy |
| post-S1 (this run, caterpillar) | **1.819** | 182 ev, 331/182 — trajectory-shifted vs pre-fix, NOT a like-for-like acceptance delta |
| chain-only (no alts, same boot image) | **2.277** | 159 ev, 362/159; accepted/draft-token 0.455 vs caterpillar 0.202 |
| native MTP-5 BI=1 | 3.047 | 127 ev (greedy); t06 2.843 |
| native MTP-5 BI=0 (reference) | 3.154 | 123 ev |

Reading (greedy, B=1):

- **Alts currently NET-NEGATIVE ~0.46 accept/event** (chain 2.277 vs caterpillar 1.819) despite the realized
  branch-accept upside (61 full-accepts include [0,2]/branch winners). On lockstep pairs the same sign holds (chain
  1.73 vs caterpillar 1.39-1.50 against native 2.2-2.59). Until the m1 verify-side contamination is fixed, the
  caterpillar pays more in spine degradation than it gains from alt acceptance — consistent with the GDN
  tree-superset memory (path0 degraded by shared state).
- **Chain-only still −0.77 vs native BI=1 (2.277 vs 3.047)**: with alts, BI asymmetry, and S1 all removed/refuted,
  this residual is the m3 / S2-class tree-arm forward-drift share (gen_pos-1 full-depth flips + baseline mean|d|
  0.43-0.47 + the episodic S2 corruption events).
- **BI-flag share of the native bar itself**: 3.154 (BI=0) vs 3.047 (BI=1) — ~0.11 of any gap quote is reference
  flag-state, not tree deficit; future gates must pin BI on both arms.
- S2 contributes discrete served-stream forks (p0 pos16 this run) on top of the drafter-flip deficit; it is the only
  remaining LOSSLESS-class violation identified at greedy (S1 healed; S3 is q-side/accept-rate, not p-lossless).

## Problems / caveats

1. Boot2 GPU_UTIL deviation: planned 0.88 failed engine init (102.58/117.51 GiB free); relaunched 0.86 — KV-cache
   budget only, B=1 128-tok probes unaffected (`run_header.json`).
2. S2 conv/window graph-safe diagnostic re-boot not executed this session; state-level evidence is the trigger
   context only (no GDN state capture this boot). Corruption is deterministic + fp32-captured on both arms.
3. `chain_t06` window was never produced (lost to the spend-limit interruption); the chain-only discriminator is
   greedy-only.
4. accept/event deltas across the S1 fix are trajectory-confounded (served streams change from early events).
5. Native capture call indices contain trailing request-boundary 6-row forwards with no spec-trace row — handled via
   per-request alignment maps (`build_align_map.py`; greedy strict 1.0 on all 4 prompts, t06 bases 0.92-1.0);
   `fr13_disc_lib.walk_native_events` needed a clipped-tail fix (acc=5 with <5 slots before the 128 cap). Both live
   in the gitignored run dir; numbers banked here and in `FR13_LADDER_LOG.md`.
6. boot1/boot1c artifacts reused from the interrupted run (same HEAD 4d45be27 substrate), not re-booted; the s2/s3
   caterpillar numbers vs native BI=1 pair a reused tree window with a fresh native window.
7. The s1 lockstep `mean_accept_on_pairs` values (1.39-1.73) are pair-conditioned (identical-prefix events only) and
   are NOT comparable to the whole-window accept/event numbers above.

Method: all numbers reproducible CPU-only from the run dir (event walks assert emitted == served token_ids; logit
pairs compared only at identical-prefix identical-input rows via the alignment maps). Flag state per boot in
`run_header.json`; this bind quotes it in the header table.

# FR13 garble: OUTPUT-PROB PROBE — the garbled token is NEAR-IMPOSSIBLE, scan-drift REFUTED

Date 2026-07-12. Direct output-level measurement (user's suggestion). Ship cat9 config, port 9955,
scripts/fr13_garble_repro_and_probe.py: reproduce garble (wcs_slice seed1, undefined names
dropped_dimension_index / pixel_keep_boolean = missing _map/_mask suffix), teacher-force the continuation
(max_tokens=1 clean single-token distribution), compare tree-committed token vs clean argmax per position.

## THE GARBLE POSITIONS (the undefined-name truncations)
  pos 123: tree committed '\n' (truncates identifier) | clean argmax '_map'  lp=-0.00 (~prob 1.0) | committed clean lp=-15.69 (~1e-7) | GAP 15.69 nats
  pos 138: tree committed '\n'                        | clean argmax '_mask' lp=-0.00 (~prob 1.0) | committed clean lp=-6.63  (~1e-3) | GAP  6.63 nats

## VERDICT
The tree committed tokens the CLEAN distribution rates ~1e-7 to ~1e-3, with 6.6-15.7 nat gaps. The
~9e-4 GDN scan drift amplified even x492 is ~0.4 nats — 15-40x TOO SMALL to flip these. The
scan-drift-flips-the-accept hypothesis is REFUTED at the output level. Garble is a GROSS structural
effect (many nats), not a ~1-ULP realization drift. (Prose small-gap flips at pos 27/94 ' world' 0.25 nats
are normal temp-0.6 sampling, NOT the garble; my auto-verdict misclassified — raw numbers above are canonical.)

## REDIRECT
Gross corruption at MULTI-token-accept positions (deep in building a long identifier => num_accepted>1).
STRONG candidate: the OPEN bug 'conv prior-window wrong bank-row at num_accepted>1' (project_fr13_conv_priorwindow_root)
— a discrete wrong-state read = many nats, fires exactly at num_accepted>1. Alt: spec-accept/rejection-sampler
logic accepting a draft the target rejects. NEXT: measure the tree's ACTUAL spec-verify target distribution at
pos 123 (verify-corruption vs accept-logic), and audit conv prior-window bank-row routing at num_accepted>1.
NOTE: conv COMPUTE is bit-exact (0.0); this is the conv STATE ROUTING at num_accepted>1, a separate issue.

## Sharpened localization (2026-07-12): committer col-0 deposit at num_accepted>1 X branches
FR13_TREE_RUNROW_INIT=1 (ship): next-step conv+ssm prior read from col 0 = the committed leaf deposited by
the post-accept committer (RUNROW_COMMIT, fr10_gdn_tree_kernel.py L877-891: `state` at replay loop-exit
stored to col 0 = spec_state_indices[:,0]). SPINE (chain5) also has num_accepted>1 and is GARBLE-FREE =>
corruption is num_accepted>1 X BRANCHES (co-residency): the committed-path replay's `state` at loop exit is
grossly wrong when branches are co-resident (wrong ring/node indexing or wrong committed leaf), so col 0 gets
a wrong prior -> next verify grossly corrupted -> commits near-impossible token. GROSS wrong-state pick, not
~1-ULP realization. NEXT: (a) confirm the replay `state`/ring indexing follows the committed path (not
position) under branches; (b) the fix = route the committed-path replay through native so col-0 prior is
native-correct (synth's recommendation); (c) gate live temp-0.6 + cache-ON, same-boot vs native 0%.

## Red-team refinement (2026-07-12): replay READS by node-id (correct) => bug is upstream at FILL time
Read _tree_gdn_replay_kernel loop (L793-891): node = accepted_paths[pid_b, t-1]; reads k_ring[node]/
v_ring[node]/a_ring/b_ring by NODE-ID (not position); state=where(active,new,state) ends at committed leaf;
deposits to col 0. The replay READ is co-residency-clean (node-id indexed). => the gross corruption is NOT
the replay indexing but UPSTREAM: (i) accepted_paths built wrong under branches, (ii) the ring buffers
(k/v/a/b) FILLED wrong for committed nodes at verify time (position-vs-id co-residency), or (iii) the verify
scan _tree_gdn_kernel itself. The finding is robust (gross 15-nat, num_accepted>1 X branches, STATE-CARRY not
verify-compute since 1-step teacher-force is correct). DEFINITIVE TEST = route the committed-path state
through native (fused_sigmoid_gating per committed path) so col-0 is native-correct regardless of which
upstream op leaks; if garble->0 the state-carry chain was the bug. Cheaper bisect: capture tree col-0 state
post-multi-accept vs prefill state, OR toggle-bisect the fill/scan/committer flags (each a reboot).

## Committer-native fix gate result (2026-07-12): ENGAGED but garble PERSISTS -> verify scan implicated
Boot: ship EAGER_PACK=1 + FR13_COMMITTER_NATIVE=1, EAGER. ENGAGED confirmed (print fired, non-vacuous).
Garble gate (11 samples, eager crawls due to 48 per-layer host-syncs): undefined-name-rate=2.86%,
3/11 samples STILL garble (final_reconciliation_rows, applied_entries_list = near-neighbor garbles).
=> committer col-0 state-carry fix (validated bit-exact 1.19e-7) ENGAGED but did NOT eliminate garble.
The committer col-0 (state for NEXT step) was NOT the sole root; the VERIFY SCAN (_tree_gdn_kernel, THIS
step's accept decision using the branched tree scan from col-0) is the remaining gross corruption.
Reduction 9.3%->2.9% is cross-boot (unconfirmed vs same-config baseline; boot variance +-3-6%); the ROBUST
signal is garble PERSISTS. NEXT: route the VERIFY scan per-path native too (multi-seq convention). Also:
eager committer host-syncs make it too slow to gate -> batch the .item() calls for a usable gate.

## Prob probe on COMMITTER-FIXED config (2026-07-12): garble STILL GROSS -> SSM committer EXONERATED
Boot ship EAGER_PACK=1 + FR13_COMMITTER_NATIVE=1 (batched syncs), ENGAGED confirmed. Reproduced garble
(wcs_slice seed1: 'pixel_keep' = truncated pixel_keep_boolean_mask). Teacher-force: pos 132 committed '\n'
| clean argmax '_boolean' lp=-0.00 (~prob 1.0) | committed rank=None (OUT of clean top-20 = near-impossible).
=> the surviving garble is STILL GROSS (~unchanged from the pre-fix 15-nat gap). The bit-exact-correct SSM
committer col-0 did NOT change the grossness => the SSM state-carry was NOT the garble source. The gross
corruption is in THIS step's verify INPUTS the committer doesn't touch: the CONV prior-window STATE ROUTING
(memory project_fr13_conv_priorwindow_root: wrong bank-row at num_accepted>1, OPEN; conv COMPUTE is bit-exact
0.0 but the state ROUTING is separate) or the verify scan. NEXT: audit + fix the conv committed-path state
routing at num_accepted>1 (analogous to the SSM committer). Keep SSM committer default-OFF (validated but not
the cure). Prose flip at pos 27 ' sliced' rank2 gap0.38 = normal sampling, not garble.

## Strategic reframe (2026-07-12): accept-LOGIC hypothesis (2 validated state-fixes now ineffective)
Two bit-exact-correct fixes have now ENGAGED but left the gross garble UNCHANGED: in_proj_ba (earlier) and
the SSM committer (now). The garble is a GROSS wrong-accept of a near-impossible token (out of clean top-20).
For a rejection sampler to accept an out-of-top-20 token, EITHER (a) the tree's VERIFY target logits are
grossly wrong (state corruption: conv routing / verify scan) OR (b) the accept LOGIC itself accepts a
drafter's low-prob proposal that the target rejects (the custom FR13_DEVICE_MULTIDRAFT rejection sampler).
The pattern of validated-but-ineffective STATE fixes raises (b) -- if the sampler is the bug, NO state fix
helps (matches every observation). DECISIVE next step: capture, at the accept, the tree's VERIFY target
rank/logit of the ACCEPTED token vs the argmax (the sampler HAS these logits). Accepted tokens with LOW
target rank => sampler bug (b). HIGH target rank => state corruption (a), and then conv-routing/verify-scan.
Do NOT build another blind state fix until (a) vs (b) is settled. See memory feedback_garble_pin_accept_not_rates.

## CAG greedy diagnostic (2026-07-12): garble is STATE (verify argmax wrong), NOT accept-logic
FR13_COMMIT_ARGMAX_GATE armed (via sidecar; worker drops the env flag) + FR13_DEVICE_MULTIDRAFT=0 (Python
committer, since CAG lives in the greedy committer bypassed by the device committer). 1275 greedy commits:
big-margin mismatch (committed != verify argmax, |margin|>2 nats) = 0; only 15 argmax-TIE mismatches.
=> the committer FAITHFULLY serves the verify argmax; it NEVER commits a low-logit token. ACCEPT-LOGIC
REFUTED. YET matrix prompt garbled at GREEDY (expected_rows for expected_row_count, undefined=2) => the
garbled token WAS the verify argmax => the verify FORWARD produces a grossly-wrong argmax => STATE corruption.
Greedy reproduces the garble DETERMINISTICALLY (no temp-0.6 sampling) = a clean repro + per-token
verify-argmax classifier for testing fixes. in_proj/conv-compute/SSM-committer already exonerated => remaining
verify-forward state ops: conv STATE routing (prior-window num_accepted>1) or the verify SCAN (_tree_gdn_kernel)
or attention. NEXT: test each at greedy+CAG (does the matrix verify argmax become expected_row_count?).
NOTE: wcs greedy clean, ledger greedy syntax-err, matrix greedy garbles => garble is prompt/position-specific.

## Committer-native greedy re-test (2026-07-12): BLOCKED by crash; committer stays OUT
Boot COMMITTER_NATIVE=1 + DEVICE_MULTIDRAFT=0 + CAG + greedy crashed on first request: EngineDeadError,
RuntimeError in _lumo_tree_path_lcp_max_greedy_sample path-LCP log "expanded size (6) must match existing
(10) at dim 0" — a committer-native x greedy-committer interaction bug (default-OFF code). Last cycle's
greedy+CAG WITHOUT committer-native did not crash. Not debugging (committer is out on 2 prior evidences:
temp-0.6 persist + prob-probe grossness unchanged). NEXT: build the CONV state-routing native fix (the
untested half of col-0 state, num_accepted>1 prior-window = memory's OPEN bug) and test at greedy+CAG
WITHOUT committer-native (that combo works, produced records + the deterministic matrix garble). Conv
committer infra scouted: prepare_committed_path_conv_rows / gather_committed_path_conv_prior
(fr13_tree_conv_fused.py), replay_conv_state_linear_remap (fr13_replay_conv_remap.py).

## Strategic reassessment (2026-07-12): conv committer looks correct on read -> definitive next = per-node native VERIFY
Read _fr13_conv_commit_to_col0 (patcher L7056): copies accepted-leaf conv window _ssi[b,leaf_node] -> col-0;
looks CORRECT (leaf's stored window IS its path-window; RUNROW_INIT reads col-0). So conv-routing wrong-bank-row
not obvious (likely fixed by the STATELESS-TREE rework). Every individual op is now either bit-exact
(in_proj/conv-compute), within-floor (attention 2-ULP), algebraically-clean+~1e-5 (verify scan realization),
or looks-correct (conv committer) -- yet garble is GROSS + deterministic. Two untested possibilities: (a) a
DISCRETE index bug in the verify scan per-node h0/node read at num_accepted>1 x branches; (b) the SSM committer
fix doesn't correct col-0 in the LIVE path (validated offline 1.19e-7, never verified live => "ineffective"
could mean "buggy wiring" not "wrong op").
DEFINITIVE NEXT (the deliverable, tests both at once): replace the verify scan node outputs with PER-NODE
native single-seq recurrence (each node's ancestor path through fused_sigmoid_gating = validated Test-A
mechanism, NO unsolved multi-seq convention; slow N-launches but correctness-only, gated). Fix site =
launch_tree_gdn_prepared (fr10_gdn_tree_kernel.py L1723 -> _tree_gdn_kernel L1904). Test at greedy+CAG
(deterministic matrix repro): garble->0 => GDN verify was the corruption; persists => GDN ruled out (=> the
committed-STATE carry, needing live col-0 verification). Do NOT build another blind sub-op fix.

## VERIFY_NATIVE result (2026-07-12): GDN scan COMPUTE ruled out -> corruption is the col-0 STATE feeding it
Boot ship + FR13_VERIFY_NATIVE=1 + DEVICE_MULTIDRAFT=0 + CAG, EAGER. ENGAGED (per-node native verify tree_n=10,
non-vacuous). Matrix greedy: undefined 2->1 (output CHANGED deterministically, so the fix TOOK EFFECT), but
the core garble (expected_rows truncation) PERSISTS. CAG: 1397 records, 0 big-margin mismatches => committer
still faithful, verify argmax STILL wrong. => replacing the GDN scan node OUTPUTS with bit-exact-native ones
does NOT fix the verify argmax => the GDN scan COMPUTE is NOT the corruption. My fix fed the scan the NATIVE
inputs (query_spec/key_spec/value_spec, which native/no-spec uses and is garble-free) + col-0 h0 => the
corruption is in the GDN INPUTS: the col-0 running STATE (SSM h0 read at spec_state_indices[b,0] AND/OR the
conv prior-window that produces value_spec/q/k). 4th compute-fix ineffective (in_proj, conv-compute, SSM
committer, GDN scan) => the bug is STATE/DATA-ROUTING, not compute. NEXT: directly verify col-0 correctness
(SSM + conv) in the live path via a targeted col-0 dump vs a native forward, OR combine correct SSM+conv
committers (COMMITTER_NATIVE crashes at greedy -- fix the path-LCP 6v10 first). value_spec/q/k are the native
tensors (native is clean) so they're likely correct => the SSM col-0 h0 is the prime suspect, and the SSM
committer fix being ineffective may mean it does NOT actually correct live col-0 (never verified live).

## RED-TEAM CORRECTION (2026-07-12): VERIFY_NATIVE rules out GDN-scan COMPUTE only; ATTENTION not ruled out
Prior note over-concluded "corruption is col-0 state". VERIFY_NATIVE only replaced the GDN linear-attn outputs;
the verify forward is GDN layers + FULL-ATTENTION layers (Qwen3.6 hybrid). "garble persists with native GDN"
=> corruption is EITHER (a) the col-0 running STATE feeding the GDN, OR (b) the FA2 TREE ATTENTION at
num_accepted>1 x branches -- which the GDN fix does NOT touch. The FA2-fork "byte-exact 14/16, 2 single-ULP"
(project_fr13_fa2_fork_nocopy_floor) was measured at a specific config (likely num_accepted=1); the garble is
num_accepted>1-specific, so the tree attention at MULTI-ACCEPT is UNTESTED. Both (a) and (b) remain open.
DECISIVE LOCALIZER: capture the live col-0 (SSM+conv) at the deterministic garble step vs a native forward's
state -- col-0 WRONG => (a) state; col-0 CORRECT => (b) attention (then apply the same per-node-native
treatment to the FA2 tree attention). Do this BEFORE any more fixes.

## KEY RE-DIRECTION (2026-07-12): branch-specific + VERIFY_NATIVE-persists => ATTENTION, not col-0/GDN
Chain of facts: (1) garble is BRANCH-SPECIFIC -- spine-only is garble-free (M-invariance baked), only the
tree/branches garble (reference_garble_within_floor_closed, garble_batch_exonerated_empirical). (2)
VERIFY_NATIVE computes each node's GDN output from its OWN root->node ancestor path, ISOLATED from sibling
branches => removes GDN branch co-residency. (3) garble PERSISTS with VERIFY_NATIVE. => the branch
contamination is NOT in the GDN (isolating it didn't help) and NOT in col-0 (col-0 = pre-branch committed
prefix, built identically for spine-only which is garble-free). => the contamination is in the FA2 TREE
ATTENTION, where sibling branches co-reside via the tree mask (strict_mask/visible_mask). The FA2-fork
"byte-exact 14/16" validated the NO-COPY optimization vs a same-tree copy reference -- NOT mask correctness
vs a branchless/native reference, and likely not at the garble's num_accepted>1 branch config. => col-0 was
the WRONG prime suspect. NEXT: audit the tree attention mask construction+application for a node attending to
a SIBLING branch's KV (visibility leak); then apply per-node-native isolation to the FA2 attention (analog of
VERIFY_NATIVE) OR fix the mask.

## FA2 MASK-LEAK REFUTED (agent audit, 2026-07-12) -> paradox: ALL verify ops validated, garble persists
Explore audit (3 independent ways): (1) decoded the LIVE captured tree_attn_bias 10x10 on a genuine
multi-level branched tree parent=[-1,0,1,1,2,2,4,4,6,6] -- every row visible set = {self}U{ancestors}, NO
sibling/cousin ever visible; (2) strict/visible masks are GDN-ONLY, never touch FA2; (3) branched-vs-branchless
per-node FA2 validator fr13_fa2_tree_path_ref.py WAS RUN + PASSED to bf16 floor (max_abs 2^-10, one elem;
a real leak = O(1)). => FA2 mask sibling-leak REFUTED. So NOW every verify-forward op is validated correct
(in_proj bit-exact, conv bit-exact, GDN scan VERIFY_NATIVE, FA2 mask ancestor-only) YET verify argmax is
grossly wrong (CAG). PARADOX => the gross error is NOT in the verify-forward compute; it is in the CARRIED
PREFIX/STATE the (correct) forward consumes -- KV cache + col-0 GDN (SSM h0 + conv prior) built by the
num_accepted>1 branch-path commit -- OR my foundational "branch-specific" assumption is mismeasured for THIS
deterministic matrix repro. Agent's own pointer: away from *which KV is visible*, toward *value realization /
physical row-order alignment on the branched layout at num_accepted>1* (fr13_patch_fa2_tree_bias.py:57
context_len mapping; sorted tree_choices order the bias assumes). NEXT: the winner_spine test -- does the
matrix garble COMMIT on a branch node (winner_spine>0, num_accepted>1 branch path) or the spine (==0)? That
verifies branch-specificity for the real repro AND tells whether the corrupt prefix is built by a branch commit.

## CRUX RE-READ (2026-07-12): CAG proves argmax-of-INDEXED-row, NOT that the indexed row is the RIGHT node
Reconciling the paradox (all verify ops validated correct, yet verify argmax grossly wrong): CAG's ch1_match
= (committed == argmax of the ROW THE COMMITTER INDEXED). 0 big-margin mismatches proves the committer
faithfully serves the argmax of WHATEVER row it read -- it does NOT prove that row is the correct node for the
committed path position. A DISCRETE WRONG-ROW-INDEX (routing) bug -- committer/verify reads node X's output
when the accepted-path node at that depth is Y (X!=Y) -- is CONSISTENT WITH EVERY OBSERVATION:
  (1) CAG-clean: committed == argmax(logits[X]), a faithful serve of the (wrong) indexed row.
  (2) VERIFY_NATIVE persists: making ALL rows bit-exact-native fixes logits[X]'s VALUES but X is still the
      wrong node => garble persists; the small value change explains the 2->1 undefined shift.
  (3) branch-specific: wrong-row-index only manifests with branches (multiple rows to confuse); spine has one
      path so no row ambiguity.
  (4) all ops validated: the COMPUTE is correct; the READ (which row) is wrong.
=> STRONGEST UNIFYING HYPOTHESIS: a discrete row-index/routing bug picks the wrong node's verify logits for a
committed position at num_accepted>1 with branches. This is NOT a compute bug (why 4 compute-fixes all missed).
CHECK: log, per committed token, the NODE INDEX the committer indexed, and verify it equals the accepted-path
node at that depth (and that argmax(logits[correct_node]) == the non-spec ground-truth token). Instrument the
committer's row selection, not the compute. Aligns with agent's "physical row-order vs sorted tree_choices"
pointer and memory's "conv1d_out wrong bank-row at num_accepted>1".

## LOCALIZED to a committer step (2026-07-12, winner-log correlation, deterministic matrix greedy)
Ran matrix_build at temp 0 (greedy seed 0) on locked cat9 + DEVICE_MULTIDRAFT=0 + winner-log
(LUMO_TREE_PATH_LCP_LOG) + CAG. Deterministic garble reproduced: undefined=['expected_rows',
'crpix_reference_value','expected_rows']. Correlated each garble token to its committer step (winner log
tree_path_lcp_max.jsonl, 50 steps; 35/50 are BRANCH commits winner_leaf!=spine_leaf, acc 3-5). The garble
'_rows' (truncating expected_row_count) is emitted at step 12 AND step 35, BOTH identical:
  winner_path=[0,2] (root->node2=(0,1), a BRANCH), accepted_len=0, all path_scores lcp=0.
  context ends '(expected'. DRAFTER drafts node0='_row', node1='_count' => 'expected_row_count' (CORRECT).
  parent_target_ids[0]='_rows' = the VERIFY/target argmax after '(expected' = '_rows' (WRONG; correct='_row').
  => verify REJECTS the drafter's correct '_row' (draft != parent_target => acc=0), commits bonus '_rows'.
So the garble = the TREE VERIFY FORWARD's target-argmax at the ROOT node0 is '_rows', overriding the correct
draft '_row'. NOT a committer accept-logic bug (committer faithfully served the verify argmax) -- the verify
FORWARD's node0 argmax is wrong. node0=root=spine node, but computed in the M=10 tree forward (co-resident
with 9 branch nodes). DECISIVE OPEN Q: is verify-argmax '_rows' the M=10 co-residency CORRUPTION, or the
model's genuine clean argmax (making the drafter's '_row' the anomaly)? => teacher-force the clean prefill
argmax after '(expected': clean='_row' + tree='_rows' => M-dependent root corruption CONFIRMED; clean='_rows'
=> not a garble. Note: VERIFY_NATIVE (GDN per-node) didn't fix it => if confirmed, root corruption is in the
M=10 FA2 attention or the prefill-vs-tree logits path at node0, NOT the GDN scan.

## CONFIRMED garble is REAL + corruption is TRAJECTORY-ACCUMULATED STATE (2026-07-12, teacher-force)
Teacher-forced the clean prefill dist after the exact prefix ending '(expected' (chat continue_final_message,
max_tokens=1, top-20 logprobs, on the spec server -- first decoded token = prefill dist):
  '_row'  lp=-0.000 (~prob 1.0)  = CLEAN argmax
  '_rows' lp=-13.875 (~prob 1e-6) = the garbled token, near-impossible
=> 13.9-nat gap. The garble is REAL (clean overwhelmingly wants '_row'; the model is NOT quirky). YET the tree
verify at step 12 committed '_rows' (parent_target[0]). SAME question (next token after '(expected'), DIFFERENT
answer: isolated teacher-force = '_row' (CLEAN); step-12-in-trajectory = '_rows' (CORRUPT). The ONLY difference
is the STATE the two forwards consume: teacher-force builds mamba/GDN + KV FRESH from the prefix prefill;
step 12 reads state ACCUMULATED over 11 spec-decode commits (35/50 branch commits, num_accepted 3-5).
=> CORRUPTION IS TRAJECTORY-ACCUMULATED CARRIED STATE (col-0 GDN mamba recurrent state and/or KV cache written
by the spec-decode committer at num_accepted>1 branch commits), NOT the verify COMPUTE (VERIFY_NATIVE reads the
same corrupt state => can't fix), NOT the prefill/model (fresh state = clean). This UNIFIES: branch-specific
(branch commits accumulate the corruption), VERIFY_NATIVE-persists, all-verify-ops-validated, CAG-faithful.
NEXT: localize mamba col-0 vs KV -- (a) matrix greedy with APC cache OFF (spec ON): still garbles => mamba
col-0 state; clean => APC KV. (b) which commit writes the corrupt state (step 11 branch acc=3 precedes the
step-12 garble). Then a compute-only fix to the state write.

## conv-prior flag test CRASHED (2026-07-12) + suspect sharpened to CONV col-0 (VERIFY_NATIVE logic)
FR12_TREE_CONV_NATIVE_PRIOR_READ baked non-vacuous (verified _FR12_NPR=True in emitted gdn_linear_attn.py,
read+needle same module) but the request 500'd: EngineCore UnboundLocalError '_fr13_tcf_prep' in "FR10 tree
state linear remap". Root cause: native_prior_read SKIPS the fused-prep block (sets _fr13_tcf_prep at 2422)
but the remap USES it (2652-2666) => genuinely incompatible with FR13_TREE_CONV_FUSED=1 (the exclusion at
patcher L868-880 is REAL; TCF_DIAG_OVERRIDE just moved the init-raise to a runtime crash). Needle=0 = crashed
before reaching it. So this diagnostic can't run on the locked (fused) config without a real remap-guard fix.
BUT the suspect is now SHARP: VERIFY_NATIVE (verified-engaged) makes the SSM/GDN scan OUTPUT native-correct per
node yet garble persists, AND VERIFY_NATIVE does NOT touch the CONV state (only the GDN linear-attn scan). So
the carried-state corruption lives in the CONV col-0 (write and/or read), not the SSM. The col-0 for step 12's
verify is written by step 11's commit (branch [0,1,4] acc=3, leaf node 4): the committer copies leaf node 4's
CONV window to col-0. NEXT: audit the CONV col-0 WRITE/commit (_fr13_conv_commit_to_col0 + the tcf-fused
snapshot) for a wrong-node / wrong-window / wrong-bank-row read at num_accepted>1 branch paths -- read-only,
no config change, no crash. A compute-only fix there is the ship-fix direction.

## RED-TEAM CORRECTION (2026-07-12): VERIFY_NATIVE exonerates SSM scan COMPUTE, NOT the col-0 STATE write
Prior note over-sharpened to "CONV col-0 only". VERIFY_NATIVE recomputes each node's GDN OUTPUT with
inplace_final_state=False (NO state write-back) and READS col-0 h0 (carried SSM) as its initial state; the
node output also uses value_spec (post-conv, which reads the col-0 conv window). So VERIFY_NATIVE persisting is
consistent with EITHER a corrupt col-0 SSM h OR a corrupt col-0 conv window feeding it -- it validated the
scan COMPUTE (per-node recurrence math == native), NOT the carried col-0 STATE that step-11's committer wrote.
The carried col-0 STATE (SSM h + conv window) written by the branch-leaf commit is the suspect, and NEITHER
half is cleanly exonerated (COMMITTER_NATIVE "garble unchanged" may have been vacuous -- worker-env drop, like
FR12_NPR/CAG/DEVICE_MULTIDRAFT all were). => audit BOTH col-0 writes: conv window AND SSM h, for a
wrong-node/window/bank-row read at num_accepted>1 branch leaf node4=(0,0,1). The clean fix-test is to make the
FULL col-0 write native (SSM h + conv window) with VERIFIED engagement + no fused/greedy crash.

## CONV col-0 WRITE audit result (2026-07-12): STATIC-CORRECT for B=1 cat9; remaining conv risk = runtime physical-row
Audit traced write->read loop for winner [0,1,4] (leaf node4=(0,0,1) acc=3). _fr13_conv_commit_to_col0
(L7112-7162): _leaf_node = _accepted_path_buf[b, acc_len-1] = GDN col 5 (=sampler node4, +1 GDN offset;
col0=running row). Reads _ssi[b,5]=node4's tree-correct branch window (ancestry [0,1,2,5] via
build_tree_conv_state_src_indices), writes to col-0 (_ssi[b,0]). NO sibling swap (node3 at col4 not read), NO
linear-position indexing, NO wrong bank-row. The linear "flat_source" table (the contiguous-block bug) exists
but is consumed ONLY in the FR10_METRICS diagnostic block, not the live path. Bank is live (self.
_fr13_replay_conv_state=conv_state), spec-idx same-row-order snapshot. RUNROW_INIT=1 forces the next read to
col-0 (two comments at L2364/L2734 claiming leaf-node-col read are STALE, not bugs). => CONV col-0 write is
static-correct for B=1; not the wrong-node bug.
LATENT bug (INERT for matrix garble, candidate for B=4 AGENTIC): commit early-fill _accepted_path_buf is in
SAMPLER-row order but _ssi is SPEC-row order; the compact remap _fr13_src_i is applied only to the LATE refill
(after commit). At B>1 reordered batches (sampler row b != spec row b), the commit pairs req b's leaf id with a
DIFFERENT req's spec-idx row = cross-request wrong-bank-row commit. Inert at B=1 (sampler==spec order). This is
a strong candidate for the separate B=4 agentic-degradation carrier (carrier B=concurrency).
REMAINING B=1 conv suspects (need LIVE CAPTURE): (1) physical col-0 row stability -- commit writes _ssi[b,0]
snapshotted THIS step; next step reads spec_state_indices[b,0] NEXT step; if block_table[b,0] reassigned
between steps the read consumes a different physical row (orphaned carrier). (2) page-safe remap
(replay_conv_state_linear_remap_prepared) writing into col-0 before re-commit. Instruments: FR13_CHASE_DIAG H6
conv tap (L2536-2559, prior window bytes as-read), FR13_TCF_SELFCHECK=1 (L2484-2523 byte A/B of
read_cols/bank_rows/prior_bank). Capture _leaf_node/_src/_dst at commit + col-0 bytes before/after + byte-join
vs next step's H6 window.

## BOTH col-0 write audits CLEAN (2026-07-12): node-selection correct -> narrowed to NUMERICS/ALIASING (runtime)
SSM audit: RUNROW_COMMIT (fr10_gdn_tree_kernel.py:892-906) writes leaf `state` to col-0 row
(spec_state_indices[b,0]); leaf selected by NODE id via +1 anchor ring gather (chain [anchor,0,1,4]=ring
[0,1,2,5]); scratch linear cols burned; M-invariant by construction; custom kernel gathers IDENTICAL node set
as _fr13_native_committer_replay. NO wrong-node/sibling/bank-row/M defect. So BOTH conv+SSM col-0 writes are
static-correct => bug is NOT node-selection. Two runtime leads BOTH audits name:
 (a) custom-kernel NUMERICS/HANDOFF vs native committer (state+0.0 norm :829, prev_lens snapshot :782,
     h0-row aliasing) -- testable via FR13_COMMITTER_NATIVE (native col-0, bit-exact-to-no-spec, needle L973).
 (b) col-0/col-k ROW ALIASING: RUNROW_COMMIT writes col-0 (:892) THEN BURN_NODE_BANK zeroes rows
     spec_state_indices[b,1..SPEC_COLS-1] (:907-922); if col-0 row aliases a burned node row => BURN destroys
     the committed leaf. NOTE (b) would corrupt SPINE commits too (spine also RUNROW_COMMIT+BURN) => since
     spine is garble-free, (b) is LESS likely UNLESS aliasing is branch/step-specific. (a) is more plausibly
     branch-specific (num_accepted>1 branch numerics).
DECISIVE TEST: re-run FR13_COMMITTER_NATIVE with VERIFIED needle + matrix greedy, NO winner-log (the prior
greedy crash was in the path-LCP LOG, gated on LUMO_TREE_PATH_LCP_LOG/FR10_METRICS -- OFF => no crash). If
garble clears => custom kernel numerics/burn is the bug + COMMITTER_NATIVE is the fix. If persists (engaged)
=> SSM col-0 exonerated => conv runtime or KV.

## COMMITTER_NATIVE greedy re-test CRASHED (2026-07-12): diagnostic path-LCP block plumbing bug
Boot ship + FR13_COMMITTER_NATIVE=1 (sidecar present, worker-drop-proof) + matrix greedy (device committer
default). Request 500'd, needle=0 (crashed before engaging). Root cause: at greedy (all_greedy), the Python
_lumo_tree_path_lcp_max_greedy_sample runs; its diagnostic committer block (patcher L8938-9030, the replay/
durable-AB/apc-leaf publish, wrapped in the "FR10 tree path-LCP log" try) invokes the native committer replay
with a WRONG-SHAPED spec_state_indices (size-10 tree vs [B,SPEC_COLS]) => _fr13_prepare_committer_layout
(fr10_gdn_tree_kernel.py:951 `ssi[b,:]=col0[b]`) RuntimeError "expanded size (6) must match existing size (10)"
=> fail-loud re-raise at L9030-9037 kills EngineCore. This IS the summary's "COMMITTER_NATIVE greedy 6v10
crash" -- a plumbing bug in the COMMITTER_NATIVE+greedy DIAGNOSTIC wiring, NOT the committer math. The
deterministic-greedy path for COMMITTER_NATIVE is blocked until this is fixed.
PLAN: COMMITTER_NATIVE at TEMP 0.6 via fr13_garble_gate.py avoids all_greedy => no path-LCP block => no crash;
COMMITTER_NATIVE still engages in the GDN forward (needle confirms). Compare the tree garble RATE with
COMMITTER_NATIVE=1 to the known tree rate (8-11%) / native (0%). If it drops to ~0% => custom RUNROW_COMMIT
kernel numerics is the garble + COMMITTER_NATIVE is the fix. Boot-variance (2.96-9.56%) means eliminate-to-0
is the detectable signal. ALT (fully deterministic, offline): compare custom _tree_gdn_replay_kernel col-0 vs
_fr13_native_committer_replay col-0 for branch path [0,1,4] acc=3 (SSM audit lead #2) -- kernel-vs-kernel, no
server, no crash; gross divergence => custom kernel bug.

## COMMITTER_NATIVE result (2026-07-12): SSM col-0 write EXONERATED (non-vacuous, needle fired)
Boot locked cat9 (APC-off, eager) + FR13_COMMITTER_NATIVE=1 + temp-0.6 garble gate (tree arm). Needle
[FR13_COMMITTER_NATIVE ENGAGED] num_spec_decodes=1 FIRED (non-vacuous). Tree undefined-name rate = 8.04%
(15/33 early; matches known tree baseline 8-11%). => making the SSM col-0 STATE write fully native
(fused_sigmoid_gating replay, bit-exact-to-no-spec) does NOT reduce the garble => SSM col-0 write is NOT the
bug. Settles the prior "unchanged" as VALID (needle confirms). Combined w/ VERIFY_NATIVE (GDN scan compute
exonerated), the ENTIRE SSM/GDN path is clean (scan compute + col-0 state write). NOTE: _fr13_native_committer_
replay ALSO burns node cols (burn_node_bank param) => if the bug were BURN-aliasing it would survive
COMMITTER_NATIVE too (not distinguished here); and COMMITTER_NATIVE does NOT touch the CONV col-0.
=> REMAINING carried-state suspects (teacher-force proved trajectory-accumulated): (1) CONV col-0 (write
compute -- fused snapshot never replaced-native+tested since native_prior_read crashes w/ fused; runtime
physical-row/aliasing), (2) col-0/col-k BURN aliasing (cheap host check: is spec_state_indices[b,0] in
[b,1:SPEC_COLS]?), (3) KV cache (least likely: committed tokens all correct). NEXT: cheap host-side aliasing
detector on the live committer launch (real code, no worker-drop/fused-crash) + a fused-compatible conv-col-0
native test.

## MAJOR REFRAME (2026-07-12): mamba FULLY exonerated -> corruption is the ATTENTION KV cache
Confirmed conv col-0 READ = col-0 via RUNROW_INIT (fr13_tree_conv_fused.py:317-322 forces read_node_cols=0 =>
bank_rows=spec_state_indices[b,0]=col-0), IDENTICAL to what FR12_TREE_CONV_NATIVE_PRIOR_READ reads => that
conv test was VACUOUS, and the conv col-0 read is exonerated (read col-0 + write static-correct [audit] +
compute bit-exact). So the ENTIRE mamba state (SSM col-0 + conv col-0) is exonerated across scan compute,
col-0 write, col-0 read, and value. Yet teacher-force proves the carried state is corrupt. => the corrupt
carried-state component is the ATTENTION KV CACHE (committed-prefix K/V), the ONE carried component
VERIFY_NATIVE + COMMITTER_NATIVE never touch (both GDN/mamba-only). Consistent: the garbling verify is at the
ROOT node0, whose full-attention layers read the committed-prefix KV; if that KV is corrupt, node0 garbles and
no mamba-side native fix helps. LEADING KV hypothesis: a KV WRITE-POSITION / slot-mapping routing bug for
committed BRANCH tokens (step 11 branch commit [0,1,4] writes the committed KV; step 12 node0 reads it). This
is branch-specific (branch node KV positions) AND survives every VALUE-correcting native fix (VERIFY_NATIVE
corrects values not positions). NOT a small drift (garble is gross 15-nat). NEXT: audit the tree KV
slot-mapping / commit for committed branch tokens (tree_attn.py fork + the FR13 KV commit) for a wrong-position
write; then a KV-position capture or native-attention test. The FA2 "byte-exact 14/16" validated the fork
OUTPUT, NOT the committed-KV write POSITIONS across steps.

## RED-TEAM CORRECTION (2026-07-12): "mamba fully exonerated" OVER-CLAIMED — conv col-0 WRITE runtime untested
COMMITTER_NATIVE natively replaced the SSM col-0 WRITE (exonerated). But the CONV col-0 WRITE
(_fr13_conv_commit_to_col0, copies node4's node-bank window -> col-0) was only AUDITED static-correct
(node-selection right), NEVER replaced-with-native-and-tested. The conv audit itself flagged a RUNTIME risk it
could not settle statically: (1) physical col-0 row stability across the two steps, (2) whether the node bank
row read for node4 holds this-step's fresh window or a STALE value, (3) remap disturbing col-0. The
native_prior_read test only tested the conv READ source (=col-0, vacuous), NOT the conv WRITE value. So the
honest suspect set for the trajectory-accumulated carried state is: SSM col-0 EXONERATED; CONV col-0 WRITE
RUNTIME VALUE (co-suspect); ATTENTION KV / seq-position (co-suspect, audit running). Both remaining suspects
are best settled by the DEFINITIVE test I've deferred: a DIRECT STATE CAPTURE at the deterministic step 12 --
dump col-0 conv window + the committed-prefix KV + position_ids, and compare to a fresh-prefill of the same
committed ids. Whichever component DIFFERS from fresh is the corruption (mamba SSM should match = confirms
exoneration; if conv col-0 or KV differs = localized). Instruments exist: FR13_CHASE_DIAG H6 (conv-prior bytes
as-read), and a KV/position dump to add. This ends the indirect component-guessing.

## STRONG UNIFYING HYPOTHESIS (2026-07-12, KV audit): attention KV NOT re-linearized (GDN state IS)
KV audit (agent, 180k tok) decisive lead: the GDN/mamba carried state IS explicitly re-linearized along the
accepted path -- launch_tree_state_linear_remap (fr10_gdn_tree_kernel.py:354-388): "column k must contain the
state for accepted_paths[b,k]". The ATTENTION KV cache is NOT: tree verify writes each node's K/V at FLAT tree
slots (slot_mapping = num_computed+arange, linear node order; tree_attn.py reshape_and_cache_flash), NO
attention-side remap for accepted branch nodes (grep: accepted_paths never touches kv/block_table/slot; no
index_copy/scatter/gather on KV in src/). So accepted non-spine nodes' K/V stay at flat slots while the next
step reads LINEAR committed slots => reads a REJECTED sibling / root-scratch slot instead of the accepted leaf.
RECONCILES ALL FACTS: (1) mamba exonerated <=> mamba HAS the remap, attention doesn't (explains why
VERIFY_NATIVE + COMMITTER_NATIVE failed -- both mamba-only). (2) branch-specific: cat9 spine nodes at flat
slots 0,1,3,5,7 (branches 2,4,6,8 interleave) => ANY accept non-contiguous => misplaced; chain5 spine
contiguous 0,1,2,3,4 => clean. Branches' PRESENCE creates the non-linearity. (3) earlier branch commits clean:
committed TOKEN unaffected; misplaced KV corrupts a LATER read. (4) both garbles [0,2] acc=0: that step commits
nothing, just READS the committed prefix the preceding [0,1,4] len=3 commit populated (first flat!=linear leaf
accept). CAVEAT (must confirm before fixing): assumes stock vLLM does NOT re-point block_table to accepted flat
slots (agent found no such code, not proven). DECISIVE CAPTURE (hook exists: _fr13_tree_attn_op_capture patcher
15122): after [0,1,4] len=3 commit + at [0,2] len=0 verify, dump KV rows at (a) accepted flat tree slots vs (b)
linear committed slots slot(num_computed-accepted_len..num_computed-1); (a)!=(b) for the leaf => confirmed. Also
check RoPE mrope-vs-regular anchor mismatch (patcher 10703-10717: _fr10_state_base=num_computed-1 vs
_fr10_mrope_base=num_computed). FIX direction (no-HBM-tax): correct the committed-KV ADDRESSING (block_table/
read positions) for non-linear accepts = metadata, not a KV move; OR add the attention analogue of
launch_tree_state_linear_remap.

## RED-TEAM of the KV hypothesis (2026-07-12): it OVER-PREDICTS -> must capture, not build
The "no attention KV remap => accepted non-spine nodes misplaced" mechanism predicts PERVASIVE corruption: in
cat9 the spine is at flat slots 0,1,3,5,7 so EVERY multi-token accept is non-contiguous -- e.g. spine accept
[0,1,3,5,7] => linear slots 2,3,4 read nodes 2,3,4 (rejected siblings) instead of accepted 3,5,7 => ~3/5
committed-KV positions wrong on EVERY accept => garble ~most tokens. But observed garble is only 8-11%. So
EITHER (i) stock vLLM DOES re-point the block_table / the slot_mapping accounts for the tree (=> KV correct,
hypothesis REFUTED), OR (ii) the misplacement is an EDGE CASE (only certain accepts, e.g. acc=0 or the
len>0->len=0 transition), which the agent's "all non-spine accepts" mechanism does NOT predict. The 8-11%
non-pervasive rate is strong evidence the KV is MOSTLY correctly placed. => DO NOT build the KV-remap fix on
this unconfirmed+over-predicting mechanism. SETTLE with the decisive capture (_fr13_tree_attn_op_capture): at
the [0,2] verify, does linear committed slot hold the accepted leaf's KV (correct => refuted) or a rejected
sibling's (misplaced => confirmed, edge-case)? If refuted, attention KV is exonerated too and the suspect
returns to conv col-0 WRITE runtime (direct capture) or the RoPE mrope-anchor mismatch (patcher 10703-10717).

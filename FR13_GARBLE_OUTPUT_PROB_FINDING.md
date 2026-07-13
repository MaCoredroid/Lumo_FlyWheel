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

## RoPE mrope-anchor lead evaluated (2026-07-12): likely secondary (consistent off-by-one = relative-invariant)
Patcher 10703-10724: for a spec row with sched==tree_n, self.positions (=_fr10_depth_pos, => slot_mapping) uses
base=num_computed-1+depth_offsets; mrope_positions (=_fr10_mrope_depth_pos, => RoPE for Qwen3.x uses_mrope)
uses base=num_computed+depth_offsets. A CONSISTENT +1 in RoPE is relative-invariant (all tree tokens shifted
equally) => likely HARMLESS in isolation; only a SLOT-vs-RoPE mismatch (KV stored at slot num_computed-1+d
but RoPE'd with num_computed+d) or an old-committed-vs-new-tree base mismatch would bite. Secondary to the KV
misplacement.
STATUS: indirect leads EXHAUSTED. All mamba components exonerated (non-vacuous); attention KV-misplacement
hypothesis OVER-PREDICTS (8-11% not pervasive) so likely refuted-or-edge-case; RoPE likely harmless; conv col-0
WRITE runtime never native-tested. ONLY a DIRECT CAPTURE settles which carried component actually differs from
a fresh prefill. DECISIVE NEXT (the endgame): boot matrix greedy + eager, capture at the deterministic [0,2]
garble step -- (1) committed-prefix KV via _fr13_tree_attn_op_capture (does linear slot hold accepted leaf or
rejected sibling?), (2) col-0 conv window (FR13_CHASE_DIAG H6), (3) RoPE positions (FR10_METRICS) -- and diff
vs a fresh prefill of the committed ids. The component that DIFFERS is the corruption; SSM should match
(confirms exoneration). This ends the component-guessing; build the fix ONLY on the confirmed component.

## KV-misplacement REFUTED by over-prediction (2026-07-12); pivot to conv col-0 WRITE runtime
Dispositive: if committed-prefix KV were ~3/5-wrong on EVERY accept (agent's "no attention remap" => all
non-spine accepts misplaced), the model would condition on a badly-corrupted history and produce GARBAGE.
Observed = coherent code, 8-11% undefined names. node0 (the garbling node) attends over the WHOLE prefix each
step, so pervasive recent-KV corruption would garble node0 CONSTANTLY, not 8-11%. => the accepted KV IS
correctly placed (a stock-vLLM block_table re-point the agent missed; vLLM not host-readable to confirm, but
the code working at 8-11% proves it). Agent's mechanism over-predicts => REFUTED. RoPE mrope-anchor = secondary
(consistent off-by-one relative-invariant). => LEADING SUSPECT (by elimination) = the CONV col-0 WRITE runtime
value: _fr13_conv_commit_to_col0 copies node4's node-bank window -> col-0; node-SELECTION audited correct +
conv COMPUTE bit-exact, but the RUNTIME copy (is the node-bank row read this-step's fresh node4 window or a
STALE value? physical-row stability? remap disturbing col-0?) was NEVER native-tested (COMMITTER_NATIVE = SSM
only; native_prior_read tested the READ not the WRITE). TEST: (a) FR13_CHASE_DIAG H6 conv tap dumps the conv
col-0 window as-read at step 12 -> check for GROSS anomaly (zeros/NaN/discontinuity, self-evident, no fresh ref
needed); or (b) build a CONV-COMMITTER-NATIVE (recompute conv col-0 fresh over accepted path [0,1,2,5], write,
needle-gate) analogous to COMMITTER_NATIVE -- garble clears => conv write bug. If conv ALSO exonerated => the
carried-state framing itself needs re-examination (teacher-force tokenization confound recheck).

## ENDGAME localizer FOUND (2026-07-12): adapt the per-layer LADDER for the node0 matrix garble
All indirect leads exhausted/refuted (mamba exonerated; KV over-predicts=refuted; RoPE secondary; conv-write
plausibly-correct: BURN rewrites node4 window fresh each step, only physical-row-stability untested). The
DEFINITIVE localizer is the DIRECT per-layer capture, and the infra EXISTS: scripts/fr13_node5_ladder_drive.py
+ fr13_nodestep_realization_ladder.py + fr13_ladder_table.py. It's a same-boot TOP-DOWN PER-LAYER GATE:
 - PHASE LIVE: drive prompt (tree_mtp), FR10_LAYER_HIDDEN captures the tree-verify NODE-ROW hidden at EVERY
   layer + FR13_FINAL_LOGIT the projected logits (node5 ladder used ROWS=6=node5; for the garble use ROWS=0=
   node0 root at the [0,2] acc=0 garble step).
 - PHASE CLEAN: teacher-force the accepted prefix (max_tokens=1), FR10_ROOT_HIDDEN captures the clean
   last-row hidden at EVERY layer + FR10_ROOT_LOGIT. This is the RIGHT fresh reference.
 - Diff LIVE vs CLEAN per-layer => FIRST divergent layer localizes the corruption to a specific op (a mamba
   layer => mamba after all; a full-attn layer => attention KV/pos; the norm/mlp => elsewhere).
ADAPTATION: matrix_build prompt, ROWS=0 (node0 root), target the [0,2] acc=0 garble step (SKIP window), cut =
committed prefix through 'expected'. Existing captures: output/fr13_node5_ladder/{clean_prefix_layers.pt,
live_node5_finallogits.pt}. This ENDS the component-guessing with a per-layer ground truth; build the fix on
the first divergent layer's op only. Next: find the ladder BOOT config (which sets FR10_LAYER_HIDDEN/
FR10_ROOT_HIDDEN/ROWS/SKIP), adapt for node0+matrix, boot, drive, analyze.

## FOUNDATION CONFIRMED (2026-07-12, ladder boot): teacher-force is NOT a tokenization confound
Ladder drive (chat-based, cat9 eager) reproduced garble (undefined=2 expected_rows), clean teacher-force argmax
'_row' lp=-0.000 vs '_rows' -13.88 (13.9-nat gap re-confirmed). Tokenization boundary check: the garble region
tokenizes as [' =',' (','expected','_rows',','] -- 'expected' is a CLEAN token boundary immediately followed by
'_rows'. So the teacher-force prefix (ending 'expected') asks the EXACT SAME question (next after 'expected')
as the trajectory (which committed '_rows'). => teacher-force VALID, NOT a confound => the carried-state
framing HOLDS (fresh state '_row' vs trajectory '_rows' at the identical boundary). Captured 30 node0 per-layer
LAYER_HIDDEN forwards (~2.7MB each, non-vacuous) for the per-layer localization. NEXT: identify the garble
forward (node0 argmax='_rows', ~committer step 12), capture the CLEAN root per-layer (FR10_ROOT_HIDDEN re-run),
diff per-layer => first divergent layer = corrupt op.

## GARBLE FORWARD PINPOINTED (2026-07-12, ladder): call12 node0 pos190 argmax '_rows'
Projected each of 30 LIVE node0 captures' final_norm_hidden through lm_head: call12 (node0 pos190) argmax=
'_rows' (10630) <<GARBLE; call15 (pos193) also '_rows'. Adjacent calls clean (call11 pos185 '_slice', call14
pos192 ' expected'). So step-12's node0 (predicting pos190) is the garble, reading the state committed by
step-11 (branch [0,1,4] acc=3). Capture structure: 64 layers, pattern [GDN,GDN,GDN,full_attn]x16 (linear_
attention vs full_attention), each layer entry = {layer_idx, layer_type, hidden, residual}. => per-layer diff
of call12 vs a CLEAN node0-at-pos190 (teacher-force, argmax '_row') => FIRST divergent layer localizes: a GDN
layer => mamba (I mis-exonerated); a full_attn layer => attention KV; norm/mlp => elsewhere. Captured call11-15
to output/fr13_garble_ladder/live/. NEXT: same-boot re-run with teacher-force FIRST (CLEAN node0=call0) + LIVE
(garble), diff per-layer.

## PER-LAYER LADDER RESULT (2026-07-12): first divergence at layer 0 (GDN) => REOPENS mamba col-0
Same-boot node0 per-layer diff, CLEAN (call1 pos187 '_row') vs GARBLE (call21 pos191 '_rows'):
 - input_hidden IDENTICAL (rel_l2=0.0000 cos=1.00000) => node0 input token matches, NO embedding/upstream
   confound.
 - FIRST divergence at LAYER 0 (linear_attention/GDN, rel_l2=0.5887 cos=0.857), stays high (0.5-1.4) all 64
   layers. GDN layers have NO RoPE => a GDN-layer-0 divergence is NOT a position artifact => points at the
   col-0 MAMBA state the GDN reads. This REOPENS the mamba col-0 (contradicts the COMMITTER_NATIVE exoneration
   -- which only replaced the SSM col-0 WRITE; something in the mamba col-0 read by GDN layer 0 still differs).
CAVEAT (must resolve): CLEAN at pos187 vs garble pos191 = 4-token offset (chat-template assistant markers). IF
those 4 tokens are CONTENT the mamba processes, col-0 differs legitimately (confound). IF they're role-markers
only, the mamba (content-only, no position) divergence = the corruption. rel_l2=0.5887 at L0 is LARGER than a
4-token re-tokenization of the same text should cause, leaning toward real corruption -- but not conclusive.
NEXT: confirm the CLEAN prefix tokens == the LIVE committed tokens (4-gap = markers vs content); if content
matches => mamba col-0 CONFIRMED corrupt => re-examine what COMMITTER_NATIVE missed (conv col-0? the h0 read?).

## LADDER CONFOUND CONFIRMED (2026-07-12): CLEAN re-tokenization != LIVE committed => diff not clean
Positions: LIVE call20 (pos187) predicts '_shape'; CLEAN call1 (pos187) predicts '_row' -- SAME absolute
position, DIFFERENT content. So the chat-teacher-force prefix 'computed_slice_shape = (expected' re-tokenizes
4 tokens SHORTER than the LIVE committed sequence (CLEAN reaches 'expected' at pos186, LIVE at pos190). The
mamba states differ from the re-tokenization itself => the GDN-layer-0 divergence is CONFOUNDED; cannot
conclude mamba from it. ROOT BLOCKER: a clean per-layer diff needs the CLEAN to use the EXACT LIVE committed
token IDs (not re-tokenized text). cat9 (fr13_launch_locked) exposes ONLY /v1/chat/completions (no
/v1/completions with prompt=ids, which the node5 ladder used on fr10_launch_speed_server). => to resolve:
either (a) enable /v1/completions on the cat9 boot, or (b) capture the LIVE committed token IDs and teacher-
force them via a token-id path. The ladder INFRA works (captures + projection + diff all functional); the only
gap is exact-token CLEAN. The GDN-layer-0 lean toward mamba is SUGGESTIVE (rel_l2 0.5887 > a 4-token re-tok
should cause, and GDN has no RoPE) but NOT proven.

## EXACT-TOKEN LADDER (2026-07-12): points at mamba/GDN col-0 (layer 0), reopens it -- 1-pos caveat
Exact-token diff (teacher-force exact committed IDs via /v1/completions, 4-token re-tok confound ELIMINATED):
CLEAN node0 (pos192 '_row') vs GARBLE node0 (pos191 '_rows'). input_hidden IDENTICAL (cos=1.0) => same input
token, no upstream confound. FIRST divergence at LAYER 0 (linear_attention/GDN, rel_l2=0.5868), stays high --
SAME profile as the confounded 4-token diff. Since the 4-token confound is gone yet the profile persists =>
NOT a tokenization artifact; GDN has no RoPE => points at the col-0 MAMBA state GDN layer 0 reads. REOPENS the
mamba col-0 (COMMITTER_NATIVE only replaced the SSM col-0 WRITE; VERIFY_NATIVE also READS the same col-0 h0 and
didn't fix -- consistent). CAVEATS (not 100% clean): (1) residual 1-position offset (LIVE re-run committed 1
fewer token than the saved exact_tokens; CLEAN pos192 vs garble pos191); (2) NON-MONOTONIC profile (layer 2
GDN drops to 0.32 while 0/1 are 0.58/0.89) -- a pure col-0 corruption should propagate uniformly. NEXT:
position-matched CLEAN (recapture the CURRENT LIVE garble prefix, CLEAN at the exact garble pos) to eliminate
the 1-token offset; if layer-0 GDN divergence persists pos-matched => mamba col-0 CONFIRMED => re-examine the
col-0 h0 read (SSM vs conv) that COMMITTER_NATIVE/VERIFY_NATIVE both consume.

## MAMBA col-0 CONFIRMED (2026-07-12): layer-0 GDN divergence is RoPE-free, offset explained away
LIVE re-run garble = gen idx 65 / abs pos192 (SAME as saved exact_tokens) => CLEAN(committed[:192]) and garble
process the IDENTICAL 192-token prefix. The node0 "position" difference (CLEAN 192 vs garble 191) is a
draft-position-vs-prefix-last CONVENTION artifact = pure RoPE, affects ONLY the full_attention layers. KEY:
layers 0,1,2 are ALL linear_attention (GDN), BEFORE the first full_attention (layer 3), so they have NO RoPE
=> their diff is a CLEAN comparison (identical tokens + input_hidden identical + no RoPE). Layer 0 GDN diverges
rel_l2=0.5868 => the col-0 MAMBA state the spec-decode built != the sequential-prefill col-0 = CORRUPT.
CONFIRMED without a pos-matched re-boot (the 1-pos offset only inflates the attention-layer diffs, which I now
DISCOUNT). Per-layer non-monotonic (L0=0.58,L1=0.89,L2=0.32) = per-layer col-0 corruption (each GDN layer's own
col-0). => mamba col-0 REOPENED + CONFIRMED. RECONCILE: SSM col-0 WRITE exonerated (COMMITTER_NATIVE native =
same), so the corrupt col-0 that GDN L0 reads is the CONV col-0 PRIOR WINDOW (feeds conv->SSM) -- audited
static-correct but NEVER native-tested. VERIFY_NATIVE reads the same col-0 => inherited, didn't fix (consistent).
NEXT: test the conv col-0 prior directly (native conv replay / capture the conv col-0 vs sequential); a
compute-only fix to the conv col-0 write/read is the ship-fix direction.

## Cumulative-onset test INCONCLUSIVE (2026-07-12): capture-matching fouled it
Cumulative diff (nearest-by-position CLEAN vs LIVE) gave input_hidden rel_l2=1.25/1.38 (NOT identical) => it
matched captures with DIFFERENT tokens (CLEAN-continuation forwards don't land at predictable call numbers;
nearest-by-position is unreliable). So the cumulative-vs-step-specific question is UNRESOLVED by this run --
honest null, not a signal. The clean result stands: exact-token diff (auto-matched by argmax '_row'/'_rows' =>
input IDENTICAL) shows layer-0 GDN divergence => mamba col-0 CONFIRMED corrupt. The ladder capture-gating is
too fragile for multi-position matching (CLEAN continuations capture unpredictably); the SINGLE-position
exact-token diff is the reliable instrument. CONSOLIDATED STATE: the corruption is the col-0 MAMBA state built
by the stateless-tree commit (task #11), NOT attention/verify-compute (both ruled out by the ladder + prior
tests). COMMITTER_NATIVE (native col-0 replay) agrees with the custom kernel => both corrupt from a shared
input (the previous-step col-0 h0 they replay from = CUMULATIVE, or the tree-scan leaf final_state at M=10).
FIX DIRECTION: correct the col-0 mamba state build so it equals the sequential-prefill col-0 (the ladder CLEAN
reference). Next: re-examine the col-0 WRITE value (leaf final_state copied, and whether it carries the prior
col-0 corruption) -- the memory's OPEN project_fr13_conv_priorwindow_root + the confirmed col-0 corruption
converge here.

================================================================================
## ROOT CAUSE FOUND + FIXED (2026-07-12): conv committer +1 off-by-one on the
## NODE-INDEXED conv bank

**The bug (code-proven, exhaustive trace across all 3 consumers + adversarial-
verified surviving candidate).** `_fr13_conv_commit_to_col0`
(scripts/fr10_phase4_patch_vllm_tree_gdn.py, in the `helper` r''' block) commits
this-step's accepted-leaf conv window into col-0 (the running row). It reads the
source column as `_src = _ssi[b, _leaf_node]` where `_leaf_node =
accepted_path_buf[b, acc_len-1]`. The caller (L8726 packed / L9612 sampled)
passes `_accepted_path_buf` = `_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR`, filled from
`_gdn_path = [int(node_id)+1 ...]` (L8473-8497) = the GDN/SSM **+1-ANCHORED**
path.

But the CONV bank is **NODE-INDEXED, no anchor**:
- Forward write: `for node_i in range(tree_n): window(path_node_tensors[node_i])
  -> spec_state_indices[b, node_i]` (L3482, L3542). path_node_tensors[i] = root-
  to-node-i path (L255-263), node0's path = [0] (NO empty-anchor prepend).
- Committed-path read (`prepare_committed_path_conv_rows`, L310): indexes with
  the RAW node id `accepted_paths[b, len-1]`, comment "node-indexed layout".

So the committer borrowed the SSM's +1 convention (correct there: the SSM/GDN
bank IS +1-anchored, col0=running-row anchor, realized by prepend-0 in
_fr13_prepare_committer_layout) and MISAPPLIED it to the anchorless conv bank ->
reads node(leaf+1)'s window into col-0. Its own docstring is wrong ("Leaf node id
= accepted_paths[b, acc_len-1]") -- the caller passes the +1 GDN buffer, not raw
accepted_paths.

**Reconciles every prior fact:** COMMITTER_NATIVE / VERIFY_NATIVE are SSM-side
(recompute via prepend-0, correct) and NEVER touch the conv col-0 prior -> can't
fix. Compute bit-exact (only the selected bank ROW is wrong; every node's conv
window CONTENT is correct). Node-selection audits passed (the correct node id IS
chosen; the +1 is a SEPARATE column-routing step). Cumulative (each commit writes
a corrupt col-0 that next step's RUNROW_INIT reads).

**Branch specificity (why 8-11%, not 100%):** interleaved (len,path) numbering.
Spine leaves 1,3,5,7 -> +1 = siblings 2,4,6,8 (last-tap-only diff => within
argmax floor => spine stayed lossless "by luck"). Branch leaves 2,4,6 -> +1 =
3,5,7 = different/deeper subtree => GROSS wrong window => garble. Branch nodes
live at depth>=2 so every branch commit is num_accepted>1 (matches the multi-
token-accept specificity).

**BASELINE (live fr13-lad, buggy, cache OFF, DEVICE_MULTIDRAFT=1, exact-token
greedy /v1/completions of matrix_build):** garble present at gen positions 64,67
(token '_rows' 1748/10630 where '_row' is correct), gen_len=221. Deterministic.

**FIX:** conv committer uses the node-indexed leaf column
`_conv_col = (_leaf_node - 1).clamp(min=0)` (alen>0 -> leaf node id; alen==0 ->
col-0 running row). Compute-only, no HBM tax, spine strictly improves (exact node
instead of within-floor sibling). Legacy +1 restorable via env
`FR13_CONV_COMMIT_PLUS1_LEGACY=1` for same-config A/B. Needle:
`[FR13_CONV_COMMIT_NODEIDX ENGAGED]`.

GATE (pending reboot): greedy garble -> GONE; ladder col-0 GDN-layer diff -> ~0;
temp-0.6 garble gate -> 0% (native parity); spine argmax lossless; live SWE-
Verified with cache ON (FR13_ENABLE_APC=1).

================================================================================
## REFUTED (2026-07-12): the conv-committer +1 was NOT the bug — I analyzed the
## DEAD non-fused path. Reverted.

Same-config A/B on the locked cat9 boot (GPU_UTIL=0.78, cache OFF), ONLY the conv
committer column differs (needle legacy_plus1 confirms which):
- **legacy +1 (baseline)**: greedy _rows_garble=TRUE but CLEAN well-formed code;
  temp0.6 n=15 syntax_bad=0/45, undef_samples=29/45 (the _rows identifier garble).
- **my -1 (proposed fix)**: greedy _rows_garble=FALSE (gone) BUT temp0.6 n=15
  syntax_bad=114/120 (95%) = DEGENERATE (repetition loops, digit spam, truncated
  signatures). undef=2/120 only because the scorer can't detect names in
  unparseable code.

=> The -1 is STRICTLY WORSE (trades mild identifier garble for gross degeneracy).
+1 is the correct ship column and was REVERTED.

**Root of my error:** I read the NON-FUSED conv forward (the `else:` branch at
L3482, `_fr10_tree_n = len(parent)`) and concluded "conv bank node-indexed, +1 is
off-by-one". But the ACTIVE ship path is `FR13_TREE_CONV_FUSED=1`, whose bank is
**+1-ANCHORED** (boot log: `gdn_linear_attn.py:196 conv emulation engaged:
fused=1 tree_n=10` — col0=anchor, tree-node k -> col k+1, same as GDN). So
accepted_path_buf=node_id+1 is the CORRECT column and the -1 reads the wrong
node. The workflow's "surviving candidate" cited the same dead non-fused L3482 —
also refuted. A code-proof on the wrong code path is not a proof.

**Genuine clue retained:** the -1 DID remove the _rows garble (29/45 -> ~0), so
the conv committer column DOES couple to _rows. But since +1 (correct column)
STILL has _rows garble, the _rows root is NOT the committer column selection --
it's the WINDOW CONTENT at the (correct) column, or the SSM col-0, on the FUSED
path. The per-layer ladder col-0 corruption (GDN layer-0) STANDS; only its
attribution to the committer +1 is refuted.

**NEXT:** re-hunt the _rows col-0 corruption on the ACTIVE fused conv path
(src/lumo_flywheel_serving/fr13_tree_conv_fused.py: build_tree_conv_state_src_
indices + prepare_committed_path_conv_rows) and the SSM col-0 -- NOT the committer
column. Ladder must capture on the fused path (FR13_TREE_CONV_FUSED=1), which is
what actually runs.

================================================================================
## BREAKTHROUGH (2026-07-12): garble KILLED by BATCH_INVARIANT + BI_TREE_ATTN
After refuting committer / GDN-scan / GDN-state (whole-GDN-native) / in_proj_ba (bmm), booted
BATCH_INVARIANT=1 + FR13_BI_TREE_ATTN=1 (graph, cache OFF, GPU_UTIL=0.78). Deterministic greedy matrix
garble GONE: _rows_garble=FALSE, clean coherent code, decode 7.4s (graph, no speed collapse). BI patches
confirmed applied (tree_attn.py/batch_invariant.py True). => The garble IS an M-dependence / co-residency
effect (directive was right), killed by making GEMMs+reductions+tree-attention batch-invariant. It is a
DIFFERENT M-dependent op than in_proj_ba (bmm alone did NOT fix it). GDN scan/state already exonerated
(native), so the carrier is a GEMM/reduction/attention that BATCH_INVARIANT covers.

NEXT = BISECT for a TARGETED compute-only fix (full BATCH_INVARIANT takes the GB10 REDUCED path = speed
cost; not shippable as-is):
(1) BATCH_INVARIANT=1 + FR13_BI_TREE_ATTN=0 -> garble gone? YES=GEMM/reduction carrier (bisect which);
    NO=the TREE ATTENTION (BI_TREE_ATTN) is the carrier (FA2 MMA-grouping M-dependence, 2/16 single-ULP).
(2) then localize the single op + a targeted batch-invariant fix for it, gate temp-0.6 + live SWE cache-ON.

## BISECTION-1 (2026-07-12): carrier is a GEMM/REDUCTION, NOT the tree attention
BATCH_INVARIANT=1 + FR13_BI_TREE_ATTN=0 (graph, cache OFF): greedy garble STILL GONE (_rows_garble=False,
decode 7.4s). => the tree attention is NOT the carrier; the carrier is a GEMM or reduction that
VLLM_BATCH_INVARIANT overrides. Since in_proj_ba bmm alone did NOT fix it, it is a DIFFERENT op. Prime
suspects (run every layer, M-dependence amplifies ~492x): RMSNorm reduction, o_proj/MLP bf16 GEMM,
in_proj_qkvz (though fp8 per-row = memory says M-indep). NEXT: enumerate the batch_invariant overrides +
bisect to the single op, then a TARGETED batch-invariant fix (no full-BI speed tax).

## RETRACTED (2026-07-12): BATCH_INVARIANT "breakthrough" was a GREEDY-GATE CONFOUND
The greedy _rows-at-idx-65 gate showed garble GONE with BATCH_INVARIANT. But the temp-0.6 per-prompt gate
on the SAME boot: matrix_build 15/15 garbled, token_ledger 13/15, wcs_slice 2/15 (overall 11.90% undefined
= baseline 9.56%, native 0%). => BATCH_INVARIANT does NOT fix the garble. The greedy flip was a confound
(perturbed one near-tie clean; mechanism intact -- matrix_build 15/15 at temp0.6 PROVES it).

**METHODOLOGY FIX (critical):** the greedy deterministic _rows/idx-65 gate has FALSE NEGATIVES (greedy
clean != fixed; matrix_build greedy-clean but temp0.6 15/15 garbled on the SAME prompt). Do NOT trust
greedy=False as "fixed". USE the temp-0.6 per-prompt gate (matrix_build undefined rate: 15/15 garbled =
present, ~0 = fixed) as the garble metric. NOTE: prior greedy=TRUE refutations (whole-GDN-native,
in_proj_ba bmm, committer) still stand -- those failed to fix even the greedy, so definitely not temp0.6.
BATCH_INVARIANT (GEMM+reduction batch-invariance) + BI_TREE_ATTN (attention) BOTH insufficient => the
garble is NOT a batch-invariance issue in GEMMs/reductions/attention (and not GDN/in_proj). NEXT: re-gate
on temp-0.6 matrix_build; the carrier is elsewhere (routing/logic? a non-batch-invariant M-dependence?).

## STATE-DRIFT CONFIRMED via the ACTIVE device-multidraft committer code (2026-07-12)
Live-path boot log: temp-0.6 (ship/reliable gate) runs FR13_DEVICE_MULTIDRAFT (device temp>0 committer),
NOT the greedy Python committer my prior tests used. fr13_device_multidraft_kernel.py docstring: accept
DISTRIBUTION proven identical to host reference (per-node accept-prob match); the committed near-neighbor
garble is because "the drift depresses the correct argmax to ~0.80" and inflates the garble branch from
true ~1e-6 to ~0.2 => the ACCEPT rule faithfully commits DRIFTED verify probs. So the accept committer is
CLEARED (again, on the ACTIVE path). Garble = gross FORWARD DRIFT (correct ~0.80 / garble ~0.2, ~12-nat
inflation) in the verify probabilities. Compute exonerated on the reliable gate (BATCH_INVARIANT temp-0.6
15/15) => the drift is STATE corruption feeding the verify at branch num_accepted>1.

**TOOLING BLOCKER (the real wall):** the reliable gate is temp-0.6 (graph mode, device-multidraft), but
the state diagnostics (VERIFY_NATIVE/COMMITTER_NATIVE) are EAGER-ONLY, and eager itself degrades quality
(confounds temp-0.6). The greedy gate is graph-compatible but has false negatives (BATCH_INVARIANT
confound). So I cannot cleanly test a state fix on the reliable gate. NEXT = close this gap: a
GRAPH-COMPATIBLE capture of the verify's INPUT state (GDN col-0 h0 + conv col-0 running window) at a branch
num_accepted>1 commit, CLEAN(sequential) vs garble(tree), on the temp-0.6 path, to localize WHICH state
component drifts (GDN col-0 / conv col-0 / physical-row spec_state_indices routing). Confirmed NOT: forward
compute, GDN scan, in_proj_ba, attention, accept committer.

## whole-GDN-native EXONERATED on the RELIABLE gate (2026-07-12, definitive)
Via canonical harness (scripts/fr13_garble_test.sh) EAGER + FR13_VERIFY_NATIVE=1 + FR13_COMMITTER_NATIVE=1
(both needles FIRED: VERIFY_NATIVE tree_n=10, COMMITTER_NATIVE num_spec_decodes=3; DEVICE_MULTIDRAFT engaged
= ship accept path). temp-0.6 matrix_build = 15/15 garbled = NOT FIXED. Upgrades the earlier greedy-only
finding to the reliable gate. ALSO confirmed eager is NOT a garble confound (eager baseline matrix_build
15/15 == graph). So on the RELIABLE gate, EXONERATED: forward compute (BATCH_INVARIANT), GDN scan+state
(whole-native), attention (BI_TREE_ATTN), in_proj_ba (bmm greedy), accept committer (device-multidraft code
proven faithful). REMAINING (uncovered by ALL of the above): the CONV col-0 running window (COMMITTER_NATIVE
rebuilds GDN col-0 from POST-conv ring activations, so a corrupt conv col-0 is INHERITED not fixed) and the
KV cache content/routing (attention COMPUTE is BI but KV CONTENT for branch nodes is not). NEXT: test the
conv col-0 (running window) on the reliable gate; then KV routing.

## UNIFY commit path (user directive): design verified, Tier2 needs a losslessness PROOF
Workflow wf_6b781ea9 (7 agents, adversarial): greedy uses _lumo_tree_path_lcp_max_greedy_sample (LCP-path-
max); temp>0 uses device-multidraft (stochastic softmax+multinomial). Routing greedy through multidraft AS-IS
is NOT lossless (Case B). Tier1 (single dispatch entry, delegate greedy internally) = byte-identical + CAG
free, but does NOT run reject-sampling for greedy (keeps LCP-max) => does not honor "reject sampling even
greedy". Tier2 (multidraft reject runs greedy via one-hot-p deterministic branch + deterministic tie-break +
device kernel OFF) = the literal unification, but adversarial CORRECTNESS lens: "sequential argmax-descent ==
global max-LCP path" for multi-node BRANCHES is UNPROVEN. GATING QUESTION (offline, decisive): does the
deterministic multidraft descent accept the SAME path as the greedy LCP-max committer on branching trees?
Prove => ship Tier2 (one reject path); disprove => the greedy/temp>0 accept rules genuinely differ (a real
reason the greedy gate was unfaithful). Adversarial fixes for either tier: delegate BEFORE the
FR13_FORCE_SPINE_COMMIT raise (L9120); unified guard on tree_parent_indices ONLY (tree_token_ids is
greedy-only, L9860 -> temp>0 batches would silently lose the committer).

## KEY INSIGHT (2026-07-12): greedy & temp-0.6 garble = SAME drift, different accept manifestation
Read the two accept rules. Greedy (_lumo_tree_path_lcp_max_greedy_sample) commits the target ARGMAX
(one-hot). temp>0 device-multidraft (sample_deterministic_multidraft_rejection_step) SAMPLES the spread
softmax p via rng (source=rng.choice(weights), accept~min(1,p/q_mix), residual~p). BOTH read the SAME
drifted verify logits. The drift depresses the correct token to ~0.80 and inflates the garble to ~0.2, so:
  - GREEDY: argmax(drifted) -> commits garble ONLY when drift flips the argmax (>50%). Lower sensitivity.
  - TEMP-0.6: samples p -> commits garble ~20%/token -> accumulates to matrix_build 15/15.
=> The greedy gate is NOT "unreliable", it is a LOWER-SENSITIVITY drift detector; the BATCH_INVARIANT
"false negative" was the perturbation nudging that ONE argmax back to correct without zeroing the drift
(temp-0.6 still sampled it). CONSEQUENCE: the drift (STATE corruption) is common to both; the greedy
per-layer LADDER localization (first divergence = GDN LAYER 0 col-0) TRANSFERS to the temp-0.6 garble.
So the conv col-0 running window (the only layer-0 state uncovered by whole-GDN-native) remains the prime
suspect, now on solid footing. A fix must ZERO the drift (temp-0.6 ~0/15), not just flip the greedy argmax.
UNIFY COROLLARY: routing greedy through the multidraft is lossless IFF p is forced one-hot (argmax delta);
then multidraft accept_prob=1 for the argmax + residual->argmax == greedy argmax-accept (== LCP-max on
distinct-draft trees like cat9). Tier2 = one-hot-p greedy branch, flag-gated, byte-identical accepted-set gate.

## Garble is EARLY / near-single-step, not long-cumulative (2026-07-13)
First-garble line across 32 garbled tree.jsonl samples: min=2, median=4, max=7 (of ~12-line functions).
=> the garble hits the FIRST long-identifier reuse (line 2-4), NOT accumulating over a long generation.
So the drift is a near-single-step effect: the first branch commit (num_accepted>1) corrupts col-0 and the
NEXT verify drifts. Fix shape: no long-accumulation handling needed; target the single branch-commit ->
col-0 -> next-verify path. DECISIVE next test unchanged: conv-col0-native recompute (does rebuilding the
conv running window natively from the committed tokens at the branch commit zero the drift?). Static
analysis of the conv col-0 committer/window/compute is EXHAUSTED (all static-correct, drift persists) =>
must be DATA/constructive, not another code read.

## UNIFY proof + garble corollary (2026-07-13, offline scripts/fr13_lcp_vs_descent_equiv.py)
200k random cat9 trees + random deterministic targets: LCP-max (greedy) vs deterministic-descent (one-hot-p
multidraft) => committed TOKENS identical 100%; accepted PATH differs 1.6% = spine-vs-branch TIES (two nodes
draft same token; LCP-max credits branch, descent credits spine; tokens identical). => Tier-2 unify is
TOKEN-LOSSLESS (gate on committed tokens, NOT accepted_tree_rows which differ harmlessly on ties). GARBLE
COROLLARY: the garble token is committed as the BONUS/RESIDUAL (= the drifted verify argmax) when the correct
draft MISSES the drifted argmax -- BOTH accept rules do this identically. So the garble is the DRIFT, not the
accept rule; unifying the committer will NOT fix garble (confirms accept-exoneration independently). The 1.6%
tie path-divergence feeds accepted_tree_rows -> GDN replay state route: harmless IFF the state advance is
token-correct (state=f(tokens)); if state were f(node) it would matter -- a latent check for the state fix.

## REDIRECT (2026-07-13): conv col-0 REFUTED; by elimination -> ATTENTION KV content
Red-team of my own conv-col0 convergence: the conv committer copies col(leaf+1)->col-0, and col(leaf+1) IS
the fused conv's leaf-node window = pure-torch (elementwise+gather, M-INVARIANT) + byte-exact to native
causal_conv1d. So the conv col-0 is CORRECT. With COMMITTER_NATIVE making GDN col-0 correct, the WHOLE col-0
state is correct => the ladder's rel_l2=0.6-0.9 "layer-0 divergence" is IMPOSSIBLE if state+compute are right
=> the LADDER IS AN ARTIFACT (compared different logical positions: spec draft-node0 vs seq committed-col0).
The conv-col0-native build is NOT warranted.
ELIMINATION: whole-GDN-native (all GDN layer outputs native) + BATCH_INVARIANT (MLP/lm_head GEMMs + norm/mean
reductions) + BI_TREE_ATTN (attention COMPUTE) ALL fail the reliable temp-0.6 gate. The ONLY thing none of
these covers = the ATTENTION KV CONTENT/ROUTING (which node's/token's K,V is written to and read from the KV
cache for branch nodes at num_accepted>1). BI_TREE_ATTN makes the attention COMPUTE batch-invariant but does
NOT fix which KV is stored/read. NOTE: "KV-misplacement REFUTED (over-predicts)" was a specific row-swap
hypothesis; a subtler branch-KV routing/content bug at num_accepted>1 is UNTESTED on the reliable gate.
NEXT: localize the tree-attention KV write/read for the committed branch path (FA2 fork + KV cache indices).

## PRECISE LOCALIZATION (2026-07-13, by elimination + M10-vs-M1 attention argument)
At the garble step, node0 (root/running position) verify differs at M=10 (tree) vs M=1 (clean native). Given:
embedding identical, col-0 state correct (conv=torch-M-invariant leaf window; GDN=COMMITTER_NATIVE), compute
batch-invariant (BATCH_INVARIANT + BI_TREE_ATTN + whole-GDN-native) => the ONLY thing that can differ is the
KV node0 ATTENDS TO. => the tree writes DIFFERENT KV for the committed branch path than native does (wrong
position/RoPE, or wrong node's K/V, at the num_accepted>1 branch commit), OR node0's read slot_mapping/mask
at M=10 pulls different KV than the M=1 clean read. This is the tree-attention KV CACHE path (write at branch
commit / read at verify) -- uncovered by EVERY compute/state test run so far. Prime sub-suspects: (1) branch
node POSITION IDs (depth) feeding RoPE -> stored K wrong; (2) which node's K/V is written to the committed
running slot at a branch commit; (3) the verify attention MASK/slot_mapping at M=10. NEXT: capture/compare the
committed-branch KV (or positions) tree-vs-native at a branch commit on the reliable gate; or a native-KV
(recompute attention KV for the committed path) test = the attention analog of whole-GDN-native.

## LEADING HYPOTHESIS (2026-07-13): MISSING KV linear-remap for accepted BRANCH paths
launch_tree_state_linear_remap (fr10_gdn_tree_kernel.py:354, called patcher L2689) re-linearizes the GDN+conv
state after accept: "the committer publishes accepted_paths as tree NODE columns; vLLM's GDN/conv consumers
read by LINEAR accepted-token position, so column k must contain accepted_paths[b,k]." The ATTENTION KV is
the IDENTICAL situation (KV written per node-slot; next-step block_table reads linear) but has NO such remap
(memory + grep confirm). Layout = caterpillar spine-first => SPINE accepts land on linear slots (clean, no
remap needed) but BRANCH accepts land on NON-LINEAR slots => the next step reads the WRONG slot's KV for a
committed branch node => node0 verify drifts. MATCHES EVERYTHING: branch-specific (spine linear=clean),
num_accepted>1 (multi-token where branch diverges), KV (attention), drift, UNCOVERED by every compute test,
and explains why whole-GDN-native FAILS (GDN state IS remapped/correct; the KV is NOT). FIX = add a KV
linear-remap mirroring launch_tree_state_linear_remap (copy accepted nodes' K,V from flat slots to linear
committed slots; only num_accepted rows = no HBM tax). GATE: reliable temp-0.6 matrix_build -> ~0/15, +
whole-config live SWE cache-ON. VERIFY layout order + that vLLM doesn't already remap before wiring.

## CAVEAT (2026-07-13) on the KV-remap hypothesis: weakened for cat9; KV PATH still localized
Red-team: accepted_tree_rows (written per-commit, "the next forward's input", patcher L8414/8455/8465) IS
used by vLLM to reference the accepted path's KV. For cat9, EVERY accepted branch path = [spine
intermediates] + [branch LEAF] (nodes 2,4,6,8 are leaves; 1,3,5,7 spine), so intermediates land on linear
slots and the leaf is handled by accepted_tree_rows => the KV may be CORRECTLY handled, i.e. the "missing
remap" may NOT fire for cat9. Also the garble step itself ([0,2] acc=0) commits only a single BONUS token
(no branch KV kept) -- the corruptor is the PREVIOUS branch commit. So: the ELIMINATION to the KV path is
sound (everything else exonerated on the reliable gate), but the exact KV mechanism (remap vs
accepted_tree_rows-correctness vs slot/position) RESISTS static analysis. DECISIVE NEXT = DATA, not more
reading: capture the KV cache rows at the committed positions after a BRANCH commit, tree-vs-native decode
of the same committed tokens; the first divergent (position, layer) is the bug. Then build the targeted fix.
Do NOT build a KV-remap on the unconfirmed hypothesis (lesson: conv-committer +1 built on a dead path).

## DECISIVE (2026-07-13): kitchen-sink confirms garble is KV CONTENT/ROUTING, not compute/state
Reliable temp-0.6 gate, ALL fixes co-armed + all needles FIRED (VERIFY_NATIVE + COMMITTER_NATIVE +
BATCH_INVARIANT + BI_TREE_ATTN + device-multidraft): native GDN outputs + correct col-0 + batch-invariant
GEMMs/reductions + batch-invariant attention COMPUTE. Result: matrix_build 15/15 = UNCHANGED. => the garble
is EMPIRICALLY, decisively NOT compute, NOT GDN scan/state, NOT attention compute. By airtight elimination it
is the ATTENTION KV CONTENT/ROUTING for branch nodes (which K/V is written/read/attended, at which slot, for
num_accepted>1 branch commits) -- the ONLY thing none of these fixes covers. My cat9-accepted_tree_rows
red-team (KV "might be handled") is REFUTED by this result: the KV is NOT correctly handled. KV values (RoPE
position depth-correct, in_proj exonerated) appear static-correct, so the bug is the SLOT/BLOCK-TABLE routing
or the attention MASK for the committed branch path, not the K/V values. NEXT (fully justified): capture the
KV cache rows / block_table / attention mask at a BRANCH commit, tree-vs-native; the first divergence is the
bug; then targeted fix. OR native-attention recompute for the committed path (attention analog of
COMMITTER_NATIVE) as a constructive fix-test.

## ROOT CAUSE LOCALIZED (2026-07-13): missing ATTENTION-KV linear remap (GDN/conv have one; attention doesn't)
Convergent from FOUR independent lines:
 1. KITCHEN-SINK (today): all compute+state+attn-COMPUTE fixes co-armed, needles fired => garble in KV
    content/ROUTING, not compute/state/attn-compute.
 2. CODE-GREP: `launch_tree_state_linear_remap` re-linearizes GDN ssm_state + conv_state (tree NODE columns ->
    LINEAR accepted-token positions: "column k must contain accepted_paths[b,k]"). grep for any
    attention-KV relinearize/compact/rewind/remap => ZERO. The attention KV cache has NO post-accept remap.
 3. BANKED H3 (FR13_CHASE_FIXA_BIND.md, topological): foreign_slot by depth d2..d5 = 167/130/84/76 foreign
    vs 16/17/22/4 clean; d1 203/0 clean. Pattern: full-SPINE accepts read FOREIGN KV at every depth>=2
    (same-depth tree rows collide; branch-leaf = last writer wins the slot); alt-leaf clean; chains never
    foreign. Explains chain5-clean / cat9-garbled.
 4. FLOW: verify writes each tree node's K/V at its flat/verify slot; after accept vLLM advances seq_len and
    the NEXT forward reads the accepted tokens at LINEAR positions [seq..seq+acc-1]. The accepted path (e.g.
    spine [0,1,3,5,7] flat slots 0,1,3,5,7) is NON-CONTIGUOUS, so linear position seq+d reads node@flat-d's
    K/V (a sibling / branch-leaf near-neighbor), NOT the accepted node's. FOREIGN.
Reconciles the "gross wrong-accept" vs "8-11% flip": foreign KV = a SIBLING node (near-neighbor token, same
drafter/depth) => SMALL per-position drift that COMPOUNDS over later attention (amplification physics ~492x)
into a gross wrong-accept with near-neighbor character ('_row'->'_rows'); flips 8-11%.
FIX (constructive, needle-gated): add the attention analog of launch_tree_state_linear_remap -- after accept,
before the next forward's KV writes, COPY each accepted node's K,V from its flat verify slot -> the linear
committed slot, per full-attn layer. Small KV copy (acc rows x layers), NO weight read => same cost-class as
the existing GDN/conv remap (NOT a full re-forward). Engagement needle: count foreign copies (src_slot !=
dst_slot) + assert byte-diff on foreign; vacuous(0 foreign) => mechanism refuted, pivot. Gate = reliable
temp-0.6 matrix_build tree-vs-native.

## FIX BUILT + UNDER TEST (2026-07-13): FR13_ATTN_KV_REMAP (default OFF)
launch_attn_kv_linear_remap (fr10_gdn_tree_kernel.py) + 2 gated vLLM patches: (A) capture full-attn
kv-cache group's per-step slot_mapping/layers/qsl during attn-metadata build (non-mamba/GDN builder);
(B) sample_tokens after _sample (fresh KERNEL committer paths, NROWS>0 gate) copy each committed node's
K/V flat verify slot -> linear committed slot per full-attn layer. Indexing (H3-derived, unit-tested
n_foreign=3): src offset=accepted_paths[b,m] (flat verify row 1..9), dst offset=m+1; foreign=ap!=m+1.
Guard: first N batch reqs all full trees (qsl spacing==tree size) else safe skip (no wrong-slot copy).
RED-TEAM THE GATE RESULT ON:
 (a) path0 log = flat rows starting at 1 (spine [1,2,4,6,8], foreign=3) NOT node-idx [0,1,3,5,7]
     (foreign=4) -- the latter means +1 convention wrong (fix: shift).
 (b) foreign>0 (engaged, not vacuous).
 (c) matrix_build garbled ~0/15 (vs kitchen-sink 15/15, native 0/15) = FIXED.
 (d) no fail-loud crash (capture populated, convention matched).

## FIX WORKS — cat9 branched garble ELIMINATED (2026-07-13): 15/15 -> 0/15, engaged, clean
FR13_ATTN_KV_REMAP=1, cat9 branched, temp-0.6, ENFORCE_EAGER=1, cache OFF. Reliable gate:
  matrix_build 0/15 (was 15/15), token_ledger 0/15 (was 10-11/15), wcs_slice 0/15 (was 1-3/15).
  undefined-name-rate 0.00%, samples-with-undef 0/45, syntax-errors 0/45 (CLEAN read, not degenerate).
ENGAGED foreign_first>0 (DIAG n=4 cols=5 byreq=True path0=[1,2,4,6,8] spine + [1,2,4,6,9] BRANCH).
=> the missing attention-KV re-linearization WAS the garble root cause (CONFIRMED by the fix). Branches
KEPT (branch-path commits garble-free) = the deliverable, NOT chain5/reshape. Causal: the ONLY change vs
the prior vacuous 15/15 runs is the remap actually copying foreign rows (guard-blocked=15/15 -> guard-fixed
foreign>0 = 0/15); flag gates only the remap block.
REMAINING for SHIP (honest): (a) cache ON (APC) slot layout; (b) graph mode (capture/apply timing); (c)
live SWE-Verified WITH cache ON; (d) general spec-row->batch alignment (guard currently restricts to
all-tree uniform-span batches); (e) confirm cat8/cat6 (same generic mechanism); (f) speed tax (small KV
copy per step, no weight read).

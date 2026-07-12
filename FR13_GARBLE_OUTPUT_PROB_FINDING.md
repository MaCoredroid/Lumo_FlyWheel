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

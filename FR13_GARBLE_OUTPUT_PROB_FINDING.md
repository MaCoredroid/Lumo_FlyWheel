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

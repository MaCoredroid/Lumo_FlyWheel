# FR13 CLEANUP + BAKE PLAN (2026-07-27)

User directive: clean up first — bake the proven work, physically remove refuted/dead
attempts — THEN fix the double-temperature bug as its own milestone. User decisions
(explicit): (1) cleanup lands byte/band-gated BEFORE the temp fix; (2) delete
refuted+dead paths (git preserves history), diagnostics stay flag-gated default-OFF;
(3) tail6 + the proven stack = the canonical launch config, cat9 locked script retired
as historical. No delegation: all edits authored directly.

## #72 CLOSED: same-driver A/B verdict (arms landed 2026-07-27)
- armA s1ab_m3 (=3):  accept 4.3766 | eps 2.0934 | tps 33.399 | step 336.99ms | pf .394 | 2P/2F
- armB s1ab_m2 (=2):  accept 3.6973 | eps 1.6568 | tps 25.106 | step 309.98ms | pf .296 | 2P/2F
  (12907+13236 pass, 13033+13398 fail — SAME tasks pass/fail in both arms)
- Pooled-line residuals: =3 −1.5ms, =2 −7.0ms → both arms ON the line (235.5+49.2·eps).
  The =2 one-graph region adds nothing at the ±14ms cross-arm noise floor; behavior
  band identical. DECISION: do NOT bake =2 (or =3). FR13_STEP_GRAPH default stays 0
  (staged); the S1 capture machinery stays in-tree behind the flag as proven-working.
  Accept gap 4.38→3.70 is workload composition (13398 solo-tail 4543s in arm B;
  suffix-tail composition variance), NOT a mode effect — comb basis in band.

## BAKE (flag default flips → proven stack becomes the default boot)
In scripts/fr13_launch_forked_fa2_tree_server.sh (shape-independent, all live-proven
in the A/B stack; each previously gated individually):
- FR13_ENABLE_APC          0→1   (APC lossless proven; MISS==HIT; blocksize fix baked)
- FR13_TAW                 0→1
- FR13_PARENT_GATHER       0→1   (byte-identical selfcheck-gated)
- FR13_COMMITTER_GRAPH     0→1
- FR13_CONV_PREGATHER      0→1
- FR13_FLAGS_INKERNEL      0→1
- FR13_SUBTREE_PARALLEL    0→1   (#60: byte-exact + graph-safe + B=1 +4.7%)
- FR13_DRAFTER_GRAPH       0→1   (R4)
Already-baked (no change): FR13_COMMITTER_BATCHED=1, FR13_DRAFT_VOCAB_K=65536,
FR13_APC_BURN_NODE_BANK=0 (now deleted outright, below).
Shape-coupled flags stay bundled with the tail6 KIND in
fr13_bigdenom_swe_serve_variant.sh (FR13_TAIL_MODE=1, FR13_DRAFT_SOURCE=merged,
FR13_TREE_GDN_GEOM_OVERRIDE=BV=8, TAIL6_TREE 21-node) — KIND now defaults to tail6.
Rationale: a launcher-level TAIL_MODE default could mis-pair with a non-tail tree.

## DELETE (refuted/dead; each is default-OFF/dormant → served path unchanged)
1. HEAD-MERGE seam (FR13_MERGED_DRAFTER_SEAM, patcher ~12424-12472) + decide_and_fill
   (fr13_merged_drafter.py:225-332) + merge-only STATS/env knobs
   (FR13_MERGED_TREE_SPEC / FR13_MERGED_FLAVOR / FR13_MERGED_SKIP_MIN_PROB) + the
   merge-mode unit-test sections. Dormant-by-design in tail mode (gate:
   `not /logs/fr13_tail_mode.arm`); merge/Front-2 closed as no-go. ALSO rewrite the
   ENGAGED needle to lead with TAIL[...] counters and delete the "match_full>0 is the
   proof" docstring — the exact trap that caused the boot-54-era misread.
   KEEP: lifecycle (note_new_requests/ingest_from_sequence/retire_requests), prewarm,
   decide_tail + tail branch machinery, TAIL stats.
2. COMMITTER BURN path: drop do_burn param+branch in _fr13_conv_commit_to_col0
   (patcher 6885), env reads at 7307/7398/8092, callers 7328/7405/8114. Burn was
   proven redundant for the served path (baked off 2026-07-21). The legacy
   runrow=0 path (where burn was load-bearing) becomes INVALID → the tri-flag guard
   now fail-louds if COMMIT_TO_RUNNING_ROW/TREE_RUNROW_INIT != 1 (legacy path
   retired). Launcher env pass-through removed.
3. FR13_HC_INTERNAL (queued lever, never gated in; incompatible with baked
   PARENT_GATHER): mechanism killed — hc_internal_on() hard-returns False with a
   RETIRED note; launcher/canonical_env/required_tree_flags wiring removed; patcher
   preseed try-block removed (subtree preseed UN-NESTED and kept — it was inside the
   HC try). DEVIATION (deliberate): the HC_MASK constexpr branches inside the Triton
   kernel body (fr10_gdn_tree_kernel.py 1470-1690, trace-time dead at HC_MASK=0) are
   NOT excised in this pass — kernel-body edits carry the bit-exact bar + pinned
   kernel-lineage rule and deserve their own gate cycle. FOLLOW-UP recorded.

## RECLASSIFIED (scouted, NOT deleted — reasons)
- FR13_FORCE_SPINE_COMMIT: KEEP — it is a greedy-committer DIAGNOSTIC with a
  fail-loud guard against sampled-run misuse (patcher 7527), not the shelved #44 lever.
- FR13_STEP_GRAPH=1: KEEP-FLAGGED — =1/=3 share the TAW-walk capture machinery
  (=2 is the separate full-_sample wrapper); carving out =1 risks the proven scaffold
  for zero measured gain. All modes default-OFF via STEP_GRAPH=0.
- Multi-spine, drafter meta-reuse (#59a), SNAP_FIX/pb, es_ckpt: code ALREADY GONE
  (verified — only doc/comment references remain). Nothing to delete.
- Diagnostics (CAPDBG, SG_LIVEPAIR, PIN_UNIFORMS, span timers, RDAB, boundary logs):
  KEEP flag-gated default-OFF per user decision.

## RETIRE
- scripts/fr13_launch_locked.sh: HISTORICAL header (cat9-era gold gate, superseded by
  canonical tail6 via the variant harness); its FR13_APC_BURN_NODE_BANK=1 export
  removed (flag no longer exists).
- FR13_STEP_GRAPH_DESIGN.md: outcome note added (projections refuted by pooled
  regression + A/B; S1 fusion moved < detection floor).

## GATE (after edits, before temp fix)
Offline: patcher dry-run against the image's vllm tree (docker cp, no GPU) — all
anchors must match; py_compile every patched output + edited scripts; bash -n
launchers; merged-drafter unit tests (trimmed) pass.
Live: one 4-task boot (subset_b4_four, clean run, graph mode, temp 0.6): engagement
needles (TAIL[hit]>0, tok_per_draft=21, PG/subtree selfchecks, capture lines),
garble eyeball at codex_trace altitude, band vs 2P/2F reference, accept/eps/step on
the pooled line. Deletions are all default-OFF/dormant code, so the served path must
be behaviorally identical — any band/accept shift = STOP and bisect.

## FOLLOW-UPS (recorded, not in this pass)
- Kernel-body HC_MASK excision (own bit-exact gate cycle).
- Legacy runrow=0 branches inside the conv committer (now unreachable behind the
  fail-loud guard) — remove in a dedicated conv-committer pass (CONV_COMMITTED_PATH
  memory: touch that code only deliberately).
- Unused helpers in fr13_arctic_suffix_adapter / fr13_mtp_suffix_assembly /
  fr13_merged_fill orphaned by decide_and_fill's removal (arctic_match_confidence,
  assemble_cat33333, build_cat33333_columns, arctic_flat_tree_to_suffix_rel).
- Then: double-temp fix (single apply + tree_self asymmetry reconciled) + same-subset
  A/B re-base at true 0.6.

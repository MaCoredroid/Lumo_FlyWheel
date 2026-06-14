# FR13 — in_proj_ba pad BAKED into locked cat9 (B=1 HOLD): lossless + accept~native + SPEED-NEUTRAL; new fingerprint

Date 2026-06-14. Bake workflow `wf_1e06a327-873` (task wc9kiwfi7), VERDICT HOLD. Raw:
`research/fr13_workflows/bake_b1_wc9kiwfi7.raw.json`. The locked cat9 build now runs WITH the in_proj_ba
pad-to-fixed-M (LUMO_FB_KERNEL_ROWS=1 + LUMO_FB_PROJ_PAD_ROWS=16 in fr13_launch_locked.sh:34-35, commit
a666f9ec, pushed). user-approved bake.

## B=1 VerifyGate (baked locked cat9, all 6 gates PASS)
- workerEnvConfirmed True (LUMO_FB_KERNEL_ROWS=1 in the EngineCore worker; PROJ_PAD_ROWS dropped by curated
  mp/spawn env but defaults to 16 = intended = benign).
- engaged tok/draft=9, within_boot_det [T,T,T,T] (class-8 same-boot), regular-decode PRISTINE (spec-path-only).
- **flips = 21** [3,6,6,6] vs unbaked banked 22 / same-boot-OFF 26 (prior baked boot 18 [4,4,4,6]). DID NOT
  go UP = lossless-preserving. The 18↔21 is the GB10 cross-boot autotune fork floor (±3-4), NOT a regression.
- accept/event = **3.1513** ~ native (leaf edge intact, NOT collapsed to leaf-free spine 2.66).
- **SPEED-NEUTRAL**: s/fwd 0.2248 s vs flag-OFF clean 0.2249 s; decode_tps 17.3 (cat9 band). The pad runs
  16*row_len rows vs ~33 real, but the extra GEMM compute is HIDDEN behind the bf16 in_proj_ba weight read
  (GB10 B=1 bandwidth-bound) - confirmed.
- **NEW locked baseline fingerprint**: served_lens [104,116,128,128]; stream sha1[:8] =
  [d32193ec,4df82e33,7c068e7e,b39b0580]; flip vector [3,6,6,6]=21. The fingerprint CHANGED = a LOSSLESS
  change (feedback_no_cross_boot_byte_gate); do NOT gate on reproducing the old [6,6,4,6]=22 or [4,4,4,6]=18.

## Follow-ups (banked)
1. STALE BLOCKER comment in fr13_launch_forked_fa2_tree_server.sh (~L118-123) falsely claims the LUMO_FB pad
   block is NOT inserted / INERT - FALSE (HEAD patcher contains the full insertion, this boot proved it live).
   Delete it (done in the chase-down ApplyRebuild).
2. The curated mp/spawn worker env (PROJ_PAD_ROWS dropped) CONFIRMS [[project_fr13_active_worker_codex_fr15]]'s
   SUBOP_MAB root-cause: the SUBOP_MAB rebuild must use the SIDECAR (/logs flag) as the reliable env channel,
   not bare -e (FR13_SUBOP_MAB_REBUILD.md keeps both).

## NEXT (re-sequence, user 2026-06-14): chase-down BEFORE final B=1
Apply the 5 SUBOP_MAB rebuild EDITs -> run the rebuilt L0-GDN A/B (empirical conv/scan M10-vs-M5; predicts ~0
depth-intrinsic per FR13_RESIDUAL13_RESOLVED_DEPTH_INTRINSIC) -> final B=1 -> OPT-1/OPT-A speed -> B=4
(FR13_ENDGAME_ROADMAP). Pairs with [[reference_gdn_kernel_lineage_table]] (new locked baseline = lineage
update, reported), [[project_fr13_22flip_carrier_l0gdn]], [[feedback_flag_gate_metrics_reuse_infra]].

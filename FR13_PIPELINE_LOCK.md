# FR13 PIPELINE LOCK (canonical — read before touching the serving path)

Locked 2026-06-13 (user: "stick to the most current tested version, lock a pipeline").

## THE LOCKED BASELINE = main HEAD (gold-gate build + the 2 new gates)
The locked pipeline is **main HEAD**, which equals the **B=1 SWE-Verified GOLD-GATE build
`b7887c89`** (the build that ran 1-2 full uncapped 30-min SWE-Verified tasks and **FAILED on the
22-flip lossless gap**) plus exactly **two default-OFF diagnostic gates** added since:
- `0b5de164` — FR13_COMMIT_ARGMAX_GATE (the channel-1/2 splitter; default OFF)
- `813cb9fd` — FR13_HIDDEN_SUBSTITUTE (causal layer-splice for localization; default OFF)

**VERIFIED:** `git diff b7887c89..HEAD` on the serving code is only those two default-OFF
instruments — the **default-ON serving path is byte-identical to the gold-gate build** (the only
default-ON-affecting line is `_FR13_HSUB_DONE`, a guard *inside* the OFF-gated splice hook). So
running main with no env overrides reproduces the exact SWE-tested pipeline.

cat10 (root-sibling, no_help) was **uncommitted working-tree cruft** — never on main; archived
to remote `fr13-cat10-archive` and discarded from the working tree. main no longer references it.

## THE LOCKED CONFIG (cat9, gold-gate)
Tree = **cat9, num_spec 9, TREE_ATTN**, descriptor
`[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)]` (5-node spine +
top-2 leaf on depths 1-4; NO root leaf). qwen3.6-27b-fp8.

PIPELINE-ON (the gold-gate serving path; defaults in scripts/fr13_launch_forked_fa2_tree_server.sh):
FR13_DRAFTER_SINGLE_LOGITS=1 (FIX-1), FR13_EAGER_PACK=1 (FIX-2), FR13_TREE_CONV_FUSED=1 (FIX-3),
FR13_TREE_SAMPLE_ROW=1 (FIX-A), FR13_REPLAY_ROUTE=1 (replay route, patcher default), FR13_FA2_TREE_BIAS=1,
FR13_FA2_PREFILL_NATIVE=1, FR13_TREE_ATTN_EXP2_SOFTMAX=1, FR13_CONV_COMMITTED_PATH=1 + the non-det
fixes (FR13_TREE_REMAP_SEQ, FR13_TREE_PER_REQ_GEN, FR13_TREE_REQKEY). BATCH_INVARIANT=0.

DIAGNOSTIC-OFF (kept on main, armed only for the chase): FR13_COMMIT_ARGMAX_GATE,
FR13_HIDDEN_SUBSTITUTE, FR13_FORCE_SPINE_COMMIT, FR13_FIX1_SELFCHECK, FR13_CHASE_DIAG,
FR13_BI_TREE_ATTN, the op-capture flags, FR10_METRICS. None alter the default-ON path.

DEPENDENCY INVARIANTS (do not break): FR13_TREE_CONV_FUSED=1 **requires** FR13_REPLAY_ROUTE=1
(patcher :827); FR13_EAGER_PACK is replay-coupled (:5464). So replay + conv-fusion + eager-pack
move together; **replay route is ALWAYS ON** (FR13_KERNEL_STATUS.md). WY kernel (`_tree_gdn_wy_kernel`)
is NOT on HEAD — it lives only on remote `fr13-wy-archive` (parked).

NATIVE BASELINE = TRUE E5 = scripts/fr10_launch_speed_server.sh num_spec=5 FLASH_ATTN (accept
~3.16). NEVER naive_mtp (~1.36 — the cat10-gate's wrong-baseline mistake).

## LOCKED LAUNCHERS
- `scripts/fr13_launch_locked.sh` — boots the EXACT cat9 gold-gate pipeline with every flag
  pinned explicitly (no env-default ambiguity), diagnostics forced OFF unless `--arm <FLAG>`.
- Native E5 = `scripts/fr10_launch_speed_server.sh` with num_spec=5 FLASH (the TRUE baseline).

## REPO HYGIENE (2026-06-13)
- **Local = only `main`.** All other branches (lineage fr9/10/11/12, round-f, replay-route,
  accept-only-wip, wy-archive, autoresearch sprints, leftover worktree-*/agent branches, and the
  cat10 archive) are **preserved on the remote** (27 branches on origin) and removed from local.
  Nothing lost; recover any with `git checkout -b <b> origin/<b>`.
- The known gap on this locked baseline: it FAILED the SWE-Verified gold gate on the **22-flip =
  channel-2 verify-forward** defect (committer exonerated, 0b5de164 0/944). The drafter is a proven
  alt-free spine, so the cat10 −28 was verify-side too. **The chase runs FROM this locked baseline.**

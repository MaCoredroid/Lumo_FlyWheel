# FR13 spine-commit (architectural committer) — DE-RISK RESULT + verdict

## Idea (user's "fuse into forward like native")
Native avoids replay: its draft is a LINEAR chain, so the forward advances state token-by-token (fused),
keeps ~5 per-position states, and the committer SELECTS h_k (7ms). The tree discards all 21 node-states
(13.7GB to keep). But the tree's SPINE (children[0] chain, ~11 nodes) is linear+native-like. Plan: export
just the spine states from _tree_gdn_kernel h_cache; committer copies spine_state[depth] on a spine-prefix
accept (no replay); branches fall back to the 72ms replay.

## DE-RISK (byte-neutral counter in fr13_device_multidraft_commit, 2000 commits, temp 0.6, tail6 B=4)
  spine-prefix accepts = 790,  branch accepts = 695,  empty (no accept) = 515
  => P(spine | accepted) = 53.2%   (branch = 46.8%)

## Verdict: MARGINAL — the naive spine-commit does NOT clearly pass the cost-gate.
- Replay is skipped only 53% of the time (branches, which contribute the tree's accept edge, are accepted
  47% — a GOOD thing for accept, per [[feedback_chain5_reshape_not_a_solution]], so we can't reshape them away).
- Cost model: committer 77ms -> ~46ms on spine-accepts (skip 66.5ms replay), BUT the spine-state export
  (~11 nodes x 48 layers x ~10.5MB fp32 = ~5.5GB write ~= 20ms) is paid in the FORWARD every step.
  Net step saving ~= 0.53*66.5 - 20 ~= +15ms (or ~25ms at bf16 export, correctness-risk; or more if the
  write overlaps the forward — uncertain on GB10's shared bandwidth). ~5-10% step, for a correctness-critical
  kernel+committer rewrite. Below the "clear win" bar.
- The TRUE native-like fix (independent of P(spine)) is the stateless-tree/piggyback: re-process the accepted
  path in the NEXT forward (~free, HBM-bound) instead of replaying — but that is the BIGGEST rewrite (KV/SSM
  cache rollback+re-setup, native cache owns it; task #11), not this session's scope.

## DECISION: pivot to the QUEUED accept>5 (task #45) — the CHEAPER, cleaner path to tree>native.
Per the task-42 break-even (tree>native above accept ~5.76; prewarm at 5.15), raising accept directly hits
the goal, vs a marginal ~15ms committer trim. Spine-commit shelved (de-risk-refuted as a clear win);
stateless-tree remains the deep-committer option if a full rewrite is later sanctioned.

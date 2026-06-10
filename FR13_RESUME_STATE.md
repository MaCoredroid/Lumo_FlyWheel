# FR13 RESUME STATE — written 2026-06-10 during spend-limit pause (uncommitted until Bash recovers)

**STATUS 2026-06-10 (later): RESOLVED — campaign completed by resumed run; results in FR13_LADDER_LOG.md "S1/S2/S3 discriminators COMPLETE"; this file is historical.**

## Why paused
User hit the monthly spend limit mid-campaign; additionally the harness tmp filesystem hit 0 MB
(ENOSPC — Bash output capture broken; a Claude Code restart clears it, or set CLAUDE_CODE_TMPDIR).

## What is SAFE (committed + pushed)
- main @ `4d45be27` (S1 fix: bonus row = accepted leaf self-target; sampled committer audited CLEAN;
  spine_path_idx diagnostics fixed; 7 tests) — then `7cce7015` (replay remediation banked).
- Branch `fr13-replay-route` @ `50ac5f5a` — CPU-cleared (remediation re-verify holds=True), awaits GPU
  gates per `FR13_REPLAY_GATE_TRANSFER_MATRIX.md`.
- All raw workflow results in `research/fr13_workflows/` (committed).

## What was IN FLIGHT when the limit hit (workflow w7thg2cif / run wf_2bc205f4-4a3)
- Fix phase: COMPLETE (cached on resume).
- Discriminate phase: PARTIAL — artifacts on disk in `output/fr13_s1s2s3_discriminate/`:
  `s1_regate.json` (182 events, 21 [0,2]-winners, superset_violations=0);
  `s3_chain_vs_nativeBI0.json` identical-rate **0.533** vs `s3_caterpillar_vs_nativeBI0.json` **0.348**
  ⇒ m1 (alt co-residency) REAL but PARTIAL; boot2 (native BI=1 = the m2 discriminator) was mid-run,
  BI-equalized comparisons NOT produced. GPU containers torn down at pause.
- Reduce + verify phases: NOT run.

## RESUME PROCEDURE (first tick after capacity returns)
1. Bash sanity: `df -h /tmp` — if still ENOSPC, ask the user to restart Claude Code (frees held FDs;
   startup cleanup clears the tasks dir). Session state survives via git + memory + this file.
2. Verify the ladder-log partial-results commit landed (`git log --oneline -3` should show
   "S1/S2/S3 campaign interrupted" on top of 7cce7015) — if ABSENT, re-append that entry to
   FR13_LADDER_LOG.md (content: fix complete 4d45be27; discriminate partial with the numbers above;
   resume pointer) and commit this file + the ladder log (pathspec).
3. Resume the campaign:
   `Workflow({scriptPath: "<session>/workflows/scripts/fr13-s1-fix-s2s3-discriminate-wf_2bc205f4-4a3.js", resumeFromRunId: "wf_2bc205f4-4a3"})`
   — fix returns cached; discriminate re-runs (boots fresh, reuses run dir + analyzers).
   If agents fail with the spend-limit error again: HOLD, reply one line, wait for the next tick.
4. Once the campaign completes: bank raw + bind + verify-gated next steps as usual
   (queue after it: the replay-branch GPU gates per the matrix; Method-A leftovers are superseded).
5. Restore the monitor cron to 10-minute cadence with the standard FR-13 duties prompt.

## Standing context for the resumed loop
Workflows are the worker; codex stood down. Lossless bar = within native same-seed floor 0.113 +
superset accept/event ≥ same-shape native. NO reward-hack. Pathspec commits. One GPU, serialized.
Key docs: FR13_DIRECTION_AND_NUMBERS.md, FR13_ACCEPTANCE_LADDER_BIND.md (+riders),
FR13_REPLAY_GATE_TRANSFER_MATRIX.md, research/fr13_workflows/INDEX.md.

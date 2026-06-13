# FR13 combined localization campaign — 22-flip (verifier) + the "drafter" −28

Prep workflow `wf_6f7eb9b5-4d7` (CPU, 4 agents). Raw:
`research/fr13_workflows/flip_drafter_prep_wf_6f7eb9b5.raw.json`. Adversarial verify
**holds=TRUE, readyToExecute=TRUE**. HEAD 9aa28ce5. Replay route ALWAYS ON (FR13_KERNEL_STATUS.md).

## FINDING 1 — the 22-flip is CHANNEL 2 (verify-forward); committer EXONERATED
The channel-1/2 conflict is RESOLVED (trust 0b5de164 over the earlier FR13_GOLD_MARGIN_BIND):
- 0b5de164 ran FR13_COMMIT_ARGMAX_GATE in-process over 944 served records: **0 clear-margin
  channel-1 violations** (the 10 ch1_match=false are exact zero-margin topk ties, not row bugs).
  The 2 gold flips (p2 code/files, p3 Let/codefence) are ch1_match=TRUE (committer faithful) but
  the verify-forward argmax itself is wrong vs a clean teacher-force.
- The splitter is CLEAN: it uses the in-process post-constraint logits the committer's argmax
  consumed (`:7427`/`:5564`), NOT streamed logprobs (which are off-by-one). gold-margin was an
  earlier HYPOTHESIS on streamed logprobs that named this exact gate as the decisive test; the
  gate then falsified the committer suspicion.
- The `path0_native_bonus` row-map suspect is already dead code (FR13_TREE_BONUS_SELF default-ON).
- ⇒ the 22-flip carrier is the **verify-forward GDN argmax** (channel 2). fp8/conv-tap/
  conv-window already RULED OUT (FR13_DRIFT_LOCALIZE_BIND); the open seam = GDN scan
  num_warps=8/BV=16 vs native 4/BV=32, or TREE_ATTN deep-row vs FLASH_ATTN.

## FINDING 2 — the "drafter −28" is NOT drafter co-residency; it is VERIFY-side
CPU-proven on HEAD: the FR10_CATERPILLAR_NATIVE_SPINE_TOP2 drafter is a **pure causal MTP spine,
alt-free by construction**. The depth loop feeds back ONLY `input_ids=_fr10_spine_tokens[-1]`
(`:9882`); each step's leaf `topk(_fr10_step_logits,2)[:,1]` (`:9979-90`) NEVER enters the
forward — it is only woven into the PACKING ORDER (`:9994-10018`), which shapes the VERIFY tree,
not the drafter forward. The cat10 root sibling is a rank-2 READ of the root logits (`:9665-67`),
never fed forward. **The hypothesized "2-row depth batches" (S3) do NOT exist in this path** (they
belong to stock propose_tree, not engaged). ⇒ the −28 enters VERIFY-side: either the
branch-commit state-advance handoff (m3: the winner-path h0/conv-window degrades the next event's
rollout) or the 10-row verify forward co-residency (m1). FR13_FORCE_SPINE_COMMIT settles it.

## THE CAMPAIGN — corrected to REPLAY-ON (user: replay always on)
The prep designed it REPLAY-OFF (from the pre-correction cat10 bind); OVERRIDE to REPLAY-ON
(FR13_REPLAY_ROUTE=1, the shipped path): (a) user directive, (b) replay-off would trip the
`FR13_TREE_CONV_FUSED=1 requires FR13_REPLAY_ROUTE=1` raise (`:827`), (c) the node-7 ladder
localizes whether the replay-scan is the carrier in-place. The replay-ON-vs-OFF rider is DROPPED
(we keep replay on, not deciding to flip it). All boots eager (diagnostics fail loud under
capture), B=1 greedy seed 1313, 4 pinned probes (output/fr13_acceptance_ladder/prompts_swe4.json).

Minimal decisive set = 4 GPU boots, serialized (2-workflow cap, recover_host_memory between):
- **Boot 1 — non-MTP ORACLE** (no-spec, FR12_NO_SPECULATIVE_CONFIG=1, FLASH, eager): the CLEAN
  teacher-forced per-position argmax reference (both tracks). NOT a prefill.
- **Boot 2 — TRUE E5** (fr10_launch_speed_server.sh num_spec=5 FLASH): the accept baseline (~3.16,
  NOT naive_mtp 1.36) + the native-chain per-sub-op reference for the ladder.
- **Boot 3 — cat9 22-FLIP LOCALIZATION** (TREE_ATTN num_spec=9, REPLAY_ROUTE=1, eager,
  FR13_COMMIT_ARGMAX_GATE=1 + node-7 op-capture + gold-margin probe): re-confirm channel-2 (0 ch1
  clear-margin violations on HEAD) + the per-sub-op ladder (pre_conv→conv1d_out→scan→gate→o_proj,
  tree-spine vs native-chain/oracle) → FIRST NONZERO sub-op = the channel-2 carrier.
- **Boot 4 — cat9 −28 A/B** (FR13_FORCE_SPINE_COMMIT=1, full verify tree, commit pinned to spine):
  does pinning the commit recover the deep-spine accept? recover ⇒ branch-commit handoff (m3);
  persist ⇒ 10-row verify co-residency (m1).
- Verify (CPU): red-team both localizations.

Class 8 (within-boot rep1==rep2 byte-identical) + class 9 (engagement: gate armed, tok/draft==9,
TREE_ATTN, eager) are MANDATORY first gates on every live boot. Within-floor / per-depth-argmax
bars, NOT abs-0.0. Then FIX the localized carriers (verify-forward seam → align to native, e.g.
GDN scan BV=8; branch-commit handoff → the handoff wiring) and re-gate with the per-token argmax
probe (the gate the prior SCALAR superset missed — 30d749a4).

# FR13: bake-for-all-trees analysis — BAKE NOTHING (workflow wi6thjr3b, 2026-06-17)

User asked to bake any flags all tree arms need so they aren't passed each boot. Adversarial
bake-safety workflow (6 agents) + source-verified at HEAD. VERDICT: no patch edits.

The 4 load-bearing-default-OFF flags (from cat6_cat9_canonical_config.md), each checked for
BEHAVIOR-PRESERVING-across-all-arms incl native E5 (non-tree, must stay byte-identical):

- **FR11_TREE_CONV_NATIVE_BF16_TAPS (+FR12 alias)** — ALREADY default-ON: patch
  `_fr11_native_bf16_taps = env != "0"` default "1" (fr10_phase4_patch L2790). Quad-gated inside
  `if use_fr10_tree_conv:` (FR10_ENABLE_TREE_GDN==1 AND decode_mode==tree_mtp AND tree_parent not None
  AND num_spec_decodes>0); E5 is a chain (tree_parent None) -> stock causal_conv1d_update. No edit.
- **LUMO_FB_PROJ_PAD_ROWS** — ALREADY default "16" (L5705) = deployed value; a VALUE not a gate, read
  only after the KERNEL_ROWS gate. No edit.
- **LUMO_FB_KERNEL_ROWS — DO NOT BAKE (the trap).** Gate `if os.environ.get("LUMO_FB_KERNEL_ROWS") != "1":
  return None` (L5670) is NOT tree-gated, NOT backend-gated. The pad site is the Qwen3-Next GDN bf16
  in_proj_ba in forward_cuda's main else branch that EVERY GDN arm runs, incl native E5. `num_spec_decodes`
  counts co-resident SEQUENCES not tree nodes, so E5 (MAX_NUM_SEQS=4, B>1) trips it. Patch IS applied at
  E5 boot (fr10_launch_speed_server.sh). Baking default-ON re-pads E5's in_proj_ba at fixed M -> different
  cuBLASLt Split-K -> ~1-bf16-ULP-different (math-equal, NOT byte-identical) -> CHANGES NATIVE E5 BYTES =
  our lossless baseline. Leave env-gated; launchers pin =1 for trees only.
- **FR13_FA2_TREE_BIAS / FR13_FA2_PREFILL_NATIVE — DO NOT BAKE.** Live in fr13_patch_fa2_tree_bias.py (NOT
  the main patch). The `"0"` default literal IS the re-patch IDEMPOTENCY ANCHOR (L564 guard / L575-584
  upgrade match); flipping it desyncs already-patched detection -> patcher breakage. FR13_PIPELINE_LOCK.md
  L63-66 left them env-gated ON PURPOSE for this. Inert for E5 (its launcher never runs that patcher;
  FLASH_ATTN has no tree_attn_bias). Baking buys nothing, risks the patcher.

NET: 2 already baked, 3 have concrete blockers (1 = E5 byte-change, 2 = patcher anchor). Keep all four
passed explicitly via the launchers (current state). No integration boot needed (nothing changes).
Optional/out-of-scope: baking the FA2 flags would require lockstep-rewriting the idempotency anchor
strings — a dedicated re-patch-verification task, NOT a one-boot safe bake; against the PIPELINE_LOCK decision.

## ADDENDUM (user 2026-06-17): FR13_DEVICE_MULTIDRAFT BAKED default-ON
Separate from the 4 flags above (which the workflow rejected). The device multidraft committer IS the
deployed cat6/cat9 committer (b1 container_env FR13_DEVICE_MULTIDRAFT=1, set by fr13_b4_campaign_driver.sh:46).
The user asked to bake it so it's the default for all trees instead of a campaign-driver flag.
BAKE-SAFE (verified): (1) STRUCTURALLY tree-gated — the env is read INSIDE _lumo_tree_canonical_multidraft_sample
(the tree committer, dispatch gated on tree_self_logits); native E5 (chain -> stock rejection_sample, separate
fr10_launch_speed_server.sh launcher with no reference) NEVER reads it -> E5 byte-baseline untouched. (2) Single
read site (L8231); NOT a re-patch idempotency anchor (no `in text` guard). (3) Deployed cat6/cat9 set =1
explicitly -> byte-identical post-bake. (4) Device path is distribution-lossless vs host-ref (offline-proven,
fr13_device_multidraft_offline_gate.py). EDITS: patch L8231 default '0'->'1'; forked launcher L319 :-0 -> :-1
(the launcher always passes -e, so the patch default alone wouldn't take through it). CONSEQUENCE (flagged): the
DEFAULT committer is now the device path (distribution-lossless, NOT byte-identical to HEAD's host-ref);
FR13_DEVICE_MULTIDRAFT=0 restores the host-reference for A/B. Verify: forked-launcher boot engages device by
default (device_multidraft_commit in the profile) + the [6,6,4,6] drafter fingerprint (committer-agnostic) holds.

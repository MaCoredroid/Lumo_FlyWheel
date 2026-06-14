# FR13 — endgame roadmap (user 2026-06-14): residual -> bake -> B=1 -> verify 2 speed fixes -> B=4

User-set sequence after the localization grind wraps (user expects the L0-GDN A/B is the LAST residual piece):

1. RESIDUAL CHASE-DOWN (expected last): L0-GDN sub-op A/B re-run (task wl043ivfu, env-fixed) =
   conv1d_out/scan_out M10-vs-M5 discriminator. Code evidence predicts ~0 (depth-intrinsic) => the +13
   residual after in_proj_ba is the diffuse/depth floor + FA2 (not paddable). +2 spine already RESOLVED
   (cascade artifact, FR13_PLUS2_DECASCADE). 

2. BAKE in_proj_ba (CONFIRMED, user-approved): flip LUMO_FB_KERNEL_ROWS=1 + LUMO_FB_PROJ_PAD_ROWS=16 in
   fr13_launch_locked.sh (workflow scripts/fr13_bake_inproj_ba_workflow.js, queued behind the A/B re-run to
   avoid confounding it). Same-boot lossless gate + record new fingerprint. Fires when the A/B frees the GPU.

3. B=1: the bake VerifyGate IS the B=1 re-measure (baked cat9 per-token argmax flip count ~18 vs unbaked
   ~22-26, accept/event ~native, det [T,T,T,T], regular-decode pristine, new fingerprint).

4. VERIFY THE 2 PREVIOUS SPEED FIXES (lossless AND fast): 
   - OPT-1 = FR13_GPU_COMMITTER (GPU-resident tree committer, kills main-thread DtoH+sync, restores run-ahead
     ~4-6ms; commit 10ebccac, default-OFF, never GPU-verified).
   - OPT-A = FR13_GB10_FP8_GEMV_CFG (GB10/sm_121 fp8 GEMV config, lossless-by-construction BLOCK_SIZE_K=128
     pinned; commit e90de7ef, default-OFF, never GPU-verified).
   GPU-verify each: lossless (same-boot det + per-token argmax unchanged + regular-decode pristine) AND fast
   (B=1 decode TPS / s-per-forward vs native; the goal is sub-native B=1, FR13_BEAT_NATIVE_SPEED_DESIGN).

5. B=4 (REMEMBER - the final gate): the deployable lossless+superset gate = B=4 + CUDA-graph-captured +
   SWE-Verified 4 tasks (NOT eager/B1/toy). Re-confirm lossless (within E5 self-noise floor) + superset
   accept/event >= native at B=4 (co-residency changes at B=4). vs E5 (FLASH native MTP-5) per
   [[feedback_fr12_subkernel_zero_gate]] measurement-regime.

Pairs with [[project_fr13_speed_first_lossless_gate]], [[feedback_dont_handroll_speed_defer_tuning]],
[[feedback_flag_gate_metrics_reuse_infra]], FR13_WIDTH_CARRIER_INPROJ_BA_BIND.md.

## RE-SEQUENCE (user 2026-06-14): empirical chase-down AFTER bake, BEFORE final B=1 (gate now ready)
Now that the SUBOP_MAB gate is rebuilt (FR13_SUBOP_MAB_REBUILD.md, c8061eda - 5 EDITs), the order is:
1. Bake in_proj_ba (running wc9kiwfi7) -> verify bake holds (lossless + speed + new fingerprint).
2b. APPLY the 5 rebuild EDITs to the patcher (post-bake, when it frees the patcher): plain -e env (NOT the
    inert ray allowlist - the worker is mp/spawn, inherits pid-1 environ) + 4 ERROR-level FR13_SUBOP_STAGE
    markers + hoist asserts out of the swallowing try.
2c. EMPIRICAL CHASE-DOWN with the working gate: rebuilt L0-GDN A/B = conv1d_out/scan_out M10-vs-M5 deep-spine.
    Worker-env gate + stage markers confirm engagement first. Predicts ~0 (depth-intrinsic) but get the REAL
    number (or find a paddable op). This is the empirical close of the residual the 4 infra failures blocked.
3. FINAL B=1: baked cat9 flip count + accept + fingerprint + speed (re-measure if 2c found a fix).
Then OPT-1/OPT-A speed verify -> B=4.

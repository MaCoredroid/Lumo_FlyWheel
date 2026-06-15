# FR13 — Scan-align Phase-2 verify: INCONCLUSIVE (both binding instruments broken)

Date 2026-06-15. Workflow w77rygxwf (verify holds=false). The committed scan-alignment (5e56b7aa) is NOT
verified as the lossless win — but NOT refuted either: the two binding instruments were BROKEN/VACUOUS
(playbook #9), so we measured nothing decisive. The FIX itself is intact + engaged.

## SOLID PASS (the one clean result)
- **default-OFF binary byte-identical = TRUE**: direct GPU A/B, OFF-arm scan == captured serving_out
  byte-exact (max_abs 0.0, int_eq True). The committed kernel `fr10_gdn_tree_kernel.py` has ZERO git diff
  from 5e56b7aa. The DCE/locked-path claim holds on GPU — the locked path does not move when FR13_SCAN_ALIGN
  is unset.
- **The recompute alignment GENUINELY ENGAGES** (not silent-OFF): worker env-curation dropped bare
  FR13_SCAN_ALIGN (14/72 FR13 vars reach the spawned EngineCore = playbook #9 again) → defeated via a
  dist-packages sitecustomize re-setting it at interpreter startup (verified EngineCore sees recompute).
  arm=us_tree_RECOMPUTE: US stream diverges from OFF at toks 11/25/34/61 → 334 token diffs / 4 prompts;
  det [T,T,T,T] on BOTH boots; accept/event 730/222=3.29 (healthy, not collapsed). Seam delta vs OFF ~1.22e-4.
  So recompute produces a DIFFERENT deterministic stream — it is NOT byte-lossless vs the deployed path.

## INSTRUMENT 1 BROKEN — int-view gate VACUOUS (corrects a monitor error)
`scripts/fr13_native_packed_decode_ref.py::native_packed_decode_per_path` returns **ALL-ZEROS output**
(native_out norm=0.0): the fused_recurrent_gated_delta_rule_packed_decode kernel updates STATE (norm 25.9)
in-place but writes ZEROS to `out` — because the ref passes **q=0** (q is irrelevant to the durable state),
and the kernel's output is `b_o = sum(b_h * b_q)` → 0. So every arm "mismatches" by `max_abs == |serving_out|`
(identical 0.0317–0.0552 across ALL 6 arms per case), int_view trivially False, and the +0.5 neg-control is
VACUOUSLY "powered" (negControlPowered=false in the verdict). The gate as written CANNOT show int-view 0.0.
- **CORRECTION of the monitor's prior-tick claim:** "carrier confirmed real — deployed scan diverges from
  native-packed 0.0317" was WRONG. The 0.0317 is `|serving_out|` vs an all-zeros ref, NOT a real scan-vs-native
  gap. **scan-vs-native-packed is UNMEASURED.** (My earlier "constexpr-caching" hypothesis was also wrong —
  the real cause is the zeros reference.) `feedback_check_artifact_before_concluding`, bug-class #9.
- VALID side-channel (vs the validated serial-torch ref `native_update_serial_per_path`): OFF/body/recompute
  ALL byte-match serial-torch; OFF==serving_out; recompute & body each differ from OFF by only ~1.22e-4. So
  the seams ENGAGE but move the scan output ~1e-4, NOT to any measured aligned target.
- **FIX:** compare the STATE (h, norm 25.9) not the output (o), OR feed real q so `out` is meaningful. The
  scan STATE-FEED is the h, not the o — the gate compared the wrong tensor.

## INSTRUMENT 2 BROKEN — e2e flips UNMEASURED (the binding number never computed)
The same-boot no-spec oracle is IMPOSSIBLE on this spec-configured build: per-request **non_mtp AND
naive_mtp BOTH CRASH** the EngineCore (`EagleProposer has no attribute 'positions'` in propose_tree). So
`fr13_gold_margin_probe` US-vs-no-spec-oracle could not run → flips_before/flips_after never computed. Needs
a **SEPARATE non-spec server boot** for the oracle (a 2nd boot, which the only-boot rule precluded here).
NOTE: this is also how the ORIGINAL 21 was measured — re-establish that mechanism for the re-run.

## VERDICT + reward-hack
holds=false; the scan-state-feed is **NEITHER confirmed NOR refuted as the e2e carrier** — re-open, NOT
cleared. Reward-hack CLEAN: native packed-decode used as A/B oracle only (no served-path splice); committed
kernel zero git diff; the only uncommitted change is the diagnostic gate n_pad=4 IndexError fix.

## NEXT (continue the grind, not a dead-end)
1. FIX `fr13_native_packed_decode_ref.py`: compare the durable STATE h (or feed real q + capture the
   downstream o/output-scale) so the int-view gate is a REAL comparison + the neg-control is genuinely powered.
2. Establish a working no-spec oracle (separate non-spec server boot — the mechanism the original 21 used),
   run gold_margin_probe US-vs-oracle for BOTH OFF and recompute → the binding flips_before/flips_after.
3. THEN the discriminator: recompute int-view 0.0 vs native-packed-STATE + flips 21→~3 + lossless gate =
   the win; else quantify. No bake/close until both instruments report (user's call).

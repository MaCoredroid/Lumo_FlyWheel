# FR13 sharp localize — the bake-in is FINE; the cross-boot check was the wrong instrument

Workflow `wf_1ea62c3f-4d7` (2 boots planned, Boot2 correctly skipped). Raw:
`research/fr13_workflows/sharp_localize_wf_1ea62c3f.raw.json`. Adversarial verify **holds=FALSE**
(it REFUTED Boot1's causal claim). HEAD 5a203dd3.

## DO NOT REVERT THE BAKE-IN — the STOP was a confounded cross-boot artifact
Boot1 (cat9/BV=16 via fr13_launch_locked.sh + COMMIT_ARGMAX_GATE + path-LCP) found the served
stream diverges from the banked cat9 capture (p0@17, p1@11, p2@60, p3@77) and recommended
reverting 219d41de/a09ef5b5/45dc05a2. **That recommendation is WRONG.** Three independent proofs:
1. **Logical equivalence:** the bake-in diff is env-read→constant where every constant == the
   locked default; the locked-config behavior is provably the same function (verify holds=FALSE
   *because* Boot1's "numerics changed" claim is unproven).
2. **No Triton codegen change from the bake:** every baked site is Python control flow (e.g.
   `fr10_gdn_tree_kernel.py:211 if True: # REMAP_SEQ` is a launch wrapper, NOT inside `@triton.jit`).
   So the bake edits zero kernel source — it cannot change numerics.
3. **The cross-boot byte-gate is INVALID on GB10** (the instrument error, mine): per the GB10
   byte-gate facts, fresh boots at B=1 fork from ANY reference at tokens 11–71 (boot-level
   autotune/kernel-selection, outside batch-invariance). Boot1's fork positions (17/11/60/77)
   ARE that floor. Comparing a fresh boot's free-running stream to a banked stream is not a valid
   bake-in test. The VALID byte-exact instrument is the **in-process same-boot gate** — and it
   reconfirmed **channel-2 / committer-exonerated 0/944 clear-margin on the baked build d85e42b2**
   (the 10 ch1 mismatches are exact fp ties). The bake-in is fine.

**Lock-doc correction:** the FR13_PIPELINE_LOCK "[6,6,4,6] cross-boot reproduce" integrity check
is REPLACED by the in-process gate (same-boot, byte-exact) + structural logical-equivalence; a
fresh-boot stream NEVER reproduces a banked stream byte-for-byte on GB10.

## BV=8 is NOT the carrier — drop it (my misread, verify-corrected)
FR13_BV_SPILL_VERDICT.md: the SEQ scan kernel is ALREADY bit-exact at BV=16 (e4a6a2f2); BV=8 is a
**spill/speed** change, bit-exact-equivalent — it CANNOT be the 22-flip numerics fix.
FR13_DRIFT_LOCALIZE_BIND §3 rates the num_warps/BV codegen seam "~1-bf16-ULP, no depth-growth,
unlikely to BE the carrier alone." So "the one un-closed seam = BV" was wrong framing. Keep
BV=16/num_warps=8 frozen. The real carrier = the channel-2 verify-forward (diffuse L0–L58 / the
current-HEAD first-diverging sub-op, UNKNOWN) — localize via the per-sub-op node-7 ladder, NOT BV.

## Gate-blindspot — QUANTIFIED (definitive)
The prior gate missed the 22-flip because of **scalar-blindness + coherent-output-masking, NOT
test-too-shallow**: 22 flips / 482 = 4.6% rate; accept/event 3.198 > native (every scalar passes);
9/22 at clean-confidence >95% (true losses, deviation median 3.31 nat); the flips form a COHERENT
valid command (`find|grep` vs `ls`) so pass-rate/eyeball/no-crash all miss it. Flips span pos
17–118 at adequate depth ⇒ depth was sufficient. The ONLY catching instrument = the per-token
argmax-vs-clean teacher-forced probe.

## CORRECTED localization plan (the cross-boot fork forces a fresh oracle)
The "reuse the banked oracle" optimization was INVALID — cross-boot fork means the current build's
stream ≠ banked, so the banked clean-argmax is off-stream. The localization must run on the
CURRENT build's OWN stream, same-boot/floor-bracketed:
1. Tree boot (current build) → its served stream + the path-LCP trace (node_type per position).
2. A FRESH no-spec oracle teacher-forced on THAT stream (the gold-margin probe's reference) → the
   channel-2 flips on the current stream.
3. Tag each flip's committed node → deep-SPINE vs deep-BRANCH (the still-open question).
4. The per-sub-op node-7 ladder (FR13_DRIFT_LOCALIZE "DECISIVE NEXT": pre_conv→conv1d_out→scan_out→
   gate_out→o_proj_out, first-nonzero = carrier) on the deepest flipping node.
NO BV=8. The oracle is needed (I was wrong to kill it for "redundancy" — it must be on the current
boot's stream, not the banked one).

## What Boot1 DID validly establish (in-process, same-boot)
- channel-2 reconfirmed on the baked build (committer exonerated 0/944 clear-margin).
- engagement PASS (TREE_ATTN, num_spec=9, eager, gate armed, traces non-empty), within-boot det 4/4.
- node census on this boot's stream: spine 730 / branch-leaf 116 / corrections 98; branches DO
  win+commit at depths 1–4 (node2/4/6/8 = 16/16/22/4) ⇒ the cat9 tree engages its leaves (not
  vacuous linear). The deep-spine-vs-branch classification of the 22 flips remains OPEN.

# FR13 — baseline RESOLVED: native 3 vs cat9 22 (clean full-stream), cat9 genuinely-worse-but-diffuse, lever = tree-reshape

Date 2026-06-14. Closes the "is native ~3 measured or assumed?" question opened by
`FR13_NODE5_LADDER_DIFFUSE_BIND.md`. Census `wkbjl7xln` (`wf_cfa1c1f7-062`) + its adversarial verify
(**holds=FALSE on the census's overstatements, verdict SOUND**). Raw:
`research/fr13_workflows/flip_census_baseline_wkbjl7xln.raw.json`.

## THE CLEAN BASELINE (measured, like-for-like, NOT assumed)
Full-stream per-token argmax-vs-clean teacher-force probe (`scripts/fr13_oracle_stream_teacher_force.py`,
the BINDING instrument per [[reference_scalar_metric_per_token_blindspot]]), every served position,
max_tokens=1, threshold 1.0 nat, SAME 4 pinned prompts (`prompts_swe4.json`, verified byte-equal across arms):

| arm | n_positions | per-prompt clear-margin flips | total | source |
|---|---|---|---|---|
| **native E5** (num_spec5 FLASH) | [128,128,128,128] | [0,1,1,1] | **3** | `output/fr13_verify_decisive/q3_native_classify.json` |
| **cat9** (locked build) | [78,128,128,128] | [5,7,4,6] | **22** | `output/fr13_verify_decisive/q3_tree_classify.json` |

→ **cat9 is genuinely ~7× worse than native, same method. cat9 does NOT clear native's self-floor.**
The standing "native ~3 flips" bar is REAL and MEASURED. The 3 native flips are at high-entropy
template/quote/code-fence boundaries (p1 pos94 `Let`→```` ``` ````, p2 pos33, p3 pos68) — native's own
irreducible floor, not zero.

## CORRECTION: the "native = 95" alarm (my prior turn) was a CONFOUNDED measurement
`output/fr13_cat10/native_e5/flips_vs_oracle.json` reports 95 native flips — **do NOT cite it.** 9/95 of
its flips have >10-nat deviations (rec0: pos12 served `/` vs clean ` to`, dev **17.56**); a
distributionally-lossless native spec-decode arm physically cannot have 17-nat clear-margin flips vs its
own target. These are the **streamed-logprob off-by-one / misaligned-oracle** artifact the directive warns
about. The clean q3 run (corrected no-spec teacher-force, in-process) = 3, not 95. Lesson reinforced:
flip-count instruments are only valid when the oracle alignment is the corrected per-position
teacher-force; the older naive_mtp `flips_vs_oracle` path is off-by-one.

## Carrier = DIFFUSE (no sharp residual) — confirmed, with two census caveats noted
The per-layer ladder HOLDS adversarially (`output/fr13_node5_ladder/per_layer_maxabs.json`): input
byte-exact (0.0), divergence born inside L0 GDN (3.9e-3), smooth monotonic to L63, no isolated
~0→argmax spike → no fixable single op. **Census overstatements the verify caught (do NOT carry):**
(1) the per-record **num_accepted distribution was FABRICATED** — all 11 `channel2_flip_records` have
`num_accepted=None` and `node_index=None`; the "10/11 at num_accepted≥3" table is back-filled, not data.
(2) `verify_top2_margin` (tree-verify's own internal top-2) was conflated with the binding
clear-margin-vs-clean. The diffuse + deep-spine picture still stands (from the ladder + q3 + the
decisive_flip_targets depth-4 spine nodes), but not from the census's null fields.

## DISPOSITION: tree-reshape (user-approved 2026-06-14) — test cat-tree shapes through the B=1 SWE gold gate
The diffuse drift grows with **committed-spine depth** (deeper rank-1 tree-scan chain = larger
chunk-vs-recurrent state-feed divergence, born at L0, ratcheting L41-L63). The lever
([[project_fr13_tree_reshape_unifying_lever]]) = **shallower committed depth + root-sibling width**
(recover accept/event via the d0-rescue). NOT a per-op patch (the grind is exhausted: GDN sub-ops already
bit-exact; the residual is the rank-1-tree-vs-no-spec-roll algorithm realization, not a roundable seam).

**Mechanism for the reshape:** `TREE` is a runtime env var (`fr13_launch_locked.sh:15`,
list-of-tuples) — shape variants need NO code change. cat9 = 5-spine
`[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]` + 4 leaves `[(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)]`.
`FR13_CAT10_ROOT_SIBLING` flag is NOT built — variants constructed via `TREE`.

**GPU campaign (serialized, MAX 2 wf):** sweep shape variants (shorter spine + root-sibling/width) through
the gold gate = full-stream flip count (`fr13_oracle_stream_teacher_force.py`) AND accept/event, vs the
two fixed bars: **native 3 flips / accept-event 3.16** (the target) and **cat9 22 flips / ~3.18**. WIN =
a shape with flips→~3 (within native floor) AND accept/event ≥ native 3.16. Tension to resolve
empirically: deep accepts give cat9 its accept edge AND cause the flips — does shallow-wide recover the
accept via width without the deep-accumulation? Do NOT self-close; bring the flips-vs-accept frontier to
the user. Reward-hacks banned. Pairs with [[reference_diffuse_gdn_accumulation_explained]],
[[feedback_no_cross_boot_byte_gate]] (same-boot in-process gate), [[project_fr13_pipeline_lock]].

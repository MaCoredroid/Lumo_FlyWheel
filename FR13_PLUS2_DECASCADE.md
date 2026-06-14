# FR13 — the +2 spine drift, DE-CASCADED: chain5 +2 IS a class-12 cascade artifact (2 independent ≤ native 3); chain3 is NOT (5 independent > native 3) ⇒ a SMALL REAL diffuse residual exists, but the deep-spine "+2" the recurrent re-score banked is the inflated raw count

Date 2026-06-14. CPU-only de-cascade of the banked flip records. READ-ONLY; no GPU. Inputs:
`output/fr13_shape_sweep/chain5_flips.json`, `chain3_flips.json`,
`output/fr13_verify_decisive/q3_native_classify.json`. Threshold = clear-margin (deviation_nat ≥ 1.0),
oracle = no-spec recurrent/chunked argmax, binding metric per [[reference_scalar_metric_per_token_blindspot]]
(per-token argmax vs no-spec oracle), deployable arbiter = accept/event.

## The de-cascade rule (applied IDENTICALLY to all 3 arms)
A clear-margin flip at pos i is an INDEPENDENT divergence event iff the served prefix up to i−1 matches the
oracle stream (oracle dev=0.000 in the immediately preceding positions, i.e. no upstream flip in a short
window). A flip is a CASCADE consequence iff an upstream flip at i−k (small k) already diverged the served
prefix that the oracle is teacher-forced on — then the oracle's argmax at i is conditioned on the
now-diverged served prefix and reads as a "flip" though only the ONE upstream decision diverged. The
signature of a cascade: a CONTIGUOUS run of flips (gap=1) that RE-CONVERGES to dev=0.000 immediately after
the format/JSON sub-tree closes. Soft (non-clear) neighbors of a clear flip are cascade tails, not events.

This is exactly bug class **#12 Measurement traps** ("per-pos counters … whole-window … single upstream fork
inflates the count") and class **#4 Spine-only-valid column arithmetic on branch winners** ("EPISODIC
whole-forward corruption AFTER branch commits; transient, recovers") in FR13_BUG_CLASS_PLAYBOOK.md — the
binding instrument is the per-token argmax probe (class #12 blind-spot), and the scalar count is
necessary-never-sufficient.

## chain5 (deep 5-spine `[(0,)…(0,0,0,0,0)]`) — RAW 5 → 2 INDEPENDENT (cascade-inflated, BELOW native)
shape_sweep per_prompt = **[0, 0, 5, 0]** — all 5 clear flips are in prompt 2, in ONE window. Full audit:

| pos | clear | served | oracle_argmax | dev | gap | classification |
|---|---|---|---|---|---|---|
| 25 | CLEAR | `tool` | `bash` | 2.750 | — | **INDEPENDENT** — tool-call FORMAT fork; near-tie high-entropy boundary: served's own top-2 = `tool` −0.21 / `bash` −2.21; oracle's = `bash` −0.10 / `tool` −2.85. The two realizations cross at a genuine ambiguity. |
| 26 | CLEAR | `_call` | `\n` | 2.750 | 1 | cascade (oracle conditioned on served `…```tool`) |
| 27 | CLEAR | `\n` | `_name` | 4.250 | 1 | cascade (served `…```tool_call`) |
| 28 | CLEAR | `<` | `` ``` `` | 4.000 | 1 | cascade (served `…```tool_call\n`) |
| 29 | soft | `function` | `invoke` | 0.750 | 1 | cascade tail (`<function` vs `<invoke`, sub-clear) |
| 34 | soft | `_b` | `_shell` | 0.250 | 5 | cascade tail inside the tool-call arg region (sub-clear) |
| 43 | CLEAR | `"{` | `{"` | 2.938 | 9 | **INDEPENDENT** — JSON brace/quote-order fork; isolated, re-converges at pos 44 (dev 0.000) |

Proof of cascade: the served stream re-converges to **dev=0.000 at pos 30** (`= tool = execute …` byte-identical
to oracle) immediately after the `<function=…>` tool-call header closes. If 26–28 were independent per-forward
defects the stream would not snap back to 0.000. pos 25–29 = ONE independent event (the format selection);
pos 43 = a second. **chain5: 5 raw clear → 2 INDEPENDENT events. 2 < native 3 = AT-OR-BELOW native.**

## chain3 (shallow 3-tree `[(0,),(0,0),(0,0,0)]`) — RAW 5 → 5 INDEPENDENT (NOT cascade-inflated, ABOVE native)
shape_sweep per_prompt = **[0, 1, 3, 1]** — the 5 clear flips are DISPERSED across prompts 1/2/3. Full audit:

| prompt | pos | served | oracle | dev | gap-from-prev-clear | classification |
|---|---|---|---|---|---|---|
| 1 | 24 | `\n` | `bash` | 5.250 | — | INDEPENDENT — fence-language fork; re-converges pos 25 (`find`=`find`, dev 0) |
| 2 | 25 | `json` | `tool` | 4.437 | — | INDEPENDENT — `` ```json `` vs `` ```tool `` format fork; re-converges pos 26 |
| 2 | 31 | `tool` | `name` | 1.625 | 6 | INDEPENDENT — distinct JSON-key boundary; re-converges pos 32 |
| 2 | 45 | `paths` | `name` | 2.625 | 14 | INDEPENDENT — distinct JSON-key boundary; re-converges pos 46 |
| 3 | 71 | `from` | `#` | 1.625 | — | INDEPENDENT — Python comment-vs-import fork; re-converges pos 73 |

Soft tails (NOT events): pos 48 (`"/` vs `\n`, dev 0.375, gap 3 from 45) and pos 72 (` pathlib` vs ` IP`,
dev 0.375, gap 1 from 71) are sub-clear cascade tails of their upstream clear flips. Every CLEAR flip here is
isolated (min gap between clears = 6), at a DISTINCT boundary, each re-converging immediately ⇒ NO contiguous
cascade. **chain3: 5 raw clear → 5 INDEPENDENT events. 5 > native 3 = +2 REAL.**

## native E5 (FLASH_ATTN MTP-5) — RAW 3 → 3 INDEPENDENT (apply the SAME rule)
per_prompt_clear = **[0, 1, 1, 1]**, one per prompt at distinct, isolated boundaries:

| prompt | pos | served | clean_argmax | dev | context | classification |
|---|---|---|---|---|---|---|
| 1 | 94 | `Let` | `` ``` `` | 1.875 | `head -5\n```\n\n` | INDEPENDENT — prose-vs-codefence continuation fork |
| 2 | 33 | ` "` | ` '` | 6.375 | `Name: find\nArguments:` | INDEPENDENT — quote-style fork |
| 3 | 68 | `Let` | `` ``` `` | 8.375 | `-20\n```\n\n` | INDEPENDENT — prose-vs-codefence fork |

No contiguity, no cascade. The native classify file records clear flips only — so any native cascade TAILS are
sub-clear and excluded, exactly symmetric to how chain5's pos 29/34 and chain3's pos 48/72 soft tails were
excluded. The comparison is like-for-like. **native: 3 raw → 3 INDEPENDENT.**

## COMPARISON (de-cascaded, same rule on all arms)
| arm | tree | raw clear | INDEPENDENT events | per-prompt independent | vs native |
|---|---|---|---|---|---|
| native E5 | MTP-5 spine | 3 | **3** | p1:1 p2:1 p3:1 | — |
| chain5 | deep 5-spine | 5 | **2** | p2:2 | **−1 (below)** |
| chain3 | shallow 3-tree | 5 | **5** | p1:1 p2:3 p3:1 | **+2 (above)** |

## VERDICT (the two clues reconciled honestly — NOT a convenient "resolved")
The two binds were BOTH right and are NOT in tension once de-cascaded:
- The recurrent re-score (FR13_ORACLE_FRAME_CLOSED_BIND) reported the RAW count 5/5 byte-identical across
  chunked AND recurrent oracles. That is true: the cascade is a deterministic teacher-forcing artifact, so it
  reproduces identically under any honest oracle frame. "5/5 across oracles" proves the count is NOT a
  chunk-vs-recurrent FRAME artifact; it does NOT prove the 5 are 5 INDEPENDENT defects. RAW count = honest;
  INDEPENDENT count = the per-forward defect measure. These are different quantities.
- The +2-align research (FR13_PLUS2_BV_FIX_REFUTED_BIND) localized chain5's 5 raw to ~2 independent (pos25
  fork + pos43 brace). CONFIRMED here, decisively, by the re-convergence-to-0.000 at pos 30/44.

**For the DEEP SPINE the user asked about (chain5), the "+2" is a CASCADE/MEASURING artifact: 2 independent ≤
native 3 = AT-OR-BELOW native. RESOLVED for chain5.** The +2 banked by the re-score is the cascade-inflated raw
count.

**BUT a small REAL residual exists, surfaced by chain3:** with a shallower 3-tree the 5 clear flips are 5
genuine, dispersed, isolated independent crossings = +2 over native, and de-cascading does NOT shrink them
(no contiguity to collapse). So the per-forward divergence IS real at the ~1-event-per-prompt-extra level — it
is just diffuse and below the cascade noise on the deep spine. This is consistent with
[[reference_diffuse_gdn_accumulation_explained]] (per-layer ~1-bf16-ULP realization diffs compounding over
~48 GDN layers until argmax flips at a high-entropy boundary; native same-model+fp8 drifts less = existence
proof). It is NOT a single fixable seam (see below). It is alignment-territory small.

So the honest answer is BOTH: the deep-spine +2 is a cascade artifact (resolved); the underlying per-forward
realization gap is real but small (~+2 independent on a shallow tree, diffuse), and the binding arbiter remains
e2e accept/event, not the raw flip count.

## IF REAL — where? (the candidate unfound seam, given everything on the spine is bit-exact or inert)
Everything verified on the branchless spine: scan RAW 0.0 to native at BOTH BV geometries (D16=D32=N_PAD1=16,
FR13_BV_GEOMETRY_NOT_THE_SEAM_BIND — the BV/warps re-proposal is REFUTED, silicon shows 0.0); conv done
(ex2-silu bf16-tap grind succeeded); in_proj bit-exact; both fp8 GEMMs M-invariant; FA2 tree-bias
ARITHMETICALLY INERT on a branchless spine (`_prepare_tree_attn_bias(chain5) == lower-triangular causal`).
So a REAL residual is NOT a wiring/geometry seam — it is one of:
1. **The FA2-fork 2-ULP MMA-grouping floor on the spine** (FR13_FA2_FORK floor: 14/16 calls whole-tree 0.0, 2
   single-ULP in ~1M, irreducible MMA grouping, no theorem). On a branchless spine the bias is inert but the
   MMA reduction grouping still differs from native FLASH_ATTN at the 2-ULP floor → can flip a near-tie
   high-entropy boundary (exactly the `tool`/`bash` −0.21-vs-−2.21 class of crossing). MOST LIKELY carrier.
2. **The tree-GDN-vs-native-MTP recurrence realization** (per-layer ~1-bf16-ULP, the diffuse class
   [[reference_diffuse_gdn_accumulation_explained]]). The q1 deep-row finding (L0 GDN 0.000854 vs recurrent
   oracle ≈ 1 ULP at floor) is this class at L0; it SURVIVES the recurrent reframe (FR13_ORACLE_FRAME bind)
   and does NOT translate 1:1 to e2e flips (9.14× L0 reduction → same e2e crossings) ⇒ diffuse accumulation,
   not a single localizable op.
The two are not separable at this magnitude and neither is a paddable carrier; this is the "alignment
territory" residual, below the +17 leaf co-residency which is the real target (bf16 in_proj_ba, M-keyed).

## Disposition
- chain5 deep-spine "+2" front: CLOSED as a class-12 cascade-inflation artifact. 2 independent ≤ native 3.
- The small REAL diffuse residual (chain3 +2 independent): PARK. It is FA2-2-ULP-floor / diffuse-GDN, not a
  fixable seam; revisit ONLY via e2e accept/event after the +17 leaf (in_proj_ba) fix lands. If post-+17
  cat9 accept/event ≥ native, the residual is moot (deployable arbiter, not the raw flip count).
- Reward-hacks BANNED; no recurrent-oracle adoption (FR13_ORACLE_FRAME: ours-only shave = reward-hack
  signature). The de-cascade applied the SAME rule to native (3→3); it did NOT cherry-pick to reach "resolved".
Pairs with [[reference_scalar_metric_per_token_blindspot]], [[reference_diffuse_gdn_accumulation_explained]],
[[project_fr13_fa2_fork_nocopy_floor]], FR13_PLUS2_BV_FIX_REFUTED_BIND, FR13_ORACLE_FRAME_CLOSED_BIND.

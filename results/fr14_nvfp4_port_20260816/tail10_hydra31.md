# TAIL10 (hydra31_fixed32) — design, cascade, and prepared package

**PREPARED, NOT LANDED.** The round-5 quiet-tree window is open; nothing here has touched a
tracked file. Everything below is in the scratchpad and lands on the signal.

---

## 1. What tail10 is

hydra27 arms 27 of 31 physical drafts. The four it disarms are the deep rank-2 side-branches
`(2,0,0,0)`…`(2,0,0,0,0,0,0)` at depths 4–7, masked off by `0x7ABDFFFF`. tail10 **respends
them as spine continuations `0^12..0^15`** and arms them, so the Arctic tail runs **6 → 10**
and all 31 drafts are live.

| | hydra27 | hydra31 (tail10) |
|---|---|---|
| physical drafts / verify rows | 31 / 32 | **31 / 32 — unchanged** |
| armed nodes | 27 | **31** |
| valid mask | `0x7ABDFFFF` | **`0x7FFFFFFF`** |
| spine max depth | 11 | **15** |
| Arctic main chain | 6 | **10** |
| Arctic requested tokens | 12 | **16** |
| rescue path columns | 10 | **6** |
| rescue carry slots | 4 | **0** |
| walk cap (`max_depth+1`) | 12 | **16** |

## 2. THE CASCADE — enumerated before code

### 2.1 The finding that decides the shape of the change

**This is not a mask change. The physical tree itself differs.** Sorting choices by
`(len, path)` puts `0^12..0^15` at the end, so **draft ids ≥ 17 acquire different paths** —
14 of 31 ids move:

```
id 17: (2,0,0,0)        -> 0^5           id 24: (2,0,0,0,0,0)   -> 0^9
id 18: 0^5              -> (0,0,0,0,1)   id 25: 0^7             -> 0^10
id 22: (2,0,0,0,0)      -> 0^7           id 30: 0^11            -> 0^15   ...
```

Therefore `DRAFT_PARENT`, `PHYSICAL_PARENT` and **both digests** change:

| digest | hydra27 | hydra31 |
|---|---|---|
| `PHYSICAL_PARENT_SHA256` | `7abd25e3…a06ddd` | `101c590e…3e698e2` |
| `TREE_ANCESTRY_SHA256` | `90873d81…946dad` | `5b33c46a…187b427` |

Editing hydra27 in place would silently change what **every banked hydra27 credential
attests**. A new named profile is not a preference here, it is required.

### 2.2 The cascade, counted

Mechanically (`cascade.py`): **205 pinned sites** touch a constant tail10 moves.

| constant | today → tail10 | sites |
|---|---|---|
| mode name `hydra27_fixed32` | *(unchanged — new profile added beside)* | **77** |
| walk depth 12 (`loop_iterations`/`walk_levels`/`critical_path`) | 16 | 36 |
| main tail length 6 | 10 | 35 |
| arctic requested 12 | 16 | 24 |
| carry slots 4 | 0 | 14 |
| `MAX_PHYSICAL_DEPTH` 11 | 15 | 7 |
| active nodes 27 | 31 | 6 |
| valid mask | `0x7FFFFFFF` | 4 |
| parent / ancestry digests | new | 2 |

**The 77 mode-name sites are exactly why the new-profile recommendation is right**: adding
`hydra31_fixed32` beside hydra27 leaves all 77 alone. Editing in place would touch every one.

### 2.3 What does NOT change — and why this stays host-side

`PHYSICAL_DRAFTS` 31, `PHYSICAL_ROWS` 32, `SAMPLER_MAX_FANOUT` 3, root fan-out 3. The kernels
see the **same 32-row geometry**; no kernel recompiles. What changes is the ancestry *table*
(data, built at boot) and the walk *depth* (a loop bound). Stated precisely because "host-side"
is doing real work in this argument: the tree-bias and GDN path-program tables are rebuilt from
the profile at boot, not baked into a shipped binary.

`merge_fill_columns` is **coincidentally unchanged at 16** (10 tail + 6 rescue = 6 tail + 10
rescue). Worth naming so nobody reads its stability as evidence that nothing moved.

## 3. Two findings the cascade turned up

### 3.1 The committer is not flat — and tail10 survives it anyway

`WALK_CAP = MAX_PHYSICAL_DEPTH + 1` goes **12 → 16**, and `cfwd` is level-proportional
(`taw.loop_iterations = walk_levels = 12` invariant; ~192 full-vocab ops over 12 levels). The
banked §8 table priced tail10 "at unchanged 207.9 ms", i.e. assuming a flat committer.

| committer assumption | step ms | TPS | vs control |
|---|---|---|---|
| flat (banked §8) | 207.87 | 27.35 | +7.7% |
| **+33% (fully level-proportional)** | **214.75** | **26.48** | **+4.3%** |
| +17% (half-proportional) | 211.31 | 26.91 | +6.0% |

**Break-even: cfwd may grow to 36.7 ms (+78%) before tail10 stops being a gain.** So unlike
the seam move, tail10 is robust to its weakest cost assumption. Accept reproduces the banked
figure exactly (4.686 vs 4.686; my control ladder gives 4.2779 against the measured 4.2774).

Caveat carried forward unchanged: positions 11–14 rest on the **simulated** ladder
(`r7..r10` = .930/.929/.937/.948). Nothing past position 10 has ever been measured — which is
precisely what tail10's own serve produces.

### 3.2 The sequenced follow-on breaks a device-side cap

`n=3 + tail14` reaches depth **17** → `WALK_CAP` **18**, against `COMMIT_PATH_CAP = 16`.

tail10 sits at **exactly** the cap (16 = 16, zero headroom). tail14 exceeds it, and 16 is not
a host constant — it is a **tensor width**: `accepted_paths.shape[1]`, `slot_paths.shape[1]`,
`spec_paths.shape[1]`, `commit_bank_alias_groups`, `commit_row_guard_{compare,path}_capacity`.

**So n=3+tail14 is not a host-side change like tail10; it needs a committer path-capacity
widening.** Flagging now because it changes the cost of the sequencing decision, not after
tail10's A/B.

## 4. Losslessness, and the gate interplay

**Padding.** The four respent slots are spine continuations fed from Arctic, and a cold cache
pads them by repeating the previous spine token — `fr13_mtp_suffix_assembly`'s last-resort pad,
whose committer tie convention is proven on device by `fr13_greedy_pointmass_dup_gate`. A
repeat can never match a distinct model token, so the committer stays monotone and tail10
**cannot regress accept**: it only adds candidates. This is the same never-regress argument the
banked §8 makes for tail extension, and it is why tail10 (unlike a seam move) trades no
measured accept for time.

**Gate interplay — clean.** A gated step still hands off two head depths earlier, so under
tail10 its Arctic chain is 12 rather than 10. The freed depth-4/5 runner-up columns are still
duplicate-sibling padded; that mechanism is untouched. The pack identity holds in both shapes:

```
ungated: 15 + (10 - 0) + 6 = 31        gated: 15 + (12 - 2) + 6 = 31
```

`LEGAL_HANDOFF_SHAPES` becomes `((4,10),(2,12))`.

## 5. Staged landing

The 205 sites do not have to move at once, and they should not.

| stage | content | risk |
|---|---|---|
| **1 — prepared here** | profile table + hydra31 definition + `validate_tail10_contract()` + tests. **No serving path touched.** | none: hydra27 byte-identical, asserted |
| 2 | make the four consumers profile-aware: `decide_fixed32`, census `MODE_SEMANTICS` + TAW literals, patcher `_FR13_FIXED32_MODES`, launcher mode validation | moderate |
| 3 | arm `FR13_FIXED32_MODE=hydra31_fixed32`, run the A/B | serve |

**Stage 1 is provably inert:** with the prepared topology and every other file tracked, the
paired-contract sweep reports **13 pairs, 0 stale**, and all three contracts validate
(`validate_contract`, `validate_gate_contract`, `validate_tail10_contract`).

`validate_tail10_contract()` asserts, among others: hydra27's digest and mask have **not**
moved; the four respent paths are exactly `0^12..0^15`; 31 drafts, all armed; parent-before-
child; root fan-out 3; fan-out ≤ `SAMPLER_MAX_FANOUT`; `WALK_CAP ≤ COMMIT_PATH_CAP`; the pack
is 31 columns in both gate shapes; nothing is carried; the two profiles have distinct digests.

## 6. The A/B plan

**Arms.** `hydra31_fixed32` (tail10) vs `hydra27_fixed32` (current control). Everything else
pinned identical: radixark `K=0`, B=1, `FR14_SUFFIX_PASS_GATE=0` on **both** arms so tail10 is
measured alone.

**Pairing.** Paired on the identical canonical exact4 task set, **≥20 000 decode steps/arm**.
Same-arm accept has varied ±10% run-to-run, and the predicted effect (+4.3%) sits inside that
band unpaired.

**Headline instrument — the per-position ladder past position 10.** This is the measurement
the whole tail programme is missing: *nothing past draft position 10 has ever been measured*.
The control arm's counters end at 10; tail10's must be non-zero through **14**. Those four
survivals retire the largest simulated input in the banked model and are the direct input to
the sequenced `n=3+tail14` economics.

**Secondary instruments.** `step_wall_ms` and `s_per_fwd_gpu`; `FR13_DFWD_SPLIT=1` on both
arms. **cfwd is the one to watch** — §3.1's 12→16 walk predicts +33%, and that prediction is
falsifiable in this serve.

**Pre-registered readings.**

| quantity | predicted | falsified by |
|---|---|---|
| accept/step | 4.28 → **4.69** (+0.41) | < +0.20 kills the simulated deep ladder |
| per-position counters | **non-zero through position 14** | zeros past 10 mean the tail never armed |
| `cfwd` | **20.6 → 24–27.5 ms** | > 36.7 ms erases the gain (§3.1 break-even) |
| `step_wall_ms` | +3 to +7 ms | |
| net | **+4.3%** (range +2 to +7.7%) | |
| `active_nodes` / `verify_rows` | **31 / 32 every step** | any other value means the profile did not take |

## 7. Prepared package

| artifact | state |
|---|---|
| `diffs/01-topology-hydra31.patch` (219 lines) | ready, validates, lint-clean |
| `tail10_design.md` (this note) | ready |
| `cascade.json` — the 205-site inventory | ready |
| `econ.py`, `derive.py` — reproducible derivations | ready |
| stage-2 consumer diffs | **not written** — deliberately, see §5 |

## 8. Honest bound

Stage 1 is inert and proven so. Stages 2–3 are where the 128 non-mode-name sites live, and I
have not written them: the cascade shows they cluster in four consumers, but "clusters in four
consumers" is a claim about a diff that does not exist yet. The A/B cannot run until stage 2
lands, and stage 2 should be reviewed against the same lints before it does.


---

# 9. STAGE 2 — the four consumers (2026-08-19)

Landed. The gate is refused-final and default-OFF; tail10 composes with it off, and the gate's
constants are still read through the same profile so nothing diverged.

## 9.1 What each consumer became

| consumer | change |
|---|---|
| `decide_fixed32` | takes **every** width from `profile(fixed32_mode())` — main tail, gated tail, requested tokens, rescue chains, carry slots. It imports no flat tail constant any more. |
| census | `MODE_SEMANTICS` += hydra31; `shape_profile(mode)` supplies every mode-varying expectation; tail6 maps to hydra27 (same physical tree, different mask). |
| patcher | `_FR13_FIXED32_MODES` and the live `_fr13_fixed32_topology_needle` mask map both learn `hydra31_fixed32 -> (0x7FFFFFFF, 31)`. |
| launchers | `FR13_FIXED32_MODE` is now whitelisted, and hydra31 **refuses** the twelve levers qualified on hydra27's tree. |

The mode reaches the worker through the `/logs/fr13_fixed32_mode.flag` sidecar the launcher
already writes — the worker-env-drop-proof pattern — defaulting to hydra27 when absent or
unrecognised, so nothing can silently promote a serve onto tail10.

## 9.2 Derive-don't-hardcode, applied to the two hard cases

**The GDN subtree schedule.** hydra27 ships `SUBTREE_LEVELS` as a hand-written literal;
hand-writing a second one would be the same hardcode twice. The *rule* is written down instead
— level 0 is the spine prefix of `N_MTP_HEAD_DEPTHS` rows, level 1 is the maximal descending
chain from every child of a level-0 node — and `validate_tail10_contract()` **proves it
reproduces the shipped hydra27 table exactly** before it is trusted for hydra31. hydra31
derives to `(1, 11)` paths, `(5, 11)` max lengths, sum 16 = its walk cap, all 32 rows covered.

**The TAW call table.** `TAW_TENSOR_CALL_CENSUS` is entirely walk-proportional
(`2×walk` gathers, `3×walk` normalizations, …). It is now derived, and
`taw_tensor_call_census(12)` reproduces the shipped literal exactly. This mattered: the banked
serve runs the **base** route (`fixed32_pytorch_exact_float_triton_integer_commit`,
`walk_levels = 12`), so under hydra31 every one of those counts moves to walk 16.

Also derived rather than restated: `merge_fill_columns` (main + rescue — 16 under **both**
profiles, for opposite reasons: 6+10 and 10+6), `rescue_carry_slots` (4 → 0), `child_lanes`,
`uniform_slots`, `row_scatter_slots`, `path_scatter_slots`, `loop_iterations`.

## 9.3 Verification

* **1 000 banked hydra27 events validate unchanged** (isolating lane 3's in-flight
  `taw.source_contract_sha256` re-attestation, which is not this lane's).
* A full **hydra31 event validates end-to-end** through the census, and hydra27 shapes are
  **refused** under the hydra31 mode.
* The launcher preconditions were **CPU-walked, not asserted**: hydra27 passes, hydra31 passes
  and announces itself, an unknown mode is refused, hydra31 + a hydra27-qualified lever is
  refused, and hydra27 + that same lever still works.
* **Symbol-resolution sweep** on the patcher: both edited sites resolve, and the needle's mask
  map is still defined before use.
* Paired-contract + shape-literal lints: **16 pairs, 0 stale**, with three new pairs —
  profile table ↔ patcher mode table, ↔ blob topology needle, ↔ census modes.

## 9.4 Two line-number fragilities this round exposed

My own patcher edits shifted the injected blobs, which turned two position-keyed things stale
for no semantic reason: the replay adjudication key `(39286, fn, ordinal)` and a lint test
asserting `lineno == 39286`.

Both now key on **content**: the adjudication is `(function, ordinal)` — function names are
unique across all injected blobs, asserted — and the test finds the flush blob by looking for
`_fr13_f32_flush_reconcile` in it. Round 4 warned that an allowlist which drifts on every edit
trains people to refresh it without reading; this is that warning arriving.

## 9.5 Ready for the A/B

`FR13_FIXED32_MODE=hydra31_fixed32` vs `hydra27_fixed32`, gate OFF on both, paired canonical
exact4, ≥20 000 steps/arm. Headline instrument: the **per-position ladder past position 10** —
counters must be non-zero through 14. Watch `cfwd`: §3.1 predicts +33% from the 12→16 walk,
and that is now the census's expectation too, so a serve that disagrees fails validation rather
than quietly reporting a wrong number.


---

# 10. STAGE 2b — the serve VEHICLE (2026-08-19)

Round 13 refused **pre-boot at zero GPU**: stage 2 taught seven files and not
`fr13_bigdenom_swe_serve_variant.sh`, which is what actually launches an arm. Env could not
route around it either — the kind block bakes `FR13_FIXED32_MODE` into `XFLAGS`, which the
vehicle exports *after* the caller's environment.

Fifth instance of one shape: **consumers taught, selector not.** My stage-1 cascade enumerated
205 sites across seven files and never enumerated the thing that starts the serve.

## 10.1 What the vehicle already had right

It does **not** hand-copy the topology. A `mapfile` block runs the topology module and emits
the contract, under the comment *"Fixed-32 has one topology authority. Do not duplicate the 31
paths or masks in shell."* So the clean shape existed; hydra31 just had to join it. The block
now emits nine fields instead of five, calls all three validators before emitting, and the
shell asserts the runner's handover (`0x7fffffff` / 31 / 31) against the derivation rather than
trusting it.

## 10.2 The catch that mattered: hydra31 needs its own TREE

The obvious kind block would copy hydra27's and swap the mask. That would be wrong, and it
would **boot**.

`TREEARG` carries the 31 paths. hydra31 is a *different physical tree* — sorting by
`(len, path)` moves **14 of 31 draft ids** (id 30 goes from `0^11` to `0^15`). Passing
`$FIXED32_TREE` under mask `0x7fffffff` would arm four rank-2 side branches as if they were
spine continuations. So the authority emits hydra31's own path list and the kind block passes
`$FIXED32_HYDRA31_TREE`. A test asserts the two kinds dispatch different trees.

## 10.3 Audit of the vehicle's other per-kind exports

Of everything the cascade named as profile-varying — main tail length, arctic requested tokens,
carry slots, rescue columns, walk cap, loop iterations, critical path, max depth, both digests,
GDN geometry — **none appears in the vehicle at all**. They are derived downstream from the
mode and the tree. The only profile-varying exports are the four now handled
(`FR13_FIXED32_MODE`, `VALID_MASK`, `ACTIVE_NODES`, `PHYSICAL_DRAFTS`) plus `TREEARG`.
`EXPECT_RATIO=31` is the *physical* draft count and is correct for every profile.

Three generic fixed32 gates (OFFLOAD_AGENT, private arm dirs, committer layer-batch) now admit
hydra31. The BV64/4-warp gate stays hydra27-only — it is a hydra27-qualified lever, and
refusing hydra31 there is the same discipline the launcher guard applies.

`FR13_FIXED32_PHYSICAL_DRAFTS=31` was the last hand-copied literal in the file; all three kinds
now derive it. There are zero left.

## 10.4 The parity detector

`tests/test_fr14_vehicle_profile_parity.py`, 18 cases, pure source + local execution, no GPU.
For **every** profile in `fr13_fixed32_topology.PROFILES` it asserts the vehicle has a dispatch
kind whose exported mode / mask / active-nodes / drafts / tree match that profile — and it
**executes** the vehicle's own authority and `case` to check, rather than reading them.

It fails on the *next* profile anyone adds. Four mutations prove it can fail: an unknown
profile, a re-literalised mask, a kind reusing the wrong tree, and a generic gate that forgot a
profile.

## 10.5 Status

346 tests, 16 pairs / 0 stale. Round 14 — hydra31 vs hydra27, both arms same HEAD, gate OFF
both, paired exact4, ladder-past-position-10 as the headline instrument — can launch.


---

# 11. STAGE 2c — the in-container preflight (2026-08-19)

Round 14: H27 banked clean, H31 refused at a **second** mode table — the launcher's own
in-container preflight, distinct from the outer whitelist stage-2b fixed.

## 11.1 Three profile-varying things in one twelve-line block, not one

The runner named the table and, by **reading rather than booting**, the tree comparison three
lines below it. Reading the rest of the block found a third:

| line | compares | hydra27 | hydra31 |
|---|---|---|---|
| mode table | mask / active nodes | `0x7abdffff` / 27 | `0x7fffffff` / 31 |
| `tree != topology.FIXED32_CHOICES` | the dispatched tree | ancestry `90873d81` | **`5b33c46a`** |
| `walk_cap != topology.WALK_CAP` | committer walk depth | 12 | **16** |

Teaching only the table moves the refusal three lines down; teaching table + tree moves it nine
lines further. All three are now keyed on the mode through `topology.profile(...)`.

**The block is mirrored in THREE launchers**, including `fr14_leg3_launch_nomiddleware.sh`,
which has none of my earlier FR14 work — my "both launcher families" assumption was wrong by
one. All three are fixed.

## 11.2 The walk cap had to be *supplied*, not just checked

H27's banked `container_env.txt` carries `FR13_FIXED32_TAW_WALK_CAP=12`, minted upstream by
the runner env, not by the vehicle. Making the preflight expect 16 for hydra31 without
supplying it would just relocate the refusal again. So every fixed32 kind block now exports its
profile's walk cap, derived from the authority alongside mask and nodes — and `XFLAGS` are
exported *after* the caller's environment, so the arm's value wins over the runner's 12.

Verified by **executing the real preflight**: all three modes pass; hydra31 carrying hydra27's
tree is refused (`TREE differs from the hydra31_fixed32 choices`); hydra31 with walk cap 12 is
refused (`shape mismatch ... walk_cap=12`); an unknown mode is refused.

## 11.3 The detector, in its closing form

`scripts/fr14_mode_table_parity.py` answers the question wherever it is asked, from
`PROFILES`:

1. **every dict keyed by fixed32 mode** in any embedded python block across all four shell
   sites → must have a row for every profile;
2. **every comparison against a profile-varying topology constant** → must be keyed on the
   mode, never made unconditionally (this is the half a table-only fix misses);
3. **every bash whitelist naming two or more modes** → must name every profile. Single-mode
   lever preconditions are left alone: hydra27-qualified levers legitimately refuse hydra31.

Five mutations prove it fires — the missing table row, the unconditional tree compare, the
unconditional walk-cap compare, and a dropped profile in either a launcher or the vehicle
whitelist — with both unmodified sources clean. Folded into the paired-contract sweep as a
17th pair.

This is the third place the same question gets asked, and the detector now covers all three:
consumers (contract pairs), vehicle dispatch (`test_fr14_vehicle_profile_parity`), preflight
(`test_fr14_mode_table_parity`).

## 11.4 Status

362 tests, 16 pairs / 0 stale. H31 is launchable.

One gap reported rather than silently widened: `fr14_leg3_launch_nomiddleware.sh` now has the
profile-aware preflight but still lacks the suffix-gate guards and the **promoted** fused
draft top-k default. If that twin is a live serving path, it would serve without the promoted
kernel. Flagged for a decision rather than expanded into this change.


---

# 12. STAGE 2d — the third launcher family (2026-08-19)

`fr14_leg3_launch_nomiddleware.sh` is a live serving path (arm B's profile-chain legs) and
carried **none** of the FR14 work. It now carries all of it, byte-identical to its siblings.

## 12.1 What was missing, and why it mattered most for the promoted lever

| | before | after |
|---|---|---|
| fused draft top-k (**PROMOTED**, default ON) | absent | default `1`, pinned `.so` + sha, host-side refusal |
| suffix-gate guards (refused-final, but must be guarded) | absent | strict `0\|1`, sidecar, incompatibility refusals, split-graph interlock |
| hydra31 lever refusals | absent | present |

The fused top-k is the sharp one. It is **promoted**, so two families served the promoted
kernel and the third would have served the unfused path — silently, with no flag anywhere
saying so. A promoted default present in two of three families is not a promotion; it is an
unlabelled A/B.

The 142-line FR14 region was ported **verbatim** from a complete twin at the identical anchor,
so the three files are now the same text rather than three independent transcriptions.

## 12.2 CPU-walked in the third twin, not assumed

Eight cases run against the ported blocks in `fr14_leg3` itself: plain launch arms the promoted
kernel; `=0` opts out; a missing `.so` and a sha mismatch each refuse; a **stale env trying to
arm the refused gate is refused** (it lacks the tail seam); a gate typo is fatal; the hydra31
arm announces itself; and hydra31 + a hydra27-qualified lever refuses.

## 12.3 The roster, in one place

This is the sixth round in which "both launcher families" was wrong by one — and this time the
one-short commit was **mine** (`8fe896720`, the promotion). So the enumeration now exists
exactly once, as `fr14_mode_table_parity.LAUNCHER_FAMILIES`, and every consumer imports it:
the paired-contract sweep, the gate wiring tests, the split-K arm tests. A test asserts no file
re-enumerates the families by hand.

`scan_family_parity()` checks eight FR14 markers — the promoted default and its credential,
the gate guards and interlock, both incompatibility guards, the tail10 profile, lane 4's arm —
and fails when any is not identical across all three. Mutation-proven on three of those markers
by stripping each from the third family and asserting the detector names both the marker and
the file.

## 12.4 Status

414 tests, 16 pairs / 0 stale, family parity clean. Round 15 can launch.

Still absent from `fr14_leg3` and reported rather than folded in: `FR13_DFWD_SPLIT` /
`FR13_LFWD_GPU_TIMER` forwarding, which the other two families have. That is an
**instrument** gap, not a serving-state gap — an arm B leg would serve correctly but without
the drafter split timer. Named here so the next A/B on that path knows before it needs the
number.

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

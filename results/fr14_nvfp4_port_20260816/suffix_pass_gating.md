# Suffix-aware MTP pass gating (`FR14_SUFFIX_PASS_GATE`)

FR14 greenlit lever 2. Branch `codex/fr14-nvfp4-port-20260816`.

**Status of this file at first commit: PRE-REGISTRATION ONLY.** §§1-4 were written and
committed before the calibration simulator was run and before any GPU was taken. Everything
below §5 is appended afterwards and is dated.

---

## 1. The lever

The fixed32 drafter runs **5 MTP passes per step** (1 initial + 4 inside one CUDA-graph
replay; `drafter.mtp_forward_calls = 4`, `drafter_runtime.graph_replays = 1`). Kernel
attribution on the banked nsys sqlite prices passes 2-5 at **10.3 ms each**
(`seam_move_economics.md` §5), so passes 4 and 5 cost **20.6 ms of a 207.87 ms step**.

The banked seam study (`results/fr14_nvfp4_port_20260816/seam_move_economics/`) measured what
those two passes buy. Its finding, restated exactly:

- moving the MTP/suffix seam from draft position 5 to position 3 costs **one slot** — the
  handoff at position 3, where MTP's measured survival is `s3 = 0.8083` and the suffix
  proposer's unconditional cold-start hit rate is `q1 = 0.473`;
- **every slot deeper than the handoff is at least as good under suffix as under MTP**
  (position 4: MTP 0.8169 vs suffix chain slot 2 at 0.803 = 98% coverable; position 5:
  0.5972 -> 0.858, an improvement);
- verdict **MARGINAL**: +6.5% central, but the pessimistic case is +1.7%, because `q1 = 0.473`
  is a bad cold proposer.

**The gate's claim is that 0.473 is the wrong number to pay.** 0.473 is the *unconditional*
cold-start rate — averaged over contexts the suffix cache has never seen and contexts it has
seen fifty times. The banked data already shows the conditional structure is enormous:
`q1 | prev 1 hit = 0.782`, `| 2 = 0.848`, `| 5 = 0.920`. If the seam moves **only on steps
where the cache demonstrably has a strong match**, the handoff is paid at the conditional
rate, not the marginal one. On every other step the drafter runs unchanged, all 5 passes, and
the arm is byte-identical to today.

So: **run 3 MTP passes when the suffix match is strong, 5 otherwise.**

### 1.1 Correction to the brief, recorded because it is load-bearing

The greenlight framing says "deep-position draft mass is ~98% suffix-coverable (position-4
finding), so passes 4-5 mostly re-derive what the suffix proposer would fill anyway." That
overstates it in one specific way: the 98% at position 4 is the coverability of a suffix
**chain slot** — it is conditional on the position-3 handoff having hit. It is not evidence
that position 4 is coverable on its own. The honest statement is the one above: the entire
accept exposure of this lever is concentrated in the **single** handoff slot at position 3,
and the gate exists to make that one slot's conditional hit rate beat 0.8083. This note is
built on that statement, not on the 98%.

## 2. Shape of a gated step (well-formedness)

Topology is `hydra27_fixed32` (`scripts/fr13_fixed32_topology.py`): 31 physical drafts, 32
verify rows, 27 active nodes = 15 MTP head nodes (depths 1-5, spine + 2 runner-ups each) + 6
Arctic main-tail spine nodes (depths 6-11) + 6 Arctic branch-chain nodes (rank-1 length 4,
rank-2 length 2); 4 physically present nodes are already masked inactive.

Skipping MTP passes 4 and 5 leaves the **6 head nodes at depths 4 and 5** unfed. They are
disposed of as follows:

| node | path | disposition on a gated step |
|---|---|---|
| depth-4 spine | `(0,0,0,0)` | **Arctic-filled** — becomes suffix chain slot 1 (the new handoff) |
| depth-5 spine | `(0,0,0,0,0)` | **Arctic-filled** — suffix chain slot 2 |
| depth-4 runner-ups | `(0,0,0,1)`, `(0,0,0,2)` | **padded-invalid** |
| depth-5 runner-ups | `(0,0,0,0,1)`, `(0,0,0,0,2)` | **padded-invalid** |

The two spine nodes MUST be filled, not padded: the entire depth-6..11 tail hangs off the
depth-5 spine node, and padding it would orphan six live suffix nodes. This is done by
extending the Arctic main tail request from 6 tokens to 8 and re-anchoring it at depth 4, so
the gated tree reaches the same maximum draft position 10 as the ungated tree. Verify rows
stay 32, the committer's walk depth stays 12, and `PHYSICAL_PARENT` is untouched — a gated
step is a **mask change plus a fill-source change**, never a shape change.

Gated active nodes = **23** (= 27 - 4), which is exactly the `TAIL6_ACTIVE_DRAFTS` count the
committer already runs at, on a mask that is a strict subset of `HYDRA27_VALID`. Losslessness
of the 4 padded nodes is the monotone-committer argument already banked for tail padding
(`seam_move_economics.md` §8: "padded non-matching tail nodes are lossless by the monotone
committer").

**Hard invariant, enforced fail-closed:** no step may reach the verifier with the depth-4 or
depth-5 spine slot unfilled. If the Arctic lookup returns fewer than 8 main-tail tokens on a
step the gate selected, the gate **reverts to 5 passes for that step** (it is decided before
the drafter runs; see §3) or, if the shortfall is discovered after the fact, the affected
nodes are masked invalid together with every descendant. Conservative bias everywhere: when
uncertain, run 5 passes.

## 3. THE PREDICATE (pre-registered, written before measurement)

### 3.1 When it is evaluated

**Before the drafter runs**, from the committed context only. Not after pass 3.

The alternative — decide after pass 3, using the actual MTP-proposed prefix — is strictly
more informative, and it is rejected on cost grounds: it requires a blocking D2H readback of
the spine tokens in the middle of the drafter, and the campaign has already measured a
blocking 4-byte D2H on this path at **2.91 ms/step** (`host_dfwd_characterization.md` §6,
item F). Spending 2.91 ms to save 20.6 ms on half the steps is a 28% tax on the lever, and it
opens a GPU bubble on ungated steps. Deciding up front costs nothing: the host already holds
the committed token stream, and the Arctic cache is already updated host-side each step.

### 3.2 The rule

Let `Sigma[:j]` be the committed prefix at the step whose first draft position is `j`. Let the
Arctic suffix cache be queried at `j` for its longest match:

- `L(j)`   = length of the longest suffix of `Sigma[:j]` that occurs earlier in `Sigma[:j]`,
             searched over the ladder `L in {24,16,12,8,6,4,3,2}` (the banked study's ladder);
- `n(j)`   = number of earlier occurrences of that match considered (capped at `VOTE_CAP=64`);
- `a(j)`   = agreement = (votes for the winning continuation) / `n(j)`.

**The gate fires iff `L(j) >= L*` and `a(j) >= a*`.**

`L*` and `a*` are selected by the rule in §3.3 from the sweep
`L* in {2,3,4,6,8,12,16,24}` x `a* in {0.0, 0.5, 0.75}`. Ties are broken toward the **larger**
`L*` (more conservative).

### 3.3 Selection rule and acceptance bar

The gate is accept-safe on a gated step iff the handoff it creates is no worse than the MTP
slot it replaces:

> **BAR: `q1_gated = P(suffix proposes Sigma[j+3] correctly | gate fired) >= 0.8083`**
> (= `s3`, the measured MTP survival at draft position 3 on the K0 serve).

Select the **smallest `L*`** (largest warm rate) whose `q1_gated` clears the BAR, with
`a* = 0.0` unless the agreement term is needed to clear it.

Because the simulator is unselected (it averages over all emitted positions, whereas a real
gated step is additionally selected by having survived 3 correct MTP tokens), its `q1_gated`
is a **lower bound** on the served value. The banked study measures that selection premium
directly at the depth-5 handoff: simulated unconditional 0.473 vs measured 0.5972 =
**+0.124**. Verdict bands, pre-registered:

| verdict | condition |
|---|---|
| **FAVORABLE** | `q1_gated >= 0.8083` at a threshold whose warm rate >= 0.20 |
| **MARGINAL** | `q1_gated >= 0.8083 - 0.124 = 0.684` (the premium could close it) at warm rate >= 0.20 |
| **UNFAVORABLE** | no threshold reaches 0.684 at warm rate >= 0.20 |

An UNFAVORABLE outcome means the lever ships default-OFF with the negative result recorded,
and no serve is requested.

### 3.4 Pre-registered numeric predictions

Stated before running the simulator so the first run is a test and not a readout.

| quantity | prediction | basis |
|---|---|---|
| warm rate at `L* = 8`, `a* = 0` | **0.25 - 0.40** | `q1 = 0.473` unconditional; a long exact match is rarer than a correct proposal |
| warm rate at `L* = 4`, `a* = 0` | **0.45 - 0.65** | brief's 47-60% band |
| `q1_gated` at `L* = 8` | **0.80 - 0.92** | banked `q1 \| prev 2 hits = 0.848`, `\| 5 = 0.920`; a long match is a stronger condition than 2 prior hits |
| `q1_gated` at `L* = 4` | **0.70 - 0.82** | weaker condition |
| selected threshold | `L* = 8`, `a* = 0` | |
| average step saving | **20.6 ms x warm rate** = 5-8 ms/step | 2 passes x 10.3 ms |
| ladder-validation gate (reimplementation check) | simulated `r2..r6` within 0.10 of `.8032 .8578 .8875 .8979 .9071` on >= 4 of 5 slots | reproduces the banked study's own gate |

If the ladder-validation gate FAILS, this simulator is not a faithful stand-in for the shipped
Arctic proposer and every number it produces is reported as indicative only, with no verdict
issued.

## 4. What is being built

| deliverable | scope |
|---|---|
| `FR14_SUFFIX_PASS_GATE` | env flag, **default 0**, byte-identical when off |
| `FR14_SUFFIX_PASS_GATE_MIN_LEN` / `_MIN_AGREE` | the pre-registered `L*` / `a*`, defaulted from §3 |
| split drafter graph | a 2-iteration graph and a 2-iteration twin, replayed 1x (gated) or 2x (ungated) |
| gated validity mask | `HYDRA27_VALID` minus the 4 depth-4/5 runner-ups, a compile-time literal |
| census fields | `drafter.gate_fired`, `drafter.gate_match_len`, `drafter.mtp_forward_calls` (2 or 4) |

Step-time claims are made only via `step_wall_ms` / `s_per_fwd_gpu` / the `FR13_DFWD_SPLIT`
span instruments. Never TPS from a client.

---

# RESULTS (appended 2026-08-18, after §§1-4 were committed unchanged at `841517b18`)

## 4bis. Amendments to the pre-registration, declared

Two things in §§1-4 did not survive contact with the code. Both are recorded here rather than
edited above, so the pre-registered text stays exactly as committed at `841517b18`.

**(a) §2's "padded-invalid" became "duplicate-sibling padded", and the mask does NOT change.**
§2 proposed masking the four depth-4/5 runner-up nodes invalid, which would have made a gated
step a 23-active-node arm and required a per-step validity mask. That is not implementable:
the mask is a boot-time constant (`_FR13_FIXED32_VALID_MASK`) and the committer's child tables
are built once (`committer.pointer_table_rebuilds = 0` on 20 579/20 579 steps), so a per-step
mask change would invalidate the committer graph. The mechanism actually used is the one the
codebase already deploys: the four columns are filled by repeating their parent's spine token
— `fr13_mtp_suffix_assembly`'s last-resort pad — whose committer tie convention is already
proven on device by `scripts/fr13_greedy_pointmass_dup_gate.py`. Consequence, and it is a
strict improvement: `active_nodes` stays **27**, `valid_mask` stays **0x7abdffff**, the
31-column pack is unchanged, and a gated step is a **fill-source change only, never a shape
change**. `validate_gate_contract()` asserts it (padded ids are leaves, spine ids keep their
subtree, pack width invariant).

**(b) §3.2's predicate is computed by the gate itself, not read from Arctic.**
`SuffixDecodingCache.speculate()` returns `token_ids`/`parents`/`probs`/`score` and exposes
**no matched-pattern length**; under the fixed32 setting `use_tree_spec=False` the adapter
`arctic_draft_to_suffix_rel` discards `probs` before the drafter sees them, and
`arctic-inference` is pip-installed inside the container at prelaunch so its `max_spec_factor`
policy is not even inspectable on this host. Building an acceptance-affecting predicate on an
uninspectable internal was rejected. `scripts/fr14_suffix_pass_gate.py` therefore keeps its own
fixed-length n-gram recurrence index: O(1) per committed token, O(1) per step. Since "longest
match >= L*" and "the L*-gram was seen before" are the same event, this measures the
pre-registered predicate exactly; the only refinement is that agreement is measured at `L*`
rather than at the longest match, and the sweep below reports it that way throughout.

## 5. Offline calibration

`scripts/fr14_suffix_gate_calibration.py` over the K0 serve's four banked SWE-bench
trajectories (`output/fr14_b1_stock_20260817T054447Z/tail6_fixed32_b1radix`), 12 000 sampled
emitted positions. Raw output: `suffix_gate_calibration.json`.

### 5.1 Gate 1 — does this simulator reproduce the shipped Arctic tail? PASS

This is a reimplementation of the banked seam study's stand-in proposer, so it must first
reproduce its validation gate.

| slot | simulated `r_m` | observed (K0 serve) | delta |
|---|---|---|---|
| r2 | 0.7741 | 0.8032 | 0.029 |
| r3 | 0.8399 | 0.8578 | 0.018 |
| r4 | 0.8732 | 0.8875 | 0.014 |
| r5 | 0.9073 | 0.8979 | 0.009 |
| r6 | 0.9159 | 0.9071 | 0.009 |

**5 of 5 slots within 0.10; max |delta| = 0.029.** The gate demanded 4 of 5. Unconditional
cold start `q1 = 0.4578`, against the banked study's 0.473 — the same number by an
independent implementation.

### 5.2 An independent re-derivation of the selection premium

The simulator's suffix cold start evaluated at draft position 5, on the same unselected
population, reads **0.4622** against the serve's **measured 0.5972** handoff. The gap is the
selection premium — conditioning on 5 correct MTP tokens picks easier regions:

**+0.135**, where the banked study reached **+0.124** by a different route (linear
interpolation in the number of preceding MTP-correct tokens). Two routes, one number.
Every "unselected" figure below is therefore a *lower bound* on what a real gated step sees.

### 5.3 The gate sweep (the implementable predicate)

`L(j) >= L*` and "the L*-gram was seen before" are the same event, so the shipped fixed-length
index measures the pre-registered predicate exactly; only the agreement term's measurement
point differs (measured at `L*` rather than at the longest match), which is stated here as the
one refinement §3.2 needed to be implementable.

| `L*` | `a*` | warm rate (position-weighted) | **`q1_gated`** | E[accept] on a gated step |
|---|---|---|---|---|
| 6 | 0.00 | 0.314 | 0.7535 | 5.21 |
| 6 | 0.75 | 0.279 | 0.7733 | 5.33 |
| 8 | 0.00 | 0.253 | 0.8013 | 5.53 |
| 8 | 0.50 | 0.249 | 0.8069 | 5.56 |
| **8** | **0.75** | **0.234** | **0.8202** | **5.64** |
| 12 | 0.00 | 0.191 | 0.8631 | 5.97 |
| 16 | 0.00 | 0.158 | 0.9032 | 6.30 |

**Pre-registered selection: `L* = 8`, `a* = 0.75`** — the smallest `L*` clearing the BAR
(`q1_gated = 0.8202 >= 0.8083 = s3`) at a warm rate above 0.20. Note `a* = 0` at `L* = 8`
misses the bar by 0.007, so the agreement term is load-bearing, exactly the case §3.2
anticipated by putting `a*` in the sweep.

**Verdict: FAVORABLE.** And it is the *unselected* reading — §5.2 says the served value
carries roughly another +0.135 of selection premium on top.

### 5.4 Scorecard against the pre-registered predictions (§3.4)

Recorded whether or not they flatter the lever.

| prediction | predicted | measured | |
|---|---|---|---|
| warm rate at `L*=8, a*=0` | 0.25 - 0.40 | 0.253 | **hit** (bottom edge) |
| warm rate at `L*=4, a*=0` | 0.45 - 0.65 | 0.434 | **MISS** (below band) |
| `q1_gated` at `L*=8` | 0.80 - 0.92 | 0.801 | **hit** (bottom edge) |
| `q1_gated` at `L*=4` | 0.70 - 0.82 | 0.675 | **MISS** (below band) |
| selected threshold | `L*=8, a*=0` | `L*=8, a*=0.75` | half-hit |
| average step saving | 5 - 8 ms | **4.02 ms** | **MISS** (below band) |
| ladder-validation gate | >=4/5 within 0.10 | 5/5, max 0.029 | hit |

The pattern is consistent and worth stating: **every miss is in the optimistic direction.**
The predicate is a sharper discriminator than predicted at long `L` and a weaker one at short
`L`, and the warm rate is lower than the brief's 47-60% band by a factor of ~2.5.

## 6. Step-weighted economics — the correction the brief needs

Position-uniform sampling over-represents easy regions. A decode loop is a **renewal
process**: it spends more *steps* in hard regions because it accepts fewer tokens there. So
the warm-POSITION rate is not the warm-STEP rate.

`renewal_simulate()` walks each stream the way the engine does — advance by (accepted + 1) —
with the MTP survivals drawn from the measured ladder and the suffix part evaluated at the
real context. Both arms walk identical streams; 3 seeds x 4 tasks; ~83 000 simulated steps
per arm.

| | gate OFF | gate ON |
|---|---|---|
| simulated steps | 83 484 | 81 485 |
| **warm-step rate** | — | **0.1951** |
| accept/step | 4.0108 | 4.1337 |

**Calibration check:** the gate-OFF arm predicts accept/step **4.011** against the serve's
**measured 4.2774** — 6.2% low. The simulator therefore *under*-predicts accept, so it is
conservative, and the residual is the honest cost of the 73% emitted-mass reconstruction
(per-task ratios 0.948 / 0.743 / 0.738 / 0.660, unchanged from the banked study's failure of
its own fidelity gate).

**Warm-step rate 0.195, not 0.47-0.60.** Average saving = 20.6 ms x 0.195 = **4.02 ms/step**
of a 207.87 ms step = **1.9%**.

## 7. The counterfactual — does the gate LOSE accept on the steps it fires?

`E[accept | gated] = 5.64` against the unconditional 4.28 is **not** a fair comparison, and
this note refuses to make it: a gated step is selected for being easy, so MTP would also have
done better than its unconditional survival there.

Draft positions 0, 1, 2 are the *same three MTP passes* in both arms, so they cancel. The
entire comparison is what happens after position 2, conditional on reaching it (probability
0.9472 x 0.8443 x 0.7709 = 0.6165):

- **gated:** suffix chain over positions 3..10, survivals
  `0.8202 0.9184 0.9338 0.9463 0.9454 0.9542 0.9644 0.9674` → **5.311** expected tokens
- **ungated counterfactual:** MTP at 3 and 4 with survival `m` (unmeasurable offline), then
  the *measured-on-the-gated-population* handoff chain from position 5,
  `0.7906 0.9225 0.9316 0.9465 0.9551 0.9623` (note: 0.79, versus 0.5972 unconditional — an
  independent confirmation that the gate is selecting genuinely easier steps)

| assumed MTP survival on gated steps | ungated E | gated E | delta / gated step | **TPS vs today** |
|---|---|---|---|---|
| 0.8083 (its unconditional value) | 4.106 | 5.311 | **+0.743** | **+4.9%** |
| 0.8800 | 4.789 | 5.311 | +0.522 | **+3.3%** |
| **0.9310 (break-even)** | 5.311 | 5.311 | **0.000** | **+2.0%** |
| 0.9500 | 5.506 | 5.311 | **-0.120** | **+1.5%** |

**This is the lever's real risk statement.** The gate is accept-positive iff MTP's survival at
draft positions 3-4 *on strong-match steps* is below **0.931**. Unconditionally it is 0.808 /
0.817. Nothing offline can measure it — no MTP model exists in the banked data — so it is the
one question the A/B is for.

**The lever is positive across the whole range**, because the 20.6 ms is not at risk:

- accept-neutral floor: **+2.0%**
- central: **+3.3%**
- optimistic: **+4.9%**

Stated in the doctrine's units rather than TPS: **-4.02 ms/step average, -20.6 ms on gated
steps**, with an accept effect bounded in **[-0.023, +0.145] tokens/step**.

## 8. The graph question, measured

`scripts/fr14_suffix_gate_graph_microbench.py`, on the GB10, synthetic MTP block byte-sized to
the pinned floor ledger (1.577 GB/pass vs the ledger's 1.565 GB). Raw:
`suffix_gate_graph_microbench.json`.

Two shapes were possible. **TWIN** = an independent 4-pass and 2-pass graph per batch size.
**SPLIT** = one 2-pass graph `lo` + one 2-pass graph `hi` sharing pool and static buffers;
ungated replays `lo` then `hi`, gated replays `lo` alone. SPLIT wins outright, because the
ungated path still executes exactly four MTP forwards, so every invariant that counts four
per step is untouched — and the microbench says it costs essentially nothing:

| quantity | measured |
|---|---|
| **split output == single-graph output** | **bit-exact** (`torch.equal`, full output) |
| gated replay fills only the first two passes, leaves the rest untouched | true |
| capture cost, single 4-pass graph | 3.26 ms |
| capture cost, `lo` + `hi` | 3.53 ms → **+0.275 ms once, at boot** |
| pool memory, single 4-pass graph | 25.2 MB |
| pool memory, `lo` + `hi` | 46.1 MB → **+21 MB** |
| extra graph launch at replay | **-0.07 ms** (i.e. nothing measurable) |
| replay, 4 passes / 2 passes | 29.41 ms / 14.63 ms |
| **proportionality** | **14.63 / 29.41 = 0.497** |

That last row is the one that matters: replay time is proportional to pass count, which is
what makes the 2 x 10.3 ms saving structural rather than hoped-for. (The synthetic pass runs
7.39 ms against the real 10.3 ms because it is a GEMM chain without attention or top-k; the
microbench prices graph *mechanics*, not the block. The 10.3 ms is already kernel-confirmed
by nsys attribution.) An incidental corroboration: eager 4-pass 30.50 ms vs graphed 29.48 ms
= 1.02 ms of launch overhead, against FR13's independently measured 1.243 ms `dfwd` host-idle.

## 9. What is built, and the one thing that is not

| piece | state |
|---|---|
| `scripts/fr14_suffix_pass_gate.py` — predicate, O(1) index, fail-closed | **landed, 25 tests** |
| predicate == the calibrated predicate | **landed** (`test_online_matches_offline_predicate`) |
| `scripts/fr14_suffix_gate_calibration.py` — offline evidence | **landed** |
| `fr13_fixed32_topology.validate_gate_contract()` — well-formedness | **landed** |
| `fr13_merged_drafter.decide_fixed32(gated=True)` — 8-token Arctic ask | **landed, 8 tests** |
| launcher arming: validated flag, /logs sidecar, refusals | **landed** |
| graph-split microbench | **landed, measured** |
| **drafter split-graph (`lo`/`hi`) in the patcher** | **NOT landed — interlocked** |

### 9.1 Why the drafter graph half was not landed blind

1. **It cannot be executed once in this session.** Host torch has no CUDA (the GPU is reached
   only through the container), and validating a CUDA-graph capture change means a full serve
   boot loop — which belongs with the scheduled A/B, not ahead of it.
2. **It re-issues a shipped credential.** The drafter graph manifest signature is pinned as a
   sha256 *literal* in `_fr13_dh_m32_note_production_replay:28283` and
   `_fr13_dh_u8_note_production_replay:28446`. Changing the manifest to carry a pass count
   changes that hash. Re-issuing an attestation blind, in a shared tree, next to a concurrent
   agent editing the same 43k-line file, is the exact move the blanket-add doctrine
   (`f592e86b9`) exists to prevent.
3. **The microbench already retired the discovery risk.** Bit-exactness, capture cost, pool
   cost and launch cost are all measured (§8). What remains is integration against 15
   fail-closed literals, each of which reports itself on a boot — cheap in the boot loop,
   guesswork outside it.

**The lever is interlocked so it cannot be armed half-integrated.** Both launchers refuse
`FR14_SUFFIX_PASS_GATE=1` unless the patcher carries the sentinel `FR14_GATE_SPLIT_GRAPH`,
because arming it today would hand `decide_fixed32` a 3-depth MTP head while the drafter still
ran four forwards — a malformed tree at the verifier. The interlock clears itself when the
split lands; it needs no second edit.

### 9.2 The integration, specified

All anchors in `scripts/fr10_phase4_patch_vllm_tree_gdn.py` unless noted. Every change is
inside a gate-armed branch; with the sidecar absent, every literal keeps today's value and the
manifest signature is byte-identical.

| # | site | change |
|---|---|---|
| 1 | `:29511` `int(_fr10_spine_steps) == 4` | allow the split shape |
| 2 | `:29562-29620` replay branch | replay `lo`, then `hi` only if ungated; unpack `range(2)`/`range(2,4)`; the `+4` seq-len fixups become `+2` per replay |
| 3 | `:29628-29690` capture branch | capture `lo` (iters 0-1) then `hi` (iters 2-3) into the SAME pool and the SAME static buffers; capture-then-replay ordering as today |
| 4 | `:6434` `batch in _FR13_FIXED32_DRAFTER_GRAPH_BY_BATCH` | re-key the registry `(batch, passes)` |
| 5 | `:6587-6590`, `:6607-6610` capture-end | `4` → the context's own pass count; schema `v2` → `v3-split` **only when armed** |
| 6 | `:6676-6683` replay | same parameterization; accumulate `graph_replays` and `mtp_forward_calls` across the two replays instead of asserting `== 0` / setting `= 4` |
| 7 | `:6906-6907`, `:6938-6939`, `:6976-6977` proposal-end + census payload | accept (4 calls, 2 replays) ungated and (2, 1) gated; `main_tail_length` 6 → 8 when gated |
| 8 | `:6800-6816` `_fr13_fixed32_drafter_observed_arctic` | `main_lookup_tokens` `6*B` → `8*B`, `main_tail_columns` 6 → 8 when gated |
| 9 | `scripts/fr13_fixed32_work_census.py:1345,1365,1600` | make the pinned literals conditional on a new optional `drafter.gated` key (optional, so all 35 banked runroots still validate) |
| 10 | `:30351-30359` `decide_fixed32` call site | pass `gated=`; place Arctic columns 0-1 into head depths 4-5 and duplicate-pad the four runner-up columns |

The armed serve is incompatible with `FR13_DRAFT_HEAD_M32/U8/FP8_PRODUCTION` (their pinned
signature literals) and with `FR13_TAIL_BRANCHES` (already refused by the launcher).

## 10. The A/B serve plan, for the coordinator to schedule

**Do not run this without scheduling.** Requirements, from the variance doctrine
(`seam_move_economics.md` §9.4: the same arm banked accept 3.81 / 4.04 / 4.28 across three
runs, ±10%):

- **PAIRED on an identical task set**, >= 20 000 decode steps per arm. A predicted +2 to +5%
  sits well inside single-run variance; only pairing measures the lever rather than task mix.
- **Arms:** `FR14_SUFFIX_PASS_GATE=0` vs `=1` at `NGRAM=8 MIN_AGREE=0.75`, everything else
  pinned identical (radixark `K=0`, hydra27 fixed32, B=1).
- **Instruments:** `step_wall_ms` and `s_per_fwd_gpu` from the reducer, plus `FR13_DFWD_SPLIT=1`
  on both arms so the drafter's model/head/other split is visible — the gated arm must show
  `dfwd` falling by ~20.6 ms on gated steps and being unchanged on cold ones. **No TPS from a
  client.**
- **Census:** `drafter.mtp_forward_calls` must be exactly 2 on gated steps and 4 otherwise,
  with no third value across the whole serve; `active_nodes` must stay 27 and `verify_rows` 32
  on every step, gated or not.

### 10.1 Pre-registered readings, so the first armed serve is a test

| quantity | predicted | falsified by |
|---|---|---|
| warm-step rate (census `gate_fired` / steps) | **0.15 - 0.25** | anything outside; the brief's 0.47-0.60 is already refuted offline (§6) |
| `dfwd` on gated steps | **-20.6 ms** vs cold steps | < 15 ms would mean the split is not skipping real work |
| average `step_wall_ms` delta | **-4.0 ms** (1.9%) | |
| accept/step delta | **-0.02 to +0.15** | a larger loss falsifies §7's break-even at m = 0.931 |
| `active_nodes` / `verify_rows` | **27 / 32 on every step** | any other value means the gate changed shape, which it must never do |

The single question the serve exists to answer: **is MTP's survival at draft positions 3-4, on
strong-match steps, below 0.931?** Everything else is already measured.

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

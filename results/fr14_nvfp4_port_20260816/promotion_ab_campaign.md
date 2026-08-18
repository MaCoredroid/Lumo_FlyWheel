# FR14 promotion A/B campaign — three arms, and what the GPU said about each

Run 2026-08-18 03:32Z–07:50Z on Mark's existing greenlight, under his condition
**"eyeball traces for degenerate"**. **Evidence only — nothing here promotes
anything, and nothing was relaxed to make an arm run.**

HEAD throughout: `4a59ca2b46ec76a9f8c0690a954e03c737d2970b` (pass 53). No commit
was taken between the first boot and the last drain.

## Result in one table

| arm | lever under test | outcome | what it cost |
|---|---|---|---|
| **C** control | `FR14_FUSED_DRAFT_TOPK=1` | **DRAINED CLEAN**, 4/4 tasks, 53 446 steps | 3 h 43 m |
| **G** gated | `+ FR14_SUFFIX_PASS_GATE=1` | **FAIL-CLOSED at the first armed boot**, 0 steps | 6 min |
| **G-iso** attribution | gate ON, top-k **OFF** | **IDENTICAL refusal** — attributes it to the split-graph alone | 6 min |
| **S** split-K | `LIVE_AB_ARM=gqa_pair_splitk` | **REFUSED AT THE LAUNCHER** — and the refusal is a defect | 5 s |
| step 0 | re-earn the `gqa_pair` credential at HEAD | **REFUSED** on a stale census mirror | 14 min |

**Three of the five windows ended in a refusal, and every refusal is a finding.**
Two of them (§0.4, §2.1) are defects that block work having nothing to do with
this campaign.

Artifacts, all in this directory: `promotion_ab_arm.sh`,
`promotion_ab_arm_s_splitk.sh`, `promotion_ab_regate_gqa_pair.sh`,
`promotion_ab_reduce.py`, `promotion_ab_eyeball.py`, `promotion_ab_pair.py`,
`promotion_ab_workshape.py`, `promotion_ab_arm_s_reachability_probe.py`, and the
`promotion_ab_*.json` / `*_container_tail.log` / `*_refusal.log` each wrote.

---

# 0. Findings that landed before, and instead of, the measurements

## 0.1 The promoted stack was not serviceable at HEAD

`gqa_pair` is the promoted B1 FA2 arm (2026-08-13, Mark's "B1 flip Yes") and arms
only from a credential whose `FR13_FA2_QROW32_B1_SOURCE_COMMIT` **equals**
`git rev-parse HEAD` — the launcher checks *serviceability*, not presence
(`fr13_launch_forked_fa2_tree_server.sh:1313-1338`). The banked pointer was
earned at `05987f682`; HEAD has moved three times since (passes 51, 52, 53). So
step 0 of the campaign was to re-earn it. **Step 0 failed — see §0.4.**

Consequence, recorded rather than worked around: arms C and G ran with the B1
production arm **NAMED EMPTY**, serving the **qrow16 incumbent**. That is the
launcher's documented deliberate opt-out, it is attested in each arm's
`container_env.txt`, and it is identical in both arms by construction rather than
by inference. **The C-vs-G comparison is unaffected** — the FA2 verifier kernel
is not the variable under test — but ARM C's absolute `step_wall` is **not**
comparable to the banked composed anchor (210.700 ms), which carried gqa_pair.

## 0.2 The task budget is 9000, not the briefed 5400 — and ARM C proved it

The brief asked for `FR13_CAMPAIGN_TASK_BUDGET_S=5400` *"so a wall-timeout is an
accounted capped terminal, not a voided arm."* At 5400 that is exactly what does
not happen on this arm shape: pass 23 ran it, `13398` hit the cap, the cap fired
correctly — and the arm still returned `swerc=13` with **three** tasks, because
the truncated-trace validator union a capped terminal needs was never built (pass
24 flagged it as provenance-core surgery awaiting Mark; it has not landed). Pass
24's own ruling was *"Future marathon arms: budget 9000s"*, which is what
`fr14_run_b1_max_stack_serve.sh` — the pattern of record for this exact arm shape
— uses.

**ARM C settled it empirically: `astropy__astropy-13398` took 5 784.657 s.** At
the briefed 5400 it would have been capped and ARM C would have been the void the
brief was trying to avoid. `AGENT_WALL_S` was held equal to the budget so the
declared budget was provably the binding limit.

## 0.3 The eyeball Mark asked for is not obtainable from this HEAD

See §2. The split-K plumbing that landed in pass 50 arms a route that **cannot
serve a single token** through the split-K kernel. `splitk_fa2.md` §10's claim
that "the plumbing removes the reason the probe could not be run" is **wrong**;
§7's condition is exactly as undischarged as it was, and there is now a second,
independent reason it cannot be discharged (§2.1).

## 0.4 A STALE CENSUS MIRROR BLOCKS EVERY fixed32 SERVE AT THIS HEAD

**This is the campaign's most broadly damaging finding and it is not about any
lever.** Step 0's gate booted, served, and resolved its task
(`swe_orchestrator_rc: 0`) — then died in the terminal chat-traffic audit:

```
fr13_fixed32_work_census.CensusError: …/fr13_fixed32_work_census.jsonl:1
  .taw.source_contract_sha256:
  expected '484babd7a883c81c7317ef23862940143c248dcbc1b66c9d4ac6775ff5a0fa93',
  got      '68b289aee5773edf1134f184c37551a90ec8543430d768a05066bc1341473c6d'
```

Attribution, exact:

* `scripts/fr13_device_multidraft_kernel.py:1667` — the **emitter** — was
  re-attested to `68b289ae…` on 2026-08-18 by FR14 lane 3, coordinator-sanctioned,
  when the batched-softmax cache was wired in (pass 44; the comment at :1660-1666
  records the prior value and the sanction).
* `scripts/fr13_fixed32_work_census.py:157` `TAW_SOURCE_CONTRACT_SHA256` — the
  **mirror the validator reads** — still carries `484babd7…`.

The emitter *recomputes its digest from live source and self-asserts it on every
boot* (`_fr13_fixed32_taw_source_contract`, :3113-3118), so `68b289ae…` is not a
claim — it is what the served source **is**, proven in-container. The census
literal is simply a mirror nobody synced.

**Effect:** every fixed32 serve at this HEAD returns non-zero from its terminal
audit and **mints no credential**. It hit step 0 (rc=16) and ARM C (rc=16)
identically. It is why `gqa_pair` could not be re-earned, and it will block any
future gate re-earn until it is fixed.

Pass 44 reported *"all 24 digest-drift failures cleared"* — true of the **tests**.
The census validator's literal is a 25th site, and it is exercised **only by a
live serve**, so no CPU test could have caught it. Same family as pass 51's
stale-fixture finding and as §2.1 below: **a constant re-attested in one place and
not in its mirror.**

**NOT FIXED HERE, deliberately.** The census validator is a credential
instrument; editing one so that my own measurement could proceed is the exact
move `splitk_fa2.md` §2(a) refused ("a guard relaxed to make a build pass").
Flagged for the credential owner. The campaign did not need it: the serve itself
completes and writes every instrument, so ARM C's evidence is intact and only its
*credential* is absent — which an evidence-only campaign does not claim.

---

# 1. What was pinned identical across arms C and G

One variable: `FR14_SUFFIX_PASS_GATE`. Everything else pinned in
`promotion_ab_arm.sh` and attested in each `container_env.txt`:

* topology `hydra27_fixed32` — **required**, because §11.7 pre-registers
  `active_nodes == 27` on every step and tail6 is 23
* K0 full-vocabulary drafting (`ROOT=0 K=0`, `NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0`,
  profile `full_vocab`); `B=1`, `SWE_CONCURRENCY=1`,
  `CUDAGRAPH_MODE=FULL_AND_PIECEWISE`, `FR10_METRICS=0`,
  `FR13_HOST_TAIL_PREP_BAKE=1`
* canonical exact4 subset sha `0e37b713…e853f5`
* `FR13_DFWD_SPLIT=1` **and** `FR13_LFWD_GPU_TIMER=1` in both arms, plus the
  sfwd/dfwd/cfwd span timers
* `FR14_FUSED_DRAFT_TOPK=1` in **both** arms (byte-exact selection; holding it on
  in both is what keeps C-vs-G to one variable)
* network boundary as the pattern carries it: `OFFLOAD_AGENT=1`, proxy + codex on
  alienware, GB10 vLLM-only, forced temp 0.6, raw dumps disabled
* `FR13_CAMPAIGN_TASK_BUDGET_S=9000` with `AGENT_WALL_S=9000` (§0.2)

---

# 2. ARM S — split-K cannot serve a token, for three independent reasons

The mandatory degenerate eyeball on split-K **cannot be discharged from this
HEAD**. Not "we ran out of time" — not reachable, and now proven three ways.

## 2.1 It does not even reach the runtime: pass 50's branch has a mirror gap

The live boot refused in 5 seconds, at the **launcher**:

```
FAIL: launcher rc=1
fixed32 qrow32 B1 binary identity is not qualified
```

`fr13_launch_forked_fa2_tree_server.sh:4296-4319` is a **second, in-container
Python qualification map** that resolves the pin arm and looks up its expected
binary identity. Its table has entries for `visibility` and `gqa_pair` only, and
`.get(b1_pin_arm, <split2 defaults>)` **falls through to split2's pins** — so the
split-K binary is compared against split2's sha/size and refused. Confirming it:
`scripts/fr13_fixed32_contract.py` contains **no split-K constants at all** (zero
matches for `SPLITK`), so the map has nothing to look up even if the branch
existed.

Pass 50 landed the split-K pins in the **bash** pin case
(`_FR13_FA2_QROW32_B1_PIN_ARM`, :2214-2280 — which this boot would have passed,
including the SASS digests and the re-hash that separates the PID-shifted twin)
and **not** in the Python mirror twenty lines of comment away, whose own comment
says it *"mirrors the bash pin case"*. It no longer does.

Pass 50 claimed the branch is *"EXECUTED in tests, not grepped"*, 39 cases. True —
of the **bash** branch. The Python qualifier is reachable only on a live boot, and
this was the first one. **Same defect family as §0.4.**

## 2.2 Even qualified, the live-A/B route refuses split-K by construction

`promotion_ab_arm_s_reachability_probe.py` `exec`s the **deployed** selector helper
blob out of the FA2 patcher and drives it on CPU (the technique
`suffix_pass_gating.md` §11.4 uses). Executed, not read
(`promotion_ab_arm_s_reachability.json`):

```
route_production : REFUSED — "FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty or
                   one of nosplit, gqa_pair; got 'gqa_pair_splitk'"
route_live       : name ACCEPTED, then
                   "FR13 qrow32 B1 raw-byte qualification requires identical
                    reduction topology: reference_partitions=1
                    candidate_partitions=4"
control          : gqa_pair at the same reference topology — NOT refused
```

The control line matters: the refusal is specific to split-K's `num_splits=4`, not
an artefact of the probe.

## 2.3 And the live-A/B route never returns candidate output anyway

`_fr13_fa2_qrow32_b1_production_begin` is the **only** deployed path that ever sets
`candidate_served: True`; the live-A/B path is a **one-shot shadow byte
comparison** that runs reference and candidate side by side, writes a JSON, and
serves the reference. The K0 gate runner states it in its own banked
`launcher_meta.txt`: `qrow16_reference_served=1`, `candidate_returned=0`.

So even with §2.1 fixed and §2.2 waived, **the served text would be the
incumbent's.** There is no split-K trace to eyeball because no split-K token can
exist.

**What is needed before Mark's condition can be met:** a route that serves the
split-K kernel — which means admitting it to `_FR13_FA2_QROW32_B1_PRODUCTION_ARMS`
(deliberately refused today, because the raw-byte gate structurally cannot qualify
it) or building a new *gate-only serving* route. That is a design decision for
Mark, not a plumbing gap.

---

# 3. ARM C — control: drained clean, and the fused top-k kernel served for the first time

`output/fr14_promoab_C_20260818T035118Z`, 03:51:18Z → 07:34:11Z.

`swe_orchestrator_rc: 0`, **4/4 tasks, no timeouts, no caps**, 2/4 resolved
(12907 R, 13033 F, 13236 R, 13398 F). Serve window 13 027 s. **53 446 census
steps** — well past the ≥20 000/arm the variance doctrine demands.

**Lever 1 engaged live for the first time in a serve** (container log, verbatim):

```
[FR14_FUSED_DRAFT_TOPK] ready K0 full-vocab width3 rows=1 blocks=64 launches_per_head=1 stock_argmax_topk=0
[FR14_FUSED_DRAFT_TOPK] engaged stock_argmax_topk=0
```

`stock_argmax_topk=0` is the load-bearing part: the ATen `argmax` + `topk(3)` pair
is gone from all five head reads, replaced by one launch each.

## 3.1 Instruments

| instrument | ARM C |
|---|---|
| `step_wall_ms` | **219.765** |
| `s_per_fwd_gpu` | **0.135519** |
| `floor_ratio` | 2.3592 (floor 93.152 ms) |
| drafter GPU ms/step | 56.082 |
| committer GPU ms/step | 20.552 |
| `overhead_other` ms/event | 7.611 |
| accept/event | 4.0984 · committed/event 5.0984 |
| per-request decode TPS | 24.454 (**reported, not a verdict**) |
| measured wall steps | 53 228 |

Per-task brackets (`promotion_ab_arm_c_per_task.json`) — retained because they are
the only basis on which a future ARM G could be paired instance-by-instance:

| instance | step_wall ms | s/fwd_gpu | accept | drafts |
|---|---|---|---|---|
| astropy-12907 | 206.12 | 0.12802 | 3.964 | 1 203 |
| astropy-13033 | 221.22 | 0.13800 | 4.356 | 11 686 |
| astropy-13236 | 217.25 | 0.13428 | 4.099 | 16 277 |
| astropy-13398 | 221.42 | 0.13553 | 3.981 | 24 280 |

**ARM C's 219.765 ms must not be compared to the banked 210.700 ms.** Two
declared differences: it serves the **qrow16 incumbent** rather than gqa_pair
(§0.1; gqa_pair's own banked value is −4.4 ms), and it carries two extra
instruments (`FR13_DFWD_SPLIT`, `FR13_LFWD_GPU_TIMER`) the anchor did not. A
lever-1 delta was **not** measurable from this campaign: the pre-registered
instrument for it is the **paired dfwd span against a top-k-OFF twin**, and that
twin was never run — ARM C's own dfwd span has nothing to be a delta against.

## 3.2 Acceptance, and a free corroboration for lever 2

Aggregate over the four per-task brackets: 53 446 drafts, 1 656 826 draft tokens,
219 045 accepted → **4.0984 accepted/event**.

Per-position cumulative survival:

```
p0 0.942  p1 0.788  p2 0.608  p3 0.476  p4 0.378  p5 0.219
p6 0.175  p7 0.149  p8 0.133  p9 0.120  p10 0.110  p11+ 0.000
```

Converting to **conditional** survivals — which is what
`suffix_pass_gating.md` §7's break-even is stated in:

```
m0 0.942  m1 0.837  m2 0.772  m3 0.783  m4 0.794  m5 0.579
```

Two things fall out, both useful and neither the question ARM G existed to answer:

* **m3 = 0.783 and m4 = 0.794** against the doc's unconditional 0.808 / 0.817 —
  measured a shade **lower**, i.e. slightly *more* favourable to lever 2 than the
  numbers its economics were built on.
* **m5 = 0.579** corroborates `seam_move_economics`'s 0.5972 unconditional
  handoff survival on an independent serve.

The break-even question itself — *is MTP's survival at positions 3-4 **on
strong-match steps** below 0.931?* — is **unanswered**. It is conditional on the
gate firing, and the gate never fired.

## 3.3 Work-shape census — all pre-registered invariants hold

53 446 events: `active_nodes` **27 on every step**, `verify_rows` **32 on every
step**, `mtp_forward_calls` **4 on every step** with `(4, 6)` the only
(calls, tail) pair, `graph_replays` **1 on every step**, and the drafter graph
signature is the banked `d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c`
on every step. Lever 1 is armed throughout — so **the fused top-k kernel does not
move the drafter graph identity**, which is the shape claim it needed to survive.

## 3.4 EYEBALL — ARM C: no degeneration signature

Read from `qwen_trace.jsonl`, the real served agent trajectories
(`promotion_ab_eyeball_arm_c.json`):

| instance | turns | words | ttr | max line | top 8-gram | tail-rep | non-ASCII | tool calls | malformed |
|---|---|---|---|---|---|---|---|---|---|
| 12907 | 35 | 2 519 | 0.332 | 9 | 5 | 0.051 | 0 | 12 | **0** |
| 13033 | 68 | 25 965 | 0.140 | 112 | 40 | 0.330 | 0 | 32 | **0** |
| 13236 | 284 | 24 585 | 0.193 | 87 | 16 | 0.201 | 0 | 99 | **0** |
| 13398 | 230 | 32 999 | 0.184 | 58 | 21 | 0.166 | 0 | 85 | **0** |

**228 tool calls, zero malformed. Zero non-ASCII characters in 86 000 words. All
stop reasons `tool_use`** — no truncation-shaped terminals.

Every repetition signature was chased to its source and every one is benign:

* the most-repeated line in **all four** traces is the markdown fence ```` ``` ````
  (112× in 13033) — structural, not generated text;
* the top 8-grams are repeated *source lines from the file under edit*:
  13033 `'raise ValueError( f"{self.__class__.__name__} object is invalid - expected'`,
  13398 `'# form the Topocentric ITRS position topocentric_itrs_repr ='`,
  13236 `'# Structured ndarray gets viewed as a mixin'`.

Verbatim tail of 13236 (284 turns, the longest trajectory):

> Structured ndarray → `Column`; explicit `.view(NdarrayMixin)` → still
> `NdarrayMixin` ✓ … Full `astropy/table` + `astropy/utils` suites: **2288
> passed**; the only new failure is `test_ndarray_mixin`, which asserts the exact
> old behavior this issue removes (the grader's test patch updates it — I
> confirmed all 9 other failures are pre-existing on the base commit via
> `git stash`).

Verbatim tail of 13398:

> All four directions transform correctly with **exact roundtrips**. …
> Cross-checked against the old ICRS route: agrees to ~0.2 arcsec — precisely the
> topocentric-aberration difference the problem statement describes.

Coherent, calibrated, self-auditing prose that distinguishes pre-existing failures
from its own. **No repetition loop, no gibberish, no mid-word break, no malformed
tool call. ARM C's eyeball is clean.**

(The `midword` column of the tool is a deliberate over-reporter — 407–5 415 hits —
and every sample inspected was ordinary prose like "is a" or "to be". It is a
pointer to where to look, never a verdict; that is why the excerpts above exist.)

---

# 4. ARM G — the first armed boot of the split-graph integration FAILED CLOSED

`output/fr14_promoab_G_20260818T073427Z`, 07:34:27Z → 07:40:28Z. Zero census
events. Full container tail: `promotion_ab_arm_g_container_tail.log`.

Pass 48's caveat was *"none of the drafter integration has executed on GPU — every
invariant is fail-closed, the first armed boot is the test"*. **The test ran, and
it refused.**

## 4.1 The §11.7 checks, as far as the boot got

| pre-registered check | result |
|---|---|
| `[FR14_SUFFIX_PASS_GATE] armed ngram=8 min_agree=0.75` printed once | **PASS** — verbatim: `[FR14_SUFFIX_PASS_GATE] armed ngram=8 min_agree=0.75 min_history=256` |
| launcher writes `/logs/fr14_suffix_pass_gate.cfg` | **PASS** — contents `8 0.75 256` |
| two half-graph captures begin | **PASS** — `graph_captures: 2`, segment 0 signature `2da8c56a0c7da5ea9262b0020a30c925138c18eba95ab2b01ef2a4f1ff22da42` |
| half-graph is a distinct artifact from the shipped 4-pass graph | **PASS** — `2da8c56a…` ≠ the banked `d9a4dd…6150c` (§11.3's v3-split intent holds) |
| registry: two rows `passes=2`, `segment` 0 and 1 | **NOT REACHED** |
| `graph_replays` 2 cold / 1 gated | **NOT REACHED** — the engine died during *capture*, before any replay |
| `mtp_forward_calls` only in {4, 2} | **NOT REACHED** |
| `active_nodes`/`verify_rows` 27/32 every step | **NOT REACHED** — zero steps |
| warm-step rate 0.15–0.25 | **NOT REACHED** |

## 4.2 The refusal, and its exact mechanism

```
RuntimeError: FR13 fixed32 drafter tree-attention work drift:
('mtp.layers.0.self_attn.attn', 1, (1, 1),
 {'mode': 'hydra27_fixed32', 'batch_size': 1, …, 'measured': False,
  'forward_step_index': -1, 'mtp_execution_basis': 'unbound',
  'mtp_forward_calls': 0, 'graph_id': None, 'graph_signature': None,
  'graph_replays': 0, 'graph_captures': 2,
  'captured_graph_id': 276890437994352,
  'captured_graph_signature': '2da8c56a0c7da5ea9262b0020a30c925138c18eba95ab2b01ef2a4f1ff22da42'},
 {'graph_id': 276890437998192, 'mode': 'hydra27_fixed32', 'batch_size': 1,
  'passes': 2, 'segment': 1, 'capturing': True, 'mtp_forward_calls': 0, …})
```

Raised at `gdn_linear_attn.py:3201 _fr13_fixed32_observed_tree_attn`, called from
`tree_attn.py:563 _fr13_tree_attn_op_capture`, inside the MTP drafter's forward
during CUDA-graph capture (`vllm/v1/spec_decode/eagle.py:5132 propose`).

Read the two dicts against each other and the mechanism is unambiguous:

1. segment **0** is captured first; the observed drafter event records
   `captured_graph_id` = segment-0's graph and `captured_graph_signature` =
   `2da8c56a…`;
2. capture of segment **1** then begins — a *different* graph
   (`graph_id: …98192`, `passes: 2`, `segment: 1`, `capturing: True`);
3. the MTP layer-0 tree-attention op fires inside segment 1's capture and
   `_fr13_fixed32_observed_tree_attn` compares it against the observed event's
   `captured_graph_id`, **still pointing at segment 0** — and refuses.

**The split made `captured_graph_id` a per-segment quantity; the tree-attention
work observer still treats it as one-per-step.** It is an 11th integration site,
beyond the 10 enumerated in §9.2 and the 2 the tests found — and it is a site
§11.4's CPU validation **structurally could not reach**: that harness `exec`s the
drafter blob and drives its state machine, while `_fr13_fixed32_observed_tree_attn`
is called from vLLM's tree-attention backend inside a real CUDA-graph capture.
There is no CPU path to it. The interlock did its job: it refused a malformed
binding rather than serving one.

## 4.3 Attribution: the split-graph alone, not the lever combination

A second boot with `FR14_SUFFIX_PASS_GATE=1` and **`FR14_FUSED_DRAFT_TOPK=0`**
(`output/fr14_promoab_Giso_20260818T074147Z`,
`promotion_ab_arm_g_iso_container_tail.log`) produced the **same refusal at the
same site**, with `passes: 2, segment: 1, capturing: True, graph_captures: 2` and
— decisively — the **same segment-0 signature `2da8c56a…`** while the graph
*addresses* differed between boots.

So: the refusal is the split-graph integration alone; lever 1 is exonerated; and
the half-graph signature is deterministic across processes. Run because "a
mechanism that EXPLAINS a failure is not evidence that it CAUSED it" (pass 30) —
six minutes of GPU instead of an argument.

**Nothing was relaxed and no retry was attempted**, per the fail-closed rule.

---

# 5. Recommendations, per lever

## Lever 1 — `FR14_FUSED_DRAFT_TOPK` (fused K0 draft top-k): **MORE EVIDENCE**

What this campaign added, and it is real: the kernel **served a full 4-task
production-shape serve for the first time**, engaged on every head read
(`stock_argmax_topk=0`), across 53 446 steps, with **zero** work-shape drift — 27
active nodes, 32 verify rows, 4 MTP forwards, 1 graph replay, and the banked
drafter graph signature `d9a4dd…6150c` unchanged on every step. Traces are clean
(§3.4). Nothing about it destabilised a 3 h 43 m serve.

What is still missing is the **only thing that could promote it**: its
pre-registered instrument is the **paired dfwd span** against an otherwise
identical top-k-OFF arm, because `fused_draft_topk.md` §8 prices the whole lever
at **0.3078 ms against a 49–53 ms span** — far too small for `step_wall`, let
alone TPS, to resolve. This campaign held it ON in both arms by design (to keep
C-vs-G to one variable), so it has no twin. **One paired serve, top-k 0-vs-1,
`FR13_DFWD_SPLIT=1` both sides, is the whole remaining ask** — and it can ride
whatever serve re-tests lever 2.

## Lever 2 — `FR14_SUFFIX_PASS_GATE` (suffix-aware MTP pass gating): **REFUSE (for now) — fix the 11th site first**

Not a verdict on the lever's economics, which remain untested: **the integration
does not run.** The first armed boot fails closed during half-graph capture on a
tree-attention work binding that the split made per-segment and the observer did
not (§4.2), reproducibly, with lever 1 out of the picture (§4.3).

Required before it is servable again:
1. make `_fr13_fixed32_observed_tree_attn`'s `captured_graph_id` binding
   segment-aware (or re-point it as each segment's capture begins);
2. add a test that reaches it — the §11.4 CPU harness structurally cannot, so this
   needs either a capture-path fake or an explicit "first armed boot" smoke arm;
3. then re-run §11.7 in full: the four checks that were **NOT REACHED** are still
   unproven, and the warm-step rate (pre-registered 0.15–0.25) and the
   break-even question are still entirely open.

Encouraging, and worth carrying forward: ARM C's independently measured
unconditional MTP survivals at the two positions the lever trades away —
**m3 = 0.783, m4 = 0.794** — come in *below* the 0.808 / 0.817 the lever's
economics assumed, and well below the 0.931 break-even. That does not answer the
conditional-on-strong-match question, but it moves no evidence against the lever.

## Lever 3 — split-K FA2 (`gqa_pair_splitk`): **REFUSE to promote; the eyeball condition CANNOT be discharged from this HEAD**

Mark's condition is not "not yet done" — it is **not reachable**, for three
independent reasons (§2.1 launcher mirror gap, §2.2 reduction-topology refusal,
§2.3 the live route never returns candidate output). No split-K token can be
generated at this HEAD, so no trace can be read, so the condition cannot be met.

`splitk_fa2.md` §10 should be corrected: the plumbing did **not** remove the
reason the probe could not be run.

Two separable asks:
* **a defect to fix** (§2.1): the split-K identity exists in the launcher's bash
  pin case and in the patcher's arms table but **not** in
  `fr13_fixed32_contract.py` nor in the launcher's in-container Python
  qualification map, which silently falls through to `split2`'s pins. Cheap, and
  it blocks the arm before anything interesting happens;
* **a decision for Mark** (§2.3): a *serving* route for a Tier-B arm that the
  raw-byte gate structurally cannot qualify. Until one exists, the offline
  evidence (2× speedup, deterministic, closer to exact than the served kernel)
  stands and the eyeball stays open.

## Not a lever, but the highest-priority item here: **§0.4, the stale census mirror**

It blocks **every** fixed32 serve's credential at this HEAD, including any gate
re-earn. One constant, `fr13_fixed32_work_census.py:157`, against the value the
emitter already proves on every boot. It is credential territory, so it is flagged
rather than touched — but nothing else in this repo can mint a fixed32 credential
until someone does.

---

# 6. Campaign discipline, stated for the record

* **No commit was taken between the first boot and the last drain.** The brief
  suggested committing between arms; HEAD-bound credentials make that
  self-defeating (a commit invalidates the credential the next arm needs), so
  everything is committed once at the end, with a `results/` pathspec on both
  `add` and `commit`.
* **docker-empty verified before every boot; zero containers after every drain.**
  Two arms (G, G-iso) left one exited container each after their engine died —
  the teardown's attestation path cannot remove a container whose engine ledger
  never materialised — and each was removed explicitly before the next boot. No
  foreign container appeared at any point.
* **Timebox:** ARM C ran 3 h 43 m against the briefed 3 h soft cap. It was allowed
  to finish rather than be killed: the declared 9000 s/task budget already bounded
  it deterministically, and killing it mid-task would have destroyed the whole
  arm's census and spans to save 43 minutes. Recorded as an overrun, not hidden.
* **Nothing was relaxed.** Two blocking defects (§0.4, §2.1) were attributed and
  reported, not patched, even though patching either would have unblocked this
  campaign's own measurements.

---
---

# ROUND 2 (2026-08-18 08:05Z–15:05Z) — the re-run on the fixed HEAD

Coordinator fixed both round-1 blockers and asked for a four-window re-run.
HEAD throughout: **`f7fde8e1b455c4baafc477e3699ad69e59e3265c`** (pass 56). Same
discipline: **no commit between the first boot and the final drain**, budgets
9000, identical canonical exact4 set, `FR13_DFWD_SPLIT=1` everywhere.

| window | outcome |
|---|---|
| **1. gate re-earn** | **PASS, rc=0, 12.8 min** — the census fix verified end-to-end |
| **2. ARM C'** gqa_pair + top-k **ON** | drained 4/4 tasks, 52 507 steps; `swerc=13` (13398 capped at the 9000 s budget) |
| **3. ARM G'** + suffix gate | **FAIL-CLOSED AGAIN — at a NEW, 12th site** |
| **4. ARM C''** gqa_pair + top-k **OFF** | drained clean, `swerc=0`, 4/4 tasks, 43 708 steps |

## R1. The census fix is verified end-to-end

The gate that failed in round 1 now passes at HEAD:

```
status PASS · schema fr13.fixed32.fa2_qrow32_gqa_pair_full_vocab_b1_live_verification.v1
source_commit f7fde8e1b455c4baafc477e3699ad69e59e3265c · qualification_profile full_vocab
credential pointer written: output/fr13_b1_gqa_pair_credential.env
```

That is the strongest possible check on the twelve-mirror sweep: the failure mode
was *only* reachable through a live serve's terminal audit, and the live serve's
terminal audit now completes. **Both C' and C'' then ran with
`FR13_FA2_QROW32_B1_PRODUCTION_ARM=gqa_pair` armed from that credential** — the
production kernel, which round 1 could not reach.

## R2. THE FUSED TOP-K VERDICT — the 0-vs-1 span pair

Instrument, pre-registered in `fused_draft_topk.md` §8.2: *"a stack-level dfwd
delta of 0.3078 ms is small against a 49–53 ms span, so the serve A/B needs the
span timer, not the step total."* The bracket the lever moves is the drafter
split's **`lmhead`** term. `promotion_ab_fused_topk_verdict.json`:

| term | top-k ON (C') | top-k OFF (C'') | delta | delta % |
|---|---|---|---|---|
| **`lmhead` ms** (the lever's bracket) | **3.47499** | **3.54597** | **−0.07099** | **−2.00 %** |
| `model` ms (null control) | 6.59517 | 6.57394 | +0.02123 | +0.32 % |
| `cfwd` ms/step (null control) | 20.46451 | 20.47216 | −0.00765 | −0.04 % |
| `dfwd` ms/step (containing bracket) | 55.44677 | 55.13937 | +0.30740 | +0.56 % |
| `step_wall_ms` | 214.75872 | 214.75191 | +0.00681 | **+0.00 %** |

**Read it in this order.**

1. **The two null controls are flat** (+0.32 %, −0.04 %). The MTP model forward
   and the committer are terms the lever cannot touch, and they did not move.
   That is the internal control that makes the third line worth reading.
2. **The targeted bracket moved, in the predicted direction, by −0.071 ms/step
   (−2.00 %).** The lever works in a live serve.
3. **It is 23 % of the −0.3078 ms predicted.** The campaign's oldest pattern
   holds again: *every miss is optimistic*. This is now the tenth.
4. **`step_wall` moved +0.003 % — indistinguishable**, and the *containing* dfwd
   bracket moved +0.56 %, i.e. the **wrong way by four times the lever's size**.
   That is not a contradiction; it is the pre-registration being vindicated. A
   0.07 ms effect cannot be seen in a 55 ms bracket, let alone a 215 ms step, and
   anyone reading `dfwd` totals or TPS here would have "measured" a regression.

**Acceptance — the byte-exactness claim, kept falsifiable.** Paired on the
identical task set: **3.88547 (ON) vs 3.90137 (OFF) = −0.41 %**, far inside the
±10 % band (`seam_move_economics` §9.4: the same arm banked 3.81 / 4.04 / 4.28).
`per_request_decode_tps` −5.54 %, also inside the band and also not a reading.

**Work shape — the strongest single result of the round.** The C'-vs-C'' census
diff over **96 215 steps** (`promotion_ab_workshape_topk.json`):

```
identical counter paths: 268
expected-different paths:   0
UNEXPECTED-different:       0
```

**Zero.** Not one of 268 per-step counters differs between top-k ON and top-k
OFF. Byte-exact selection, asserted offline over 6 840 configurations, now holds
across two full production-shape serves. Both arms also carry `active_nodes` 27,
`verify_rows` 32, `mtp_forward_calls` 4, `graph_replays` 1 and drafter signature
`d9a4dd…6150c` on **every** step.

### Verdict on lever 1: **PROMOTE-ELIGIBLE on the evidence; my recommendation is PROMOTE**

Byte-exact by gate and now by a 268-counter live census diff; acceptance-neutral
inside variance; the targeted bracket improves −0.071 ms/step with both null
controls flat; two clean multi-hour serves with clean traces. The honest caveat
to carry into any promotion note is that **the win is 23 % of what was briefed**
— −0.071 ms/step, roughly 0.03 % of a 215 ms step. It is real, it is safe, and it
is very small. Promote it for correctness-preserving hygiene (one launch instead
of a multi-kernel ATen radix chain), not for the number.

## R3. ARM G' — fail-closed AGAIN, at a NEW 12th site

`output/fr14_promoab_Gp_20260818T115449Z`, 11:54:49Z → 12:00:55Z, zero census
events. Tail: `promotion_ab_arm_g_prime_container_tail.log`.

**The 11th site is genuinely fixed** — the boot got *past* the tree-attention work
drift that killed round 1, and further into the step. It then refused at:

```
File ".../mamba/gdn_linear_attn.py", line 6689, in _fr13_fixed32_drafter_proposal_end
    raise RuntimeError(
RuntimeError: FR13 fixed32 drafter proposal evidence drifted
→ FR13 fixed32 prior sample failed: sample raised before fixed32 proposal seal
```

**Attribution, from the deployed source** (`fr10_phase4_patch_vllm_tree_gdn.py`,
the `_fr13_fixed32_drafter_proposal_end` blob). The function was made pass-aware
in its **census half** — §11.5's work is visible at :7034-7048, which reads
`_fr14_calls = int(proposal["mtp_forward_calls"])` and
`int(arctic.get("main_tail_columns", 6))` dynamically. Twenty lines later its
**runtime-evidence half** is still hardcoded to the single-4-pass-graph world:

```python
observed["drafter_runtime"] = {
    …
    "graph_replays": 1,          # :7074
    "mtp_forward_calls": 4,      # :7076
}
…
if (… or int(evidence.get("matching_replays", -1)) != 1 …):   # :7114
    raise RuntimeError("FR13 fixed32 drafter proposal evidence drifted")
```

§11.1 states the armed shape plainly: *"an ungated armed step replays `lo` then
`hi` (4 forwards, **2 replays**)"*. So the first armed **ungated** step reports
`matching_replays = 2`, the check demands exactly 1, and the engine dies. Since
the gate cannot fire until 256 tokens of history exist (`min_history=256`), every
early step is ungated — which is why it died ~5 minutes in, on the first real
request.

**This is a 12th site, and it is inside the very function §11.5 says was
updated.** One half of a paired structure made pass-aware, its mirror left
4-pass-hardcoded.

**That is now three consecutive findings of the same shape:** the TAW census
mirror (round 1 §0.4, which turned out to be twelve mirrors), the launcher's
in-container qualification mirror (round 1 §2.1), and now `proposal_end`'s
runtime-evidence mirror. **The recurring defect in this integration is not any
one site — it is that paired structures are being updated on one side only.** A
targeted sweep for "expected-value dicts that hardcode `graph_replays`/
`mtp_forward_calls`/pass counts anywhere in the drafter blob" is likely to find
the 13th before the next boot does.

No GPU was spent on an isolation twin this time: the mechanism is legible in the
deployed source, the raise site is exact, and the quantity in dispute
(`matching_replays` 2-vs-1) has nothing to do with lever 1 — which round 1
already exonerated on a dedicated boot.

### §11.7 checks reached this round

| check | result |
|---|---|
| gate armed line printed once | **PASS** (`armed ngram=8 min_agree=0.75 min_history=256`) |
| `/logs/fr14_suffix_pass_gate.cfg` written | **PASS** (`8 0.75 256`) |
| tree-attention work binding (round 1's 11th site) | **PASS — fixed, boot proceeded past it** |
| registry two rows `passes=2` segment 0/1 | **NOT REACHED** |
| `graph_replays` 2 cold / 1 gated | **NOT REACHED** (refused *on* the 2-replay step) |
| `mtp_forward_calls` only {4,2} | **NOT REACHED** |
| 27/32 every step · warm-step rate 0.15–0.25 | **NOT REACHED** — zero steps |

### G' acceptance vs C' under the ±10 % doctrine

**There is none, and there cannot be.** ARM G' produced **zero decode steps** —
no drafts, no accepted tokens, no census events, no `/metrics` bracket. The
±10 % doctrine needs two paired populations; this round has one. Any number
placed against C' here would be fabricated, so none is.

### Verdict on lever 2: **REFUSE — unchanged, and for a new reason**

The economics are still untested. Required, in order: fix the 12th site; sweep
the drafter blob for the same one-sided-mirror class rather than waiting for the
13th boot to find it; add a test that reaches `proposal_end`'s runtime-evidence
half with a 2-replay step; **then** re-run §11.7 in full, and only then is the
break-even question (MTP survival at positions 3–4 on strong-match steps vs
0.931) reachable.

## R4. ARM C' and C'' — the arms themselves

| | ARM C' (top-k ON) | ARM C'' (top-k OFF) |
|---|---|---|
| runroot | `fr14_promoab_Cp1_20260818T081918Z` | `fr14_promoab_Cp0_20260818T120217Z` |
| serve rc | 13 (13398 capped at 9000 s) | **0** |
| tasks in health record | 3 of 4 (+1 capped) | **4 of 4** |
| census steps | 52 507 | 43 708 |
| `step_wall_ms` | 214.759 *(ungated reduce)* | 214.752 *(census-gated)* |
| `s_per_fwd_gpu` | 0.131262 | 0.131601 |
| accept/event | 3.8855 | 3.9014 |
| eyeball | clean | clean |

**ARM C' is `swerc=13` and the reason is worth recording:** `13398` exceeded even
the 9000 s budget (round 1 it took 5 784 s; C'' completed it in 8 920 s — a
1.5× spread on one instance across three runs). Its capped terminal then hit the
same never-built truncated-trace validator union that pass 23 met at 5400, so the
task dropped out of the health record and the census-gated reduce fired class-9
(bracket 52 500 steps vs census 52 507 — a 7-event gap from the truncated
bracket). **C' was therefore reduced ungated**, which is disclosed everywhere it
is quoted.

None of that touches the round's deliverable: **the fused-top-k verdict is read
from the dfwd span sidecars, which are cumulative per-step GPU timers and are
completely independent of the bracket reduction.** That is why §R2 stands at full
strength while C's aggregate carries a caveat.

**Budget note, now measured three times:** 13398 ran 5 784 s / >9 000 s / 8 920 s.
The briefed-5400 correction from round 1 was right, and 9000 is itself not
comfortably above this instance's spread. Anyone scheduling this arm shape should
either accept an occasional capped terminal or finally build the truncated-trace
union.

## R5. EYEBALL — both arms clean, including the capped trajectory

`promotion_ab_eyeball_arm_c_prime.json`, `..._c_dprime.json`.

| arm | instance | turns | words | ttr | max line | tail-rep | non-ASCII | tools | malformed |
|---|---|---|---|---|---|---|---|---|---|
| C' | 12907 | 38 | 2 580 | 0.375 | 6 | 0.000 | 0 | 14 | **0** |
| C' | 13033 | 62 | 11 585 | 0.189 | 52 | 0.317 | 0 | 25 | **0** |
| C' | 13236 | 137 | 9 599 | 0.219 | 43 | 0.217 | 0 | 51 | **0** |
| C' | 13398 *(capped)* | 312 | 46 104 | 0.178 | 81 | 0.097 | 0 | 113 | **0** |
| C'' | 12907 | 47 | 5 936 | 0.303 | 32 | 0.377 | 0 | 16 | **0** |
| C'' | 13033 | 32 | 2 618 | 0.275 | 10 | 0.304 | 0 | 11 | **0** |
| C'' | 13236 | 74 | 2 066 | 0.244 | 2 | 0.000 | 0 | 24 | **0** |
| C'' | 13398 | 272 | 60 325 | 0.136 | 88 | 0.059 | 0 | 101 | **0** |

**355 tool calls across the two arms, zero malformed. Zero non-ASCII in ~140 000
words. No degeneration signature in either arm.**

The one trace that most needed reading is **C's capped 13398** — a task killed at
a 9000 s wall is exactly where a decode loop would hide. It is not one. Its
tail-repeat fraction is **0.097, the lowest of C's four traces**, and the final
text before the kill is coherent, methodical debugging:

> Reproduced under pytest: `viaitrs` is wrong but inputs (SUN ITRS, LOC ITRS) are
> identical to the passing plain-Python case. The transform body itself must
> behave differently. Let me instrument the exact transform steps in the diag:

Its top repeated 8-gram is a rotation-matrix expression (`'φ cos λ, -sin φ sin λ,
cos'`) from the coordinate transform under test. **The cap fired on genuine
difficulty, not on degeneration** — which is the distinction that decides whether
a capped terminal is an instrument problem or a model problem. It is an
instrument problem.

## R6. Consolidated recommendations after two rounds

| lever | verdict | why |
|---|---|---|
| **`FR14_FUSED_DRAFT_TOPK`** | **PROMOTE** | −0.071 ms/step on its own pre-registered bracket with both null controls flat; 268/268 census counters identical over 96 215 steps; accept −0.41 % (inside ±10 %); clean traces in two multi-hour serves. Promote for hygiene, and state plainly that the win is 23 % of briefed and ~0.03 % of a step. |
| **`FR14_SUFFIX_PASS_GATE`** | **REFUSE** | Second consecutive fail-closed first boot, now at a 12th site inside the function §11.5 says was updated. Economics still untested; break-even question still unreachable. |
| **split-K FA2** | **REFUSE** | Unchanged from round 1: the eyeball condition is not dischargeable — the arm cannot serve a token. Needs a launcher/contract identity fix *and* a Tier-B serving policy decision from Mark. |

**Process finding for the integration owner, offered as the most useful thing
this campaign produced:** three separate blockers in two rounds were all the same
shape — a paired structure updated on one side only (TAW digest × 12 mirrors;
launcher bash pin case vs its in-container Python twin; `proposal_end`'s census
half vs its runtime-evidence half). Two of the three were reachable *only* from a
live serve. The class deserves a sweep and a lint, not three more boots.

---
---

# ROUND 3 (2026-08-18 15:17Z–15:40Z) — ARM G' at the swept HEAD: a 14th site, and the gate FIRED

HEAD: **`2925119b731498a954df73721e43fea06799a59a`** (pass 58). Two windows, same
discipline. The coordinator's honest bound — *"a 14th site in an unenumerated
structure would still reach the boot"* — is exactly what happened.

| window | outcome |
|---|---|
| **1. gate re-earn** | **PASS**, rc=0, **14.2 min** (15:17:43Z → 15:31:53Z) |
| **2. ARM G'** gqa_pair + top-k + gate | **FAIL-CLOSED at a 14th site**, 0 census events, 6 min |

## R3.1 The boot verdict — §11.7, and the milestone in it

`output/fr14_promoab_Gp3_20260818T153209Z`, boot 15:32:11Z, refusal 15:38:31Z.
Tail: `promotion_ab_arm_g_round3_container_tail.log`.
Arm attested in `container_env.txt`: `FR14_SUFFIX_PASS_GATE=1`,
`FR14_FUSED_DRAFT_TOPK=1`, `FR13_FA2_QROW32_B1_PRODUCTION_ARM=gqa_pair`,
`FR13_DFWD_SPLIT=1`.

| §11.7 check | round 2 | **round 3** |
|---|---|---|
| gate armed line printed once | PASS | **PASS** |
| `/logs/fr14_suffix_pass_gate.cfg` written | PASS | **PASS** (`8 0.75 256`) |
| 11th site — tree-attention work binding | PASS (fixed) | **PASS** |
| 12th site — `proposal_end` runtime evidence | **REFUSED** | **PASS — survived every armed UNGATED step** |
| **the gate actually firing** | not reached | **REACHED — see below** |
| registry two rows `passes=2` segment 0/1 | not reached | not reached |
| `graph_replays` 2 cold / 1 gated | not reached | not reached |
| `mtp_forward_calls` only {4,2} | not reached | not reached |
| 27/32 every step · warm rate 0.15–0.25 | not reached | not reached |

**The milestone, and it is a real one: the suffix pass gate fired for the first
time in a live serve.** Round 2 died on the first *ungated* armed step. Round 3
ran ungated armed steps successfully — which is the live proof that the 12th-site
fix works — and then refused on the first step **where the gate itself fired**.
Getting there means the gate accumulated its 256 tokens of history, found a
recurring 8-gram, measured continuation agreement ≥ 0.75, and decided to hand off
early. The predicate works on real traffic. Nothing downstream of it does yet.

## R3.2 The 14th site, attributed

```
File ".../vllm/v1/spec_decode/eagle.py", line 5924, in propose
    raise RuntimeError(
RuntimeError: FR13 fixed32 drafter work is not 15 native + 6 tail + 10 rescue = 31
```

Source, `fr10_phase4_patch_vllm_tree_gdn.py:30941-30949`:

```python
if (15 + len(_fr13_t_cols) + len(_fr13_t_paths) != 31):
    raise RuntimeError(
        "FR13 fixed32 drafter work is not 15 native + 6 tail + 10 rescue = 31"
    )
```

`_fr13_t_cols` is the Arctic **main tail** and `_fr13_t_paths` the **rescue**
paths. §11.5 is explicit that a gated step lengthens the main tail from 6 to 8
(*"a gated step hands off at draft position 3, so Arctic's main chain is 8 long
instead of 6"*), and that change **has** landed — it is read dynamically two
hundred lines away as `int(arctic.get("main_tail_columns", 6))`. But the
**`15`** in this invariant is the *native* (MTP-produced) column count for the
**ungated 5-pass** shape, and it is a bare literal. So on the first gated step:

```
15 (native, hardcoded for 5 passes) + 8 (tail, correctly grown) + 10 (rescue) = 33 ≠ 31
```

and the drafter refuses. One side of the sum was made gate-aware; the other was
not.

## R3.3 Why the sweep did not catch it — the defect class needs widening

This is the **fourth consecutive blocker of the same family**, and it is the one
that says the most, because pass 58's sweep was already looking for the family
and reported **9 pairs / 0 stale**. It missed this because the sweep's notion of
the defect is a **mirrored structure** — two dicts, two constants, two halves of a
credential — and this is not one. It is a **single arithmetic invariant with the
ungated shape baked in as an addend**:

> `15 + tail + rescue == 31`

There is no mirror to compare against. The stale value is an operand inside a
sum, and the only thing that makes it wrong is a semantic fact about what `15`
means under a 3-pass drafter.

**Recommended refinement of the class, offered as the actionable finding of this
round:** the invariant is not *"paired structures must agree"* but *"**no literal
may encode the ungated 5-pass shape**"* — including addends inside invariants,
default arguments (`arctic.get(..., 6)` is a benign instance, but only because
its call sites now always pass a value), comparison constants, and assertion
messages. The candidate set is enumerable by grepping the drafter blob for the
shape constants **15, 6, 10, 31, 5, 4, 2** in arithmetic or comparison position
and asking of each: *does this number change when the drafter runs 3 post-root
passes instead of 5?* The existing lint tests structure equality; this needs a
lint on **shape literals**.

The four blockers in one line each:

| # | site | shape |
|---|---|---|
| 11 | `_fr13_fixed32_observed_tree_attn` `captured_graph_id` | per-step binding vs per-segment reality |
| 12 | `proposal_end` runtime-evidence half | census half pass-aware, evidence half hardcoded `4 / 1` |
| 13 | `observed_build_record` (found by sweep, no boot) | mirrored structure, one side stale |
| **14** | **drafter work invariant `15 + tail + rescue == 31`** | **ungated shape as a literal addend inside a sum** |

Three of the four were reachable **only** from a live serve.

## R3.4 The pairing against round-2 C' — not performed, and why

The coordinator asked for G' paired against round-2 ARM C' (52 507 steps) on
step_wall / dfwd / accept under the ±10 % doctrine, with a note on how to account
C's capped 13398.

**No pairing is reported, because ARM G' produced zero decode steps** — no census
events, no `/metrics` post-brackets (`post-brackets=0`, deploy-speed VACUOUS), no
drafts, no accepted tokens. The ±10 % doctrine compares two populations; this
round produced one.

For the record, the accounting I *would* have used had G' drained, so the next
attempt does not have to re-derive it:

* **Arm level:** C' must be quoted as its **ungated** reduce (class-9 fired on a
  7-event bracket/census gap created by 13398's truncated capped bracket), and
  that caveat travels with every number taken from it.
* **Per-task matched basis:** pair only on instances that ran to a *complete*
  terminal in **both** arms. In C' that is `12907`, `13033`, `13236`; `13398` is
  capped and its bracket truncated, so it is excluded from the paired reduction
  and reported separately — the same matched-basis method pass 23 used for its
  three-task sighting.
* **dfwd under the gate:** the sidecar is a cumulative per-step timer and cannot
  be split by gated/cold directly, so the pre-registered *"−20.6 ms on gated
  steps"* is recovered arithmetically from the arm mean and the census warm-step
  rate `w`: `saving_per_gated_step = (dfwd_C' − dfwd_G') / w`, with `w` read as
  the census fraction of steps at `mtp_forward_calls == 2`. Both terms come from
  instruments this campaign already collects.

## R3.5 Verdict — lever 2 unchanged: **REFUSE**, with the ledger moved forward

Three armed boots, three fail-closed refusals, three distinct sites — 11th, 12th,
14th — each further into the step than the last. The integration is converging,
and the interlocks are doing exactly what they were built to do: every one of
these refused a malformed drafter rather than serving one.

What is now **proven live**: the gate arms, writes its sidecar, survives armed
ungated steps under the fixed 11th and 12th sites, and **fires on real traffic**.
What is still **entirely unmeasured**: every quantity §10.1 pre-registered — warm
rate, the −20.6 ms gated dfwd, the −4.0 ms step_wall, the accept delta — and
therefore the one question the serve exists to answer, whether MTP survival at
draft positions 3–4 on strong-match steps is below 0.931.

Before a fourth boot: fix the 14th site, then run the **widened** sweep of
§R3.3 — shape literals, not just mirrored structures — because the current
evidence is that one more boot buys one more site, and each boot costs a GPU
window plus a gate re-earn. Three of four sites were live-only, so the sweep is
the cheaper instrument by a wide margin.

---
---

# ROUND 4 (2026-08-18 15:53Z–16:19Z) — the gate SERVES A WHOLE TASK, then a 15th site at the flush

HEAD: **`7ac3bde9bd330d5db005ad083a6bc4561fa5a3f2`** (pass 60).

| window | outcome |
|---|---|
| **1. gate re-earn** | **PASS**, rc=0, **12.6 min** |
| **2. ARM G'** | **served task 12907 to completion**, then **FAIL-CLOSED at the task-boundary flush** |

## R4.1 First: the coordinator's status question — the driver was DEAD, not slow

The serve looked alive at 16:50Z because a container was up. It was an **orphan
held open by policy, not a running serve**:

```
[hydra27_fixed32_promoab_Gp4] serve rc=1 2026-08-18T16:19:01Z
FAIL: fixed32 terminal flush rc=2
fixed32 exact container preserved after engine-ledger materialization failure: 26954c65…
```

The driver exited at **16:19:01Z** — 12.5 min after boot — and the teardown
*deliberately preserved* the container because the engine ledger never
materialised. Nothing was hung: no orchestrator, no eval, no submitter was
running (`ps` clean), and `swe orchestrator rc=1 wall=430s`. The second task
directory never appeared because the campaign had already ended. Container torn
down explicitly; host memory returned 94 GiB → 12 GiB used, zero containers.

## R4.2 How far it got — the furthest yet, by a wide margin

| §11.7 check | R2 | R3 | **R4** |
|---|---|---|---|
| gate armed line + cfg sidecar | PASS | PASS | **PASS** |
| 11th site (tree-attn binding) | PASS | PASS | **PASS** |
| 12th site (`proposal_end` evidence) | REFUSED | PASS | **PASS** |
| 14th site (drafter work invariant) | — | REFUSED | **PASS** |
| gate fires on real traffic | — | reached, then refused | **PASS — sustained** |
| **a whole SWE task served under the armed gate** | — | — | **PASS — 12907 completed, 1 604 draft events** |
| task-boundary flush | — | — | **REFUSED (15th site)** |
| census / registry / replays / warm rate | — | — | **NOT MATERIALISED** (the flush is what writes them) |

Round 3 died ~6 min in on its first gated step. **Round 4 ran gated steps for
430 seconds of real agent traffic and finished the task** — `12907` completed and
finalised at 16:18:53 with the gate armed throughout. Everything the drafter does
per-step now survives under the gate. What does not survive is the **boundary**.

## R4.3 The 15th site — validator-side, at the flush

```
[FR13_FIXED32_FLUSH] failed generation 2:
RuntimeError("fixed32 drafter replay evidence did not attest event: …")
→ Fixed32BoundaryError: fixed32 post bracket failed:
  snapshot=FlushRuntimeError: runtime returned non-ok flush status 'error:RuntimeError'
```

The refused event, from the payload — an armed but **ungated** step, i.e. §11.1's
legal shape:

```
mtp_forward_calls: 4 · main_tail_length: 6 · graph_replays: 2 · matching_replays: 2
active_nodes: 27 · verify_rows: 32 · graph_signature: 7fa7d56d…a031c6
```

Raise site, `fr10_phase4_patch_vllm_tree_gdn.py:39694`, inside the `fixed_flush`
blob's `_fr13_f32_flush_reconcile()`:

```python
or evidence.get("matching_replays") != 1
```

An armed ungated step replays `lo` then `hi` — **2** replays — so the flush-time
attestation refuses a legal step. This runs **once per task boundary**, not per
step, which is exactly why the arm served 430 s of gated traffic before dying.

## R4.4 Why the sweep did not catch it — verified in the tooling, not guessed

I read pass 60's sweep rather than speculate. Two facts:

**(a) Enumeration is fine.** `all_injected_blobs()` returns **65** blobs and the
`fixed_flush` blob (patcher line 39286) is among them and parses. The blob-hole
that hid the 14th site is genuinely closed.

**(b) The candidate set has a missing dimension.** `shape_literal_scan()` reports
`41 blobs scanned (24 unparseable, textually checked), 18 literals adjudicated,
0 unreviewed` — and its candidates are

```python
magnitudes = {15, 6, 10, 5, 4, 12, 16, 8, 14, 18}
```

Every one of the 18 adjudicated literals is a **column/pass** magnitude
(`4`, `6`, `10`, `16`, `5`, `3`, `2`). The split graph moves **three** independent
dimensions:

| dimension | change under the gate | in the scan? |
|---|---|---|
| columns | 15 native / 6 tail / 10 rescue → redistributed | **yes** |
| passes | 5 → 3 (post-root 4 → 2) | partly (`4`, `5` present) |
| **replays** | **1 → 2** | **NO — neither `1` nor `2` is a magnitude** |

The 15th site's stale literal is the number **`1`**. It was scanned and could not
be flagged, because `1` is not in the candidate set. The scan was right to report
`0 unreviewed`; its universe simply excluded the value the gate most directly
changes.

## R4.5 I ran the missing dimension — 38 candidates, banked

Rather than recommend the scan again, I wrote and ran it:
`promotion_ab_replay_literal_scan.py` → `promotion_ab_replay_literal_scan.json`.
Every literal `1` or `2` in a **replay-counting position** across all 65 blobs:

```
blobs scanned: 65
replay-position 1/2 literals: 40 (38 for review)
  OK     blob@39286+379  or evidence.get("matching_replays") != 1
  STALE! blob@39286+409  or evidence.get("matching_replays") != 1
```

**The two lines are character-identical and have opposite verdicts.** At blob line
379 the check guards the **forward** graph, which really is replayed once per step
whatever the drafter does — correct, must not change. At line 409 it guards the
**drafter** — the 15th site. The other 38 are a short review list (a `1` beside
"replay" is often right, which is why the value cannot simply be banned).

**A finding about my own tool, reported because it is the sharpest evidence of the
class.** The first draft of that scan keyed its allowlist on the literal's *text*
— and immediately marked **both** twins OK, hiding the very site it was written to
expose. I had reproduced the defect, in a tool built to catch the defect, within
ten minutes of describing the defect. It is now keyed on `(blob, line)`. The
lesson generalises to the campaign's own adjudication list: **adjudicate
positions, never literals** — otherwise one correct instance vouches for a stale
twin.

## R4.6 The first acceptance evidence for lever 2 — a matched-basis SIGHTING

The census never materialised (the flush is what writes it), so warm-step rate,
registry rows and per-step replay counts are **unmeasured**. But the arm left
`metrics_before_swe.txt` / `metrics_after_swe.txt`, and its whole serve was task
`12907` — which has a matched twin in round-2 C'. Same instance, same
`/metrics` counter basis, both arms:

| | ARM G' (gate **ON**) | ARM C' (gate **OFF**) | delta |
|---|---|---|---|
| draft events | 1 604 | 1 408 | +13.9 % |
| draft tokens | 49 724 | 43 648 | |
| **tok per draft** | **31.0** | **31.0** | **0 — pack width unchanged** |
| accepted tokens | 6 015 | 5 131 | |
| **accept / event** | **3.7500** | **3.6442** | **+0.1058 (+2.90 %)** |

Cumulative survival, same instance:

```
G' ON : 0.951 0.815 0.659 0.473 0.304 0.143 0.110 0.089 0.075 0.067 0.064
C' OFF: 0.930 0.755 0.575 0.442 0.340 0.165 0.125 0.101 0.085 0.066 0.059
```

conditional:

```
G' ON : m0 .951  m1 .857  m2 .808  m3 .718  m4 .643  m5 .469
C' OFF: m0 .930  m1 .811  m2 .762  m3 .769  m4 .769  m5 .484
```

**Read carefully, because this is a sighting and not a verdict.**

* **+0.106 tokens/event lands inside the pre-registered band** `[-0.02, +0.15]`
  from `suffix_pass_gating.md` §7 — the first live datum against that
  pre-registration, and it is favourable.
* **The pack width did not move**: 31.0 draft tokens per event in both arms. The
  gate is a fill-source change, not a shape change, confirmed on served traffic.
* The shape of the win is what the design predicted: G' survives **better through
  positions 0–3** and **worse at 4–5** (m3 .718 vs .769, m4 .643 vs .769), which
  is what trading MTP passes for the Arctic suffix chain looks like — and the net
  is positive because the gate fires where the suffix is strong.
* **±10 % doctrine: +2.90 % is INSIDE variance.** One instance, one run, and the
  trajectories differ by 13.9 % in draft events. Pass 23's word for this is the
  right one: a **matched-basis sighting**, not a citable result.
* C's ungated m3/m4 here are **0.769 / 0.769**, far below §7's **0.931**
  break-even, and consistent with round 2's arm-level 0.783 / 0.794. Three
  independent measurements now agree that the unconditional survivals sit well
  under break-even. The **conditional-on-strong-match** number — the actual
  question — still requires a census, and therefore still requires a drained arm.

## R4.7 Verdict — lever 2: **REFUSE**, and the first evidence that it may be worth it

Four armed boots, four fail-closed refusals at four distinct sites (11th, 12th,
14th, 15th), each strictly further into the run than the last: capture → first
ungated step → first gated step → **a complete task**. The interlocks have refused
a malformed drafter every single time and never served one.

What is now proven live: the gate arms, fires, sustains gated decoding across a
whole SWE task, leaves the pack width untouched, and — on a matched-basis sighting
— *gains* acceptance inside its pre-registered band.

What remains unmeasured: everything the census carries — warm-step rate vs the
pre-registered 0.15–0.25, the −20.6 ms gated `dfwd`, the −4.0 ms `step_wall`,
registry rows, per-step replay counts — because the flush that writes the census
is the thing that refuses.

**Before a fifth boot:** fix `:39694` (and only `:39694` — `:39664` is correct),
then work the 38-row replay-dimension list in
`promotion_ab_replay_literal_scan.json`, and re-key the campaign's adjudication
list on positions rather than literals. The boundary path is now the frontier:
every per-step interlock has been satisfied, so the remaining sites are
concentrated in flush/reconcile/materialisation code that only a task boundary
exercises — a much smaller surface than the per-step drafter, and one this scan
now covers.

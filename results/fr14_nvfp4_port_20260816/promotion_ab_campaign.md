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

---
---

# ROUND 5 (2026-08-18 17:09Z–17:19Z) — BLOCKED: the repo moved under the gate

**No arm was run. No GPU was spent beyond one gate window. Standing by.**

## R5.1 What happened

Gate re-earn launched at HEAD `a1aceef05`. **The serve itself passed** —
`ARM_DONE … swerc=0`, `astropy__astropy-12907` resolved, container torn down,
zero containers after. The gate runner then refused at its manifest interlock:

```
[regate] rc=14
runtime/source manifest changed during qrow32 gate
```

The manifest diff names the cause exactly — **three tracked source files changed
between the gate's launch and its end**:

| file | at launch | at end |
|---|---|---|
| `scripts/fr13_fixed32_contract.py` | 166 151 B `85a185d6…` | 168 732 B `9bab69d3…` |
| `scripts/fr13_patch_fa2_tree_bias.py` | 440 309 B `1e978032…` | 447 449 B `b5b715e1…` |
| `scripts/fr13_qrow32_b1_pass_sidecar.py` | 52 215 B `5404be89…` | 65 558 B `45094311…` |

All three are the FA2 / split-K provenance surface, and a commit landed on top of
mine while my gate was running:

```
9d294733b  FR14 lane 4 PRE-REGISTRATION: the Tier-B bounds, written before the gate runner exists
a1aceef05  (mine, the round-5 preparation)
```

At the time of writing the worktree is **still dirty** with those same three files
modified-but-uncommitted, and `fr13_patch_fa2_tree_bias.py` has grown again since
the gate ended (447 449 → 453 943 B). Lane 4 is mid-edit on the split-K serving
route Mark approved in pass 64.

## R5.2 Why this stops round 5 rather than being retried

**No credential was minted.** rc=14 fires before the verification JSON is written,
so `output/fr13_b1_gqa_pair_credential.env` still points at `7ac3bde9b` — two
HEADs stale. Without a serviceable credential the arm would serve the qrow16
incumbent, which is not the arm this round is for.

**And the gate cannot even be restarted right now**: its preflight requires a
clean tracked worktree, and the tree is not clean. That refusal is correct.

**The deeper reason is not the retry cost.** A HEAD-bound credential campaign and
a concurrent editor of tracked source are mutually exclusive by construction:

* the gate asserts the manifest is byte-equal at start and end — a concurrent
  editor makes that assertion fail **by definition**, however many times it is
  retried;
* worse is the case that does *not* fail loudly: if lane 4's edits land in the
  window **between** a successful gate and the arm's boot, the arm serves a stack
  whose source no longer matches the credential it presents. The interlock is
  what prevents that, and it just prevented it.

So the honest move is to **yield**, exactly as the standing rule says for a
foreign container on the GPU. The same principle applies to the source tree: this
campaign does not own it, and it must not race an owner who does.

**Nothing of lane 4's was touched** — not stashed, not reverted, not committed.
Only `results/` paths are committed here, by pathspec, as always.

## R5.3 What round 5 still needs, unchanged

The moment the tree is settled and lane 4's work is committed:

1. gate re-earn at whatever HEAD is then current (~13 min, four consecutive
   PASSes at this point);
2. ARM G' — gqa_pair + fused top-k **via the promoted default** (the round-5
   preparation in `a1aceef05` makes the runner name nothing, so it exercises the
   promoted launcher literals rather than this campaign's own);
3. §11.7 checklist at boot, then first task-boundary survival — which is the new
   frontier, since round 4 died there after a complete task and the 16th-site fix
   makes the reconciliation per-segment;
4. drain report with warm rate vs the pre-registered 0.15–0.25 and the paired
   accept / dfwd / step_wall against round-2 C', using the accounting already
   recorded in §R3.4.

**One coordination request, and it is the only thing blocking:** round 5 needs a
window in which no other lane edits tracked source between the gate's launch and
the arm's drain — roughly gate + 3.5 h. Everything else is ready and has been
rehearsed four times.

---
---

# ROUND 5, EXTENDED (2026-08-18 17:29Z–20:12Z) — four phases at frozen HEAD `71ab122d2`

The quiet-tree window held: the worktree was byte-clean at every phase boundary
and no other lane touched tracked source. Nothing was relaxed. The Tier-B
credential was earned to a **gitignored** path so the frozen tree stayed clean,
which is why no commit was needed mid-sequence.

| phase | outcome |
|---|---|
| **1** gqa_pair gate re-earn | **PASS** rc=0, 16.1 min |
| **2** ARM G' fifth boot | **DRAINED CLEAN** rc=0, 4/4 tasks, 32 656 steps — **first ever** |
| **3** Tier-B credential re-earn | **PASS**, 9/9 pre-registered bounds |
| **4** ARM S first Tier-B serve | **REFUSED at a 17th site**; no traces; eyeball **NOT discharged** |

**Headline: ARM G' finally drained — and its traces contain a degeneration
signature. §R5.4 is the most important section of this campaign.**

## R5.1 Phase 2 — ARM G' drained, and every §11.7 check is finally measurable

`output/fr14_promoab_Gp5_20260818T174541Z`, rc=0, wall 7 533 s, 4/4 tasks
(12907 resolved; 13033/13236/13398 failed). **First task boundary survived** —
the census materialised, which is exactly what refused in round 4.

| §11.7 check | result |
|---|---|
| gate armed line printed once | **PASS** |
| registry two rows `passes=2`, `segment` 0 and 1 | **PASS** |
| `graph_replays` 2 cold / 1 gated | **PASS — exactly**, `{2: 21 365, 1: 11 291}` |
| `mtp_forward_calls` only {4, 2}, no third value | **PASS**, `{4: 21 365, 2: 11 291}` |
| `(calls, tail)` pairs legal | **PASS**, only `(4,6)` and `(2,8)` |
| `active_nodes` / `verify_rows` 27 / 32 every step | **PASS**, all 32 656 |
| warm-step rate 0.15–0.25 | **FAIL — 0.3458** (see §R5.3) |
| ungated signature `d9a4dd…` present | **flagged by my reducer — FALSE ALARM** |

The last line is a defect in **my** checker, not the integration: with the gate
armed the drafter runs two half-graphs (`7fa7d56d…`, `2da8c56a…`) and never
instantiates the 4-pass graph, exactly as §11.1 designs. §11.7's "ungated-arm
signature unchanged" is a claim about the gate-**off** arm, which C' satisfies.
My reducer applied it to the armed arm. Recorded rather than quietly dropped.

**Work shape (C' vs G', 85 163 steps):** 256 identical counter paths, **12
expected-different** (all drafter pass/tail/replay/arctic fields), **0
unexpected**. The gate moves exactly what it is specified to move.

## R5.2 Phase 2 instruments — paired against round-2 C'

| instrument | C' (gate OFF) | G' (gate ON) | delta |
|---|---|---|---|
| **`step_wall_ms`** | 214.759 | **207.794** | **−6.964 (−3.24 %)** |
| **drafter GPU ms/step** | 55.447 | **47.501** | **−7.946 (−14.33 %)** |
| `s_per_fwd_gpu` | 0.13126 | 0.13211 | +0.65 % |
| committer GPU ms/step *(null)* | 20.465 | 20.577 | +0.55 % |
| `overhead_other` *(null)* | 7.585 | 7.602 | +0.23 % |
| accept / event | 3.8855 | 3.8537 | **−0.0318 (−0.82 %)** — inside ±10 % |
| per-request TPS | 24.392 | 25.014 | +2.55 % — inside ±10 % |

Both null controls are flat and the verifier forward is unchanged (+0.65 %), which
is right: the gate is drafter-side only. **The dfwd bracket carries the whole
effect.**

**The mechanism is confirmed at its pre-registered size.** With warm rate
`w = 0.3458`, the per-gated-step saving is

```
7.946 ms / 0.3458 = 22.98 ms per gated step     (pre-registered: -20.6 ms)
```

**The step_wall win is LARGER than the −4.0 ms pre-registration only because the
warm rate is inflated by a defect.** At the pre-registered `w = 0.195` the same
22.98 ms/gated step gives −4.48 ms — i.e. the pre-registration was right and the
headline is flattered. Reporting −3.24 % as "beats prediction" would have been
precisely the optimistic mis-read this campaign keeps catching.

## R5.3 The warm rate is not a rate — the gate LATCHES PER REQUEST

| | |
|---|---|
| requests in the arm | **98** |
| fully **gated** requests | 11 (11 291 steps) |
| fully **ungated** requests | 87 (21 365 steps) |
| **MIXED requests** | **0** |

**Not one request out of 98 has a mixed interior.** A per-step predicate on real
text would make mixed requests the overwhelming norm; zero of 98 is conclusive.
The gate decision is taken **once per request and never re-evaluated**.

§11.6 specifies a per-step decision ("the decision is taken before the forward"),
and §10.1's 0.15–0.25 comes from a **renewal-process** simulation that assumes
per-step re-evaluation. So the measured 0.3458 does **not** falsify the
pre-registration — it measures a different quantity: a length-weighted average of
which requests happened to latch. Per-task warm rates make that obvious:

```
12907  w = 0.500      13033  w = 0.328      13236  w = 0.993      13398  w = 0.107
```

## R5.4 EYEBALL — A DEGENERATION SIGNATURE, ON THE TASK THE GATE LATCHED ONTO

| arm | instance | turns | words | ttr | max line | **tail-rep** | tools | patch |
|---|---|---|---|---|---|---|---|---|
| G' | 12907 | 29 | 2 243 | 0.361 | 14 | 0.256 | 11 | 504 B |
| G' | 13033 | 53 | 15 000 | 0.172 | 60 | 0.451 | 22 | 2 297 B |
| **G'** | **13236** | **7** | **16 211** | **0.067** | **130** | **0.991** | **2** | **0 B** |
| G' | 13398 | 185 | 26 960 | 0.172 | 55 | 0.140 | 70 | 3 617 B |

`astropy__astropy-13236` in ARM G' is **degenerate**. One assistant block of
**117 739 characters**, in which the same 12-gram repeats **71 times**, two tool
calls in seven turns, and **no patch at all**.

Verbatim, from the middle of the loop:

> `. Let me verify against the actual astropy v5.1 source... I've seen the astropy
> 5.1 source before. In astropy 5.1's table.py, searching for "deprecated in" near
> NdarrayMixin:`
>
> `Actually, now I'm fairly confident. In astropy 5.1, the code is:`
>
> ```python
>         # Structured ndarray gets viewed as a mixin unless already a valid
>         # mixin class
>         if (not isinstance(data, Column) and not data_is_mixin
>                 and isinstance(data, np.ndarray) and len(data.dtype) > 1):
>             warnings.warn(
> ```

— that block, and that self-talk, cycling 71 times. The trace then ends on a
**truncated tool call**:

> `            warnings.warn`
> `</parameter>`
> `</function>`
> `</tool_call>`

**Note the instrument's own limitation, found by reading:** my table reports
`malformed=0` for this trace, because it counts tool calls the runner *parsed*.
This one never became a parsed `tool_use` block — it is raw text carrying
tool-call markup, cut off mid-token. The statistic missed it; the human read
caught it. That is the whole reason §3.4's doctrine says the signatures point at
where to look and never issue the verdict.

### The association, and how far it can honestly be pushed

**13236 is the task that ran at warm rate 0.993** — 6 318 of 6 361 steps gated,
i.e. latched ON for essentially its entire life. And the same instance is
**healthy in all three ungated arms**:

| arm on 13236 | gate | ttr | tail-rep | tools | patch |
|---|---|---|---|---|---|
| **G' (round 5)** | **ON, w=0.993** | **0.067** | **0.991** | **2** | **0 B** |
| C' (round 2) | off | 0.219 | 0.217 | 51 | 1 809 B |
| C'' (round 2) | off | 0.244 | 0.000 | 24 | 808 B |
| C (round 1) | off | 0.193 | 0.201 | 99 | 4 534 B |

Across G's four tasks, tail-repeat tracks warm rate:
`w=0.107 → 0.140`, `w=0.328 → 0.451`, `w=0.500 → 0.256`, `w=0.993 → 0.991`.

**There is also a mechanism, and it is a positive feedback loop.** The gate hands
off at draft position 3 to the **Arctic suffix chain**, which proposes
continuations of previously-seen n-grams — a copier. Its own predicate is
"last-8-gram recurred AND continuation agreement ≥ 0.75". So copying makes the
text more repetitive, which makes the predicate *more* true, which keeps it
latched. Per-step re-evaluation is what was supposed to break that cycle; the
latch removes the brake.

**What I will not claim:** this is one arm, one instance, n=1, and agent
trajectories are stochastic — pass 30's rule ("a mechanism that EXPLAINS a failure
is not evidence that it CAUSED it") applies with full force. This is an
**association plus a mechanism**, not a controlled result.

**What I will claim:** it is a degeneration signature, in the gated arm, on the
task the gate latched onto, with a coherent feedback mechanism and a clean
dose-response across four tasks. Under Mark's standing condition that is a
**STOP**, and I am reporting it as one.

## R5.5 Phase 3 — Tier-B credential re-earned, 9/9 bounds

Offline, ~9 min, zero containers before and after. Earned at HEAD `71ab122d2`
(the previous credential bound `eb06fe45f`, two commits stale).

```
B1 determinism_bitwise         PASS  all cases bitwise identical, cross-process digests identical
B2 output_ulp_concentration    PASS  0.9331 >= 0.9        B3 output_max_abs_delta   PASS  1.384 <= 4.0
B4 lse_max_ulp                 PASS  4 <= 8               B5 lse_max_abs_delta      PASS  3.8e-06 <= 1e-4
B6 argmax_flips_vs_exact       PASS  1 <= 2 (no worse than incumbent)
B7 output_rms_vs_exact_ratio   PASS  0.9608 <= 1.1        B8 lse_rms_vs_exact_ratio PASS  0.8386 <= 1.1
B9 nonfinite_agreement         PASS  0
```

B7/B8 below 1.0 restate lane 4's finding independently: **split-K is closer to
exact attention than the kernel that ships.** Credential
`output/fr14_splitk_tierb_credential_71ab122d2.json`, sha
`fd77c3501d91b6bb…`, grants "live-A/B serving only".

## R5.6 Phase 4 — ARM S refused at a 17th site; the eyeball is still not discharged

Boot 20:01:23Z, engine init failure 20:05:17Z, zero served tokens.

```
RuntimeError: FR13 qrow32 B1 pinned identity drifted
```

**Attribution.** `fr13_patch_fa2_tree_bias.py:6238 _fr13_fa2_qrow32_b1_identity`
has explicit branches for `gqa_pair` and `visibility`, then a **bare fallback**
returning the incumbent split2 pins. There is **no `gqa_pair_splitk` branch**, so
the split-K arm resolves to the wrong identity and `require_identity` refuses the
real binary:

| field | resolver returns | split-K truth |
|---|---|---|
| `candidate_sha256` | `a9d8a688…` (split2) | `28570f83…` |
| `candidate_size` | 300 154 616 | 300 123 792 |
| `source_closure_sha256` | `22b8c201…` | `4ed00909…` |

**This is the exact shape of round 1 §2.1** — `.get(arm, <split2 default>)` in the
launcher's in-container qualification map — now in the sibling *identity*
resolver. Lane 4 closed the three refusals I enumerated; this is a fourth, and it
sits one layer deeper, which is why nothing before phase 4 could reach it.

**The refusal is correct and prevented a real hazard.** A permissive fallback
would have loaded the split-K `.so` while attesting the incumbent's identity —
serving one kernel while the artifact claims another, the precise failure the
launcher's own comment at `:4290` was written about.

**Everything upstream worked**: launcher host-side checks, the in-container
qualification map, `TIER_B_SERVE=1`, and the fresh credential all passed. The gap
is one `if arm == "gqa_pair_splitk":` branch.

**Mark's mandatory degenerate eyeball on split-K is NOT discharged.** No split-K
token was served, so there is still no trace to read. Fifth attempt, fourth
distinct blocker, and the honest statement is unchanged from round 1: the split-K
kernel has never produced a single served token.

## R5.7 Verdicts

| lever | verdict | change |
|---|---|---|
| **fused draft top-k** | **PROMOTED — holds** | served both arms via the promoted default this round; 0 unexpected census differences again |
| **suffix pass gate** | **REFUSE — and now on QUALITY, not plumbing** | the integration finally runs end-to-end and the arm it produced contains a degeneration signature (§R5.4) |
| **split-K FA2** | **REFUSE** | Tier-B numerics credential is excellent (9/9, closer to exact than the incumbent); the serving path still cannot run, and the eyeball is undischarged |

**On lever 2, the priority order has inverted.** The integration is essentially
complete — every §11.7 shape check passes, work shape is exact, and the −20.6 ms
mechanism is confirmed at −22.98 ms/gated step. What is now blocking is not a
16th plumbing site but **two behavioural defects**:

1. **The per-request latch (§R5.3).** The gate is specified per-step and behaves
   per-request. Everything downstream — the warm rate, the step_wall headline, and
   most likely §R5.4 — follows from it. **Fix this first; it is not a plumbing
   nit, it is the lever not being the lever.**
2. **The degeneration signature (§R5.4).** Re-test only after the latch is fixed,
   because a per-step gate cannot sustain the copy-feedback loop the latched one
   can. Then read 13236 specifically, and require a resolved-or-healthy trace on
   it before any promotion discussion.

Only after both would the pre-registered questions — the true renewal warm rate,
and whether MTP survival at positions 3–4 on strong-match steps is below 0.931 —
be worth asking again.

**For split-K:** add the `gqa_pair_splitk` branch to
`_fr13_fa2_qrow32_b1_identity`, and — given this is the fourth
fallback-to-split2 defect in the same serving path — sweep for the pattern
directly rather than boot for it: every `arm ==` / `.get(arm, …)` resolver in the
FA2 path that can silently return another arm's identity.

---
---

# ROUND 6 (2026-08-18 20:39Z – 2026-08-19 01:10Z) — the latch fix answers both release questions, and split-K FINALLY SERVES

Frozen HEAD `c5d41e364`. Quiet window held; tree byte-clean throughout; the Tier-B
credential again earned to a gitignored path so no mid-sequence commit was needed.

| phase | outcome |
|---|---|
| **1** gate re-earn | **PASS** rc=0, 11.7 min |
| **2** ARM G'' latch-fixed | **DRAINED**, 4/4 tasks, 61 497 steps, `swerc=13` (13398 capped) |
| **3** Tier-B credential | **PASS**, 9/9 bounds, earned at this HEAD |
| **4** ARM S sixth boot | **SERVED — split-K generated real tokens for the first time** |

## R6.1 Release question (b): do MIXED requests exist? **YES — the latch is gone**

| | round 5 (latched) | **round 6 (fixed)** |
|---|---|---|
| requests | 98 | **188** |
| **MIXED** (gated *and* ungated inside one request) | **0** | **181** |
| fully gated | 11 | **0** |
| fully ungated | 87 | 7 (short requests) |
| gated runs | — | **3 256**, mean **3.1** steps, longest **32** |
| MAX_RUN=32 brake engagements | — | 53 |

181 of 188 requests now transition per step. The gate is a per-step predicate
again, exactly as §11.6 specifies. The brake fired 53 times, so it is load-bearing,
but the mean gated run is 3.1 steps — the gate is self-limiting on its own.

## R6.2 The first honest warm-step rate: **0.1619 — INSIDE the pre-registered 0.15–0.25**

Round 5's 0.3458 was a length-weighted artifact of the latch. With per-step
re-evaluation restored, the renewal-process prediction from §10.1 is **confirmed**:

```
warm_step_rate = 9 956 / 61 497 = 0.1619        pre-registered 0.15 - 0.25   INSIDE
```

**Every §11.7 check now passes** (registry two rows `passes=2` seg 0/1;
`graph_replays` exactly {2 cold, 1 gated}; `mtp_forward_calls` only {4,2}; pairs
only (4,6)/(2,8); 27/32 on all 61 497 steps; warm rate in band). The single
"failure" my reducer reports is still its own known bug — the "ungated signature
`d9a4dd…`" claim belongs to the gate-**off** arm, and I apply it to the armed arm.

## R6.3 Release question (a): is `astropy-13236` HEALTHY? **YES — and it RESOLVED**

| 13236 | round 5 (latched, w=0.993) | **round 6 (fixed)** |
|---|---|---|
| turns | 7 | **146** |
| type-token ratio | **0.067** | **0.201** |
| tail-repeat | **0.991** | **0.272** |
| max line repeat | 130 | 62 |
| top 8-gram | ×71 | ×21 |
| tool calls | 2 | **51** |
| malformed | truncated call at end | **0** |
| patch | **0 B** | **1 924 B** |
| verdict | failed | **resolved** |

Read, not merely counted — its closing text:

> **Changes** (2 files): `astropy/table/table.py`: removed the auto-transform
> clause in `_convert_data_to_col` that viewed structured ndarrays as
> `NdarrayMixin`… **Verification**: Structured arrays added via `Table([...])`,
> setitem, or `add_column` now become `Column`s… All other failures (9) are
> pre-existing environment issues (IERS leap-second staleness, numpy quirks),
> unchanged by this fix.

Coherent, correct, self-auditing. **The degeneration signature of §R5.4 is gone,
and it disappeared exactly when the latch did** — which is the strongest available
evidence that the latch caused it, without ever having claimed so from n=1.

Second trace read as instructed — `13398` (243 turns, 41 773 words, tail-repeat
0.210, 83 tool calls, patch 0 B). Zero-patch here is the **9000 s budget cap**,
not a loop: it is mid-debugging when killed —

> `Confirmed pre-existing failure (fails without my change too). Import restored.
> Running the full file without -x:` … `The predicted test_gcrs_altaz_bothroutes
> regression has materialized. To understand the magnitude of the difference…`

All four G'' traces: **zero non-ASCII, zero malformed tool calls.**

## R6.4 The measurement the whole campaign was for — and lever 2 fails it

Paired against round-2 C' (both arms ungated-reduced with a capped 13398, per the
accounting pre-registered in §R3.4):

| instrument | C' (gate OFF) | G'' (gate ON) | delta |
|---|---|---|---|
| `step_wall_ms` | 214.759 | 212.388 | **−2.371 (−1.10 %)** |
| drafter GPU ms/step | 55.447 | 52.645 | −2.802 (−5.05 %) |
| `s_per_fwd_gpu` | 0.13126 | 0.13151 | +0.19 % |
| committer *(null)* | 20.465 | 20.511 | +0.23 % |
| **accept / event** | **3.8855** | **3.6832** | **−0.2023 (−5.21 %)** |
| per-request TPS | 24.392 | 23.610 | **−3.21 %** |

Derived per gated step (w = 0.1619):

```
dfwd saving  = 2.802 / 0.1619 = 17.31 ms/gated step   (pre-registered -20.6)   84% of prediction
accept cost  = 0.2023 / 0.1619 = 1.249 tokens/gated step
```

**The time side of the pre-registration holds. The acceptance side fails by an
order of magnitude.** §10.1 pre-registered the accept delta in
**[−0.02, +0.15] tokens/step**; measured **−0.2023**, i.e. **10.1× the worst
pre-registered case**, and on the wrong side.

§7 stated the risk exactly: *"the gate is accept-positive iff MTP's survival at
draft positions 3–4 on strong-match steps is below 0.931"*, and called that "the
one question the A/B is for". **The A/B has now answered it: no.** On the steps
the gate actually fires, MTP was doing better at positions 3–4 than the Arctic
suffix chain that replaces it — the opposite of the modelled assumption.

Net effect: −1.10 % step wall bought with −4.14 % committed tokens ⇒ **−3.21 %
throughput. The lever is net negative.**

**Variance discipline, stated plainly:** every one of those arm-level deltas sits
*inside* the ±10 % band, so none is citable as a verdict from a single pair. What
is not inside the band is the **pre-registered accept range**, which was written
precisely so this comparison could be decided — and it is missed by 10×. The
paired per-task accept deltas (−11.58 %, −10.12 %, −4.05 %, +1.52 %) point the
same way in three of four.

## R6.5 Phase 4 — SPLIT-K SERVES, AND MARK'S EYEBALL IS DISCHARGED

Sixth boot. The identity resolver's new `gqa_pair_splitk` branch cleared the 17th
site; boot healthy at 261 s; `swerc=0`; `astropy-12907` **resolved**.

Served-arm artifact, `fr13_fa2_qrow32_b1_gqa_pair_splitk_live_paged_ab.json`:

```
arm = gqa_pair_splitk · tier = B · status = PASS
served_return = "candidate output served (tier-b)"
```

**That line is the campaign's long pole falling.** For five attempts across two
weeks the honest statement was "the split-K kernel has never produced one served
token". It has now produced 1 027 draft events' worth.

### THE EYEBALL — verdict: **CLEAN. No degeneration signature.**

| | ARM S (split-K) | C' (gqa_pair) | best other arm |
|---|---|---|---|
| type-token ratio | **0.395** | 0.375 | 0.361 |
| max line repeat | **4** | 6 | 14 |
| top 8-gram | **2** | 2 | 3 |
| **tail-repeat** | **0.000** | 0.000 | 0.256 |
| non-ASCII | 0 | 0 | 0 |
| malformed tool calls | **0** / 10 | 0 / 14 | 0 |

**The cleanest trace in the entire campaign on every signature.** Read verbatim —
it diagnoses the real bug:

> The bug is in the `else` branch for `right`: when `right` is already a
> coord_matrix (ndarray), instead of copying the matrix into the correct position, it
> … fills its output block with all-1s (`= 1`) instead of copying the child's
> separability matrix (`= right`), making nested models' inputs/outputs appear
> mutually dependent.

and produces the minimal correct patch:

```diff
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
```

closing with a verification summary that distinguishes its own effect from
pre-existing failures. **No repetition, no gibberish, no mid-word breaks, no
malformed tool calls. Mark's condition is discharged on this trace.**

### Performance — the projection does NOT hold

| 12907, same instance | C' (gqa_pair) | ARM S (split-K) | delta |
|---|---|---|---|
| **verifier forward** (contains FA2) | 125.898 ms | **128.999 ms** | **+3.101 (+2.46 %)** |
| drafter span *(null)* | 50.440 | 50.163 | −0.55 % |
| committer span *(null)* | 22.563 | 23.294 | +3.24 % |
| accept / event | 3.6442 | 3.7692 | +0.125 (+3.43 %) |

**Expected ~−14 ms; measured +3.1 ms.** The offline 2× kernel win (13.38 → 7.00 ms
at 23k) did not transfer. Two honest caveats: n=1 task, single run, no repeat; and
the trajectories differ (1 038 vs 1 408 decode steps), though the *shorter* ARM S
trajectory implies shorter KV contexts and therefore, if anything, a cheaper FA2 —
which makes +3.1 ms the conservative direction. The null spans moving ±3 % on the
same comparison is the noise floor of a single unpaired 1-task diagnostic run, and
it is the same size as the effect, so the correct statement is: **the projected
−14 ms is not visible, and this run cannot resolve anything smaller than ~±4 ms.**

Measure, don't bank — banked as measured.

## R6.6 Verdicts after six rounds

| lever | verdict |
|---|---|
| **fused draft top-k** | **PROMOTED — holds.** Served via the promoted default in every round-6 arm; work shape unchanged. |
| **suffix pass gate** | **REFUSE.** The integration is now correct and every §11.7 check passes, the warm rate lands in its pre-registered band, and the traces are clean — but it **fails its own accept pre-registration by 10×** and is **net −3.21 % throughput**. §7's break-even question is answered: MTP survival at positions 3–4 on strong-match steps is *above* 0.931. |
| **split-K FA2** | **MORE EVIDENCE — for the first time, not REFUSE.** Numerics 9/9; it serves; **Mark's eyeball is discharged on a clean, resolving trace**. But the −14 ms projection did not appear (+3.1 ms measured on one task). It needs a *paired, multi-task* serve on the instruments before any promotion — the thing it has never had. |

**What changed this round, in one line each.** Lever 2 stopped being a plumbing
story and became a measurement, and the measurement says no. Split-K stopped being
unreachable and became measurable, and the first measurement says the offline win
does not obviously transfer — which is exactly why the eyeball condition was
attached to a *serve* and not to a probe.

---
---

# CORRECTION TO ROUND 6 — ARM S NEVER SERVED SPLIT-K, AND I REPORTED THAT IT DID

Lane 4 (`fd728e2b3`) found an **18th site**: the tier-B serving-hook installer was
keyed to the *production* selector while tier-B is spelled as a **live** arm, so the
`elif` chain installed nothing. Round 6's ARM S was **incumbent-vs-incumbent**.

**I verified the correction against my own banked artifacts rather than accept it:**

* the round-6 census carries `tree_attn.calls = 16` on **all 1 038 rows** — one call
  per full-attention layer, i.e. a single un-retagged dispatch;
* the served artifact has **no engagement counter at all** (`tier_b_engagement` did
  not exist); the two fields I quoted, `served_return = "candidate output served
  (tier-b)"` and `tier_b_serving = True`, are **derived from environment variables**;
* the artifact's own body proves what it really was — a *shadow* comparison:
  `reference_dispatch = qrow16 incumbent`, `reference_selector_sentinel = 1179791667`
  vs `selector_sentinel = 1179791671`, and **176 915 output / 11 320 LSE raw-byte
  mismatches**. Those mismatches are exactly what `splitk_fa2.md` §1 predicts for a
  changed reduction topology — they were never a serving signal.

## What this invalidates, precisely

| round-6 claim | status |
|---|---|
| "SPLIT-K SERVED … first time in six attempts" | **WITHDRAWN** |
| "Mark's mandatory degenerate eyeball: DISCHARGED, CLEAN" | **WITHDRAWN — the trace was the PROMOTED STACK's, not split-K's** |
| verifier forward +3.101 ms (+2.46 %) vs C' | **not a split-K reading** |
| Tier-B credential 9/9 bounds (phase 3) | stands — offline kernel probe, unaffected |
| ARM G'' results (phases 1–2) | stand — different arm, unaffected |

**Mark's split-K eyeball is NOT discharged. It never was.** Six attempts, and the
honest count of served split-K tokens remains **zero**.

## My share of this, stated plainly

I checked a field named `served_return` and did not check whether the kernel had
*run*. The evidence was already in my own hands: `tree_attn.calls` sat in the census
I had reduced, and in round 1 I had written a probe specifically because
`candidate_served` is the only field that means served — then in round 6 I trusted a
string. **I verified configuration and reported it as observation**, which is the
exact failure this campaign has named repeatedly ("an artifact must report what
ran"). The lane's own confession does not transfer the reporting error: I published
the discharge.

**One thing worth keeping from the wreckage.** Because round 6's ARM S was the
promoted stack measured against the promoted stack, its verifier-forward delta is an
empirical **noise-floor estimate for a single unpaired 1-task diagnostic run**: the
true value was 0 and it read **+3.101 ms (+2.46 %)**. My stated caveat — "this run
cannot resolve anything smaller than ~±4 ms" — was right, and is now calibrated
rather than asserted. Round 7 needs the full task set for exactly that reason.

---
---

# ROUND 7 (2026-08-19 01:26Z–01:58Z) — a 19th site: the installer now installs, but installs the TIER-A entry point

Lead with the engagement observation, as asked: **there is none.** The boot never
reached the engagement counter — it refused three minutes earlier, at a
production-only precondition.

| phase | outcome |
|---|---|
| **0** yield to foreign container | `fr14_sglang_dspark_calib` drained 01:32Z; GPU released, waited |
| **1** Tier-B credential re-earn | **PASS**, 9/9 bounds |
| **2** gqa_pair gate re-earn | **PASS** rc=0 |
| **3** ARM S round 7 | **REFUSED at a 19th site**, 0 served tokens, 0 engagements |

## R7.1 A process note: a commit landed inside the declared quiet window

Pass 73 (`6530b1f17`) committed at 01:33:06Z, **during** my Tier-B credential earn.
It was benign — results-only (`REDTEAM_20260816.md` + two calibration `.jsonl`), no
runtime source — and the timing was lucky: the gate read HEAD *after* the commit, so
the credential binds `6530b1f17`, and phase 2's gate binds the same. Both credentials
are consistent and the patcher digest `2d0df5f1…` matches the live file.

Recording it because round 5 was lost to exactly this and the next one may not be
results-only: **a credential earn reads HEAD at an unpredictable instant inside its
window.**

## R7.2 The 19th site

```
RuntimeError: FR13 qrow32 B1 production has no launcher attestation
  tree_attn.py:2749 forward -> tree_attn.py:2228 _fr13_fa2_qrow32_b1_production_begin
```

Lane 4's 18th-site fix works as designed: the new
`--fixed32-query-tile32-b1-tier-b-serve` modifier is a separate expansion outside the
exclusive `elif` chain, and it **does** install the serving hook now
(`fr13_patch_fa2_tree_bias.py:9498-9505` ORs it into the same installer). The hook it
installs is `_fr13_fa2_qrow32_b1_production_begin`, and that function *is* tier-aware
— it opens with `arm, tier = _fr13_fa2_qrow32_b1_serving_arm()` and handles
`tier == "B"` with the credential further down.

**The defect is the line between those two facts:**

```python
arm, tier = _fr13_fa2_qrow32_b1_serving_arm()      # tier-aware
if arm is None: return None
if os.environ.get("FR13_FA2_QROW32_B1_INTERNAL_ATTESTED") != "1":
    raise RuntimeError("FR13 qrow32 B1 production has no launcher attestation")
```

`FR13_FA2_QROW32_B1_INTERNAL_ATTESTED` is exported by the launcher **only** inside the
production-arm attestation block, which runs `--arm "$FR13_FA2_QROW32_B1_PRODUCTION_ARM"`
(`fr13_launch_forked_fa2_tree_server.sh:6996-7002`). Tier-B is spelled as a **live**
arm, so `FR13_FA2_QROW32_B1_PRODUCTION_ARM` is empty, that block never runs, and the
variable is never set. **A tier-A-only precondition sits above the tier branch in a
function both tiers now enter.**

So the defect moved one layer inward rather than being eliminated: 17th = resolver
didn't know the arm; 18th = installer didn't install the caller; **19th = the
installed caller gates on an attestation only tier A can hold.**

Two shapes of fix, for lane 4 to choose: export an equivalent tier-B attestation from
the launcher's tier-B branch, or move the check below the tier resolution so tier B is
validated by the credential it already carries (which the `tier == "B"` block does
immediately afterwards) rather than by tier A's env flag.

## R7.3 What did work, and is worth keeping

* The **new engagement machinery exists and is honest** — `tier_b_engagement` is now
  counted at the retag, and "armed but never engaged" refuses in seconds. This boot
  simply died before reaching it, so it is still unexercised, but the observability
  that would have caught round 6 is in place.
* Both credentials re-earned cleanly at the same HEAD, 9/9 bounds again.
* Fail-closed held: 4 minutes of GPU, zero served tokens, no silent wrong-kernel serve.

## R7.4 Status — unchanged, and the count is now honest

**Mark's split-K degenerate eyeball is NOT discharged. Served split-K tokens across
seven rounds: zero.** The blocker ledger for the split-K *serving path* alone now
reads:

| # | site | found by |
|---|---|---|
| 2.1 | launcher in-container qualification map falls through to split2 | round 1 (read) |
| 2.2 | live-A/B reduction-topology refusal | round 1 (executed probe) |
| 2.3 | live-A/B never returns candidate output | round 1 (read) |
| 17 | identity resolver defaulted to split2 pins | round 5 boot |
| 18 | serving-hook installer keyed to the production selector | lane 4, after my round-6 mis-report |
| **19** | **installed hook gates on a tier-A-only attestation** | **round 7 boot** |

Six distinct defects in one serving path, five of them found only by booting. The
pattern across all of them is identical and worth naming once more: **a
production-shaped assumption sitting on a path that a second, differently-spelled
arm now also takes.**

**Recommendation, unchanged in substance from round 3 and stronger now:** before boot
eight, sweep the tier-B serving path for *tier-A-only preconditions* the way the
replay-dimension scan swept for shape literals — every `INTERNAL_ATTESTED`,
`PRODUCTION_ARM`, `require_exact4` and production-allowlist reference reachable from
`_fr13_fa2_qrow32_b1_production_begin` when `tier == "B"`.

### And a 20th site, verified without booting — plus a contradiction under it

I checked the next precondition rather than predict it.
`_fr13_fa2_qrow32_b1_require_exact4()` (`:6397`) is the line immediately after the
attestation, and it reads two env pins:

```python
task_ids != _FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS
or subset  != _FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256
    -> RuntimeError("FR13 qrow32 B1 production exact4 identity drifted")
```

An ARM S run sets neither, so **boot eight refuses here**, one line further in, unless
it is fixed in the same pass. That is the 20th site and it cost no GPU to find.

**The contradiction underneath it is the more important half, and it is a design
question, not a bug.** The tier-B serving path demands the **canonical exact4**
identity; the tier-B *live-arm* predicate in the launcher demands
`FR13_FIXED32_B1_DIAGNOSTIC=1` with the **single** canonical instance
(`fr13_launch_forked_fa2_tree_server.sh`, live-A/B block), and `B1_DIAGNOSTIC=1`
forces the ingress task list to exactly one id. **As written, tier-B serving requires
a four-task identity that tier-B arming forbids.** They cannot both be satisfied.

I could have made this boot proceed by exporting the two exact4 pins by hand. **I did
not, deliberately.** The ingress would still have been one task while the env asserted
four — two attestations agreeing with each other and both disagreeing with what ran,
which is precisely the failure this campaign already banked once (the K64-shaped
credential minted by a K0 gate, pass-era red-team). Papering over it with env vars
would manufacture exactly the kind of green artifact that made round 6's report wrong.

**So the real question for lane 4 / Mark is which identity tier-B serving should
carry** — exact4 (and then the live-arm predicate must admit the four-task subset), or
the single diagnostic instance (and then `require_exact4` must not be on the tier-B
path). Boots will keep finding one precondition at a time until that is decided.

---
---

# ROUND 8 (2026-08-19 02:18Z–02:25Z) — BLOCKED at a 21st site: two quote characters in a comment truncate every fixed32 boot script

**Engagement observation, leading as asked: none — and none was reachable.** ARM S
never launched. The *gqa_pair gate* (phase 2) died first, and it died for a reason
that blocks **every fixed32 serve at this HEAD**, not just tier-B.

| phase | outcome |
|---|---|
| **0** yield to E1 | waited out `e1-dspark-capture`; a **second** E1 container (`e1-dspark-replay`) started at 02:22Z — yielded again |
| **1** Tier-B credential re-earn | **PASS**, 9/9 bounds, bound to HEAD `731c91498`, patcher `ce6a64f5…` |
| **2** gqa_pair gate re-earn | **FAILED, rc=2** — container died before health in 47 s |
| **3** ARM S | **not attempted** |

## R8.1 The 21st site, attributed to one line

The container log said only:

```
sha256sum … _vllm_fa2_C.abi3.so
no: -c: line 163: syntax error: unexpected end of file
```

I pulled the **exact script the container ran** out of `docker inspect …Config.Cmd`
and syntax-checked it. It is **truncated at 162 lines**, and its final line is:

```
  # died on has
```

The source line, `fr13_launch_forked_fa2_tree_server.sh:7063`, reads:

```bash
  # died on "has no launcher attestation". A tier-B serve needs its own
```

That comment lives inside the **double-quoted host string** that builds the
in-container boot script. Its embedded `"…"` closes the host string early, so
everything after `# died on ` is lost, and the remaining words become positional
arguments to `bash -lc` — which is why `$0` was `no` and the error printed as
`no: -c: line 163`. That prefix was the clue, not noise.

**The truncation removes the entire back half of the boot script.** Verified against
the generated text:

| the script still contains | the script has LOST |
|---|---|
| `fr13_patch_fa2_tree_bias` (earlier) | `verify-tier-b` (the site-19 fix itself) |
| | `fr10_phase4_patch…` (the drafter patcher) |
| | **`vllm serve`** — the serve command itself |

So the container starts, patches the FA2 `.so`, hits the truncation, and exits.
**Every fixed32 boot at `731c91498` dies the same way** — the gqa_pair gate, ARM C,
ARM G and ARM S alike. It is a total blocker, and the gate found it in 47 seconds.

It is also the **only** comment carrying double quotes in the new block
(`7039-7085`), so it is a one-line fix: drop or escape the quotes at `:7063`, and
add the obvious lint — *no unescaped `"` in a comment inside the in-container
string*, which is checkable without a GPU on all three launcher twins.

## R8.2 Two notes worth keeping

**The bound held, and then moved.** Lane 4's honest bound was "a 21st site would live
in vLLM's behaviour under capture, unreachable by reading". This one is neither: it is
in the launcher's own generated text and is **reachable by `bash -n` on the
generated script** — which is exactly the check nobody runs, because the script only
exists at boot. The CPU harness walks the *served path*; nothing walks the *boot
script*. That is the gap the 110-gate walk could not see, and it is cheap to close.

**Phase 1 still passed, and that matters.** The Tier-B credential re-earned cleanly at
this HEAD, 9/9 bounds, patcher digest matching — because it is offline kernel work
that never builds a container boot script. The split-K *numerics* remain in good
standing; only the *route* is broken.

## R8.3 Status

**Mark's split-K eyeball is NOT discharged. Served split-K tokens after eight rounds:
zero.** The serving-path ledger gains one more, and its shape is new:

| # | site | class | found by |
|---|---|---|---|
| 17 | identity resolver defaulted to split2 | tier-A-only default | round-5 boot |
| 18 | installer keyed to the production selector | tier-A-only installer | lane 4 |
| 19 | installed hook gates on a tier-A attestation | tier-A-only precondition | round-7 boot |
| 20 | `require_exact4` vs single-instance arming | contradiction | round-7 **read** |
| **21** | **quotes in a comment truncate the boot script** | **host/container quoting** | **round-8 boot, 47 s** |

Sites 17–20 were all one shape — a production assumption on a path a second arm now
takes. **21 is not that shape**: it is a text-assembly defect introduced *by the fix
for 19*, and it would have been caught by syntax-checking the generated script rather
than by any amount of gate-reading. Different class, different detector.

**GPU cost of this round: ~1 minute of a dead container, plus the offline credential.**
Fail-closed did its job: nothing served, nothing mis-reported, and the blocker is
named to the character.

---
---

# ROUND 9 (2026-08-19 02:26Z–02:57Z) — site 21 is fixed and confirmed; ARM S refused at a 23rd site

**Engagement observation, leading as asked: none.** ARM S refused twice before any
forward ran — once on the credential path, once on binary identity. Total GPU cost
this round: **about two minutes** across two fast refusals.

| phase | outcome |
|---|---|
| **0** yield to E1 | `e1-dspark-replay` drained 02:38Z |
| **1** Tier-B credential | **PASS**, 9/9 bounds, bound to `ac6e3ed87` |
| **2** gqa_pair gate | **PASS rc=0** — **site 21 confirmed fixed end-to-end** |
| **3** ARM S | **REFUSED**: credential path, then a **23rd site** |

## R9.1 Site 21 is genuinely fixed — verified two ways

Statically: zero quoted comments in the boot region of **all three** launcher twins.
Behaviourally: the gate container lived past 58 s (round 8 died at 47 s) and the gate
drained `rc=0`. **Every fixed32 boot works again.** That is also the first independent
confirmation that the coordinator's mechanical fix was complete across the twins.

## R9.2 The credential path — one half measured, one half still open

I ran both spellings rather than argue about them.

| `FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL` | result |
|---|---|
| **container** path (`/workspace/output/…`) | **REFUSED** by the launcher's *host-side* check — `-f`/`sha256sum` run on the host, where `/workspace` does not exist |
| **host** path (`/home/mark/…/output/…`) | host check **PASSES**, run proceeds |

The variable is host-validated at `:2383-2386`, forwarded **verbatim** at `:6704`, and
consumed **inside the container** at `:7071`. The host path clears the first hurdle;
whether the container can then read it is **untested**, because the run dies earlier
(§R9.3). So I am *not* claiming "no value satisfies both" — that would be exactly the
kind of read-not-measured claim this campaign keeps punishing. **Suspected, not
established.**

If it does turn out to need both, the launcher already carries the pattern:
`FR13_FA2_QROW32_B1_GQA_PAIR_GATE_HOST` (host) alongside `…_GATE_JSON` (container).

## R9.3 The 23rd site — the Python twin doesn't speak the new spelling

```
FAIL: launcher rc=1
fixed32 qrow32 B1 binary identity is not qualified
```

The in-container qualification map resolves the pin arm at `:4366`:

```python
b1_pin_arm = os.environ.get("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "") \
          or os.environ.get("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "")
```

It never consults the new first-class `FR13_FA2_QROW32_B1_TIER_B_ARM`. Lane 4 *did*
add the `"gqa_pair_splitk"` entry to the map (`:4385`) **and** a tier-B cross-check
("tier-b arm resolved a non-tier-b binary") — but the resolver feeding that map still
speaks only the two old spellings, so `b1_pin_arm` comes out **`""`**, the map's `""`
key returns **split2's** pins, and the mounted split-K binary is refused.

**The bash pin-arm resolver, twenty-one hundred lines earlier, does know the new
spelling** — `:2245` falls back to `TIER_B_ARM`. The Python twin's own comment four
lines above the defect reads *"This mirrors the bash pin case"*. **It no longer does.**

### The irony worth banking

The `""` entry in that map was added *deliberately*, with a comment explaining it
replaced a `.get()` default because "the `.get()` default this replaced answered for
every arm nobody had written a key for — with split2's identity — which is the 17th
site's defect shape exactly."

But `""` is precisely what the un-updated resolver produces for a tier-B arm. So the
named `""` entry now plays the exact role the old silent default played, and answers
for the tier-B arm with split2's identity — **the 17th site's defect, reconstituted
inside the fix written to prevent it.** Naming a default does not remove it; it renames
it. The removal has to happen at the *resolver*, not the table.

## R9.4 Process: HEAD moved inside the quiet window again

Pass 77 (`c0d5550f4`) landed at **02:41:33Z**, between my Tier-B earn (02:38–02:40)
and the ARM S attempts. Both credentials bind `ac6e3ed87` and are therefore **stale**
— the container-side `verify-tier-b` would have caught it, but the run died earlier at
§R9.3, so it cost nothing this time.

Second occurrence in three rounds (pass 73 in round 7, pass 77 now). It has been
harmless twice. The failure mode it invites is not harmless: a credential earned at
one HEAD and a serve validated at another is exactly the drift the whole credential
regime exists to prevent.

## R9.5 Status

**Mark's split-K eyeball is NOT discharged. Served split-K tokens after nine rounds:
zero.** The serving-path ledger:

| # | site | class | found by |
|---|---|---|---|
| 17 | identity resolver defaulted to split2 | tier-A default | round-5 boot |
| 18 | installer keyed to the production selector | tier-A installer | lane 4 |
| 19 | hook gates on a tier-A attestation | tier-A precondition | round-7 boot |
| 20 | `require_exact4` vs single-instance arming | contradiction | round-7 read |
| 21 | quotes in a comment truncate the boot script | text assembly | round-8 boot |
| 22 | credential path host-vs-container | *suspected* | round-9 boot (half) |
| **23** | **Python pin-arm resolver ignores `TIER_B_ARM`** | **bash/python twin drift** | **round-9 boot** |

Sites 17, 18 and 23 are the **same defect in three different resolvers**, and 23 is
the second time a bash/python twin pair has drifted apart in this exact file. The
cheap detector is not another boot: it is a test that asserts **the bash pin-arm
resolver and its Python twin return the same arm for every member of
`QROW32_B1_TIER_B_ARMS`** — executable on CPU, no GPU, and it would have caught 2.1,
17 and 23.

---
---

# ROUND 10 (2026-08-19 03:00Z–03:31Z) — sites 22 and 23 confirmed fixed; ARM S refused at a 24th, in the third resolver

**Engagement observation: none.** ARM S refused before any forward ran — but it got
**further than any previous attempt**: past every launcher check, into the container,
through the Arctic prelaunch install, and died at the fixed32 **contract** check.
GPU cost this round: ~1 minute.

| phase | outcome |
|---|---|
| **1** Tier-B credential | **PASS**, 9/9, bound to `b9a343d8d` |
| **2** gqa_pair gate | **PASS rc=0** |
| **3** ARM S | **REFUSED — 24th site** |

## R10.1 What is now genuinely fixed

* **Site 22 (credential path)** — the staged `_CREDENTIAL_HOST` + `_SHA256` contract
  works: the launcher accepted the host path, staged it, and derived the container
  path itself. No refusal.
* **Site 23 (pin-arm resolver)** — the total resolver works. Verified in source: the
  `""` key is abolished and "no selector named" is spelled **`"nosplit"`** explicitly,
  with the stated reason that `""` "was also what a resolver that failed to look
  produced, and the two must not be the same value." Mutual exclusion is enforced too.

Both refusals from round 9 are gone, and the run reached the deepest point of the
campaign.

## R10.2 The 24th site — a third resolver, still speaking the retired pun

```
FAIL fixed32 contract: container FA2 identity mismatch:
{'path': '/tmp/fr13_fork_fa2.so', 'size': 300123792,
 'sha256': '28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857'}
```

**The mounted binary is correct** — that size and sha *are* split-K. What disagrees is
`fr13_fixed32_contract.py::_expected_runtime_fa2_identity`, which resolves the arm at
`:3686-3687` from **`LIVE_AB_ARM`** and **`PRODUCTION_ARM`** only. It never reads
`FR13_FA2_QROW32_B1_TIER_B_ARM` — confirmed by inspection (`reads TIER_B_ARM -> False`).

I executed it on CPU with both spellings rather than infer the consequence:

| env | `_expected_runtime_fa2_identity` returns |
|---|---|
| **new** spelling (`TIER_B_ARM=gqa_pair_splitk`) | **stock** FA2 — size `299183936`, sha `f51e23c5…` |
| retired pun (`LIVE_AB_ARM=gqa_pair_splitk`) | reaches the tier-B branch (raises only for the missing `_SO_SHA256` pin my probe omitted) |
| **split-K truth** | size `300123792`, sha `28570f83…` |

So under the new spelling the function **silently answers "stock"** and the contract
refuses the correctly-mounted split-K binary.

**This function was already fixed for tier-B — for the *old* spelling.** Lane 4's §13
table lists it as B3, "every member of `QROW32_B1_TIER_B_ARMS` mapped onto split-K's
pins", and its own comment at `:3710` still reads *"Tier-B arms are LIVE-only"* — the
pun's premise. The spelling migration (passes 76 and 79) reached the bash resolver and
the in-container qualification map and **stranded this one**.

That is the precise shape of this round's defect, and it is a new variant: not a
resolver that never knew about tier-B, but a resolver that **knew about it under a name
that has since been retired**. Site 23 was "the resolver, not the table". Site 24 is
**"the other resolver, and the rename didn't reach it."**

## R10.3 The pattern, stated once more with the count

| # | resolver | how it answered wrongly | round |
|---|---|---|---|
| 2.1 | launcher in-container qualification map | `.get(arm, split2)` default | 1 |
| 17 | patcher `_fr13_fa2_qrow32_b1_identity` | bare fallback → split2 | 5 |
| 23 | that map's **arm resolution** | unread `TIER_B_ARM` → `""` → split2 | 9 |
| **24** | **contract `_expected_runtime_fa2_identity`** | **unread `TIER_B_ARM` → stock** | **10** |

Four resolvers, one question ("which binary should this arm have?"), four different
wrong answers. The detector I proposed after round 9 — *assert the bash pin-arm
resolver and its Python twin agree for every member of `QROW32_B1_TIER_B_ARMS`* —
would have caught 23 but **not** 24, because 24 lives in a third place with its own
env-reading. The stronger form, and the one worth building: **one test that feeds the
canonical tier-B env to every arm→identity resolver in the tree and asserts they all
return split-K's pins.** There are now four known; enumerating them is a grep for
`FR13_FA2_QROW32_B1_(LIVE_AB|PRODUCTION)_ARM` reads.

## R10.4 Status

**Mark's split-K eyeball is NOT discharged. Served split-K tokens after ten rounds:
zero.**

The encouraging half, stated plainly because it is real: the route is converging
monotonically. Round 7 died in the launcher, round 8 before the container script
existed, round 9 in the launcher's qualification map, round 10 **inside the container
after a successful prelaunch**. Each round the refusal moves later, and every refusal
so far has been fail-closed — no run has served the incumbent while claiming the
candidate since the observation-not-configuration doctrine landed.

---
---

# ROUND 11 (2026-08-19 03:35Z–04:06Z) — six sites confirmed fixed; ARM S refused at an INSTALLER disjunction, in the patcher

**Engagement observation: none.** ARM S refused during engine-core init — but again
deeper than before: past every launcher check, past the fixed32 **contract** check that
killed round 10, into vLLM's own module import.

| phase | outcome |
|---|---|
| **1** Tier-B credential | **PASS**, 9/9, bound to `f147b8698` |
| **2** gqa_pair gate | **PASS rc=0** |
| **3** ARM S | **REFUSED** — `ImportError` at engine init |

## R11.1 Site 24 confirmed fixed — on CPU, before spending GPU

I executed the contract resolver with the canonical tier-B env before booting:

```
new spelling + pin -> size=300123792 sha=28570f835ea72c99
split-K truth      -> size=300123792 sha=28570f835ea72c99      MATCH: True
```

Round 10's killer is gone, and the boot confirmed it behaviourally by getting past it.

## R11.2 The refusal — a producer/consumer split inside the patcher

```
ImportError: cannot import name '_fr13_fa2_qrow32_b1_production_capture_end'
             from 'vllm.v1.attention.backends.tree_attn'
```

Two patch targets, two conditions, and only one of them learned tier-B:

| patch target | what it installs | condition | tier-B? |
|---|---|---|---|
| `vllm/compilation/cuda_graph.py` | the **import + call** of `…_production_capture_end` | `elif fixed32_query_tile32_b1_tier_b_serve:` (`:10322`) | **yes** |
| `vllm/v1/attention/backends/tree_attn.py` | `FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS`, which **defines** that symbol | `if fixed32_query_tile32_b1_live_ab or fixed32_query_tile32_b1_production:` (`:9271`) | **NO** |

I verified the blob really does carry the definition rather than assume it:

```
helpers blob defines _fr13_fa2_qrow32_b1_production_capture_end: True
helpers blob defines _fr13_fa2_qrow32_b1_production_begin:       True
helpers blob defines _fr13_fa2_qrow32_b1_tier_b_arm:             True
```

So under a pure tier-B arming the **consumer is installed and the producer is not**:
`cuda_graph.py` imports a symbol that was never injected into `tree_attn.py`, and the
engine dies at import. Nothing about the kernel, the credential, the binary or the
identity is wrong — the two halves of one patch simply disagree about whether tier-B
counts.

## R11.3 Why the sweep did not catch it — a genuinely new class

Pass 81's sweep was thorough and it worked: it found **34 reads, 6 stranded, 6 fixed**,
including 25 *before it fired*. But it swept **readers of the legacy selector env vars**
and **selector-active / exclusion disjunctions in the launcher**. This defect is neither:

* it is not an env reader — the condition is on the patcher's own boolean parameters,
  `fixed32_query_tile32_b1_live_ab or fixed32_query_tile32_b1_production`;
* it is not a resolver — nothing resolves an arm here;
* it is an **installer disjunction inside the patcher**, and specifically one where the
  producer and the consumer of a symbol live in **different patch targets under
  different conditions**.

It is the same *shape* as site 18 ("the installer was keyed to the production
selector") reappearing in a **second installer** — the tree_attn helpers block rather
than the serving-hook call site. Site 18 was fixed by adding tier-B to one disjunction;
this is the sibling disjunction eleven hundred lines earlier that nobody had reason to
look at, because it installs *helpers*, not *hooks*.

**The detector that fits this class** is not another env grep: it is an assertion that
for the tier-B parameter set, **every symbol the patcher injects a call to is also
injected a definition of**. That is checkable statically on the patcher's own output —
patch a scratch copy of the two files with the tier-B parameters and `python -c
"import ast"`-resolve the cross-file names. No GPU, and it generalises past this one
symbol: `production_begin`, `production_end` and the capture-end hook are all in that
blob, so the same gap would have bitten each of them in turn.

## R11.4 Status

**Mark's split-K eyeball is NOT discharged. Served split-K tokens after eleven rounds:
zero.**

The convergence continues to be real and monotone: round 8 died before the container
script existed, 9 in the launcher's qualification map, 10 in the fixed32 contract check
inside the container, **11 in vLLM's module import** — one layer further in each time,
every refusal fail-closed, and this round's blocker is the first that is purely a
*patching* defect rather than an *identity* or *credential* one. The identity and
credential families now appear genuinely closed: nine resolvers answered correctly, the
contract check passed, and the credential staged and verified without complaint.

---
---

# ROUND 12 (2026-08-19 04:25Z–05:58Z) — **SPLIT-K SERVES, ENGAGEMENT VERIFIED, EYEBALL DISCHARGED, AND THE PROJECTION HOLDS**

Twelve rounds, ten ARM S attempts. This one engaged.

| phase | outcome |
|---|---|
| **1** Tier-B credential | **PASS**, 9/9, bound to `bdca0bd50` |
| **2** gqa_pair gate | **PASS rc=0** |
| **3** ARM S | **SERVED — 4/4 tasks, `swerc=0`, 19 901 census steps** |

## R12.1 THE ENGAGEMENT OBSERVATION — leading, as asked

From `fr13_fa2_qrow32_b1_production_engagement.json`, **counted at the retag**, not read
from configuration:

```
status                = ENGAGED
candidate_served      = True
tier                  = B          arm = gqa_pair_splitk
selector_sentinel     = 1179791671        num_splits = 4
calls_observed        = 16
tier_b_engagement.candidate_retag_calls = 16
tier_b_engagement.layers_engaged        = 16 distinct
   layer indices: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

Every one of the sixteen full-attention layers retagged to the split-K sentinel.
`candidate_so_sha256 = 28570f83…`, `fallback_allowed = False`,
`tier_b_credential_sha256 = 6b0e0161…`. **This is the observation rounds 6–11 could not
produce, and the one round 6 fabricated from env vars.**

## R12.2 Performance vs C' — the projection holds, and then some

Paired on the identical exact4 set:

| instrument | C' (gqa_pair) | **ARM S (split-K)** | delta |
|---|---|---|---|
| **`s_per_fwd_gpu`** (verifier fwd, contains FA2) | 0.131262 | **0.115035** | **−16.23 ms/step (−12.36 %)** |
| **`step_wall_ms`** | 214.759 | **196.423** | **−18.34 ms (−8.54 %)** |
| accept / event | 3.8855 | 4.1393 | +0.254 (+6.53 %) — inside ±10 % |
| committer *(null)* | 20.465 | 20.279 | −0.90 % |
| `overhead_other` *(null)* | 7.585 | 7.412 | −2.29 % |
| per-request TPS | 24.392 | **28.819** | **+18.15 %** |
| floor_ratio | 2.3216 | **2.1086** | −9.17 % |

**Per-task, all four instances, same direction and tight:**

| instance | `s_per_fwd_gpu` | `step_wall` |
|---|---|---|
| 12907 | **−10.44 %** | −6.51 % |
| 13033 | **−12.17 %** | −7.73 % |
| 13236 | **−14.12 %** | −10.16 % |
| 13398 | **−12.21 %** | −8.47 % |

**Read against my own calibrated noise floor.** Round 6's accidental
promoted-vs-promoted run measured **+3.101 ms** on this instrument where the true value
was 0. The effect here is **−16.23 ms — five times that floor — and it repeats on four
independent instances with a spread of 3.7 pp.** This is not a single-task sighting; it
is a paired multi-task result, and it is the first time in the campaign that split-K's
offline win has been shown to transfer.

Lane 4's in-serve projection was **~−14.3 ms** with the caveat "probably optimistic;
MEASURE, DON'T BANK." Measured: **−16.23 ms.** For once in this campaign a prediction
came in *better* than briefed — the first of the whole ledger.

**Acceptance is unchanged within variance** (+6.53 %, inside the ±10 % band), which is
what a numerics-faithful kernel should show, and consistent with the Tier-B bounds
(B6–B8: split-K is *closer* to exact than the incumbent).

**Work shape is identical.** C'-vs-ARM-S census diff over 72 408 steps:
**268 identical counter paths, 0 expected-different, 0 unexpected.** The kernel swap
changes timing and nothing else — 27 active nodes, 32 verify rows, 4 MTP forwards,
1 graph replay, drafter signature `d9a4dd…6150c` on every one of 19 901 steps.

## R12.3 MARK'S DEGENERATE EYEBALL — **DISCHARGED, CLEAN**

On split-K's **own** traces this time, with engagement verified before reading them.

| instance | turns | words | ttr | max line | 8-gram | tail-rep | non-ASCII | tools | malformed | patch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12907 | 50 | 2 952 | 0.344 | 12 | 3 | 0.196 | 0 | 19 | **0** | 504 B | **resolved** |
| 13033 | 71 | 7 804 | 0.202 | 41 | 20 | 0.289 | 0 | 25 | **0** | 1 017 B | failed |
| 13236 | 74 | 1 947 | 0.269 | 5 | 3 | 0.000 | 0 | 24 | **0** | 819 B | **resolved** |
| 13398 | 164 | 25 576 | 0.158 | 18 | 7 | 0.102 | 0 | 62 | **0** | 3 942 B | failed |

**130 tool calls, zero malformed. Zero non-ASCII in ~38 000 words. No repetition loop,
no gibberish, no mid-word break, no truncation.** 2/4 resolved — the same rate as the
promoted control.

Read verbatim, 13236:

> The fix is complete. I removed the block in `astropy/table/table.py` that
> automatically converted structured ndarrays to `NdarrayMixin` … Explicit
> `NdarrayMixin` usage (e.g. `data.view(NdarrayMixin)`) still works as before.

and 13033, which audits its own effect against the baseline:

> the only failures are the 3 tests asserting the *old* message strings … exactly the
> assertions the grader's test patch updates. All other failures … were confirmed
> pre-existing environment issues (expired leap-second data, no internet) via
> `git stash` comparison.

Coherent, correct, self-auditing. **Mark's condition — attached to this lane since
`splitk_fa2.md` §7 and undischarged through eleven rounds — is met.**

## R12.4 Verdicts

| lever | verdict |
|---|---|
| **fused draft top-k** | **PROMOTED — holds.** Served via the promoted default in every arm. |
| **suffix pass gate** | **REFUSE.** Unchanged from round 6: integration correct, warm rate 0.1619 in band, but it misses its accept pre-registration by 10× and is net −3.21 % throughput. |
| **split-K FA2** | **PROMOTE-ELIGIBLE — recommend PROMOTE to the tier-B serving route.** |

**The split-K case, complete for the first time:**

* **numerics** — 9/9 pre-registered bounds, twice re-earned at the serving HEAD;
  deterministic in-process and cross-process; *closer to exact than the incumbent*
  (B7 0.961, B8 0.839, B6 1 flip vs 2);
* **engagement** — observed at the retag, 16/16 layers, not inferred;
* **speed** — **−16.23 ms/step verifier forward (−12.36 %)**, **−18.34 ms step wall
  (−8.54 %)**, consistent across four instances, five times my calibrated noise floor,
  and *better* than the −14.3 ms projection;
* **acceptance** — unchanged within variance;
* **work shape** — 268/268 census counters identical;
* **generations** — read, clean, 2/4 resolved, zero malformed tool calls.

The honest caveats to carry into any promotion note: this is **one paired serve**, not
n≥2; the trajectories differ (19 901 vs 52 304 steps, prefill_frac 0.105 vs 0.072), so
the arm-level accept and TPS numbers carry ordinary trajectory drift — the
prefill-independent `s_per_fwd_gpu` is the instrument that decides, and it is the one
that moved most cleanly. Promotion to **default** remains gated on exact16 QC parity
per Mark's pass-64 ruling; what this round supports is the **tier-B serving route**,
which is exactly what the credential grants.

## R12.5 The ledger, closed

Ten ARM S attempts, nine distinct blockers, every one fail-closed:

| # | site | class |
|---|---|---|
| 2.1 / 17 / 23 / 24 | four arm→identity resolvers | wrong-arm identity |
| 18 / 25 | two installers | tier-B omitted from a disjunction |
| 19 | tier-A attestation on a tier-B path | precondition |
| 20 | `require_exact4` vs single-instance arming | contradiction (ruled) |
| 21 | quotes in a comment | boot-script text assembly |
| 22 | credential path host-vs-container | staging |

**Not one of them was in the kernel.** The split-K binary that served today is
byte-identical to the one built on 2026-08-18 — sha `28570f83…`, both C++ closures
unchanged throughout. Every blocker was in the plumbing that decides *which* kernel is
allowed to run and *how* it is attested. That is the campaign's closing lesson, and it
is worth more than the −16 ms: **the hard part of promoting a kernel was never the
kernel.**

---
---

# ROUND 13 (2026-08-19) — the tail10 A/B is NOT LAUNCHABLE: the serve vehicle cannot select hydra31

**Boot/validation verdict, leading as asked: ARM H31 cannot be booted, and I stopped
before spending GPU.** Stage 2 taught six consumers about `hydra31_fixed32`; it did not
teach **`scripts/fr13_bigdenom_swe_serve_variant.sh`** — the arm vehicle every campaign
serve goes through, including both arms of this A/B.

**No GPU was spent this round.** The blocker is static and was found by reading the
vehicle before booting it.

## R13.1 The finding

Stage 2 (`a08ceb088`) landed `hydra31` in seven files:

```
fr13_launch_forked_fa2_tree_server.sh   fr14_armb_leg3_launch_nomiddleware.sh
fr10_phase4_patch_vllm_tree_gdn.py      fr13_merged_drafter.py
fr13_fixed32_topology.py                fr13_fixed32_work_census.py
fr14_paired_contract_sweep.py
```

`fr13_bigdenom_swe_serve_variant.sh` is **not** among them, and it is the vehicle the
campaign driver and every arm runner call. Two independent consequences, both verified
in source:

1. **There is no `hydra31_fixed32` kind.** The vehicle dispatches on `$KIND`; an
   unknown kind is refused. Its fixed32 whitelists at `:215`, `:222`, `:227`, `:234`
   name only `tail6_fixed32` and `hydra27_fixed32`.
2. **Setting the env from outside does not work — the vehicle overrides it.** The
   `hydra27_fixed32` kind block hardcodes
   `FR13_FIXED32_MODE=hydra27_fixed32` into its `XFLAGS`, and `:1714` does

   ```bash
   for kv in "${XFLAGS[@]:-}"; do [[ -n "$kv" ]] && export "$kv"; done
   ```

   — an unconditional `export` into the shell the launcher inherits, executed **after**
   the caller's environment is already set. So a caller that exports
   `FR13_FIXED32_MODE=hydra31_fixed32` has it silently replaced by `hydra27_fixed32`.

The launcher's whitelist (`:6134`) and its twelve-lever hydra31 refusal block are both
correct and in place. **The mode simply cannot reach them through the vehicle.**

Worth noting *why* this is not a one-line pass-through: the kind block does not only
set the mode string, it also supplies the **shape literals** —
`FR13_FIXED32_VALID_MASK`, `FR13_FIXED32_ACTIVE_NODES`, `FR13_FIXED32_PHYSICAL_DRAFTS`
— which differ between the profiles. Choosing them is design content, so this is lane
territory, not something an operator should improvise at the call site.

## R13.2 Handover — the constants, so it is one edit

Pulled from the profile that stage 2 *did* land (`fr13_fixed32_topology.py`), so the
vehicle can derive rather than retype:

| | hydra27 | **hydra31** |
|---|---|---|
| `ACTIVE_NODES` | 27 | **31** (`HYDRA31_ACTIVE_DRAFTS = 31`) |
| `VALID_MASK` | `0x7abdffff` | **`2147483647` = `0x7fffffff`** |
| inactive draft ids | 4 | **none** (`HYDRA31_INACTIVE_DRAFT_IDS = ()`) |
| `PHYSICAL_DRAFTS` | 31 | 31 (unchanged) |
| profile name | `hydra27_fixed32` | `PROFILE_HYDRA31 = hydra31_fixed32` |

And the census already encodes the two expectations this A/B was going to test —
walk cap **12 → 16** (`:1985`, the +33 % cfwd the brief pre-registers) and rescue count
**10 → 6** (`:1768`) — so a serve that disagrees will fail validation loudly, exactly as
intended. **The validator is ready; only the vehicle is not.**

Sites to touch: the four kind guards (`:215`, `:222`, `:227`, `:234`) plus one new
`hydra31_fixed32)` dispatch block alongside the hydra27 one.

## R13.3 Why I did not run ARM H27 alone

H27 is ~3.5 h of GPU and would have booted fine. I stopped anyway, for one reason:
**a topology A/B requires both arms at the same HEAD.** Whatever commit lands the
vehicle's hydra31 kind moves HEAD, so an H27 run taken now could not pair with the H31
run taken after the fix — it would have to be repeated. Spending 3.5 h to produce a
baseline that must be re-taken is not evidence, it is a rehearsal, and I already hold a
promoted-stack baseline of this exact shape (round-2 C', 52 507 steps) for any
comparison that does not need same-HEAD pairing.

## R13.4 The shape, one more time

This is the **same class the campaign has now hit at every layer**: a new selector is
taught to the components that *consume* it and not to the one that *selects* it.

| round | the thing that didn't learn |
|---|---|
| 18, 25 | two patcher installers |
| 23, 24 | two arm→identity resolvers |
| **13 (this)** | **the serve vehicle** |

The detector that fits this instance is the cheapest yet and needs no GPU: **for every
profile in `fr13_fixed32_topology`, assert the serve vehicle has a dispatch kind whose
exported `FR13_FIXED32_MODE`, `VALID_MASK` and `ACTIVE_NODES` match that profile's
constants.** It is a pure source-and-import test, it would have caught this before the
A/B was scheduled, and it generalises to the next profile.

**Status:** tail10 A/B **not started**. Nothing about hydra31's merits is known or
claimed. The three standing verdicts are unchanged — fused top-k PROMOTED, suffix pass
gate REFUSE, split-K recommended for the tier-B serving route on round 12's evidence.

---
---

# ROUND 14 (2026-08-19 06:29Z–08:15Z) — H27 drained; H31 refused at a SECOND mode table, before the credential question could be asked

| arm | boot/validation verdict |
|---|---|
| **H27** | **PASS** — clean boot, `swerc=0`, 4/4 tasks, 21 269 census steps |
| **H31** | **REFUSED in 5 s** — `unsupported fixed32 mode: 'hydra31_fixed32'` |

## R14.1 ARM H27 — boot verdict PASS, baseline re-validated at this HEAD

Container env attests the intended arm exactly: `FR13_FIXED32_MODE=hydra27_fixed32`,
`FR13_FIXED32_ACTIVE_NODES=27`, `FR13_FA2_QROW32_B1_PRODUCTION_ARM=gqa_pair`,
`FR14_FUSED_DRAFT_TOPK=1`, `FR14_SUFFIX_PASS_GATE=0`. Zero RuntimeErrors.
`swerc=0`, wall 5 233 s, 1/4 resolved.

Instruments (the pairing basis this round was meant to supply):

```
step_wall_ms 219.240 · s_per_fwd_gpu 0.135121 · accept/event 5.2715
drafter 55.676 · committer 20.742 · overhead_other 7.700 · per-request TPS 28.912
27/32 on all 21 269 steps · mtp_forward_calls 4 · graph_replays 1 · signature d9a4dd…6150c
```

**Note the accept: 5.2715/event, against C's 3.8855 and round-12 ARM S's 4.1393** — a
+35 % swing on the same arm shape and task set, driven by trajectory mix (1/4 resolved
here vs 2/4 there). It is a fresh reminder of why the ±10 % doctrine exists and why
`s_per_fwd_gpu` is the deciding instrument; this arm's own accept figure would be
misread as a lever effect by anyone comparing across rounds rather than within a pair.

## R14.2 ARM H31 — refused, and NOT at the check you asked about

```
FAIL: launcher rc=1
unsupported fixed32 mode: 'hydra31_fixed32'
```

**The credential question is unanswered.** The run never reached the FA2 credential
check, so I cannot yet tell you whether a topology change invalidates the gqa_pair byte
credential. Per your instruction I captured and stopped rather than re-earning under
hydra31 — but the reason it stopped is upstream of that question entirely.

**The site:** `fr13_launch_forked_fa2_tree_server.sh:3718-3728`, a **second mode table**
inside the launcher's own in-container preflight, distinct from the outer whitelist at
`:6134` that stage 2 correctly updated:

```python
expected = {
    "tail6_fixed32":   (topology.TAIL6_VALID_MASK,   topology.TAIL6_ACTIVE_DRAFTS),
    "hydra27_fixed32": (topology.HYDRA27_VALID_MASK, topology.HYDRA27_ACTIVE_DRAFTS),
}
if mode not in expected:
    raise SystemExit(f"unsupported fixed32 mode: {mode!r}")
```

It already imports the topology authority and already derives hydra27's constants from
it — it simply has no hydra31 row. `topology.HYDRA31_VALID_MASK` (2147483647) and
`topology.HYDRA31_ACTIVE_DRAFTS` (31) both exist.

## R14.3 A second refusal three lines later — found without booting

The very next check is:

```python
if tree != topology.FIXED32_CHOICES:
    raise SystemExit("fixed32 TREE differs from FIXED32_CHOICES")
```

— an **unconditional** comparison against hydra27's tree. The vehicle correctly
dispatches hydra31's own tree (`FIXED32_HYDRA31_TREE`), and I verified the two trees are
genuinely different:

```
FIXED32_CHOICES (hydra27): 31 paths      TAIL10_CHOICES (hydra31): 31 paths
trees identical? False
ancestry sha: 90873d81e83c…  vs  5b33c46a2586…
```

So **fixing only the mode table moves the refusal three lines down.** Both must be made
profile-aware in the same edit, and the authority already carries what they need:
`TAIL10_CHOICES`, `TAIL10_TREE_ANCESTRY_SHA256`, `TAIL10_SUBTREE_LEVELS`.

This is the "wrong fix that would have booted" hazard your pass-87 note names, arriving
one layer lower: the mode-table fix alone is exactly the change that *looks* right,
passes review, and fails on the next line.

## R14.4 The shape, and the detector that keeps almost catching it

| round | consumer that didn't learn hydra31 |
|---|---|
| 13 | the serve vehicle (kind + XFLAGS) |
| **14** | **the launcher's in-container preflight — mode table AND tree comparison** |

Stage 2b's parity detector covers *profiles vs the vehicle*. What it does not cover is
**profile-keyed tables inside the launcher's in-container preflight**, which is a third
place the same question is asked. The detector that would close it: **enumerate every
`dict` keyed by fixed32 mode and every comparison against a topology constant reachable
from the launcher, and assert each admits every profile in
`fr13_fixed32_topology`.** Both of this round's sites are in one twelve-line block, so
the enumeration is small and the check is static.

## R14.5 Status

**tail10 A/B still not measured.** H27 is banked as a clean, same-HEAD baseline and will
pair with H31 whenever H31 can boot — provided no commit intervenes, which is the usual
constraint. The credential-under-topology-change question stands **open and explicitly
unanswered**.

Standing verdicts unchanged: fused top-k PROMOTED, suffix pass gate REFUSE, split-K
recommended for the tier-B serving route on round 12's evidence.

---
---

# ROUND 15 (2026-08-19 08:29Z–08:48Z) — the preflight is fixed; H31 stopped at the credential tripwire, and the answer is NOT "the credential refused"

| item | verdict |
|---|---|
| **pairing decision** | **(a) pair against round-14 H27** — measured, not asserted; see §R15.1 |
| **gate re-earn** | **PASS rc=0** at `ba954f512` |
| **preflight (rounds 13–14 blockers)** | **FIXED** — H31 got past both the mode table and the tree-ancestry compare |
| **ARM H31** | **STOPPED at the credential tripwire**, 5 s, captured, no re-earn |

GPU this round: one gate re-earn plus a 5-second refusal.

## R15.1 The pairing decision, and why it is (a)

You asked me to state the decision and the reasoning. I made it **measurable rather
than editorial**:

* **What changed between `7286065fb` (H27's HEAD) and now** — 17 files: results docs,
  the serve vehicle, three launchers, one new lint + its test, the sweep, two tests.
  **Untouched:** `fr13_patch_fa2_tree_bias.py`, `fr10_phase4_patch_vllm_tree_gdn.py`,
  `fr13_fixed32_work_census.py`, `fr13_fixed32_topology.py`, `fr13_merged_drafter.py`,
  `fr13_fixed32_contract.py`, and every binary.
* **What a hydra27 arm executes under** — byte-identical across the diff:

  | | round-14 H27 (observed in `container_env.txt`) | this HEAD (resolved from the authority) |
  |---|---|---|
  | `FR13_FIXED32_MODE` | `hydra27_fixed32` | `hydra27_fixed32` |
  | `VALID_MASK` | `0x7abdffff` | `0x7abdffff` |
  | `ACTIVE_NODES` | 27 | 27 |
  | `PHYSICAL_DRAFTS` | 31 | 31 |
  | **`TAW_WALK_CAP`** | **12** | **`topology.WALK_CAP` = 12** |

The one change that touches the hydra27 path is the walk cap now being **supplied by
the kind block** instead of inherited from the sequence file. **The provenance moved;
the value did not.** So the diff is pairing-neutral for a hydra27 arm, and re-running
H27 would have bought an identical baseline for 3.5 h of GPU.

Recorded because it cuts against my own habit: I have refused to pair across HEADs
repeatedly in this campaign. The difference here is that I could *enumerate* what a
hydra27 arm executes and show it unchanged, rather than argue from the commit
subjects — which is the same standard I have been holding lane 4 to.

## R15.2 The preflight fix works

H31 passed both round-13/14 blockers: no `unsupported fixed32 mode`, no
`TREE differs from FIXED32_CHOICES`. The twelve-line block is genuinely profile-aware.

## R15.3 THE CREDENTIAL TRIPWIRE — and the answer to your explicit question

The refusal:

```
FAIL: launcher rc=1
fixed32 FA2 realpath mismatch:
  /home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so
  != <repo>/output/auto_research/.../_vllm_fa2_C.abi3.so
```

**Read literally this looks like a path bug. It is the tripwire, and the mechanism is
the answer you wanted.**

> **Does a topology change invalidate the FA2 byte credential?**
> **No — and the credential is never consulted.** The gqa_pair *arm* is excluded two
> layers earlier, by a mode gate on the selector, so nothing ever reaches the
> credential to refuse it.

Verified in source, two independent gates, both keyed on the mode:

1. **The promoted-default block** (`:1275-1276`) arms `gqa_pair` only when
   `FR13_FIXED32_MODE == "hydra27_fixed32"`. Under hydra31 it does not fire, so
   `FR13_FA2_QROW32_B1_PRODUCTION_ARM` stays **empty**.
2. **The selector predicate** requires `FR13_FIXED32_MODE == "hydra27_fixed32"` too —
   so even naming the arm explicitly would refuse, and refuse on the *mode*, not on the
   credential.

With no B1 selector active, the fixed32 contract then requires the **stock** FA2 at its
canonical in-repo path — and my runner had mounted the gqa_pair candidate `.so` for an
arm that never armed. Hence the realpath mismatch. It is a *downstream symptom* of an
arm silently not arming, which is the same class as round 1 §0.1 (the stale credential
degrading to the incumbent), except here the exclusion is topological rather than
temporal.

**No re-earn was attempted**, per your instruction.

## R15.4 What this means for the A/B, and the choice it forces

An H31 arm **is** runnable today — but only with the **incumbent** FA2 (production arm
named empty, stock `.so`). That makes H31-vs-H27 a **two-variable** comparison:
topology *and* FA2 kernel, with gqa_pair's banked ~−4.4 ms sitting entirely on H27's
side. Any hydra31 win measured that way is understated by roughly that amount, and any
loss is overstated.

Three ways forward, and the choice is a design call, not an operator one:

| option | what it measures | cost |
|---|---|---|
| **A. Both arms on the incumbent FA2** (name the production arm empty in H27 too) | topology alone, cleanly | one new H27 (~3.5 h) + H31 |
| **B. Re-earn the FA2 byte credential under hydra31** | topology with the promoted kernel on both sides | a gate re-earn *and* the ruling that hydra31 may carry a hydra27-earned kernel qualification |
| **C. Widen the selector's mode gate to admit hydra31** | same as B, without a re-earn | a credential-scope decision: does a byte gate earned on hydra27's tree describe hydra31's? |

**My recommendation is A**, and only A, unless Mark rules otherwise. It is the only one
that answers the question the A/B was scheduled to answer — *what does the tail10
topology do?* — without first settling a separate and genuinely open question about
credential scope. B and C both require deciding whether an FA2 byte qualification earned
against hydra27's physical tree describes a different tree; the launcher's own comment
on the twelve refused levers says it does not, and I would not route around that
judgement to save a serve.

## R15.5 Status

**tail10 A/B still not measured.** H27's round-14 baseline stands and is now formally
established as pairing-valid at this HEAD (§R15.1) — but it is a *promoted-stack*
baseline, so under option A it would itself need re-running on the incumbent.

Standing verdicts unchanged: fused top-k PROMOTED, suffix pass gate REFUSE, split-K
recommended for the tier-B serving route on round 12's evidence.

---
---

# ROUND 16 (2026-08-19 08:50Z–10:15Z) — option A: H27i drains clean; H31i refused at a FOURTH profile-varying compare, this one inside the CONTRACT API

| arm | boot/validation verdict |
|---|---|
| **H27i** | **PASS** — `swerc=0`, 4/4 tasks, 2/4 resolved, 17 804 census steps |
| **H31i** | **REFUSED in 5 s** — `fixed32 TREE text differs from canonical contract` |

## R16.1 The fused-top-k decision, and why I opted out rather than confirmed

You offered either "opt out on both" or "confirm both arms resolve identically". **I
chose to opt out — `FR14_FUSED_DRAFT_TOPK=0` on both arms — because I could not confirm
the alternative, and the failure mode of assuming is severe.**

The lever's guard requires the drafter's wide widths to be exactly `(3,3,3,3,3)` at the
five head depths, and it raises a hard `RuntimeError` **at the first `propose()`** if
they are not. The two trees are demonstrably different in shape past depth 2:

```
hydra27 FIXED32_CHOICES depth-fanout: {0:3, 1:5, 2:5, 3:5, 4:5, 5:2}
hydra31 TAIL10_CHOICES  depth-fanout: {0:3, 1:5, 2:5, 3:4, 4:4, 5:1}
```

The guard reads the MTP *head* widths rather than the tree fanout, so those numbers do
not settle it — but they are enough to establish that I cannot settle it **statically**,
and the only way to "confirm identical resolution" would have been to boot H31 with the
lever armed and find out mid-serve. Opting out costs nothing here (this pair is a
topology-isolation experiment, not a promoted-stack measurement) and removes a variable
plus a crash risk. Stated as instructed.

## R16.2 ARM H27i — the clean incumbent baseline

Container env attests the intended arm exactly: `FR13_FA2_QROW32_B1_PRODUCTION_ARM=`
(empty), stock `_vllm_fa2_C.abi3.so`, `FR14_FUSED_DRAFT_TOPK=0`,
`FR14_SUFFIX_PASS_GATE=0`, `FR13_DFWD_SPLIT=1`, `FR13_FIXED32_MODE=hydra27_fixed32`,
`ACTIVE_NODES=27`, `TAW_WALK_CAP=12`. Zero RuntimeErrors. `swerc=0`, wall 4 541 s.

```
step_wall_ms 215.498 · s_per_fwd_gpu 0.133974 · accept/event 4.2124
drafter 53.640 · committer 20.515 · overhead_other 7.369 · per-request TPS 26.058
27/32 on all 17 804 steps · mtp_forward_calls 4 · graph_replays 1 · walk cap 12
```

**This is the pairing basis option A needs**, and it is the first baseline in the
campaign that is single-variable-ready against a topology arm: incumbent kernel, no
top-k, no gate, no split-K.

## R16.3 ARM H31i — a fourth profile-varying compare, and the API itself is the problem

```
FAIL: launcher rc=1
fixed32 TREE text differs from canonical contract
```

H31i passed everything rounds 13–15 fixed — the vehicle's kind, the preflight's mode
table, the tree-ancestry compare, the walk cap — and then stopped at
`fr13_launch_forked_fa2_tree_server.sh:4531`, in a **different block** from round 14's
(the contract-identity section, not the preflight):

```python
if tree != contract.fixed32_tree_text():
    raise SystemExit("fixed32 TREE text differs from canonical contract")
if spec_config != contract.speculative_config_text():
    raise SystemExit("fixed32 SPEC_CONFIG differs from canonical contract")
```

**And this time the profile-blindness is in the API signature, not the call site:**

```
fixed32_tree_text()        -> str      # no profile parameter
speculative_config_text()  -> str      # no profile parameter
```

Both are parameterless and both encode hydra27's tree — I verified `tree_text` carries
hydra27's choices, and `speculative_config_text()` embeds a `speculative_token_tree`
string, so it is profile-varying too.

**Consequence, and it is the round-14 pattern repeating two lines apart:** fixing
`fixed32_tree_text()` alone moves the refusal to the **very next line**. Both must
become profile-aware in the same edit — either by taking a mode argument, or by the call
sites passing one. Rounds 13–15 fixed *call sites*; this is the first blocker where the
**contract API can only ever answer for one profile**, so no call-site edit can fix it.

Sites: `:4531` and `:4533` in all **three** launcher families (the message exists at
`fr13_launch_forked_fa2_tree_server.sh:4531`,
`fr14_leg3_launch_nomiddleware.sh:4347`, `fr14_armb_leg3_launch_nomiddleware.sh:4370`),
plus the two contract functions.

## R16.4 The tally, and what it says about the detector

| round | profile-blind consumer |
|---|---|
| 13 | serve vehicle (kind + XFLAGS) |
| 14 | preflight mode table **and** tree-ancestry compare |
| 15 | *(not a compare — the FA2 mode gate, by design)* |
| **16** | **contract-identity compares — `fixed32_tree_text` / `speculative_config_text`** |

Pass 90's mode-table parity lint (the 17th pair) covers **dicts keyed by fixed32 mode**.
Neither of this round's sites is a dict — they are **equality compares against a
parameterless contract accessor**, which no key-parity check can see.

The detector that would have caught all four rounds in one shot: **enumerate every
`contract.*` accessor and module constant that encodes a tree, mask, node count, walk
cap or spec-config, and assert each either takes a profile argument or is proven
profile-invariant.** That is a static scan of one module's public surface, and it turns
"which consumer did we miss?" into a closed list rather than a boot-by-boot discovery.

## R16.5 Status

**tail10 A/B still not measured — but the baseline half is now banked and clean.**
H27i is the correct single-variable control and needs no re-running when H31i boots,
provided HEAD holds. The credential-scope question from round 15 remains parked and is
*not* on the critical path under option A.

Standing verdicts unchanged: fused top-k PROMOTED, suffix pass gate REFUSE, split-K
recommended for the tier-B serving route on round 12's evidence.

---
---

# ROUND 17 (2026-08-19 10:35Z–10:41Z) — H31i refused at a FIFTH site, and this one would have BOOTED into a silently wrong drafter

**Boot verdict: REFUSED in 5 s.** GPU cost: one 5-second container. **The ladder past
position 10 remains unmeasured.**

## R17.1 Pre-flight checks I ran before spending GPU — all three passed

Your pairing claim, verified independently rather than accepted:

* both accessors are now profile-parameterised
  (`fixed32_tree_text(profile='hydra27_fixed32')`, same for
  `speculative_config_text`);
* **the current hydra27 default is byte-identical to the tree ARM H27i actually
  executed** — I extracted the `speculative_token_tree` from H27i's own
  `container_env.txt` and compared. **My banked baseline stands.**
* `fixed32_tree_text('hydra31_fixed32')` genuinely differs from the hydra27 default.

And your floor-gate question, verified rather than assumed: `serve_variant` *does*
import from `fr13_floor_gate` (three sites), but the imported helpers —
`validate_fixed32_run_subset`, `build_fixed32_chat_traffic_audit`,
`_validate_fixed32_ingress_reports`, `load_fixed32_ingress_ledger` — **none reference
`FIXED32_MODE_SPECS`**, and my arm runner never calls the floor gate at all (0 refs).
**The missing hydra31 spec did not bite, as you predicted.**

## R17.2 The fifth site

```
File "/workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py", line 10302,
  in _fr13_fixed32_validate_patch_env
RuntimeError: fixed32 requires FR13_FIXED32_TAW_WALK_CAP=12
```

The drafter patcher's own env validator hardcodes the hydra27 walk cap:

```python
if int(os.environ.get("FR13_FIXED32_TAW_WALK_CAP", "0")) != 12:
    raise RuntimeError("fixed32 requires FR13_FIXED32_TAW_WALK_CAP=12")
```

hydra31's walk cap is **16** (`topology.TAIL10_WALK_CAP`) — the exact quantity round
14's fix taught the vehicle to supply per-profile, and the one the census expects to
move 12 → 16 (**the +33 % cfwd this A/B was scheduled to measure**). The vehicle
correctly supplied 16; the patcher refuses anything but 12.

## R17.3 The part that matters more than the literal

I checked whether this is one stale number or something deeper, and it is deeper.

```
patcher _FR13_FIXED32_CHOICES  == hydra27 FIXED32_CHOICES : True
patcher _FR13_FIXED32_CHOICES  == hydra31 TAIL10_CHOICES  : False
hydra31 mentions in the whole patcher                     : 2
   (both are mask/active-count tables — :163 and :1786; neither is the tree)
```

**The drafter patcher is bound to hydra27's tree by a module-level constant, with no
profile selection anywhere.** It builds the parent/ancestry index from
`_FR13_FIXED32_CHOICES` immediately after the walk-cap check — the next line is
`if len(_FR13_FIXED32_CHOICES) != 31`, which *passes* for both profiles because both
trees have 31 drafts.

So: **fixing the walk-cap literal alone would let H31i boot — and it would boot with a
drafter whose ancestry was built from the wrong tree.** Both trees are 31 wide, so
nothing downstream would catch it on shape; the census's `(4,6)` pair, 27/32 rows and
31 physical drafts would all still validate. That is the "wrong fix that would have
booted" hazard in its most dangerous form yet: not a refusal one line later, but a
**silent** wrong-topology serve producing numbers that look like a tail10 result.

It is the same failure family as round 6, when a serve reported a kernel it never ran —
and the campaign's answer then is the answer now: **the fix must make the binding
observable, not just permissive.** Whatever lands should assert that the drafter's tree
matches the served profile at boot, so that a mismatch refuses rather than serves.

## R17.4 The tally

| round | profile-blind consumer | would a naive fix have booted? |
|---|---|---|
| 13 | serve vehicle | — |
| 14 | preflight mode table **+** tree-ancestry compare | yes, refusal moved 3 lines |
| 16 | contract accessors (`tree_text`, `spec_config`) | yes, refusal moved 2 lines |
| **17** | **drafter patcher: walk-cap literal + hardcoded tree** | **yes — and it would have SERVED** |

Four rounds, four different layers, and the detector proposed after round 16 —
*enumerate every accessor/constant encoding a tree, mask, node count, walk cap or
spec-config and assert it takes a profile argument or is proven profile-invariant* —
would have caught **all four**, including this one: `_FR13_FIXED32_CHOICES` is exactly a
tree-encoding module constant with no profile parameter, and the walk-cap literal is
exactly a walk-cap constant. I recommend that scan cover `fr10_phase4_patch_vllm_tree_gdn.py`
as well as the contract module, since this round proves the class is not confined to one file.

## R17.5 Status

**tail10 A/B: baseline banked, treatment arm still unbootable.** H27i
(`step_wall 215.498`, `s_per_fwd_gpu 0.133974`, `accept 4.2124`, TPS 26.058, 17 804
steps) remains valid and re-verified this round against the current contract. The four
headline questions — **the ladder past position 10**, accept vs +4.3/+7.7 %, cfwd vs
+33 %, step_wall/TPS — are all still open, and all four need the same single boot.

Standing verdicts unchanged: fused top-k PROMOTED, suffix pass gate REFUSE, split-K
recommended for the tier-B serving route on round 12's evidence.

---
---

# ROUND 18 (2026-08-19 11:19Z–16:06Z) — two refusals in a NEW SOURCE ROOT; one was my own spec error, and the corrected baseline is banked

**Boot verdict: H31i refused twice.** The first refusal was **mine, not a defect**. The
second is a seventh profile-blind site, in `src/` rather than `scripts/` — a source root
no sweep has covered.

| arm | outcome |
|---|---|
| H31i (attempt 1) | **REFUSED** — `FR13_HOST_TAIL_PREP_BAKE=1` incompatible with hydra31 — **my arm spec** |
| **H27n** (corrected baseline) | **PASS** — `swerc=0`, 4/4 tasks, **69 389 census steps** |
| H31n (attempt 2) | **REFUSED** — `_FR13_FIXED32_MODES` allowlist in `src/lumo_flywheel_serving/` |

## R18.1 Pre-flight, verified before GPU

The observable ancestry binding is present (`TREE_ANCESTRY_SHA256` reduced at the bake
point), and — the check my pairing standard rests on — **the current hydra27 default is
still byte-identical to the tree ARM H27i executed**, re-verified against H27i's own
`container_env.txt`. Your pairing claim holds.

## R18.2 The first refusal was my error, and the guard was right

```
RuntimeError: FR13_HOST_TAIL_PREP_BAKE=1 requires fixed32 to be armed but
FR13_FIXED32_MODE is 'hydra31_fixed32'. The lever bakes the tree depth-position
plan as a literal … the baked plan would feed wrong RoPE depth offsets into
positions[...] = base + depth_offsets and nothing downstream would catch it.
```

`src/lumo_flywheel_serving/fr13_host_tail_prep.py:81`, admitting only
`('hydra27_fixed32','tail6_fixed32')`.

**This is not a defect — it is round 17's demand already satisfied.** The lever
genuinely bakes a hydra27 literal, and the guard refuses rather than serving a silently
wrong topology, naming the exact mechanism ("nothing downstream would catch it").

**The error was mine.** I had carried `FR13_HOST_TAIL_PREP_BAKE=1` since round 2 because
it was part of the *composed/promoted* arm shape — and then carried it into a
*topology-isolation* experiment, where it is both unnecessary and topology-bound by
construction. H27i had run with it **on**, so pairing H31 (off) against H27i (on) would
have been two variables.

I fixed my spec rather than route around the guard: **`PREP_BAKE=0` on both arms**, and
re-ran the baseline. Cost: one 4h 43m serve, charged to me.

## R18.3 ARM H27n — the corrected baseline, and it is the strongest yet

`swerc=0`, wall 16 633 s, 4/4 tasks, 1/4 resolved. Env attests exactly:
`PRODUCTION_ARM=` (stock FA2), `FUSED_DRAFT_TOPK=0`, `SUFFIX_PASS_GATE=0`,
**`HOST_TAIL_PREP_BAKE=0`**, `MODE=hydra27_fixed32`, walk cap 12.

```
step_wall_ms 218.702 · s_per_fwd_gpu 0.133693 · accept/event 3.8690
drafter 56.642 · committer 20.569 · overhead_other 7.797 · per-request TPS 25.365
69 389 census steps — 27/32 on every one, mtp_forward_calls 4, graph_replays 1
```

**69 389 steps is by far the largest census in the campaign** (3.3× C', 3.9× H27i), so
this baseline has the tightest statistics of any arm banked so far. It supersedes H27i
as the pairing basis, and unlike H27i it is single-variable-ready in *every* dimension:
incumbent kernel, no top-k, no gate, no split-K, no baked host-tail literal.

## R18.4 The seventh site — and the sweep's blind spot is a whole source root

```
RuntimeError: FR13_FIXED32_MODE: invalid fixed32 route source(s):
  env:FR13_FIXED32_MODE='hydra31_fixed32',
  sidecar:/logs/fr13_fixed32_mode.flag='hydra31_fixed32'
```

`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:843`:

```python
_FR13_FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
```

used at `:982` (route-source validation) and `:1165`. **Zero hydra31 mentions in the
entire file.** Note the refusal names *both* sources as invalid — env **and** sidecar —
so the mode plumbing worked perfectly; the allowlist that receives it simply predates
hydra31.

**Every previous site was in `scripts/`. This is the first in `src/lumo_flywheel_serving/`
— the serving package rather than the tooling.** That is why every sweep and lint so far
has missed it: they enumerate the wrong root.

I enumerated that root so the next fix can be complete rather than incremental:

| file in `src/lumo_flywheel_serving/` | hydra27 refs | hydra31 refs |
|---|---|---|
| `fr10_gdn_tree_kernel.py` | 8 | **0** |
| `fr13_sfwd_conv_postprep_fusion.py` | 7 | **0** |
| `fr13_host_tail_prep.py` | 2 | **0** |
| `fr13_gdn_gqa_group3.py` | 1 | **0** |
| `fr13_fixed32_commit_slot_scatter.py` | 1 | **0** |

Four distinct mode allowlists, all spelled `frozenset(("tail6_fixed32",
"hydra27_fixed32"))` or the tuple equivalent:

```
fr10_gdn_tree_kernel.py:843          _FR13_FIXED32_MODES
fr13_sfwd_conv_postprep_fusion.py:44 FIXED32_MODES
fr13_gdn_gqa_group3.py:18            FIXED32_MODES
fr13_fixed32_commit_slot_scatter.py:74 FIXED32_MODES
```

plus inline comparisons at `fr10_gdn_tree_kernel.py:3841`, `:4317` and
`fr13_sfwd_conv_postprep_fusion.py:961`.

**Not all of these should be widened.** `fr13_host_tail_prep.py` is correctly
hydra27-only (§R18.2) — it bakes a literal. The others must each be adjudicated:
*does this component's behaviour depend on the tree, and if so does it derive from the
authority or assume hydra27?* That is the same adjudication the shape-literal sweep
made in the patcher, applied to a root nobody has swept.

## R18.5 Status

**tail10 A/B: baseline banked at its strongest, treatment arm still unbootable.**
Seven profile-blind sites across five layers — vehicle, preflight, contract accessors,
drafter patcher, and now the serving package. The four headline questions — **the ladder
past position 10**, accept vs +4.3/+7.7 %, cfwd vs +33 %, step_wall/TPS vs H27n's
218.702 / 25.365 — remain open and still need one boot.

Standing verdicts unchanged: fused top-k PROMOTED, suffix pass gate REFUSE, split-K
recommended for the tier-B serving route on round 12's evidence.

---
---

# ROUND 19 (2026-08-19 16:45Z–16:53Z) — the eighth site, and it is the inverse of the first seven: the AUTHORITY is incomplete

**Boot verdict: REFUSED**, but deeper than ever — the engine loaded the model, passed
every launcher, contract, patcher and serving-package check, and died inside
**EngineCore runtime** during topology preseed. GPU cost: one ~6-minute container.

## R19.1 Pre-flight, verified

Site 7 is fixed in the shape described — `fr10_gdn_tree_kernel.py:865` now carries a
**widened route vocabulary** `("tail6_fixed32","hydra27_fixed32","hydra31_fixed32")`
while `_FR13_FIXED32_MODES` at `:867` stays narrow for the byte-qualified levers.
"Vocabulary widened, qualification kept", exactly as stated.

And my baseline: **the hydra27 default is still byte-identical to the tree H27n
executed**, re-verified against H27n's own `container_env.txt`. H27n stands.

## R19.2 The refusal

```
File ".../mamba/gdn_linear_attn.py", line 14170
File "/workspace/scripts/fr13_device_multidraft_kernel.py", line 3228,
  in fr13_fixed32_taw_preseed
RuntimeError: unknown FR13 fixed32 preseed mode 'hydra31_fixed32'
```

## R19.3 Why this is a class no scan has named

The consumer is **not** at fault. Read it:

```python
if mode not in topology.VALID_MASK_BY_MODE:
    raise RuntimeError(f"unknown FR13 fixed32 preseed mode {mode!r}")
expected_mask = int(topology.VALID_MASK_BY_MODE[mode])
```

`fr13_device_multidraft_kernel.py` contains **zero hardcoded mode literals** for this —
it delegates to the authority's by-mode mapping, which is **precisely the remediation
pattern rounds 13–18 kept prescribing**. It asked the authority and the authority did
not know.

I inspected the authority's own per-profile mappings:

| mapping in `fr13_fixed32_topology.py` | keys | missing |
|---|---|---|
| `PROFILES` | hydra27, **hydra31** | tail6 |
| `VALID_BY_MODE` | hydra27, tail6 | **hydra31** |
| `VALID_MASK_BY_MODE` | hydra27, tail6 | **hydra31** |

Meanwhile the authority exposes **29 standalone hydra31/TAIL10 constants** —
`HYDRA31_VALID_MASK`, `TAIL10_CHOICES`, `TAIL10_WALK_CAP`, `TAIL10_TREE_ANCESTRY_SHA256`,
`TAIL10_PHYSICAL_PARENT_SHA256`, and 24 more. **The profile is fully described and
partially indexed.**

**Sites 1–7 were consumers that failed to consult the authority. Site 8 is a consumer
that consulted it correctly and was failed by it.** That inverts the remediation:
routing more consumers through `*_BY_MODE` — the fix applied repeatedly in rounds
13–18 — *increases* exposure to this defect rather than reducing it. Every consumer
converted to "ask the authority" becomes a new way for an incomplete mapping to surface.

Note also that the gap is **bidirectional**: `PROFILES` is missing `tail6_fixed32`. So
this is not "hydra31 was added late" — the authority's three per-profile mappings have
three different key sets, and no invariant ties them together.

## R19.4 The detector — a self-check, not a scan

Every detector proposed so far scans *consumers*. This one cannot be found that way,
because the consumer is correct. The right check lives **inside the authority**:

> **Assert that every `*_BY_MODE` mapping in `fr13_fixed32_topology` has exactly the
> same key set as the module's profile roster.**

Three lines, no GPU, and structurally better than every previous detector because it is
**self-maintaining**: it needs no enumeration of consumers and no update when a
consumer is added. Any future profile that is described but not indexed fails
immediately, in the authority, before a boot.

Given `PROFILES` is itself missing `tail6_fixed32`, the roster used for the comparison
should be derived from the constants (the set of `PROFILE_*` names), not from
`PROFILES`.

## R19.5 Status

Eight sites, six layers, two source roots — and the eighth is the first that is *not* a
missing consumer update but a missing **authority row**.

**tail10 A/B: baseline banked, treatment arm still unbootable.** H27n
(218.702 / 0.133693 / accept 3.8690 / TPS 25.365, **69 389 steps**) stands, re-verified
this round. **The ladder past position 10, accept vs +4.3/+7.7 %, cfwd vs +33 %, and
step_wall/TPS remain open and still need one boot.**

Standing verdicts unchanged: fused top-k PROMOTED, suffix pass gate REFUSE, split-K
recommended for the tier-B serving route on round 12's evidence.

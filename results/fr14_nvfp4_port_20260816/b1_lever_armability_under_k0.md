# FR14 — can the built B1 levers be armed on the K0 production config?

**Answer: no, neither of them, and not for want of a credential.** Both are
blocked by the launcher's own fail-closed legality predicate, which names the
K64 shape Mark just parked. Found by reading the predicates before spending GPU;
**no GPU was spent.**

## 1. FA2 qrow32 `gqa_pair` at B1 — BLOCKED under K0

`scripts/fr13_launch_forked_fa2_tree_server.sh:2000-2014`, the predicate that
guards *any* B1 qrow32 selector (candidate **and** production — a non-zero
`_FR13_FA2_QROW32_B1_SELECTOR_COUNT` enters it):

```bash
[[ "${FR13_FIXED32_MODE:-}" == "hydra27_fixed32" \
   && "$MAX_NUM_SEQS" == "1" \
   && "${SWE_CONCURRENCY:-}" == "1" \
   && "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "${FR13_DRAFT_VOCAB_K:-65536}" == "65536" \
   && "${FR13_DRAFT_VOCAB_BLOCKS:-}" == "/workspace/scripts/fr13_dvk_subset_blocks.json" \
   && ... ]] || {
  echo "FR13 qrow32 B1 selector requires Hydra27 K64/root1 B1 and exact binary/source provenance" >&2
  exit 2
}
```

Three separate clauses make this mutually exclusive with the K0 production
config Mark ruled for:

| clause | K0 serve | verdict |
|---|---|---|
| `FR13_DRAFT_VOCAB_K == 65536` | `0` | **refuses** |
| `FR13_DRAFT_VOCAB_ROOT == 1` | `0` | **refuses** |
| `FR13_DRAFT_VOCAB_BLOCKS == …/fr13_dvk_subset_blocks.json` | unset/irrelevant under K0 | **refuses** |

and the production arm additionally requires `-z FR13_NEEDS_ALLOW`
(`:2135` family), while K0 requires `FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"` —
a fourth, independent contradiction.

**So "aggressive + K0 + gqa_pair armed" is not constructible at this HEAD.** Not
because the credential is missing, but because the launcher refuses the
combination by construction.

### A second fact that matters more than the credential

The same predicate requires `FR13_FIXED32_MODE == "hydra27_fixed32"`. The stock
B1 serve arm — the one that produced every FR14 headline, including arm A's
218.764 ms and arm B's numbers — is **`tail6_fixed32`**. Its container env
confirms it: `FR13_FA2_QROW32_B1_PRODUCTION_ARM=` (empty).

`FR13_FA2_QROW32_B1_PRODUCTION_ARM_DEFAULT=gqa_pair` exists in the canonical
registry, but the launcher applies the default only "in the one shape where it
is legal", and that shape is Hydra27. **The gqa_pair lever has therefore never
been live on the tail6 stock arm at all** — before or after the model swap. Any
plan that assumed the stock serve was already carrying it is mistaken.

### Why re-earning the gate first would have been wasted GPU

Re-earning it is *possible*: the candidate binary is on this box with the exact
pin (`/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810/`, sha
`3560cdc0…`, 299,815,552 B), its `source_closure.json` canonical sha matches the
runner's `172b5e71…` pin, and the FA2 head matches — no rebuild needed. The gate
runner (`fr13_run_b1_k64_qrow32_split2_live_gate.sh`) hardcodes **and asserts**
`ROOT=1 / K=65536 / NEEDS_ALLOW=` at `:136-148` and pins
`FR13_DRAFT_VOCAB_ROOT=1` into the credential binding at `:238`.

So the gate can only ever produce a **K64/Hydra27** credential — for a shape that
is now parked. Running it would have cost 1–2 GPU-h to earn evidence with no
production consumer.

## 2. GDN `single_launch` at B1 — NOT ARMABLE TODAY

Two independent blockers:

* **Legality shape.** `scripts/fr13_canonical_env.sh:104-114`: the launcher
  applies it "in the one shape where it is legal: fixed32 B1/B4 at matching
  concurrency, **K64/root1**, BV=8, FULL_AND_PIECEWISE, presenting a PASS
  credential bound to the exact serving HEAD." Same K64 contradiction.
* **Credential.** It is a *credentialed* selector the launcher refuses "wherever
  no HEAD-bound gate credential was presented". The existing byte-gate PASS
  (`427d8cba`) is bound to a different HEAD **and** the pre-swap checkpoint — it
  died with the model swap exactly as gqa_pair's did. Re-earning it is a
  separate GPU gate, i.e. new work, which the amendment scoped out.

`FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT` is `0` — deliberately off.

**Verdict: not armable today. Skipped, as instructed.**

TAW at B1 was already out of scope (B=1 refused by construction).

## What this means for the composed serve

The amendment's "aggressive + K0 + gqa_pair (+ single_launch)" arm cannot be
built at this HEAD. The options, in increasing cost:

1. **Serve aggressive + K0 with no B1 levers.** Constructible right now; it is
   the config Mark ruled for, and it is what the killed
   `fr14_b1_stock_20260817T051722Z` arm was. Cost: one serve.
2. **Serve aggressive + K64 + gqa_pair on Hydra27**, re-earning the gate first.
   Constructible, but it measures the config Mark parked, on the arm the
   headlines were never taken on. Cost: gate (1–2 GPU-h) + serve.
3. **Extend the legality predicate to the K0 shape**, then re-earn the gate
   under it. This is the honest path to "K0 + levers", and it is exactly the
   follow-up config train already flagged when K0 was ruled in
   (`d908924a0`): promote K=0 to the canonical default and retire the K64
   machinery from the serving path. The predicate is a credentialed safety gate,
   so widening it is a deliberate decision with its own review, and the gate
   must then be re-earned **in the new shape** — a credential earned under
   K64/Hydra27 does not describe a K0/tail6 serve.

Option 3 is the one that ends with Mark's production candidate actually
measured with its levers. It is a config train plus a gate, not a serve.

**Recommendation: run option 1 now** (it is the ruled production config and
costs one serve), and schedule option 3 as the next config train rather than
spending GPU on a parked shape.

---

# EXTENSION (2026-08-17): this is not a gqa_pair problem, it is portfolio-wide

Prompted by the B4 max-stack chain, I enumerated every launcher predicate that
requires the K64 draft vocabulary. **Twelve of them.** They gate essentially the
whole FR13 credentialed-lever portfolio:

| launcher line | lever it guards |
|---:|---|
| 885 | CUTLASS wave, `k64_root` qualification profile |
| 970 / 986 / 1001 | packed-walk node trust, active depth, node-trust production |
| 1481 | draft-head U8 (B1) |
| 1609 | draft-head M4 U8 (B4) |
| 1722 | DFWD K64 top3 |
| 1806 | draft-head padding / direct M32 |
| **1974** | **FA2 qrow32 B4 GQA-pair — timing AND production** |
| **2007** | **FA2 qrow32 B1 selector — candidate AND production** |
| 2120 | FA2 qrow16 live A/B |

Every one of them tests `"${FR13_DRAFT_VOCAB_K:-65536}" == "65536"`, and most
also test `FR13_DRAFT_VOCAB_ROOT == "1"` and/or `-z FR13_NEEDS_ALLOW`.

**So Mark's K0 ruling parks more than the K64 drafter: it parks the arming path
for every built lever simultaneously.** Neither the B1 composed serve
(K0 + gqa_pair) nor the B4 max-stack serve (K0 + padded gqa_pair + single_launch
+ TAW) is constructible at this HEAD. The launcher refuses each one at its own
fail-closed predicate — which is the gate working exactly as designed.

Re-earning any of these gates first does not help: every gate runner hardcodes
and asserts the same K64 identity and pins it into the credential binding, so it
can only ever mint a K64-shaped credential — for a shape that is now parked.

## The precedent that shows the way out — it already exists in-tree

The CUTLASS wave lever already carries **two qualification profiles**
(`:2386-2400`):

```bash
case "$FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE" in
  full_vocab)
    [[ "$FR13_DRAFT_VOCAB_ROOT" == "0" \
       && "${FR13_DRAFT_VOCAB_K:-65536}" == "0" \
       && "${FR13_NEEDS_ALLOW:-}" == "FR13_DRAFT_VOCAB_K=0" ]] || {
      echo "CUTLASS full_vocab B4 qualification requires the K0 workload" >&2
  k64_root)
    [[ -z "${FR13_NEEDS_ALLOW:-}" ]] || {
      echo "CUTLASS k64_root B4 qualification forbids a K0 override" >&2
```

So the codebase **already models "this lever, qualified under the K0
workload"** — for exactly one lever. That is the template.

## What the max-stack chain actually costs

Not "re-earn the gates, then arm them". It is:

1. **A config train** giving each lever a `full_vocab` (K0) qualification
   profile alongside its `k64_root` one, mirroring the CUTLASS pattern. This
   widens credentialed safety predicates, so it is a deliberate, reviewed change
   — not a knob.
2. **A gate re-earn per lever UNDER K0**, because a credential earned on the
   K64 workload does not describe a K0 serve. The K64-era runners cannot do this
   as written (they assert the K64 identity and pin it into the binding), so
   each needs a K0 variant.
3. Only then the composed serves.

The kernels themselves are quant-agnostic and the geometry is identical, as
Mark says — no rebuilds. The blocker is entirely in the *arming predicates and
the credential shapes*, not in the binaries.

## Recommendation

Serve **aggressive + K0, no levers** now — it is the ruled production config,
costs one serve, and is the honest FR14 K0 anchor. Take the lever chain as its
own config train (step 1 above) with Mark's explicit sign-off on widening the
predicates, then re-earn gates in the K0 shape.

Spending 4–6 GPU-h re-earning K64-shaped credentials before that decision would
produce evidence no production serve can consume.

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

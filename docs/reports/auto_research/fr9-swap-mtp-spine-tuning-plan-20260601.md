# FR9 Swap: Independent-Row MTP/Spine Tuning Before Enhanced Tree

**Date:** 2026-06-01
**Status:** Proposed next step before editing the enhanced MTP+suffix tree strategy.
**Related docs:**
- `docs/reports/auto_research/fr9-superset-closeout-20260601.md`
- `docs/reports/auto_research/fr9-independent-rows-options-20260601.md`
- `docs/reports/auto_research/round-f-enhanced-tree-mtp-suffix-strategy-20260530.md`
- `docs/unified-tree-spec-decode.md`

## 1. Decision

Before building the enhanced MTP+suffix tree, swap the branch-verification
substrate to the FR9 independent-row path and tune it directly.

The enhanced-tree plan should not start from the packed in-tree token-tree
verifier. On this Qwen/GDN hybrid, the packed tree route is capped by recurrent
state sharing: sibling branches contaminate the state scan, so the contained
top-1 chain is worse than native E5. FR9 avoids that failure mode by using
multiple native co-resident sequence rows, each with its own GDN recurrent
state.

This means FR9 is not only a closeout result. It is the right prototype surface
for branch verification:

- tune MTP depth with `--mtp`;
- tune branch width with `--spines`;
- preserve native GDN recurrent state per candidate row;
- select the longest accepted row;
- commit winner tokens and sync the winner's post-accept recurrent state back
  to sibling rows.

Enhanced MTP+suffix should be layered after this sweep, using FR9 as the
branch/verifier substrate and suffix decoding as a cheap chain extender.

## 2. What FR9 Actually Does

FR9 has two public row modes under `--config Fb`:

- `--row-mode tree`: packed `speculative_token_tree` route. This is useful as a
  diagnostic, but is closed for this cycle on GDN because free-running path0
  stayed below E5.
- `--row-mode independent`: native linear MTP on persistent hidden rows. This is
  the working FR9 path.

In independent mode:

1. the launcher creates `N = --spines` co-resident request rows;
2. row 0 is the native top-1 MTP chain, equivalent to E5 when `--mtp 5`;
3. later rows use alternate root tokens, then continue through normal linear
   MTP;
4. each row has its own native per-sequence GDN recurrent state;
5. the verifier accepts a prefix for each row;
6. winner = row with the longest accepted prefix;
7. winner tokens are committed to the public sequence;
8. winner post-accept recurrent state is copied back to sibling rows through
   vLLM's Mamba state-copy path.

The important property is state isolation. FR9 pays extra row compute, but it
does not need to invent a GDN tree scan kernel. That makes it the practical
near-term branch verifier.

## 3. Current Evidence

FR9 already proved the important correctness invariant:

| Config | Accept/Event | Direct Probe Tok/s | Notes |
|---|---:|---:|---|
| In-tree tree mode path0 | 1.751 | n/a | closed route; below E5 |
| Independent `spines=1` | 2.772 | 44.45 | E5-equivalent direct probe |
| Independent `spines=2` winner | 3.442 | 47.50 | `superset_violations=0` |

The clean speedup is therefore `47.50 / 44.45 = 1.0685x`, not the earlier
workload-confounded comparison against the agentic E5 number.

The result is modest but meaningful:

- accept/event improved by about 24%;
- direct probe throughput improved by about 6.85%;
- correctness held under greedy proof;
- production temp-0.6 stability held with `viol=0` and `missing_sum=0`;
- the agentic temp-0.6 throughput number is not comparable because that run was
  workload-confounded.

## 4. Why This Should Precede Enhanced Tree

The enhanced MTP+suffix idea depends on cheap verification. If branch
verification is built on the packed tree route, we start from the known bad
surface: shared GDN recurrent state and degraded path0.

FR9 gives us a better question:

> How much branch width and MTP depth can we afford when every branch has a
> correct native recurrent state?

Only after that answer is known should suffix decoding be added. Suffix should
extend the winning chain cheaply; it should not be used to hide an expensive or
incorrect branch verifier.

## 5. Tunable Parameters

### 5.1 MTP Depth

`--mtp` controls how many speculative tokens each row proposes.

Candidate sweep:

```text
mtp in {1, 2, 3, 4, 5}
```

Expected behavior:

- `mtp=1`: cheapest verify event, low depth, useful as a branch/root-rescue
  baseline.
- `mtp=2`: likely strong for suffix composition because MTP supplies a short
  model-grounded anchor.
- `mtp=3`: possible sweet spot if depth-3 acceptance remains high.
- `mtp=4..5`: closer to E5, but may hit diminishing returns; useful as the
  baseline to beat, not necessarily the final strategy.

Do not assume `mtp=5` is optimal. Native MTP often peaks before max depth when
deeper proposals add verification cost faster than accepted-token gain.

### 5.2 Spine Count

`--spines` controls how many independent candidate rows compete.

Launcher support exists for:

```text
spines in {1, ..., 10}
```

Practical sweep:

```text
spines in {1, 2, 3, 4, 6}
```

Expected behavior:

- `spines=1`: E5-equivalent chain baseline.
- `spines=2`: already proven useful; rescues root misses.
- `spines=3..4`: may recover more misses, but compute cost rises.
- `spines>=6`: likely only wins if higher batch/weight reuse makes extra rows
  cheap enough.

### 5.3 Branch Placement

The current FR9 prototype mainly varies the root token for each row:

```text
row 0: top1 root -> greedy MTP continuation
row 1: top2 root -> greedy MTP continuation
row 2: top3 root -> greedy MTP continuation
...
```

This is good for root misses, but it is not necessarily the best use of width.
The next useful variant is position-targeted rescue:

```text
row 0: a1 -> a2 -> a3 -> a4 -> a5
row 1: b1 -> ...                    # rescue position 1
row 2: a1 -> b2 -> ...              # rescue position 2
row 3: a1 -> a2 -> b3 -> ...        # rescue position 3
row 4: a1 -> a2 -> a3 -> b4 -> ...  # rescue position 4
```

This asks a better question than pure root width:

> At which MTP position do we lose the most accepted tokens, and can a targeted
> alternate row recover that position cheaply?

This is also closer to the future enhanced-tree shape, but still uses correct
independent recurrent state per row.

## 6. Required Plumbing Before Full Sweep

The relaunch helper already exposes the knobs:

```bash
python scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config Fb \
  --row-mode independent \
  --mtp 2 \
  --spines 3
```

The agentic driver currently exposes `--mtp` and `--row-mode`, but not
`--spines`. Before a full agentic sweep, add a driver flag:

```text
scripts/run_codex_experiment.py --spines N
```

and forward it into:

```text
scripts/swe_x86_helpers/relaunch_qwen36_round.py --spines N
```

Until then, direct relaunch/probe sweeps can use the helper directly, or the
driver can rely on `LUMO_TREE_SPINES` only if verified end to end.

## 7. Sweep Plan

### Phase A: Direct Greedy Baselines

Goal: get clean apples-to-apples token/sec and accept/event curves before
running expensive agentic workloads.

Matrix:

```text
row_mode = independent
temp = 0
mtp = {1, 2, 3, 4, 5}
spines = {1}
```

Outputs:

- accept/event;
- acc0;
- full-accept at the configured depth;
- direct probe completion tok/s;
- per-position LCP/accept;
- state-sync invariant status.

Decision:

- choose the best `mtp` depths for width sweep;
- do not carry forward depths whose marginal accept gain is eaten by tok/s.

### Phase B: Width Sweep

Matrix:

```text
row_mode = independent
temp = 0
mtp = best depths from Phase A, likely {1, 2, 3, 5}
spines = {1, 2, 3, 4, 6}
```

Outputs:

- winner accept/event;
- direct probe tok/s;
- recovery rate versus spine 0;
- superset violations;
- missing winner events;
- GPU utilization and memory headroom;
- per-row utilization.

Decision:

- keep only widths with positive tok/s gain versus `spines=1`;
- reject widths where accept improves but tok/s drops materially;
- separately track if high batch changes the conclusion.

### Phase C: Position-Targeted Spine Prototype

Implement a small prototype where extra rows rescue different MTP positions
instead of only different root tokens.

Minimum variants:

```text
root-only width:        rows rescue position 1 only
position-targeted w3:   rescue positions 1, 2
position-targeted w4:   rescue positions 1, 2, 3
```

This phase is important because enhanced tree is not just "more root width".
The useful future shape is a compact set of high-value alternate paths.

Decision:

- if position-targeted rows beat root-only rows at equal width, use that as the
  branch substrate for enhanced MTP+suffix;
- if not, keep root-width FR9 and add suffix only to the winning chain.

### Phase D: Agentic Stability Confirmation

Run only the shortlisted configurations under the same agentic settings as the
E5/FR9 comparison.

Required controls:

- same workload subset;
- same concurrency;
- same temperature/top-p;
- same wall-time and timeout policy;
- same `max_num_seqs >= concurrency * spines`;
- no mid-decode condense/removal of sibling rows.

Do not compare agentic runs if resolved rate, retry behavior, or task completion
state diverges materially. Keep direct-probe speed and agentic stability as
separate gates.

## 8. Metrics That Matter

Primary:

- `completion_tokens/sec` on direct probe;
- accept/event;
- `superset_violations = 0`;
- `missing_sum = 0`;
- no engine death, CUBLAS/CUDA errors, or illegal memory access;
- recurrent-state sync correctness.

Secondary:

- acc0;
- full-accept rate;
- recovery rate versus spine 0;
- per-position recovery;
- per-row win rate;
- GPU utilization;
- memory headroom;
- prefill/decode split;
- agentic resolved rate, only when workload is comparable.

Do not optimize only for accept/event. The existing FR9 result already shows
why: accept improved by about 24%, but tok/s improved by only about 6.85% at the
measured batch because the second row consumes real compute.

## 9. Recommended Near-Term Defaults

Use these as starting points, not final claims:

```text
baseline:      mtp=5, spines=1
proven FR9:    mtp=5, spines=2
cheap branch:  mtp=1, spines=2..4
suffix anchor: mtp=2, spines=2..3
depth probe:   mtp=3, spines=2..3
```

My expected sweet spot for enhanced MTP+suffix is:

```text
mtp = 1 or 2
spines = 2 to 4
suffix = chain extension after the selected MTP anchor
```

Reasoning:

- MTP supplies a model-grounded short prefix;
- suffix decoding can extend repeated/code/harness patterns cheaply;
- independent rows give correct GDN state for branch candidates;
- keeping MTP shallow avoids paying for deep MTP tokens that suffix may propose
  more cheaply.

## 10. How This Feeds Enhanced Tree

After FR9 tuning, the enhanced-tree doc should consume these results as
constraints:

1. choose the cheapest FR9 branch substrate that gives positive tok/s;
2. cap MTP at the measured best depth, probably not blindly at 5;
3. add suffix only as a chain extender or high-confidence narrow side path;
4. avoid packed GDN token-tree verification unless a TreeScan/STree-style kernel
   is explicitly built;
5. compare against both:
   - E5-equivalent `spines=1`;
   - best FR9 winner from this sweep.

Enhanced tree should be considered successful only if it beats the best FR9
substrate, not merely old E3.

## 11. Open Questions

1. Does `mtp=2` or `mtp=3` beat `mtp=5` in tok/s once suffix is available?
2. Does width beyond `spines=2` remain profitable at the target batch?
3. Are root alternate rows enough, or do position-targeted rows recover more
   accepted tokens per unit compute?
4. Can suffix extension reduce the need for deep MTP while preserving acceptance?
5. Does the agentic workload remain comparable enough to produce a trustworthy
   production speed number?

## 12. Bottom Line

Proceed in two steps:

1. **FR9 swap and tune:** use independent rows as the branch verifier, sweep
   `mtp` and `spines`, and optionally prototype position-targeted rows.
2. **Enhanced MTP+suffix:** layer suffix decoding on the best FR9 substrate,
   with MTP capped at the measured efficient depth.

This keeps the next work grounded in the path that is already correct for GDN
state, while still preserving the enhanced-tree goal: more accepted tokens at a
lower marginal proposal cost.

# FR9 Spine-2 Distribution-Safe Winner Spec

**Date:** 2026-06-02
**Status:** Proposed fix after `mtp5_s2_argrepair7`
**Scope:** FR9 `--config Fb --row-mode independent --spines 2` with stochastic
sampling, especially `temp=0.6` SWE-agent runs.

## 1. Problem

`fr9_b4temp06_lowmem088_mtp5_s2_argrepair7_20260602T102212Z` proved that the
independent-row spine-2 path can complete cleanly:

- `superset_violations=0`
- `copy_missing_sum=0`
- `winner_nonzero_spine_events=2267`
- `recovered_token_total=5271`

But it also exposed a distribution bug. The current public commit logic is not
vanilla MTP. It samples two independent target trajectories at `temp=0.6`, then
publishes the row with the longer accepted prefix. That is a best-of-spines
selector, not target-distribution-preserving speculative sampling.

Observed result:

| Arm | Tasks | Resolved | Accept/event | Decode TPS | Decision |
|---|---:|---:|---:|---:|---|
| `mtp5_s1 lowmem088` | 16 | 8/16 | 3.027 | 39.91 | clean baseline |
| `mtp5_s2_argrepair7` | 16 | 4/16 | 3.818 | 31.44 | clean mechanics, not speed/quality win |

The four quality flips were normal agent trajectory differences, not infra
crashes:

- `astropy__astropy-13453`: resolved -> tests failed
- `astropy__astropy-14365`: resolved -> tests failed
- `astropy__astropy-14508`: resolved -> tests failed
- `astropy__astropy-14539`: resolved -> empty patch / patch apply failed

## 2. Losslessness Boundary

Speculative decoding is lossless only when the published token stream is sampled
from the same target distribution as ordinary target-model decoding.

Primary references:

- Leviathan, Kalman, and Matias, ICML 2023: speculative decoding runs candidate
  continuations in parallel while preserving the large model's output
  distribution.
  <https://proceedings.mlr.press/v202/leviathan23a>
- Chen et al., 2023: speculative sampling combines parallel scoring with a
  modified rejection sampler that preserves the target distribution within
  hardware numerics.
  <https://arxiv.org/abs/2302.01318>
- vLLM's lossless guarantee is scoped to the implemented rejection sampler /
  greedy equality path, not arbitrary public winner selection.
  <https://docs.vllm.ai/en/v0.7.1/features/spec_decode.html#lossless-guarantees-of-speculative-decoding>
- Multi-draft speculative sampling requires a token-level selection scheme whose
  output token matches the target distribution. A naive longest-accepted
  multi-draft winner is not that scheme.
  <https://arxiv.org/abs/2410.18234>

Therefore:

```text
single-chain MTP + target verifier       => can be lossless
spine-2 best-of-accepted-rows at temp>0  => not lossless
spine-2 with target-distribution selector => can be lossless, but needs new math/plumbing
```

## 3. Current Fault

Current winner commit code selects the longest accepted row:

```text
best_idx = argmax(accept_counts[row])
commit_idx = best_idx
publish output_token_ids[commit_idx]
copy recurrent state from commit row to siblings
```

At `temp=0.6`, hidden row 1 is an independently sampled target trajectory. If it
accepts more tokens than row 0, publishing row 1 biases the public output toward
longer-accepted hidden samples. This is a best-of-two distribution shift.

That behavior is useful for a controlled superset probe, but it must not be the
default for production-quality stochastic SWE runs.

## 4. Design Goal

Make FR9 spine-2 have two explicit modes:

1. **Lossless public mode:** preserve ordinary target-model output distribution
   at `temp>0`.
2. **Best-of-spines research mode:** keep the current longest-accepted winner
   behavior, but mark it lossy and exclude it from quality claims.

The default must be lossless public mode.

## 5. Proposed Commit Policy

Add an explicit environment/config knob:

```text
LUMO_IR_PUBLIC_COMMIT_POLICY =
  lossless_spine0       # default
  deterministic_best    # allowed only when temperature == 0
  best_of_spines        # lossy research override
```

### 5.1 `lossless_spine0`

At `temp>0`, public sequence always commits spine 0.

Hidden spine 1 remains useful for measurement:

- compute `candidate_winner_spine`
- compute hidden recovery opportunity
- validate recurrent state copy
- estimate how much value a future true multi-draft selector could recover

But hidden spine 1 must not change public tokens or public recurrent state.

Commit logic:

```text
primary_idx = row with spine_id == 0
best_idx = argmax(accept_counts)

if temperature > 0:
    commit_idx = primary_idx
    commit_acc = accept_counts[primary_idx]
    suppressed_reason = "stochastic_sampling"
else:
    commit_idx = best_idx
    commit_acc = accept_counts[best_idx]
    suppressed_reason = null
```

Trace both values:

```json
{
  "winner_spine": 0,
  "winner_acc": 3,
  "candidate_winner_spine": 1,
  "candidate_winner_acc": 5,
  "hidden_winner_suppressed_reason": "stochastic_sampling",
  "policy": "lossless_spine0"
}
```

### 5.2 `deterministic_best`

At `temperature == 0`, public best-of-spines is allowed only for deterministic
probes. This is still not enough for a full stochastic quality claim, but it is
useful for checking:

- recurrent state isolation
- hidden-row scheduler stability
- path0/winner superset invariants
- direct token/sec potential

If `temperature > 0`, this policy must fail closed before launch.

### 5.3 `best_of_spines`

This preserves the current `argrepair7` behavior and must be labeled lossy.

Allowed uses:

- direct research probes
- estimating upper-bound recovery
- stress-testing state copy and parser safety

Forbidden uses:

- claiming MTP losslessness
- claiming quality parity with `spines=1`
- reporting SWE resolved rate as a target-distribution-preserving result

## 6. Implementation Plan

### Step 1: Restore Temperature-Aware Hidden Winner Suppression

The earlier repair chain briefly introduced stochastic-safe commit suppression,
then later parser-safety work removed that suppression. Reintroduce it in
`scripts/swe_x86_helpers/relaunch_qwen36_round.py`.

Required helper functions:

```python
def _lumo_ir_request_temperature(self, primary_req_id) -> float | None:
    ...

def _lumo_ir_commit_policy() -> str:
    return os.environ.get("LUMO_IR_PUBLIC_COMMIT_POLICY", "lossless_spine0")

def _lumo_ir_select_commit_row(self, primary, req_ids, indices, accept_counts):
    ...
```

Required behavior:

- identify `primary_idx` by `spine_id == 0`, not by tensor row order;
- compute `best_idx` for diagnostics in all modes;
- commit `primary_idx` for `lossless_spine0` when `temp > 0`;
- commit `best_idx` for `deterministic_best` only when `temp <= 0`;
- commit `best_idx` for `best_of_spines`, but stamp trace as lossy;
- fail closed on unknown policy.

### Step 2: Do Not Copy Hidden State Into Public State In Lossless Mode

When hidden winner is suppressed, recurrent state must be copied from public
spine 0 to hidden spine 1, not the other way around.

Invariant:

```text
if policy == lossless_spine0 and temp > 0:
    public output_token_ids == spine0 output_token_ids
    public recurrent state == spine0 recurrent state
```

Hidden recovery must remain diagnostic only.

### Step 3: Preserve Parser/Protocol Guards

Keep the Qwen parser/protocol repairs from `argrepair7`. They do not provide
losslessness, but they are still necessary to prevent hidden-row suffixes from
leaking raw protocol markers into Codex-visible text or tool arguments.

The new policy should reduce how often these guards matter, because hidden row
1 will no longer become public at `temp>0`.

### Step 4: Make The Harness Label Policy Explicit

Add the policy to:

- relaunch log
- per-request spec trace metadata if practical
- `independent_winner_trace.jsonl`
- `independent_winner_summary.json`
- `agentic_summary.json`
- final closeout docs

Minimum trace fields:

```json
{
  "policy": "lossless_spine0",
  "temperature": 0.6,
  "winner_spine": 0,
  "candidate_winner_spine": 1,
  "hidden_winner_suppressed_reason": "stochastic_sampling",
  "lossless_public_stream": true
}
```

## 7. Validation Plan

### 7.1 Unit Tests

Add/extend tests around the injected winner commit patch:

- `temp=0.6`, hidden row has higher accept count:
  - public winner remains spine 0;
  - `candidate_winner_spine == 1`;
  - `hidden_winner_suppressed_reason == "stochastic_sampling"`;
  - state copy source is spine 0.
- `temp=0`, hidden row has higher accept count:
  - `deterministic_best` may publish spine 1;
  - `lossless_spine0` still publishes spine 0 unless policy says otherwise.
- `best_of_spines` at `temp=0.6`:
  - publishes hidden winner;
  - trace marks `lossless_public_stream=false`.
- unknown policy:
  - launch fails before SWE run.

### 7.2 Direct Probe

Run direct probes:

```text
temp=0.6, mtp=5, spines=1
temp=0.6, mtp=5, spines=2, policy=lossless_spine0
temp=0.6, mtp=5, spines=2, policy=best_of_spines
```

Expected:

- `lossless_spine0` public accept/event should be close to `spines=1`;
- hidden recovery opportunity should still be reported;
- `best_of_spines` should recover more accepted tokens but remain marked lossy.

### 7.3 Agentic SWE Gate

Run the same 16-task SWE subset:

```text
fr9_b4temp06_lowmem088_mtp5_s1
fr9_b4temp06_lowmem088_mtp5_s2_lossless_spine0
fr9_b4temp06_lowmem088_mtp5_s2_bestof   # optional lossy diagnostic
```

Acceptance criteria for `lossless_spine0`:

- all per-task request metrics nonzero;
- no raw protocol marker leaks;
- `lossless_public_stream=true` for all winner events;
- `hidden_winner_suppressed_events > 0` if hidden row would have won;
- task quality not materially worse than `s1` beyond normal stochastic variance;
- decode TPS regression quantified separately from quality.

The expected speed may still be worse than `s1`, because spine 2 still computes
real extra rows. That is acceptable for the fix: this spec fixes distribution
safety first. Speed optimization is a separate decision.

## 8. What This Does Not Solve

This does not make spine 2 fast. It removes the quality regression caused by
publishing best-of-two stochastic hidden rows.

If `lossless_spine0` is slower than `spines=1`, then spine 2 should not be a
production strategy by itself. It remains useful as:

- a measurement surface for hidden recovery opportunity;
- a scaffold for deterministic branch probes;
- a stepping stone toward true multi-draft sampling or suffix-aware selection.

## 9. Future True Lossless Spine-2

A true lossless spine-2 sampler must not choose the longest accepted sequence.
It needs a token-level selector that samples from the target distribution while
using multiple draft proposals.

High-level requirements:

1. collect target probabilities/logits for the next token;
2. treat spine 0 and spine 1 proposals as candidate tokens;
3. select or reject candidates using a distribution-preserving multi-draft
   sampling rule;
4. if a candidate is rejected, sample from the corrected residual distribution;
5. only then commit public token and recurrent state.

This is a larger implementation than FR9's current row-copy prototype. It is
the right long-term path if we want both:

```text
temp > 0 quality parity
and
actual public benefit from multiple spines
```

Until then, default policy must be:

```text
temp > 0: public spine 0 only
temp = 0: deterministic best allowed for probes
lossy best-of-spines: explicit research override only
```


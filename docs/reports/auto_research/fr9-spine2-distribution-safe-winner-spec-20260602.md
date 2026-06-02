# FR9 Spine-2 Distribution-Safe Winner Spec

**Date:** 2026-06-02
**Status:** Proposed fix after `mtp5_s2_argrepair7`
**Scope:** FR9 `--config Fb --row-mode independent --spines 2` with stochastic
sampling, especially `temp=0.6` SWE-agent runs.
**Non-negotiable invariant:** every default/public FR9 spine-2 path must be
lossless with respect to the target model's token distribution. A spine-2 path
that publishes hidden-row tokens without a distribution-preserving token-level
selector is forbidden, not a diagnostic mode.

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
spine-2 with target-distribution selector => required before hidden row can publish
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

That behavior must be deleted before new spine-2 work proceeds. Hidden-row
promotion without a lossless selector must not exist as a launchable policy.

## 4. Design Goal

Make FR9 have one public commit policy:

1. **Lossless:** preserve ordinary target-model output distribution for every
   public token.

`spines=1` is the degenerate lossless case: it commits the single public spine.
`spines>1` is also `lossless`, but hidden spines may influence public output
only if the distribution-preserving multi-draft selector is implemented. Without
that selector, `spines>1` must commit spine 0 and keep hidden-spine recovery as
trace-only evidence.

## 5. Proposed Commit Policy

Add an explicit environment/config knob:

```text
LUMO_IR_PUBLIC_COMMIT_POLICY =
  lossless       # default and only public policy
```

### 5.1 `lossless`

For `spines=1`, public sequence commits the single spine. For `spines>1`, public
sequence commits spine 0 unless the lossless multi-draft selector exists and is
enabled inside the same `lossless` policy.

Hidden spine 1 remains useful for measurement:

- compute `candidate_winner_spine`
- compute hidden recovery opportunity
- validate recurrent state copy
- estimate how much value the required multi-draft selector could recover

But hidden spine 1 must not change public tokens or public recurrent state.

Commit logic:

```text
primary_idx = row with spine_id == 0
best_idx = argmax(accept_counts)

commit_idx = primary_idx
commit_acc = accept_counts[primary_idx]
suppressed_reason = "no_lossless_selector" if best_idx != primary_idx else null
```

Trace both values:

```json
{
  "winner_spine": 0,
  "winner_acc": 3,
  "candidate_winner_spine": 1,
  "candidate_winner_acc": 5,
  "hidden_winner_suppressed_reason": "no_lossless_selector",
  "policy": "lossless"
}
```

### 5.2 Multi-Draft Selector

Within `policy=lossless`, hidden spine 1 may affect public tokens only when the
multi-draft selector is implemented. It is not implemented by choosing the
longest accepted row. It must implement token-level multi-draft speculative
sampling:

1. obtain target probabilities/logits for the next token;
2. treat spine 0 and spine 1 proposals as candidate tokens for that position;
3. select an intermediate candidate with a distribution-preserving token-level
   selection rule;
4. apply single-draft speculative sampling or equivalent accept/reject logic;
5. if rejected, sample from the corrected residual target distribution;
6. commit only the selected output token and its corresponding recurrent/KV
   state.

Any configuration that tries to let hidden spines publish without this selector
must fail closed before launch.

## 6. Implementation Plan

### Step 1: Delete Hidden Winner Promotion

Remove the old longest-accepted-row public commit path in
`scripts/swe_x86_helpers/relaunch_qwen36_round.py`. There should be no
environment/config override that publishes hidden spine 1 at `temp > 0` unless
the lossless multi-draft selector exists and passes distribution tests.

Required helper functions:

```python
def _lumo_ir_request_temperature(self, primary_req_id) -> float | None:
    ...

def _lumo_ir_commit_policy() -> str:
    return os.environ.get("LUMO_IR_PUBLIC_COMMIT_POLICY", "lossless")

def _lumo_ir_select_commit_row(self, primary, req_ids, indices, accept_counts):
    ...
```

Required behavior:

- identify `primary_idx` by `spine_id == 0`, not by tensor row order;
- compute `best_idx` for trace-only hidden recovery diagnostics;
- commit `primary_idx` for `policy=lossless` unless the token-level selector is
  present and enabled;
- fail closed if hidden publication is requested before the token-level selector
  exists;
- reject any legacy `best_of_spines`, `unsafe_best_of_spines`, or
  `deterministic_best` policy before launch;
- fail closed on unknown policy.

### Step 2: Do Not Copy Hidden State Into Public State In Lossless Mode

When hidden winner is suppressed, recurrent state must be copied from public
spine 0 to hidden spine 1, not the other way around.

Invariant:

```text
if policy == lossless and temp > 0 and selector unavailable:
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
  "temperature": 0.6,
  "winner_spine": 0,
  "candidate_winner_spine": 1,
  "hidden_winner_suppressed_reason": "no_lossless_selector",
  "lossless_public_stream": true,
  "policy": "lossless"
}
```

## 7. Validation Plan

### 7.0 Required Order Of Operations

The work must proceed in this order:

1. **Prove losslessness.** Delete hidden winner promotion, enforce
   `policy=lossless`, and pass the losslessness gates in Section 7.5.
2. **Inspect kernel-level cost.** Only after the public stream is lossless, run
   the matched s1/s2 kernel profile in Section 7.4 to learn where verifier cost
   actually lands.
3. **Design speedups from evidence.** Only after the kernel attribution exists,
   choose whether to shrink row count, improve state/KV sharing, improve CUDA
   graph capture, or implement the multi-draft selector.

Do not optimize the branch verifier first and then ask whether it is lossless.
The public policy must be lossless before any speed result is meaningful.

### 7.1 Unit Tests

Add/extend tests around the injected winner commit patch:

- `temp=0.6`, hidden row has higher accept count:
  - public winner remains spine 0;
  - `candidate_winner_spine == 1`;
  - `hidden_winner_suppressed_reason == "no_lossless_selector"`;
  - state copy source is spine 0.
- `temp=0`, hidden row has higher accept count:
  - `policy=lossless` still publishes spine 0 when selector is unavailable;
  - no longest-accepted-row fallback exists.
- hidden publication before selector implementation:
  - launch fails before SWE run.
- legacy `best_of_spines`, `unsafe_best_of_spines`, or `deterministic_best`:
  - launch fails before SWE run.
- unknown policy:
  - launch fails before SWE run.

### 7.2 Direct Probe

Run direct probes:

```text
temp=0.6, mtp=5, spines=1
temp=0.6, mtp=5, spines=2, policy=lossless, selector=off
temp=0.6, mtp=5, spines=2, policy=lossless, selector=on  # once implemented
```

Expected:

- `spines=2`, selector off, public accept/event should be close to `spines=1`;
- hidden recovery opportunity should still be reported;
- selector-on path must pass token-distribution tests before any SWE
  quality claim;
- no lossy winner-promotion run should be launchable.

### 7.3 Agentic SWE Gate

Run the same 16-task SWE subset:

```text
fr9_b4temp06_lowmem088_mtp5_s1
fr9_b4temp06_lowmem088_mtp5_s2_lossless_selector_off
fr9_b4temp06_lowmem088_mtp5_s2_lossless_selector_on  # once implemented
```

Acceptance criteria for `policy=lossless`, selector off:

- all per-task request metrics nonzero;
- no raw protocol marker leaks;
- `lossless_public_stream=true` for all winner events;
- `hidden_winner_suppressed_events > 0` if hidden row would have won;
- task quality not materially worse than `s1` beyond normal stochastic variance;
- decode TPS regression quantified separately from quality.

The expected speed may still be worse than `s1`, because spine 2 still computes
real extra rows. That is acceptable for the fix: this spec fixes distribution
safety first. Speed optimization is a separate decision.

### 7.4 Kernel-Level Verify Cost Probe

The current cost evidence is aggregate-counter only. For `mtp5_s1` versus
`mtp5_s2_argrepair7`, the measured regression was:

```text
mean engine step:       354.9 ms -> 522.3 ms
decode TPS:              39.91  -> 31.44
draft tokens / step:     17.58  -> 34.07
public tokens / step:    14.16  -> 16.42
```

That is enough to say spine 2 nearly doubles row-token verifier work while only
raising public tokens per step by about 16%. It is not enough to attribute the
extra 167 ms/engine step across attention/GDN kernels, KV writes, recurrent
state copy, scheduler bookkeeping, parser/protocol guards, or CUDA launch
overhead.

After Section 7.5 passes, run one matched SWE Verified task with kernel-level
profiling before using spine 2 as a speed strategy:

```text
task: one fixed concprobe16 SWE Verified instance, preferably one that resolved
      in the s1 baseline and completed cleanly in s2
arm A: temp=0.6, mtp=5, spines=1, lowmem088
arm B: temp=0.6, mtp=5, spines=2, policy=lossless, selector=off
arm C: temp=0.6, mtp=5, spines=2, policy=lossless, selector=on, only after selector exists
```

Required capture:

- Nsight Systems or equivalent CUDA kernel timeline for the vLLM process;
- per-engine-step markers or trace ranges around draft proposal, verifier
  forward, winner selection, recurrent-state copy, and KV/cache update;
- `dgx_steptrace.jsonl`, `per_req_spec_trace.jsonl`,
  `independent_winner_trace.jsonl`, and `agentic_summary.json` for the same
  window;
- a small postprocess table that reports per-public-token and per-engine-step
  time by kernel family when names are available.

Minimum attribution table:

| Bucket | Question |
|---|---|
| target verifier forward | Does the second row mostly add matmul/attention/GDN time? |
| KV writes / cache update | Are branch-row KV appends or cache writes scaling linearly with rows? |
| recurrent state copy | Is the 96-state-unit copy visible or negligible next to forward time? |
| scheduler / row bookkeeping | Are independent-row request updates causing CPU or CUDA launch gaps? |
| parser/protocol guard | Is Qwen repair logic measurable, or only quality safety overhead? |
| uncaptured graph launches | Are s2 verifier regions falling out of CUDA graph capture more often? |

Decision rule:

- If most extra time is verifier forward, keep `policy=lossless` selector-off
  as the default and move branch value to suffix-tree trimming or the required
  multi-draft token selector.
- If copy or cache update is material, optimize state/KV sharing before adding
  more branches.
- If launch gaps or uncaptured regions dominate, prioritize CUDA-graph-compatible
  fixed buffers and capture-safe hit-only caches before changing tree shape.
- If parser/protocol guard is the only material overhead, remove the guard from
  lossless public mode after proving hidden rows cannot publish.

### 7.5 Losslessness Verification Gates

Do not run SWE quality or speed claims for `spines>1` until the losslessness
gates pass. The gate shape follows the public vLLM losslessness framing:
rejection-sampler convergence plus greedy exact equality, extended with a
multi-draft selector distribution test.

#### Gate A: Policy Fail-Closed

Launch-time checks:

- `LUMO_IR_PUBLIC_COMMIT_POLICY` must be exactly `lossless`.
- Legacy `best_of_spines`, `unsafe_best_of_spines`, `deterministic_best`, or
  unset old aliases fail before model launch.
- If `spines>1` and hidden-publication is requested, the selector must be
  present, enabled, and marked `lossless_selector_version`.
- If selector is unavailable, the only valid `spines>1` behavior is selector-off
  `lossless`: public spine 0 commits, hidden recovery is trace-only.

Required trace fields for every winner event:

```json
{
  "policy": "lossless",
  "selector_enabled": false,
  "lossless_public_stream": true,
  "winner_spine": 0,
  "candidate_winner_spine": 1,
  "hidden_winner_suppressed_reason": "no_lossless_selector"
}
```

#### Gate B: Greedy Equality

For `temperature=0`, compare target-only decode against `policy=lossless` for:

- `spines=1`;
- `spines=2`, selector off;
- `spines=2`, selector on once implemented.

Acceptance:

- exact token sequence equality for every prompt;
- exact committed recurrent-state source for every public token;
- no hidden-spine public commit when selector is off;
- same result across representative batch shapes used by SWE serving.

This is the practical vLLM-style greedy equality gate: speculative decoding must
not change greedy output.

#### Gate C: Rejection / Selector Distribution Convergence

For `temperature>0`, use small controlled vocab distributions before involving
the real model. Build synthetic target distributions and synthetic spine
proposal distributions where the expected target output probabilities are known.

Acceptance:

- selector output distribution matches the target distribution within the
  configured statistical tolerance;
- corrected residual distribution normalizes to 1 within numerical tolerance;
- candidate order, spine id, and accepted-prefix length do not change the
  output distribution;
- a deliberately naive longest-accepted-row selector fails this test, proving
  the test catches the `argrepair7` bug class.

Minimum cases:

| Case | Purpose |
|---|---|
| identical spine proposals | selector degenerates to ordinary single-draft sampling |
| one bad hidden proposal | hidden row cannot bias output toward its accepted prefix |
| hidden row accepts longer | catches best-of-spines promotion |
| low-probability residual | verifies corrected residual sampling |
| top-k plus `other` bucket | scales the test beyond tiny vocab while keeping counts stable |

#### Gate D: Target-Model Sampling Equivalence

After synthetic convergence passes, run fixed-prompt target-model sampling:

```text
target-only
policy=lossless, spines=1
policy=lossless, spines=2, selector off
policy=lossless, spines=2, selector on   # once implemented
```

Acceptance:

- `spines=1` and `spines=2` selector-off match target-only by construction for
  public commit path, aside from ordinary target sampling RNG controls;
- selector-on distribution over next-token buckets matches target-only within
  tolerance;
- sequence-level smoke prompts show no systematic drift in format, stop-token
  behavior, or tool/protocol markers.

Record enough metadata to replay failures:

- prompt id;
- seed / RNG stream id;
- temperature, top-p/top-k, and repetition settings;
- target logits checksum or top-k probability snapshot when practical;
- selector decision, residual mass, accepted token, rejected token if any;
- public recurrent-state source.

#### Gate E: SWE Admission

Only after Gates A-D pass can a `spines>1` run be used for SWE quality/speed
claims.

SWE admission criteria:

- `lossless_public_stream=true` for all winner events;
- selector-off runs have `winner_spine=0` for all public commits;
- selector-on runs have nonzero selector trace coverage and no missing residual
  metadata;
- no raw protocol marker leaks;
- per-task request metrics are nonzero;
- quality comparison is interpreted only after the losslessness gate passes.

## 8. What This Does Not Solve

This does not make spine 2 fast. It removes the quality regression caused by
publishing best-of-two stochastic hidden rows, and it makes that unsafe behavior
impossible to use as the default.

If `spines=2` with selector off is slower than `spines=1`, then spine 2 should
not be a speed strategy by itself. It remains useful as:

- a measurement surface for hidden recovery opportunity;
- a scaffold for non-public deterministic branch probes;
- a development surface for the required lossless multi-draft selector or
  suffix-aware selection.

## 9. Required True Lossless Spine-2

This is not future work for a valid public spine-2 result. It is the required
implementation if hidden spine 1 is allowed to change public output at
`temp > 0`.

A true lossless spine-2 sampler must not choose the longest accepted sequence.
It must use a token-level selector that samples from the target distribution
while using multiple draft proposals.

Current implementation requirements:

1. collect target probabilities/logits for the next token;
2. treat spine 0 and spine 1 proposals as candidate tokens;
3. select or reject candidates using a distribution-preserving multi-draft
   sampling rule;
4. if a candidate is rejected, sample from the corrected residual distribution;
5. only then commit public token and recurrent state.

This is a larger implementation than FR9's current row-copy prototype, but it is
mandatory if we want both:

```text
temp > 0 quality parity
and
actual public benefit from multiple spines
```

Until that selector exists and passes tests, default policy must be:

```text
policy: lossless
temp > 0, selector unavailable: public spine 0 only
lossy best-of-spines: deleted / fail closed
```

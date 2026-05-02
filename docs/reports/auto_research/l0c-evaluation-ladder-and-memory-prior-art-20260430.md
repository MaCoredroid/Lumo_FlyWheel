# L0c Evaluation Ladder and Cross-Round Memory Prior Art

Generated: 2026-04-30

## Scope

This document covers the L0c kernel-mutation arm of serving auto-research. It is not about L0a kernel selection, L0b autotune, L1 serving-configuration tuning, L2 request-shaping, benchmark-family authoring, or training-loop research.

## HLD Stage Context

This work sits inside `docs/HLD-Serving-Backend-AutoResearch-v0_2-L0KernelPlan.md`, Workstream 4: **L0c kernel mutation**, specifically the Karpathy-style LLM-in-the-loop mutation round for serving kernels.

Stage alignment:

- HLD §0.6 ranks DeltaNet Triton kernels as the first predicted kernel category for L0c exploration, gated by P3a roofline evidence.
- HLD §0.7 and §7 place this work after P1/P2/P2b/P5/P5b/P3a setup and inside the P7 family of L0c real rounds.
- The concrete stage is **P7a: L0c-DeltaNet real round**.
- In the current HLD executable state, P7a is the only L0c target that is actually executable; P7e fused epilogue and P7d sampling are explicitly later/wiring-dependent targets.
- The current P7a HLD text says "two-tier preflight active" and includes AR.55-AR.57 for preflight semantics, positive memory, and calibration-log integrity. This document proposes the next design correction to that stage: remove DeltaNet syntax soft-demotion, keep evaluator-corruption hard rejects, make isolated replay the primary fast correctness/throughput scorer, and reserve vLLM parity plus full serving measurement for top-ranked confirmation.
- P3a roofline remains a prerequisite for spending a full 48-hour P7a budget. If `chunk_delta_h.py` is not a top-3 self-time contributor on the actual workload, P7a should retarget before additional L0c rounds.

In HLD terms, this is not a new workstream. It is a refinement of the P7a execution contract and should eventually patch:

- §5.3 `mutate-kernel` effects/order,
- §5.5 `iteration_brief.md` prompt skeleton,
- §5.7 L0c step-level verification,
- §5.8 two-tier preflight architecture,
- §9 AR.55-AR.57 or replacement AR entries for replay-gated evaluation,
- the P7a row in the phase table.

The intended HLD transition is:

```text
P7a v0.3.4 current:
  static hard reject + DeltaNet soft demote + vLLM parity + full measurement

P7a next:
  static hard reject + advisory memory tags + small replay
  + isolated kernel replay correctness/timing over all candidates
  + vLLM parity / full serving confirmation for top-K survivors
```

The active L0c surface is:

- model family: `qwen3.5-27b`,
- workload family: `responses-sdk-adapter-cutover-heavy`,
- hardware/runtime focus: GB10 / local vLLM serving stack,
- kernel target: DeltaNet,
- concrete mutable file: `chunk_delta_h.py`,
- host workdir used by current rounds: `output/auto_research/l0c_kernel_workdir/chunk_delta_h.py`,
- in-container target path: `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_delta_h.py`,
- correctness fixture family: `responses-sdk-adapter-cutover-deltanet-v1`,
- baseline comparison: contemporaneous paired measurements against the selected L0b/vLLM-default baseline bundle for the same workload.

The L0c loop is intended to search for source-level kernel mutations that preserve model behavior while improving serving throughput or latency under the heavy decode workload. It is intentionally below API/runtime tuning: the mutation unit is a patch to kernel code, not a change to user-facing request behavior or benchmark scoring.

## Stack Under Auto-Research

The current stack has six relevant layers:

1. **Agent/proposer layer.** A fresh Codex-backed candidate worker receives the round brief, strategy memory, prior rejection rows, winner diffs, and kernel source context. It proposes exactly one `mutation.patch` plus rationale for a single candidate.
2. **Controller layer.** `L0cKernelMutationRunner` owns candidate orchestration, mutation hashing, ledgers, preflight, replay/parity/measurement routing, and final result accounting.
3. **Kernel workdir layer.** The candidate patch is applied to the isolated host-side kernel file, then bind-mounted or imported into the runtime path used for evaluation.
4. **Kernel replay scoring layer.** The isolated replay harness calls captured DeltaNet kernel invocations outside vLLM, checks output/state correctness, and records baseline-relative kernel runtimes. This is the primary high-throughput candidate scorer for L0c.
5. **Serving correctness layer.** vLLM parity compares logits and DeltaNet recurrent-state snapshots against the reference fixture. This layer catches integration, state, cache, dtype, and scheduler-sensitive failures that direct replay cannot prove away.
6. **Serving measurement layer.** Top-ranked parity-passing candidates are measured with the real serving harness against contemporaneous paired baseline rows. This is the final confirmation layer, not the primary search evaluator.

The design change in this document sits between layers 2 and 5. It adds a controller-owned replay scorer before full vLLM parity, removes syntax-based DeltaNet demotion from the controller, and changes full serving measurement from per-candidate default to top-K confirmation.

## Mutation Contract

A mutation is one candidate-local unified diff written to:

```text
candidates/<NNN>/mutation.patch
```

Allowed mutation target:

- DeltaNet kernel implementation code for the active `kernel_target=deltanet` round, currently `chunk_delta_h.py`.

Allowed mutation classes include, but are not limited to:

- address arithmetic rewrites,
- block/tile indexing changes,
- cache modifier changes,
- local temporary computation changes,
- branch/mask simplification,
- memory-access coalescing changes,
- launch-shape or local constant changes when routed through the kernel code path.

These classes are allowed because they are exactly the search space where speedups may exist. They may be wrong, but wrong kernel math should be rejected by replay/parity evidence, not by a standing regex demoter.

Forbidden mutation targets:

- parity fixture builders,
- parity-check implementation,
- L0c controller code,
- measurement-recording code,
- tests such as `tests/test_l0c_*.py`,
- rejection-ledger or filter-hit-review writer code,
- benchmark/evaluator artifacts that would make the candidate easier to pass without improving the kernel.

Required candidate metadata:

- `mutation_hash`: controller-computed SHA-256 over `mutation.patch`,
- `speed_thesis`: candidate explanation of why the patch could improve throughput or latency,
- `expected_affected_path`: one of memory traffic, launch count, cache behavior, occupancy, register pressure, instruction count, or another explicit low-level mechanism,
- `prior_failure_relation`: whether the patch resembles any prior rejected family and why it is materially different.

## Observations From Current Runs

The current design was motivated by six observations from the recent L0c canary/live rounds.

### Observation 1: Prompt-only Failure Memory Is Not Enough

The 2026-04-29 canary showed that the proposer received a strategy brief and prior rejection context, including known-bad DeltaNet mutation families, but candidate `001` still proposed a patch in that region. That established that memory must be more structured than natural-language warnings alone.

However, the follow-up conclusion is narrower than "hard-filter all similar syntax." The right correction is to preserve failure memory and route candidates through cheaper objective checks. A prior failed family should make the proposer justify the patch and should inform ranking, but it should not automatically block DeltaNet kernel math.

### Observation 2: The Soft-Demote Rules Encode Too Much Belief

The six DeltaNet soft-demote rules were derived from a small number of bad examples:

- removing shared `g` / `gk` pre-offsets,
- inlining `g` / `gk` address expressions,
- removing or inlining `t_start`,
- adding or retaining `.cg` cache hints on changed gate loads.

Those shapes may be risky, but they are not evaluator-corrupting. Some may be valid optimization attempts if they preserve state/logit behavior and improve memory traffic or instruction count. Treating them as controller demotions risks ratcheting the search away from useful kernel rewrites.

### Observation 3: vLLM Reload Is Too Expensive As The First Serious Gate

The live loop currently pays for runtime restart, kernel activation, parity probing, and real measurement for many candidates. That is defensible for final correctness but wasteful for obviously wrong math/indexing changes. The evaluator needs a ladder so cheap failures die before the vLLM stack is reloaded.

### Observation 4: Safe Candidates Are Not Necessarily Useful Candidates

The latest live status report observed mechanically healthy loop behavior: baseline measurement, candidate spawning, parity-clean candidates entering measurement, structured parity rejection, and continuation after rejection. The open problem was search quality: safe candidates were small metadata/cache/load-shape mutations that did not beat the paired baseline. Therefore evaluator-cost reduction must be paired with proposer-quality work. The ladder makes bad candidates cheaper; it does not by itself make good candidates more likely.

### Observation 5: Full vLLM Measurement Is Too Expensive As The Primary Throughput Evaluator

If every plausible candidate pays a 25-30 minute vLLM reload/parity/measurement cycle, a 72-attempt budget burns roughly 30-36 hours of wall-clock evaluation before search quality is even considered. That is the wrong cost structure for kernel autoresearch. The primary throughput signal should come from isolated kernel replay timing, where candidate cost is closer to seconds than tens of minutes. Full serving measurement should confirm the top replay-ranked survivors, not score the whole candidate population.

### Observation 6: Target Choice Must Be Revalidated Before A Long Round

The current target is justified only if P3a roofline/profile evidence shows DeltaNet `chunk_delta_h.py` is a top bottleneck for the actual heavy workload. If the dominant self-time is instead in GEMMs, MLP projections, prefix-cache overhead, or another serving/runtime component, then a clean L0c loop can still spend the budget on the wrong surface. P3a is therefore a go/no-go gate for another long P7a round, not a nice-to-have diagnostic.

## Decision

Remove the DeltaNet-only soft-demote preflight rules from the executable controller path:

- `deltanet_g_pre_offset_removed`
- `deltanet_gk_pre_offset_removed`
- `deltanet_inline_g_base_rewrite`
- `deltanet_inline_gk_base_rewrite`
- `deltanet_t_start_removed_or_inlined`
- `deltanet_gate_load_cg_hint`

These rules are useful as historical guidance, but they are not fair as controller-enforced search-space filters. They were derived from one observed bad canary family and encode local beliefs about likely failure. They should remain in auto-research memory so future agents can see which mutation shapes previously failed, but they should not stop or demote a candidate before a cheaper correctness check has tested it.

Keep the hard static preflight for safety-critical edits:

- edits to parity fixture builders or parity-check implementation,
- edits to the L0c controller or measurement-recording implementation,
- edits to `tests/test_l0c_*.py`,
- edits to rejection-ledger or filter-hit-review writer code.

Those are different from DeltaNet math rewrites: they can corrupt the evaluator, hide failures, or make future measurements untrustworthy.

This document is the design target for replacing soft-demote with a staged evaluator ladder. It is not a request to implement source code in this pass.

## Why Remove DeltaNet Soft Demote

The original DeltaNet soft-demote list was a reasonable emergency response to a live canary gap: the proposer had been told about prior failures and still repeated a known-bad mutation family. The problem is that the current list treats syntax patterns as if they were known semantic failures.

For kernel mutation search, that is too blunt. A patch that moves `g` or `gk` address arithmetic, changes `t_start`, or adjusts cache hints may be wrong, but the right answer is to test it at the cheapest sound level. The loop should learn from failures, not turn one failure family into a standing regex filter.

The correct use of those observations is:

- show them in `strategy_brief.md`,
- include them in `prior_mutations_rejected.tsv`,
- expose exact prior `mutation.patch`, `parity_check.json`, and `BLOCKED.md` evidence,
- ask the proposer to explain why a similar-looking mutation is materially different,
- let the staged evaluator decide.

## Proposed L0c Evaluation Ladder

The controller should route each mutation through the cheapest sufficient check first:

1. static preflight,
2. small recorded-output replay,
3. isolated DeltaNet kernel replay correctness and timing,
4. top-K vLLM parity probe,
5. top-K full serving measurement confirmation.

The first tier protects evaluator integrity. Tier 2 catches cheap math/state mistakes. Tier 3 is the primary search evaluator: it rejects replay-incorrect candidates and ranks replay-correct candidates by baseline-relative kernel runtime. Tiers 4 and 5 remain authoritative integration and end-to-end confirmation gates, but they should run only on top-ranked survivors or explicit fallback cases.

## Tier 1: Static Preflight

No GPU. Diff-only checks.

Purpose:

- reject attempts to modify the evaluator, tests, ledgers, or controller,
- reject duplicate mutation hashes,
- reject malformed patches or wrong target paths,
- record touched files and static risk tags.

Decision rule:

- if the patch touches evaluator-critical files, hard reject with `forbidden_mutation_family_hard_rejected`,
- if the patch is duplicate or malformed, reject with the existing structured rejection reason,
- otherwise continue.

Artifacts:

- `static_preflight.json`
- existing `parity_check.json` only for hard safety rejections
- existing `mutations_rejected.tsv` row only for real rejection

The former DeltaNet soft-demote patterns may still be emitted as advisory `risk_tags` in `static_preflight.json`, but advisory tags must not stop, demote, or queue a candidate.

## Tier 2: Small Recorded-Output Replay

No vLLM reload. No independent PyTorch reference is assumed.

Reference means recorded outputs from the unmutated known-good kernel on a small deterministic probe set. This tier is a small replay check, not a separate hand-written math oracle.

Fixture:

- `small_replay_fixture.yaml`
- `small_replay_inputs.npz`
- `small_replay_expected.npz`

Fixture contents:

- 2-4 small deterministic DeltaNet cases,
- input tensors needed by the kernel entry point,
- expected output tensor(s),
- expected recurrent-state tensor(s) where exposed,
- dtype, shape, stride, device, layout, and tolerance metadata,
- source commit/hash of the unmutated kernel used to record outputs.

Capture procedure:

1. Run the unmutated baseline kernel once inside the same container/runtime family used for L0c.
2. Capture direct kernel inputs immediately before the DeltaNet call.
3. Capture direct outputs immediately after the call.
4. Persist tensors to `.npz` with a manifest hash over YAML plus all binary blobs.
5. Verify the fixture by replaying the unmutated kernel and requiring exact or tolerance-bounded match.

Candidate procedure:

1. Apply candidate patch to the isolated kernel workdir.
2. Load `small_replay_inputs.npz`.
3. Invoke the same kernel entry point used by the captured baseline.
4. Compare candidate outputs to `small_replay_expected.npz`.
5. Write `small_replay_check.json`.

Pass/fail:

- pass if every captured output and state tensor is within fixture tolerance,
- fail with `small_replay_diverged` if output or state differs,
- fail with `small_replay_compile_failed` if the patched kernel cannot compile in the direct harness,
- fail with `small_replay_entrypoint_missing` if the current target cannot yet be called outside vLLM.

This tier should catch obvious indexing, dtype, layout, mask, and state-update mistakes before isolated replay spends GPU time.

## Tier 3: Isolated DeltaNet Kernel Replay

GPU allowed, but no full serving reload. This is the load-bearing fast evaluator for L0c.

The goal is to replay real captured DeltaNet kernel calls outside vLLM's request scheduler, compare output/state against recorded reference outputs, and collect per-kernel runtime/profiling data. For the search loop, this tier is not optional timing decoration: it is the primary throughput measurement used to rank candidates before vLLM confirmation.

### Entry Point

The replay harness must call the same DeltaNet kernel implementation that vLLM uses after kernel activation. For the current target, the entry point is the patched `chunk_delta_h.py` module mounted into the kernel workdir and then imported inside the replay environment.

Required entry-point contract:

- import the mutated module from the candidate kernel workdir, not from the host's stale site-packages,
- call the exported DeltaNet function or wrapper with captured tensors,
- preserve Triton autotune/cache behavior enough to match the serving path,
- run with the same dtype and device constraints as the live vLLM path,
- never route through vLLM request scheduling or the inference proxy.

If the upstream function is too entangled for direct invocation, add a thin replay wrapper around the existing kernel call. The wrapper should live under test/replay infrastructure, not inside the optimized kernel body.

### Fixture Schema

The current parity fixture with `probe_input`, `reference_logits`, and `reference_state` is not enough by itself. It verifies model-level behavior, not direct kernel-call replay.

Add a kernel replay section to the fixture:

```yaml
fixture_id: responses-sdk-adapter-cutover-deltanet-replay-v1
kernel_target: deltanet
kernel_entrypoint:
  module: chunk_delta_h
  callable: <resolved callable or wrapper name>
  source_ref: output/auto_research/l0c_kernel_workdir/chunk_delta_h.py
capture:
  runtime_image: <image id>
  model_id: qwen3.5-27b
  weight_version_id: <weight id>
  base_bundle_id: <bundle id>
  triton_cache_root_hash: <hash or empty>
cases:
  - case_id: token_0001_small
    inputs_ref: kernel_replay_inputs/token_0001_small.npz
    expected_ref: kernel_replay_expected/token_0001_small.npz
    baseline_runtime_ms:
      median: <baseline median>
      p20: <baseline p20>
      p80: <baseline p80>
    tolerances:
      rtol: 5.0e-3
      atol: 5.0e-3
  - case_id: token_1024_state
    inputs_ref: kernel_replay_inputs/token_1024_state.npz
    expected_ref: kernel_replay_expected/token_1024_state.npz
    baseline_runtime_ms:
      median: <baseline median>
      p20: <baseline p20>
      p80: <baseline p80>
    tolerances:
      rtol: 5.0e-3
      atol: 5.0e-3
content_hash: <manifest hash over yaml plus npz blobs>
```

Each `.npz` input case must include:

- all tensor arguments passed to the DeltaNet kernel call,
- scalar parameters such as batch/head dimensions, block size, tile offsets, and sequence boundaries,
- masks or boundary metadata used by the kernel,
- dtype and stride metadata for each tensor,
- optional debug names mapping tensors back to source variables such as `g`, `gk`, `bos`, `t_start`, state, and output.

Each expected case must include:

- output tensors,
- recurrent-state tensors,
- any auxiliary tensors that affect later decode correctness,
- tolerance policy per tensor class.

### Capture Procedure

1. Start the unmutated baseline runtime once.
2. Enable debug export around the DeltaNet kernel call.
3. Run selected probe requests that exercise:
   - first token,
   - long-context state checkpoint near token 1024,
   - at least one multi-sequence or boundary-sensitive case,
   - at least one shape matching the heavy workload.
4. Capture direct kernel inputs and outputs for each selected call.
5. Store them under `parity_fixture/kernel_replay_inputs/` and `parity_fixture/kernel_replay_expected/`.
6. Run the replay harness against the unmutated kernel and require pass.
7. Measure unmutated replay runtime with warmup/repetition policy matching `triton.testing.do_bench` semantics, storing median plus dispersion per case.
8. Record fixture hash in the round spec.

### Candidate Procedure

1. Apply `mutation.patch` to the isolated kernel workdir.
2. Run static preflight.
3. Run small replay.
4. Run full isolated replay over all captured cases.
5. Write `kernel_replay_check.json`.
6. If isolated replay passes, compute baseline-relative timing score and enqueue the candidate in the replay survivor queue.
7. If isolated replay fails, reject without vLLM reload.
8. Run vLLM parity only when the candidate is selected as a top-K replay survivor, the replay fixture is unavailable, or an operator marks the mutation as integration-sensitive.

### Replay Output

`kernel_replay_check.json` should include:

- `pass`
- `reason`
- `fixture_id`
- `fixture_content_hash`
- `kernel_target`
- `case_count`
- `first_failing_case_id`
- `first_failing_tensor`
- `max_abs_error`
- `max_rel_error`
- `compile_time_s`
- `replay_wall_s`
- `per_case_runtime_ms`
- `per_case_baseline_runtime_ms`
- `per_case_speedup`
- `aggregate_replay_speedup`
- `replay_rank_score`
- `speed_thesis_supported`
- optional profiling counters when available
- `mutation_hash`

Failure reasons:

- `kernel_replay_compile_failed`
- `kernel_replay_entrypoint_missing`
- `kernel_replay_output_diverged`
- `kernel_replay_state_diverged`
- `kernel_replay_fixture_invalid`
- `kernel_replay_runtime_fault`

### Replay Timing Policy

Replay timing should be conservative enough to avoid promoting noise:

- warm up each mutated kernel before timing,
- report median plus p20/p80 or another fixed dispersion summary,
- compare against fixture-recorded baseline runtimes from the same hardware/runtime family,
- use a robust aggregate across cases, such as geometric mean speedup with a worst-case guard,
- require no case to exceed an allowed slowdown unless the candidate's speed thesis explicitly targets a narrower measured case,
- record timing instability as `kernel_replay_timing_unstable` rather than pretending the candidate has a clean speedup.

The default survivor selector should be configurable but concrete:

```text
confirmation_top_k = 5
minimum_replay_speedup = 1.01
max_allowed_case_slowdown = 0.99
```

These defaults mean the loop can evaluate many candidates cheaply, then spend vLLM wall-clock only on the few candidates with both replay correctness and a plausible kernel-level speed signal.

### What Still Requires vLLM Parity

Passing isolated replay is not enough to ship a mutation. vLLM parity is still required for:

- runtime activation correctness,
- Triton autotune/cache state in the real serving process,
- prefix-cache interactions,
- request shaping,
- dtype routing,
- recurrent state across full real decode,
- scheduler/proxy behavior,
- sequence-boundary behavior not represented in replay fixtures.

The value of isolated replay is search throughput and early correctness. It replaces full serving measurement as the primary candidate scorer, but it does not replace integration parity or final serving confirmation.

## Tier 4: vLLM Parity Probe

Use full vLLM parity for top-K replay survivors, or immediately when replay fixture support is unavailable for the target mutation class.

Output:

- existing `parity_check.json`,
- first-diverging probe and state/logit tolerance data.

The controller should record why vLLM parity was needed:

- `top_k_replay_survivor`,
- `replay_fixture_unavailable`,
- `integration_sensitive_mutation`,
- `manual_force_vllm_parity`.

If vLLM is already warm for parity, add a small held-out perplexity or next-token-loss probe when fixture cost is low enough. This should not replace logit/state parity, but it can catch behavior drift outside the handpicked parity prompts.

## Tier 5: Full Serving Measurement

Run only after vLLM parity passes and the mutation is selected from the replay-ranked top-K survivor set, unless an operator explicitly forces full measurement for diagnosis.

Purpose:

- measure actual end-to-end serving impact,
- compare against contemporaneous paired baseline,
- confirm that replay-level speedups survive integration effects,
- avoid spending GB10/vLLM wall-clock on the full candidate population.

Output:

- existing `measurement_trace.json`,
- `measurements.tsv`,
- `results.tsv`,
- winner or null-result decision.

## Controller Routing

Per candidate:

1. Agent writes `mutation.patch` and a short speed thesis.
2. Controller computes `mutation_hash`.
3. Controller runs static preflight.
4. Controller runs small replay if fixture exists.
5. Controller runs isolated kernel replay if fixture exists and static preflight passed.
6. Controller records replay correctness, replay timing, and replay rank score.
7. Controller selects top-K replay survivors for vLLM parity.
8. Controller runs full serving measurement only for parity-passing top-K survivors.

Fallback:

- if replay fixture support is unavailable, the controller may route to vLLM parity directly, but the round should be marked `replay_fixture_unavailable_high_cost_path`;
- if P3a roofline does not show the target as top-3 self-time, the controller should require explicit operator override before launching a long P7a round;
- if no replay-correct candidate clears `minimum_replay_speedup`, the round should stop early or request proposer retargeting instead of measuring low-signal patches in vLLM.

The agent should not run full vLLM `apply-and-test` before the controller has run the cheaper gates. Agent-side work should stop at writing the patch, rationale, touched-surface tags, and expected speed mechanism.

## Cross-Round Memory Policy

Auto-research memory should stay and become more structured. Removing DeltaNet soft demote does not mean forgetting failures.

Keep and strengthen:

- `prior_mutations_rejected.tsv`: exact prior failures across rounds.
- `mutations_rejected.tsv`: current-round failures.
- `winning_diffs.md`: positive memory of successful edits.
- candidate-local `mutation.patch`, `parity_check.json`, `BLOCKED.md`, `small_replay_check.json`, and `kernel_replay_check.json`.
- strategy brief synthesis from prior successes and failures.

Change the semantics:

- prior failures are evidence and proposer guidance,
- only evaluator-corrupting edits are hard static rejects,
- DeltaNet math/address/cache-hint patterns are not controller demotes,
- repeated failures should influence ranking, prompts, and candidate prioritization, not block the search without measurement evidence.

## Proposer-Quality Mechanism

Removing soft-demote filters reduces ratcheting-filter risk, but it does not make the proposer better. Proposer quality needs a separate mechanism.

Short-term prompt/ranking design:

- require every candidate to include `speed_thesis`,
- require `expected_affected_path`: memory traffic, launch count, cache behavior, occupancy, register pressure, or instruction count,
- require `why_not_prior_failure`: explanation if the patch resembles any prior rejected family,
- rank mutation targets using P3a/roofline evidence plus prior winners,
- distill profiler output into the top three stalls with operational interpretation before showing it to the proposer,
- reject or halt after repeated candidates that are parity-safe but have no throughput-relevant thesis.

Medium-term controller design:

- maintain a mutation-family priority queue,
- enforce mutation-family scheduling so the proposer spends a bounded number of attempts in each planned family before repeating low-value safe edits,
- boost families with prior winners and profiler-supported speed thesis,
- down-rank families with repeated replay/parity/measurement failures,
- keep down-ranking advisory unless failures are evaluator-corrupting,
- surface `proposer_stuck_low_value_region` when the queue repeatedly yields measured-but-uninteresting candidates.

Longer-term training design:

- use accepted and rejected trajectories to train a local improver or discriminator,
- preserve correctness-preserving, high-gain revisions as positive examples,
- preserve structured failure traces as negative examples,
- do not turn the discriminator into an unreviewed hard rejector.

## Design Actions

| Action | Design Status | Notes |
|---|---|---|
| Remove the six DeltaNet soft-demote patterns | Ready | Small code change; removes ratcheting-filter risk. |
| Keep safety-critical hard rejects | Ready | Protects evaluator integrity. |
| Keep cross-round rejection and winner memory | Ready | `prior_mutations_rejected.tsv` and `winning_diffs.md` are the right artifacts. |
| Run/refresh P3a roofline before long P7a budget | Ready | Go/no-go check: `chunk_delta_h.py` should be top-3 self-time or L0c should retarget. |
| Static preflight for file-touch safety | Ready | Keep it narrow and hard only for evaluator corruption. |
| Small recorded-output replay | Designed here | Uses unmutated-kernel recorded outputs, not an assumed independent PyTorch reference. |
| Isolated DeltaNet kernel replay | Designed here | Primary correctness/timing scorer; requires fixture schema upgrade and direct kernel-call harness. |
| Replay-ranked top-K survivor queue | Designed here | Default `confirmation_top_k=5`; avoids measuring the whole candidate population in vLLM. |
| vLLM parity probe | Ready | Integration correctness gate for top-K or fallback candidates. |
| Held-out perplexity probe | Designed here | Optional Tier 4 add-on when vLLM is already warm. |
| Full serving measurement | Ready | Final top-K performance confirmation, not primary search scoring. |
| Proposer-quality mechanism | Designed here | Ranking and speed-thesis mechanism complements evaluator ladder. |
| Mutation-family scheduling | Designed here | Proactive exploration coverage for proposer quality. |

## Prior-Art Support

Citation verification status: the URLs below were browser-checked again on 2026-05-02. They resolved to the claimed repository, article, or arXiv paper. This section should still be treated as supporting context, not as the load-bearing proof of the design. The local evaluator economics are sufficient on their own: a 25-30 minute per-candidate serving cycle is too expensive to be the primary search scorer.

| Reference | Verification status on 2026-05-02 | Main design signal |
|---|---|---|
| Karpathy `autoresearch` | Resolved | One-GPU autonomous loop with small mutable surface and measured keep/discard decisions. |
| KernelEvolve | Resolved | Runtime diagnostics, historical signals, structured search, and persistent knowledge/memory. |
| KernelFoundry | Resolved | Diversity-preserving evolutionary search and hardware-aware feedback. |
| GPU Kernel Scientist | Resolved | Explicit hypotheses plus timing-feedback iteration. |
| Record-Remix-Replay | Resolved | Fast replay-style evaluation to avoid full application cost on every candidate. |
| Kernel-Smith | Resolved | Population/archive search with structured feedback on compile, correctness, and speed. |
| KernelBench | Resolved | 250-kernel benchmark with correctness plus speedup metrics, showing kernel generation remains hard. |
| Sakana AI CUDA Engineer / robust-kbench revision | Resolved | Evaluator robustness matters; weak correctness gates create misleading speedups. |
| Flash Linear Attention / DeltaNet | Resolved | `chunk_delta_h.py` is inherited from an actively optimized Triton/FLA lineage, so expected wins may be small and target choice must be profiled. |
| Triton autotune / `do_bench` | Resolved | Built-in timing/autotune APIs support replay-level benchmarking policy. |

### Karpathy Autoresearch

Karpathy's `autoresearch` demonstrates the useful outer-loop shape: small mutable target, fixed experiment budget, one metric, keep/discard based on measured result. The README describes the setup as self-contained, one GPU, one file, one metric, with simplified nanochat training. Source: https://github.com/karpathy/autoresearch

Relevance to L0c:

- use fixed-budget autonomous experiments,
- keep the loop simple and measurable,
- avoid broad mutable surfaces.

Limit:

- Karpathy's task is training-code optimization, not production kernel mutation. It does not justify full vLLM reload for every kernel patch.

### KernelEvolve

Meta's KernelEvolve uses dynamic prompts enriched with runtime diagnostics, hardware constraints, and historical signals. It also keeps a knowledge base with correctness constraints, optimization guidance, hardware docs, and distilled successful strategies. Source: https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/

Relevance to L0c:

- cross-round memory is supported by prior art,
- memory should include both constraints and successful strategies,
- evaluation should include correctness, performance, and profiling, not one raw runtime number.

### KernelFoundry

KernelFoundry argues that simple prompting plus feedback is not enough for GPU kernel optimization; it uses MAP-Elites quality-diversity search, meta-prompt evolution, and template-based parameter optimization. Source: https://arxiv.org/abs/2603.12440

Relevance to L0c:

- avoid collapsing exploration to one narrow mutation family,
- preserve diverse candidate strategies,
- use behavioral dimensions and hardware feedback rather than prompt-only forbids.

### GPU Kernel Scientist

GPU Kernel Scientist describes a multi-stage evolutionary loop that selects promising prior code versions, generates optimization hypotheses from code plus GPU literature, and submits candidates to an external evaluator using timing feedback. Source: https://arxiv.org/abs/2506.20807

Relevance to L0c:

- prior versions and historical evidence matter,
- hypotheses should be explicit,
- the evaluator should stay external and authoritative.

### Record-Remix-Replay

Record-Remix-Replay is the closest prior-art match for the reload-cost issue. It uses record-replay compilation to evaluate GPU kernel candidates efficiently, rather than paying full application compile/evaluation cost for every source-level candidate. Source: https://arxiv.org/abs/2604.11109

Relevance to L0c:

- use isolated kernel replay timing as the primary search scorer,
- amortize expensive runtime setup,
- treat full serving reload as a late-stage gate, not the default first serious test.

### Kernel-Smith

Kernel-Smith uses a population of executable candidates, an archive of top-performing and diverse programs, and structured feedback on compilation, correctness, and speedup. Source: https://arxiv.org/abs/2603.28342

Relevance to L0c:

- archive both successful and diverse programs,
- preserve structured execution feedback across rounds,
- train or prompt the proposer as a local improver, not a one-shot generator.

### KernelBench And Robust CUDA Evaluation

KernelBench provides the most useful public benchmark frame for this work: correctness plus speedup across a suite of GPU-kernel tasks, not just compile success or one-off runtime wins. Source: https://arxiv.org/abs/2502.10517

Sakana's CUDA work and later robust benchmarking write-up are useful mainly as a warning: evaluator weakness can produce misleading speedup claims. Source: https://pub.sakana.ai/static/paper.pdf

Relevance to L0c:

- keep logit plus recurrent-state snapshot parity; it is stronger than a single forward-output check,
- keep the evaluator external to the proposer,
- treat replay timing as a screen, not as proof that the serving stack is behavior-preserving,
- add held-out behavior probes when the vLLM process is already warm.

### Flash Linear Attention / DeltaNet

The DeltaNet target comes from the Flash Linear Attention lineage, which already contains Triton implementations for efficient linear-attention and state-space-style operators. Source: https://github.com/fla-org/flash-linear-attention

Relevance to L0c:

- `chunk_delta_h.py` is not an untuned toy kernel,
- expected per-candidate wins may be small,
- P3a profile evidence must justify the target before another long P7a round,
- replay cases must include stateful and boundary-sensitive shapes, not only tiny happy-path tensors.

### Triton Autotune And Benchmarking

Triton exposes `triton.autotune` for configuration search and `triton.testing.do_bench` for repeatable timing. Sources: https://triton-lang.org/main/python-api/generated/triton.autotune.html and https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html

Relevance to L0c:

- replay fixtures should store the timing policy, not just output tensors,
- baseline and candidate timing should use the same warmup/repetition semantics,
- cache/autotune state must be recorded enough to explain when replay results and vLLM results disagree.

## Answer: Does Prior Art Support Research-Round Memory?

Yes. The relevant prior art strongly supports memory across rounds, but not as blind hard-coded bans.

Supported memory forms:

- archives of top-performing programs,
- historical failure traces,
- runtime diagnostics,
- hardware-specific knowledge,
- correctness constraints,
- successful optimization patterns,
- structured trajectories for future training or prompting.

Unsupported or risky form:

- promoting every observed bad syntactic pattern into a permanent hard or soft controller filter without calibration.

The L0c direction should therefore be:

1. preserve cross-round memory,
2. remove DeltaNet soft-demote enforcement,
3. implement small replay and isolated kernel replay as first-class evaluator tiers,
4. make isolated replay the primary correctness/timing scorer for candidate ranking,
5. use vLLM reload only for top-K survivors, fixture-unavailable fallbacks, or explicitly integration-sensitive mutations,
6. keep full serving measurement for final top-K performance confirmation,
7. refresh P3a roofline before another long P7a round,
8. improve proposer quality through speed theses, distilled profiler hints, and mutation-family scheduling.

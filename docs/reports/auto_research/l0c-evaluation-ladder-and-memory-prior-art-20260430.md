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
- The current P7a HLD text says "two-tier preflight active" and includes AR.55-AR.57 for preflight semantics, positive memory, and calibration-log integrity. This document proposes the next design correction to that stage: remove DeltaNet syntax soft-demotion, keep evaluator-corruption hard rejects, and add replay gates before vLLM parity.

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
  static hard reject + advisory memory tags + small replay + isolated kernel replay + vLLM parity + full measurement
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

The current stack has five relevant layers:

1. **Agent/proposer layer.** A fresh Codex-backed candidate worker receives the round brief, strategy memory, prior rejection rows, winner diffs, and kernel source context. It proposes exactly one `mutation.patch` plus rationale for a single candidate.
2. **Controller layer.** `L0cKernelMutationRunner` owns candidate orchestration, mutation hashing, ledgers, preflight, replay/parity/measurement routing, and final result accounting.
3. **Kernel workdir layer.** The candidate patch is applied to the isolated host-side kernel file, then bind-mounted or imported into the runtime path used for evaluation.
4. **Serving correctness layer.** vLLM parity compares logits and DeltaNet recurrent-state snapshots against the reference fixture. This layer catches integration, state, cache, dtype, and scheduler-sensitive failures.
5. **Serving measurement layer.** Passing candidates are measured with the real serving harness against contemporaneous paired baseline rows. This is the only layer that can declare a throughput winner.

The design change in this document sits between layers 2 and 4. It adds cheaper controller-owned gates before full vLLM parity and removes syntax-based DeltaNet demotion from the controller.

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

The current design was motivated by four observations from the recent L0c canary/live rounds.

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
3. isolated DeltaNet kernel replay,
4. vLLM parity probe,
5. full serving measurement.

The first tier protects evaluator integrity. The middle two tiers reduce GPU/vLLM cost. The last two tiers remain the authoritative integration and performance gates.

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

This tier should catch obvious indexing, dtype, layout, mask, and state-update mistakes before a full vLLM reload.

## Tier 3: Isolated DeltaNet Kernel Replay

GPU allowed, but no full serving reload. This is the load-bearing cost-reduction tier.

The goal is to replay real captured DeltaNet kernel calls outside vLLM's request scheduler, compare output/state against recorded reference outputs, and optionally collect per-kernel runtime/profiling data.

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
    tolerances:
      rtol: 5.0e-3
      atol: 5.0e-3
  - case_id: token_1024_state
    inputs_ref: kernel_replay_inputs/token_1024_state.npz
    expected_ref: kernel_replay_expected/token_1024_state.npz
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
7. Record fixture hash in the round spec.

### Candidate Procedure

1. Apply `mutation.patch` to the isolated kernel workdir.
2. Run static preflight.
3. Run small replay.
4. Run full isolated replay over all captured cases.
5. Write `kernel_replay_check.json`.
6. If isolated replay passes, continue to vLLM parity.
7. If isolated replay fails, reject without vLLM reload.

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
- optional profiling counters when available
- `mutation_hash`

Failure reasons:

- `kernel_replay_compile_failed`
- `kernel_replay_entrypoint_missing`
- `kernel_replay_output_diverged`
- `kernel_replay_state_diverged`
- `kernel_replay_fixture_invalid`
- `kernel_replay_runtime_fault`

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

The value of isolated replay is cost reduction, not replacement of integration parity.

## Tier 4: vLLM Parity Probe

Use full vLLM parity after replay passes, or immediately when replay fixture support is unavailable for the target mutation class.

Output:

- existing `parity_check.json`,
- first-diverging probe and state/logit tolerance data.

The controller should record why vLLM parity was needed:

- `replay_passed`,
- `replay_fixture_unavailable`,
- `integration_sensitive_mutation`,
- `manual_force_vllm_parity`.

## Tier 5: Full Serving Measurement

Run only after parity passes and the mutation has a plausible speed thesis.

Purpose:

- measure actual end-to-end serving impact,
- compare against contemporaneous paired baseline,
- avoid spending GB10/vLLM wall-clock on candidates that failed cheaper checks.

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
6. Controller runs vLLM parity if replay passed or replay is unavailable.
7. Controller runs full serving measurement only if vLLM parity passes.

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
- reject or halt after repeated candidates that are parity-safe but have no throughput-relevant thesis.

Medium-term controller design:

- maintain a mutation-family priority queue,
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
| Static preflight for file-touch safety | Ready | Keep it narrow and hard only for evaluator corruption. |
| Small recorded-output replay | Designed here | Uses unmutated-kernel recorded outputs, not an assumed independent PyTorch reference. |
| Isolated DeltaNet kernel replay | Designed here | Requires fixture schema upgrade and direct kernel-call harness. |
| vLLM parity probe | Ready | Integration correctness gate after replay. |
| Full serving measurement | Ready | Final performance truth. |
| Proposer-quality mechanism | Designed here | Ranking and speed-thesis mechanism complements evaluator ladder. |

## Prior-Art Support

Citation verification status: the URLs below were checked on 2026-04-30 and resolved to the claimed repository, article, or arXiv paper.

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

- add isolated kernel replay before vLLM parity,
- amortize expensive runtime setup,
- treat full serving reload as a late-stage gate, not the default first serious test.

### Kernel-Smith

Kernel-Smith uses a population of executable candidates, an archive of top-performing and diverse programs, and structured feedback on compilation, correctness, and speedup. Source: https://arxiv.org/abs/2603.28342

Relevance to L0c:

- archive both successful and diverse programs,
- preserve structured execution feedback across rounds,
- train or prompt the proposer as a local improver, not a one-shot generator.

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
4. use vLLM reload only when integration-level evidence is needed,
5. keep full serving measurement for final performance truth,
6. improve proposer quality in parallel with evaluator-cost reduction.

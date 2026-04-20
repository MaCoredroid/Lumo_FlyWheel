Agreed. **Decision:** drop **Browser-preview workflows**, **Computer-use workflows**, and **Automation & maintenance** from the **canonical** CNB-55 benchmark. For a Linux-only Codex CLI harness, the portable first-class surfaces are the terminal, skills, MCP or integrations, and explicit subagent workflows: Codex CLI is a local terminal agent, skills are available in CLI, MCP is supported in CLI, plugins are available in CLI, and subagent activity is surfaced in CLI as well. ([OpenAI Developers][1])

**Decision:** replace those three tracks with agentic workloads that are both **Linux-native** and **offline-verifiable**. The benchmark pattern to imitate is: fixed corpora for research-style tasks, stateful local environments for tool-use tasks, and multi-round objective scoring for long-horizon strategic tasks. That is the cleanest lesson from BrowseComp-Plus, AppWorld, ToolSandbox, CCTU, and CodeClash. ([GitHub][2])

## Updated CNB-55 track layout

Keep the benchmark at **11 tracks x 5 families = 55 families** and swap in three new tracks:

| Track | 5 families |
| --- | --- |
| 1. Core implementation | feature implementation; bug fix; test repair; performance fix; API upgrade |
| 2. Codebase understanding | request-flow tracing; ownership/file localization; dependency/risk mapping; architecture-plan writing; migration scoping |
| 3. Refactor & modernization | dead-code removal; duplicate collapse; module decomposition; stale-pattern modernization; config/build cleanup |
| 4. Review & remediation | PR review; security review; flaky-test review; inline-comment resolution; review-to-fix follow-up |
| 5. Front-end & multimodal build | screenshot-to-code; design-system conformance; visual bug fix; accessibility remediation; asset-to-component implementation |
| 6. Evidence-grounded research & synthesis | RFC/spec retrieval; issue/PR archaeology; upgrade research pack; incident evidence synthesis; policy/docs compliance audit |
| 7. Stateful tool, policy & constraint execution | multi-tool transaction; policy-aware request resolution; constraint-heavy workflow; MCP routing/parameter control; rollback/recovery |
| 8. Skills & tooling | curated skill consumption; skill selection; skill packaging; CLI companion authoring; hooks/rules authoring |
| 9. MCP & local integrations | local issue-tracker triage; local docs/code-search MCP; log/metrics query + patch; multi-service replica workflow; artifact pipeline coordination |
| 10. Strategic management & long-horizon evolution | proposal ranking; release-note-to-plan; backlog decomposition; objective-driven improvement; multi-round software evolution |
| 11. Subagents & orchestration | parallel exploration; one-agent-per-review-axis; multi-worktree candidates; manager/worker synthesis; verifier/executor split |

Rename the old plugin/integration track to **MCP & local integrations**, because MCP is the portable tool layer across both the Responses API and Codex CLI. The Responses API is the recommended agentic interface for new projects and supports stateful multi-turn runs plus MCP/tools; Codex CLI also supports MCP directly. ([OpenAI Developers][3])

## Why these are the right replacements

### 6. Evidence-grounded research & synthesis

This replaces browser-preview with something more agentic and more reproducible. BrowseComp-Plus explicitly moves away from live-web evaluation and instead uses a fixed corpus of roughly `100K` human-verified documents so deep-research agents can be compared fairly and reproducibly. General AgentBench reinforces that general agents should be evaluated across search, coding, reasoning, and tool use in one unified setting rather than in narrow silos. ([GitHub][2])

The five families in this track are:

- RFC/spec retrieval with pinned-citation answers
- issue/PR archaeology with root-cause memo
- dependency-upgrade research pack
- incident evidence synthesis / postmortem packet
- policy/docs compliance audit

These are frontier-relevant because they test whether the agent can find the right evidence, reject distractors, synthesize across sources, and justify its answer. They are also easy to grade offline with a frozen corpus, gold supporting documents, hard negatives, and claim-level citation checks. ([GitHub][2])

### 7. Stateful tool, policy & constraint execution

This is the strongest replacement for computer use in a Linux harness. ToolSandbox shows why: it evaluates stateful tool execution, implicit state dependencies, on-policy conversational interaction, and intermediate or final milestone grading. AppWorld pushes the same idea further with simulated apps and state-based unit tests plus collateral-damage checks. CCTU adds explicit constraints, step-level validation, and failure analysis when agents must obey many rules at once. MCP-Radar is especially relevant for Codex-native work because it evaluates MCP tool use directly and scores both correctness and operational accuracy. ([arXiv][4])

The five families in this track are:

- multi-tool transactional workflow
- policy-aware maintainer or support resolution
- constraint-heavy workflow with recovery
- MCP routing and parameter fidelity
- state debugging or rollback after partial failure

This track evaluates whether a model can actually behave like an agent instead of just writing code: choose the right tool, carry state across turns, obey policy, recover from bad tool outputs, and avoid corrupting state. ([arXiv][4])

### 10. Strategic management & long-horizon evolution

This is the right replacement for automation & maintenance. SWE-Lancer signals that real software work is not only implementation; it includes managerial tasks where the model must choose among implementation proposals. CodeClash moves beyond isolated tickets into open-ended objective pursuit via multi-round tournaments. SWE-EVO is especially useful because it tests long-horizon software evolution derived from real release histories and proposes a partial-progress metric so you can measure movement instead of only binary success. ([OpenAI][5])

The five families in this track are:

- proposal ranking / managerial decision
- release-note-to-implementation plan
- backlog decomposition and dependency scheduling
- objective-driven repo improvement
- multi-round software evolution with partial credit

This is the track that separates a strong coding model from a strong engineering agent. A frontier agent should handle ambiguity, tradeoffs, prioritization, cleanup pressure, regression risk, and iterative improvement against a moving objective. ([OpenAI][5])

## What SOTA agentic capability should mean here

Define agentic capability around seven things:

1. **Evidence seeking and grounding**: can it search a fixed corpus, identify the right evidence, and cite it correctly? BrowseComp-Plus is the reference pattern. ([GitHub][2])
2. **Stateful tool execution**: can it execute multi-turn workflows where tool outputs change later decisions? ToolSandbox and AppWorld are the reference pattern. ([arXiv][4])
3. **Policy and constraint obedience**: can it follow complex operational constraints, not just reach the final state? CCTU is the reference pattern. ([arXiv][6])
4. **Strategic judgment**: can it rank proposals, pick a path, and justify tradeoffs? SWE-Lancer is the reference pattern. ([OpenAI][5])
5. **Long-horizon code evolution**: can it improve a codebase over multiple rounds without turning it into slop? CodeClash and SWE-EVO are the reference pattern. ([arXiv][7])
6. **Skill consumption and selection**: can it use curated procedural knowledge correctly, and do skills actually help? SkillsBench is the reference pattern. ([arXiv][8])
7. **Orchestration under parallelism**: can it decompose tasks and use subagents well when parallel exploration is beneficial? Codex subagents are now a real CLI surface, so this should be benchmarked directly. ([OpenAI Developers][9])

## Revised family variants

For the Linux benchmark, use:

- **V1 Clean baseline**
- **V2 Noisy reality**
- **V3 Dirty state**
- **V4 Multi-tool or multi-corpus requirement**
- **V5 Recovery-in-thread**

That variant shape is better aligned with what newer agent benchmarks are actually stressing: state carryover, constraint handling, partial feedback, and unified multi-skill settings. ([arXiv][4])

## Revised scoring for the new tracks

Add track-specific scorecards instead of a single generic formula.

### Evidence-grounded research & synthesis

- final answer correctness: `25`
- support-document recall/precision: `25`
- claim-level citation grounding: `20`
- synthesis completeness: `20`
- efficiency: `10`

This follows the BrowseComp-Plus lesson: final answer alone is not enough; you also need to know whether the agent found the right supporting evidence rather than getting lucky. ([GitHub][2])

### Stateful tool, policy & constraint execution

- final state correctness: `30`
- policy compliance: `25`
- tool-call / parameter accuracy: `20`
- intermediate milestone success: `15`
- recovery quality: `10`

This follows ToolSandbox, AppWorld, CCTU, and MCP-Radar: the benchmark should care about trajectory quality, state mutations, constraint obedience, and operational accuracy, not just whether some terminal output looks plausible. ([arXiv][4])

### Strategic management & long-horizon evolution

- proposal ranking / decision quality: `20`
- objective delta: `20`
- regression-free change: `20`
- maintainability / slop control: `15`
- plan/dependency correctness: `15`
- partial-progress metric: `10`

This is directly motivated by SWE-Lancer’s managerial tasks, CodeClash’s objective-driven multi-round format, and SWE-EVO’s partial-progress scoring. ([OpenAI][5])

## Canonical benchmark design decisions

Make the canonical benchmark entirely offline and replayable. That means:

- fixed corpora for research families
- local MCP servers or service doubles for integration families
- versioned repo snapshots for code families
- state diffs and artifact hashes for grading
- no live web and no live SaaS in the official score

That is the most consistent pattern across the strongest reproducible agent benchmarks: BrowseComp-Plus for fixed corpora, AppWorld for simulated apps and unit-tested state, ToolSandbox for conversational stateful tool use, and CCTU for executable constraint validation. ([GitHub][2])

## Bottom line

For a Linux-only Codex CLI benchmark, the three best replacements are:

- **Evidence-grounded research & synthesis**
- **Stateful tool, policy & constraint execution**
- **Strategic management & long-horizon evolution**

These are more reproducible than the removed GUI tracks, more portable to Responses API plus Codex CLI, and closer to the agentic frontier that current benchmarks say still breaks frontier systems. ([OpenAI Developers][3])

Next useful step is to turn those **15 replacement families** into concrete benchmark bundles with task specs, evaluator contracts, Codex config, repo-local skills, and benchmark-run records.

[1]: https://developers.openai.com/codex/cli "Codex CLI"
[2]: https://github.com/texttron/BrowseComp-Plus "BrowseComp-Plus"
[3]: https://developers.openai.com/api/docs/guides/migrate-to-responses "Migrate to the Responses API"
[4]: https://arxiv.org/abs/2408.04682 "ToolSandbox"
[5]: https://openai.com/index/swe-lancer/ "SWE-Lancer"
[6]: https://arxiv.org/abs/2603.15309 "CCTU"
[7]: https://arxiv.org/abs/2511.00839 "CodeClash"
[8]: https://arxiv.org/abs/2602.12670 "SkillsBench"
[9]: https://developers.openai.com/codex/subagents "Subagents"

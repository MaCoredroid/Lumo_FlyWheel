I’d scope the **Linux Codex CLI** version to **Codex Native Bench 40 (CNB-40)**: a **private, rolling, Responses-API benchmark** for real developer work, with **40 families × 5 variants = 200 tasks**. The key decision is to benchmark the **Linux CLI surface that is actually documented today**, not the broader desktop-app surface. OpenAI’s current Codex CLI docs support repository reading, editing, commands, screenshots as image inputs, local code review, skills, plugins/MCP, and subagents; the app-only surfaces such as the in-app browser, computer use, and scheduled app automations should be excluded from the Linux CLI benchmark. The Responses API is still the right substrate because OpenAI recommends it for new projects, it is agentic by default, and it natively supports multi-turn state plus tools like web search, file search, code interpreter, and remote MCP. ([OpenAI Developers][1])

## What to steal from existing benchmarks

The benchmark should deliberately borrow **different strengths from different evals**, not copy any one of them. From **SWE-bench Verified**, take human-validated repo issues and reliable solvability; from **SWE-bench Multimodal**, take visual bug/context handling; from **Terminal-Bench 2.0**, take long-horizon, realistic environments with human-written solutions and comprehensive tests; from **DevBench (2026)**, take telemetry-grounded realism and mixed evaluation signals; from **LiveCodeBench** and **SWE-rebench**, take contamination resistance and rolling refresh; from **CodeClash**, take multi-round scored improvement loops; from **AppWorld** and **ToolSandbox**, take stateful multi-app execution and intermediate milestone grading; from **BrowseComp-Plus**, take the “fixed corpus / fixed environment for fairness” idea; from **OSWorld**, **WorkArena++**, and **TheAgentCompany**, take real-computer and workplace-style workflows; and from **AgentDojo**, take joint utility-and-security evaluation. ([SWE-bench][2])

## Core design decisions

**Decision 1: make Responses API the canonical runtime.**
Reason: it is OpenAI’s recommended agentic primitive, it exposes typed output items and built-in tool loops, and it already supports the exact surfaces you need for a Codex-style harness. Use `shell` and `apply_patch` for code changes, image inputs for screenshot-driven tasks, and MCP/custom functions for external services. ([OpenAI Developers][3])

**Decision 2: benchmark Codex-native surfaces as first-class capabilities, not side channels.**
That means scoring whether the system used the right surface: local review, worktree isolation, screenshots or other image inputs when visual context matters, skills, plugins/MCP, and explicit subagent orchestration. OpenAI’s CLI docs make these all part of the current terminal workflow. ([OpenAI Developers][4])

**Decision 3: use fixed local replicas and frozen corpora for canonical scoring.**
Do not make the scored benchmark depend on the live web or live SaaS APIs. That is the main lesson from BrowseComp-Plus: fixed corpora make comparisons fairer and let you disentangle retriever/tool quality from model quality. For Slack/GitHub/Linear/Sentry/Vercel-like tasks, use deterministic service doubles or local MCP servers with frozen datasets and UI states. ([arXiv][5])

**Decision 4: evaluate both first-pass performance and recovery-in-thread.**
Codex is designed as a persistent teammate with review and subagents, not just a one-shot patch generator. The benchmark should therefore score both initial completion and one structured follow-up turn, such as inline review comments, a failed deploy log, or targeted regression evidence. OpenAI’s CLI docs already emphasize local review, image inputs, and explicit subagent routing. ([OpenAI Developers][4])

**Decision 5: publish two profiles, not one.**
Have a **Minimal Codex** profile and a **Configured Codex** profile. Minimal Codex is analogous to a standard harness comparison; Configured Codex includes benchmark-provided skills, plugins, and surface adapters. This mirrors why SWE-bench separates the broader system leaderboard from its apples-to-apples bash-only comparisons. ([SWE-bench][2])

## Benchmark definition

**Target:** real work a developer would plausibly hand to Codex in 2026.
**Unit of evaluation:** not just “did tests pass,” but **did the agent produce the right artifact, through the right surface, with acceptable churn, evidence, safety, and recovery behavior**.
**Canonical mode:** multi-turn Responses API run with stored state and a benchmark-side harness that exposes Linux-CLI-relevant surfaces: repo/worktree manager, review adapter, screenshot or image-input support, skill registry, plugin/MCP registry, and subagent manager. OpenAI’s App Server can be an **optional parity runner** for CLI smoke tests because it exposes stable JSON-RPC notifications over Codex core, but it should not be the scoring substrate. ([OpenAI Developers][3])

## The 40 families

I would split the Linux CLI subset into **8 tracks of 5 families each**. This keeps the benchmark aligned to the surfaces that are actually available in Codex CLI on Linux.

| Track                           | 5 families                                                                                                                              |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Core implementation          | feature implementation, bug fix, test repair, performance fix, API upgrade                                                              |
| 2. Codebase understanding       | request-flow tracing, ownership/file localization, dependency/risk mapping, architecture-plan writing, migration scoping                |
| 3. Refactor & modernization     | dead-code removal, duplicate collapse, module decomposition, stale-pattern modernization, config/build cleanup                          |
| 4. Review & remediation         | PR review, security review, flaky-test review, inline-comment resolution, review-to-fix follow-up                                       |
| 5. Front-end & multimodal build | screenshot-to-code, Figma-to-code, responsive repair, design-system conformance, visual bug fix                                         |
| 6. Skills & tooling             | create skill, improve skill trigger behavior, create companion CLI, CLI+skill packaging, hooks/rules authoring                          |
| 7. Plugin / integration work    | GitHub issue-to-fix, Slack task kickoff, Sentry/Linear/GitHub triage, Vercel deploy/log fix, docs/data retrieval into artifact          |
| 8. Subagents & orchestration    | parallel exploration, one-agent-per-review-axis, candidate branches across worktrees, manager/worker synthesis, scored improvement loop |

That track selection is not arbitrary: it lines up with the public Codex workflows OpenAI highlights today for CLI-shaped work, including PR review, refactoring, screenshot-driven UI work, reusable skills, plugin-backed integrations, and scored improvement loops. ([OpenAI Developers][6])

## The 5-variant pattern

I would keep the **same five variant axes across every family** so results are comparable.

* **V1 Clean baseline:** explicit prompt, clean repo/service state, no hidden blockers.
* **V2 Noisy reality:** prompt ambiguity, irrelevant context, extra logs/files, partial docs.
* **V3 Dirty workspace:** staged/unrelated changes, failing unrelated checks, partial previous work, or stale worktree state.
* **V4 Multi-surface requirement:** visual artifact, plugin/MCP source, worktree isolation, or required skill invocation.
* **V5 Recovery-in-thread:** the system gets one follow-up turn with review comments, failed deploy logs, regression evidence, or targeted test output and must fix narrowly in the same thread.

This matters because OpenAI’s current Codex CLI docs explicitly distinguish between clean code changes, inline review iteration, image-assisted reasoning, and explicit subagent orchestration. The benchmark should reflect that real work is not a single prompt-response exchange. ([OpenAI Developers][4])

## Task contract for each benchmark item

Each item should have a strict schema:

* **workspace bundle:** repo snapshot, test fixtures, local services, frozen external-service replica data
* **surface manifest:** which surfaces are available and preferred (`shell`, `apply_patch`, `review`, `image_inputs`, `skills`, `plugins/MCP`, `subagents`)
* **thread setup:** `AGENTS.md`, optional skills, optional hooks/rules, approval policy, initial response state
* **prompt pack:** primary user request, optional attachments, hidden follow-up injection for V5
* **expected artifacts:** patch, PR comments, bug report, deploy URL, report, skill files, CLI binary/help output, automation config, etc.
* **grader pack:** deterministic checks, state/collateral checks, trace checks, model graders, and human-audit flags
* **budget:** max turns, wall-clock, token budget, tool-call budget, and whether subagents are allowed

That schema is easy to operationalize with the Responses API and OpenAI’s eval tooling because the API exposes typed response items and the Evals stack supports logs-backed runs, text similarity, string checks, model graders, and Python-based graders. ([OpenAI Developers][3])

## Scoring

I would **not** make “all tests green” the only metric. Use a **hard-gate + weighted-score** design.

### Hard gates

* Forbidden write or approval violation → **0**
* Collateral damage outside allowed scope → **cap at 40**
* Missing required artifact or unverifiable claim → **cap at 60**
* Broken workspace / unrecoverable repo state → **0**

### Weighted score (base profile)

* **Outcome correctness:** 40
* **Artifact quality:** 20
* **Surface/process fidelity:** 15
* **Recovery quality:** 10
* **Safety + collateral damage:** 10
* **Efficiency:** 5

Then allow **track-specific reweighting** by ±10. For example, PR review families should weight artifact quality higher; skill families should weight trigger behavior and repeatability higher; multimodal front-end families should weight evidence and visual correctness higher.

This follows OpenAI’s eval guidance closely: outcome, process, style, and efficiency should all be explicit, small must-pass checks; graders should mix deterministic and subjective signals; logs should be tagged so you can drill into specific failure modes instead of staring at one aggregate score. ([OpenAI Developers][7])

## What “artifact quality” should mean

This is the part most coding benchmarks miss.

For **patch tasks**, grade:

* patch locality / unnecessary churn
* touched-file scope
* explanation quality
* whether tests or validation evidence match the claimed fix

For **review tasks**, grade:

* issue precision / false-positive rate
* evidence anchoring to files/lines
* prioritization and actionability

For **visual or screenshot-grounded QA tasks**, grade:

* repro steps
* expected vs actual
* severity
* screenshot/evidence completeness
* whether the right surface was used

For **skills / CLI tasks**, grade:

* trigger correctness
* `SKILL.md` quality
* command usability from another folder
* approval boundary documentation
* repeatability on a second run

OpenAI’s Codex docs already give strong hints about what good looks like here: review is line-specific and stageable, screenshots can be attached directly to the CLI, skills are gated by `name`/`description` and progressive disclosure of `SKILL.md`, and CLI/skill workflows should be installable and reusable. ([OpenAI Developers][4])

## Codex-native process grading

This is where the benchmark becomes genuinely new.

Score whether the system:

* used **worktree isolation** for parallel or risky changes
* resolved **inline review comments** instead of ignoring them
* used **image inputs or visual evidence** when screenshots or mockups were part of the task
* preferred **plugins/MCP/structured integrations** over brittle GUI clicking when those were available
* invoked the right **skill**, and avoided false-positive skill triggers
* used **subagents explicitly and appropriately** for read-heavy parallel work
* avoided parallel write-heavy chaos unless the family explicitly required it
* produced reviewable **Git artifacts**: stage/revert-ready chunks, clean diffs, clear summary

Those behaviors are directly reflected in the current Codex docs. Existing benchmarks usually grade the final state; CNB-40 should also grade whether the system behaved like a good Codex operator. ([OpenAI Developers][8])

## Runner architecture

The harness should have five layers:

1. **Responses runtime**
   Use the Responses API with stored state and `previous_response_id`, plus `shell`, `apply_patch`, image inputs, `file_search`, `code_interpreter`, remote MCP, and custom functions where needed. ([OpenAI Developers][3])

2. **Codex-surface adapters**
   Add benchmark-native adapters for review comments, worktree management, skill loading, plugin directory manifests, and subagent spawning or waiting.

3. **Replica services**
   Local GitHub/Slack/Linear/Sentry/Vercel-style services with frozen datasets and deterministic behavior.

4. **Trace logger**
   Capture response items, tool calls, shell commands, patch ops, screenshots, skill loads, plugin calls, subagent events, approvals, and artifact hashes. Store them as eval logs with metadata tags like `family`, `variant`, `surface`, `language`, `skill_id`, and `plugin_id`. OpenAI’s eval stack is built to grade from structured logs and metadata. ([OpenAI Platform][9])

5. **Grader stack**
   Deterministic execution/state graders first, then artifact/schema graders, then trace graders, then label/score-model graders, with human calibration on a sample. OpenAI’s eval tooling already supports this mixed approach. ([OpenAI Developers][10])

## One worked family example

**Family F4: responses adapter cutover**

* **V1:** migrate a small SDK wrapper to the current Responses shape and keep visible tests green
* **V2:** same task, but fixture noise and stale docs make the legacy path look partially valid
* **V3:** dirty workspace includes unrelated local edits and one stale generated fixture
* **V4:** the task also requires updating repo-local Codex config and preserving tool metadata in transcript rendering
* **V5:** same thread gets review feedback pointing at one regression in replay order and must patch only that issue

**Required surfaces:** repo patching, tests, config editing, review comments
**Artifacts:** diff, test evidence, config update, short migration note
**Graders:** visible and hidden tests, diff locality, transcript-order invariants, preservation of tool metadata, clarity of blocker reporting

## What not to do

Do **not** make this:

* a single global leaderboard number
* a bash-only harness with a few GUI tasks bolted on
* a live-web benchmark for canonical scoring
* a pure pass/fail repo-patch suite
* a benchmark where hidden skills or secret hand-built scaffolds can dominate without disclosure

That is exactly the lesson from the current benchmark ecosystem: different surfaces matter, contamination matters, hidden scaffolds matter, and stateful workflows need more than one correctness bit. ([SWE-bench][2])

## Recommended rollout

Start with a **pilot of 8 families × 5 variants = 40 tasks** across these tracks: core implementation, codebase understanding, refactor, review, front-end, skill/CLI creation, plugin integration, and subagents. Calibrate graders against human runs, then expand within the Linux CLI matrix. Keep at least one hidden holdout split and rotate a subset of families quarterly, borrowing the rolling/decontaminated logic from LiveCodeBench and SWE-rebench. ([arXiv][12])

The sharpest summary is this:

**Build CNB-40 as a Responses-API, Linux-Codex-CLI, artifact-and-trace benchmark.**
Not “can the model solve generic terminal tasks,” but **“can a Linux Codex CLI system do the work a 2026 developer actually hands to Codex, through the surfaces the CLI really exposes, with reviewable outputs and reliable recovery?”**

Next step could be a **machine-readable task schema + grader contract** for 3 pilot families so you can start implementing the runner.

[1]: https://developers.openai.com/codex/cli/features "Features – Codex CLI | OpenAI Developers"
[2]: https://www.swebench.com/verified.html "https://www.swebench.com/verified.html"
[3]: https://developers.openai.com/api/docs/guides/migrate-to-responses "https://developers.openai.com/api/docs/guides/migrate-to-responses"
[4]: https://developers.openai.com/codex/cli/features "https://developers.openai.com/codex/cli/features"
[5]: https://arxiv.org/abs/2508.06600 "https://arxiv.org/abs/2508.06600"
[6]: https://developers.openai.com/codex/use-cases/github-code-reviews "https://developers.openai.com/codex/use-cases/github-code-reviews"
[7]: https://developers.openai.com/blog/eval-skills "https://developers.openai.com/blog/eval-skills"
[8]: https://developers.openai.com/codex/cli/features "Features – Codex CLI | OpenAI Developers"
[9]: https://platform.openai.com/docs/api-reference/evals/list?ref=medhakhurana.com "https://platform.openai.com/docs/api-reference/evals/list?ref=medhakhurana.com"
[10]: https://developers.openai.com/api/docs/guides/evaluation-getting-started "https://developers.openai.com/api/docs/guides/evaluation-getting-started"
[11]: https://developers.openai.com/codex/explore/ "https://developers.openai.com/codex/explore/"
[12]: https://arxiv.org/html/2403.07974v2?utm_source=chatgpt.com "LiveCodeBench: Holistic and Contamination Free ..."

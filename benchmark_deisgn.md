I’d build this as **Codex Native Bench 55 (CNB-55)**: a **private, rolling, Responses-API benchmark** for real developer work, with **55 families × 5 variants = 275 tasks**. The key decision is to benchmark the **full Codex-native system** rather than a bash-only model loop, because OpenAI’s current Codex surface is explicitly broader than terminal coding: the app supports parallel threads, worktrees, Git review, in-app browser feedback, computer use, skills, plugins/MCP, automations, and subagents; OpenAI’s Codex use-case docs also present PR review, front-end implementation, bug triage, deployment, reusable skills, and computer-use QA as common workflows. The Responses API is the right substrate because OpenAI recommends it for new projects, it is agentic by default, and it natively supports multi-turn state plus tools like web search, file search, code interpreter, remote MCP, and computer use. ([OpenAI Developers][1])

## What to steal from existing benchmarks

The benchmark should deliberately borrow **different strengths from different evals**, not copy any one of them. From **SWE-bench Verified**, take human-validated repo issues and reliable solvability; from **SWE-bench Multimodal**, take visual bug/context handling; from **Terminal-Bench 2.0**, take long-horizon, realistic environments with human-written solutions and comprehensive tests; from **DevBench (2026)**, take telemetry-grounded realism and mixed evaluation signals; from **LiveCodeBench** and **SWE-rebench**, take contamination resistance and rolling refresh; from **CodeClash**, take multi-round scored improvement loops; from **AppWorld** and **ToolSandbox**, take stateful multi-app execution and intermediate milestone grading; from **BrowseComp-Plus**, take the “fixed corpus / fixed environment for fairness” idea; from **OSWorld**, **WorkArena++**, and **TheAgentCompany**, take real-computer and workplace-style workflows; and from **AgentDojo**, take joint utility-and-security evaluation. ([SWE-bench][2])

## Core design decisions

**Decision 1: make Responses API the canonical runtime.**
Reason: it is OpenAI’s recommended agentic primitive, it exposes typed output items and built-in tool loops, and it already supports the exact surfaces you need for a Codex-style harness. Use `shell` and `apply_patch` for code changes, `computer` for GUI workflows, and MCP/custom functions for external services. ([OpenAI Developers][3])

**Decision 2: benchmark Codex-native surfaces as first-class capabilities, not side channels.**
That means scoring whether the system used the right surface: review pane behavior, worktree isolation, browser comments, computer-use flows, skills, plugins, automations, and explicit subagent orchestration. OpenAI’s docs make these all part of the current Codex workflow, and some have clear norms—for example, use the in-app browser first for local/public unauthenticated web pages and use computer use when structured integrations or CLI checks are insufficient. ([OpenAI Developers][4])

**Decision 3: use fixed local replicas and frozen corpora for canonical scoring.**
Do not make the scored benchmark depend on the live web or live SaaS APIs. That is the main lesson from BrowseComp-Plus: fixed corpora make comparisons fairer and let you disentangle retriever/tool quality from model quality. For Slack/GitHub/Linear/Sentry/Vercel-like tasks, use deterministic service doubles or local MCP servers with frozen datasets and UI states. ([arXiv][5])

**Decision 4: evaluate both first-pass performance and recovery-in-thread.**
Codex is designed as a persistent teammate with review, inline feedback, automations, and subagents—not just a one-shot patch generator. The benchmark should therefore score both initial completion and one structured follow-up turn, such as inline review comments, a failed deploy log, or browser feedback. OpenAI’s own Codex docs emphasize inline review comments, iterative browser feedback, background automations, and explicit subagent routing. ([OpenAI Developers][4])

**Decision 5: publish two profiles, not one.**
Have a **Minimal Codex** profile and a **Configured Codex** profile. Minimal Codex is analogous to a standard harness comparison; Configured Codex includes benchmark-provided skills, plugins, and surface adapters. This mirrors why SWE-bench separates the broader system leaderboard from its apples-to-apples bash-only comparisons. ([SWE-bench][2])

## Benchmark definition

**Target:** real work a developer would plausibly hand to Codex in 2026.
**Unit of evaluation:** not just “did tests pass,” but **did the agent produce the right artifact, through the right surface, with acceptable churn, evidence, safety, and recovery behavior**.
**Canonical mode:** multi-turn Responses API run with stored state and a benchmark-side harness that exposes Codex-native surfaces: repo/worktree manager, review-channel adapter, browser preview adapter, computer-use adapter, skill registry, plugin/MCP registry, automation simulator, and subagent manager. OpenAI’s App Server can be an **optional parity runner** for desktop/CLI smoke tests because it exposes stable JSON-RPC notifications over Codex core, but it should not be the scoring substrate. ([OpenAI Developers][3])

## The 55 families

I would split the 55 families into **11 tracks of 5 families each**. This is the cleanest way to cover the current Codex surface without turning the benchmark into a random bag of tasks.

| Track                           | 5 families                                                                                                                              |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Core implementation          | feature implementation, bug fix, test repair, performance fix, API upgrade                                                              |
| 2. Codebase understanding       | request-flow tracing, ownership/file localization, dependency/risk mapping, architecture-plan writing, migration scoping                |
| 3. Refactor & modernization     | dead-code removal, duplicate collapse, module decomposition, stale-pattern modernization, config/build cleanup                          |
| 4. Review & remediation         | PR review, security review, flaky-test review, inline-comment resolution, review-to-fix follow-up                                       |
| 5. Front-end & multimodal build | screenshot-to-code, Figma-to-code, responsive repair, design-system conformance, visual bug fix                                         |
| 6. Browser-preview workflows    | local preview review, browser-comment resolution, deploy-preview verify-and-fix, regression sweep, visual artifact capture              |
| 7. Computer-use workflows       | GUI bug repro, browser-only flow, settings/config flow, multi-app workflow, simulator/native app validation                             |
| 8. Skills & tooling             | create skill, improve skill trigger behavior, create companion CLI, CLI+skill packaging, hooks/rules authoring                          |
| 9. Plugin / integration work    | GitHub issue-to-fix, Slack task kickoff, Sentry/Linear/GitHub triage, Vercel deploy/log fix, docs/data retrieval into artifact          |
| 10. Automation & maintenance    | recurring bug sweep, nightly smoke summary, telemetry report with optional patch, dependency/update sweep, stale backlog summarization  |
| 11. Subagents & orchestration   | parallel exploration, one-agent-per-review-axis, candidate branches across worktrees, manager/worker synthesis, scored improvement loop |

That track selection is not arbitrary: it lines up with the public Codex workflows OpenAI highlights today—PR review, refactoring, Figma and screenshot-driven UI work, deployment, computer-use QA, bug triage, reusable skills, CLI creation, and scored improvement loops. ([OpenAI Developers][6])

## The 5-variant pattern

I would keep the **same five variant axes across every family** so results are comparable.

* **V1 Clean baseline:** explicit prompt, clean repo/service state, no hidden blockers.
* **V2 Noisy reality:** prompt ambiguity, irrelevant context, extra logs/files, partial docs.
* **V3 Dirty workspace:** staged/unrelated changes, failing unrelated checks, partial previous work, or stale worktree state.
* **V4 Multi-surface requirement:** visual artifact, browser comment, plugin/MCP source, GUI-only step, or required skill invocation.
* **V5 Recovery-in-thread:** the system gets one follow-up turn with review comments, failed deploy logs, regression evidence, or bug repro output and must fix narrowly in the same thread.

This matters because OpenAI’s current Codex docs explicitly distinguish between clean code changes, inline review iteration, browser comment workflows, background automations, and explicit subagent orchestration. The benchmark should reflect that real work is not a single prompt-response exchange. ([OpenAI Developers][4])

## Task contract for each benchmark item

Each item should have a strict schema:

* **workspace bundle:** repo snapshot, test fixtures, local services, frozen external-service replica data
* **surface manifest:** which surfaces are available and preferred (`shell`, `apply_patch`, review, browser, computer, skill registry, plugins/MCP, subagents, automation)
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

Then allow **track-specific reweighting** by ±10. For example, PR review families should weight artifact quality higher; skill/automation families should weight trigger behavior and repeatability higher; deploy/browser families should weight surface fidelity and evidence higher.

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

For **browser / computer-use QA tasks**, grade:

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

OpenAI’s Codex docs already give strong hints about what good looks like here: review is line-specific and stageable, browser work is comment-driven, skills are gated by `name`/`description` and progressive disclosure of `SKILL.md`, CLI/skill workflows should be installable and reusable, and automations should either produce findings or cleanly no-op. ([OpenAI Developers][4])

## Codex-native process grading

This is where the benchmark becomes genuinely new.

Score whether the system:

* used **worktree isolation** for parallel or risky changes
* resolved **inline review comments** instead of ignoring them
* used the **in-app browser** for local/public unauthenticated web previews
* used **computer use** only when CLI or structured integrations were not enough
* preferred **plugins/MCP/structured integrations** over brittle GUI clicking when those were available
* invoked the right **skill**, and avoided false-positive skill triggers
* used **subagents explicitly and appropriately** for read-heavy parallel work
* avoided parallel write-heavy chaos unless the family explicitly required it
* produced reviewable **Git artifacts**: stage/revert-ready chunks, clean diffs, clear summary

Those behaviors are directly reflected in the current Codex docs. Existing benchmarks usually grade the final state; CNB-55 should also grade whether the system behaved like a good Codex operator. ([OpenAI Developers][8])

## Runner architecture

The harness should have five layers:

1. **Responses runtime**
   Use the Responses API with stored state and `previous_response_id`, plus `shell`, `apply_patch`, `computer`, `file_search`, `code_interpreter`, remote MCP, and custom functions where needed. ([OpenAI Developers][3])

2. **Codex-surface adapters**
   Add benchmark-native adapters for review pane comments, browser preview comments, worktree management, automation scheduling, skill loading, plugin directory manifests, and subagent spawning/waiting.

3. **Replica services**
   Local GitHub/Slack/Linear/Sentry/Vercel-style services with frozen datasets and deterministic behavior.

4. **Trace logger**
   Capture response items, tool calls, shell commands, patch ops, screenshots, skill loads, plugin calls, subagent events, approvals, and artifact hashes. Store them as eval logs with metadata tags like `family`, `variant`, `surface`, `language`, `skill_id`, and `plugin_id`. OpenAI’s eval stack is built to grade from structured logs and metadata. ([OpenAI Platform][9])

5. **Grader stack**
   Deterministic execution/state graders first, then artifact/schema graders, then trace graders, then label/score-model graders, with human calibration on a sample. OpenAI’s eval tooling already supports this mixed approach. ([OpenAI Developers][10])

## One worked family example

**Family F4: deploy preview and fix**

* **V1:** build app, deploy preview, return URL, verify healthy
* **V2:** same task, but env var missing; correct behavior is to detect blocker and report it cleanly
* **V3:** deploy fails; read build logs and patch narrowly
* **V4:** deploy succeeds, but mobile layout is broken; open preview, inspect rendered page, fix, redeploy
* **V5:** same thread gets review/browser feedback and must address only the flagged issue

**Required surfaces:** repo patching, deploy plugin/MCP, browser preview, review comments
**Artifacts:** diff, build command, preview URL or blocker report, final verification note
**Graders:** build/deploy status, HTTP health, screenshot/visual delta, diff locality, correct use of structured deploy integration before browser clicking, clarity of blocker reporting

That example is intentionally Codex-native: it matches OpenAI’s documented deployment, browser-preview, and front-end workflows, and it would be badly measured by a terminal-only benchmark. ([OpenAI Developers][11])

## What not to do

Do **not** make this:

* a single global leaderboard number
* a bash-only harness with a few GUI tasks bolted on
* a live-web benchmark for canonical scoring
* a pure pass/fail repo-patch suite
* a benchmark where hidden skills or secret hand-built scaffolds can dominate without disclosure

That is exactly the lesson from the current benchmark ecosystem: different surfaces matter, contamination matters, hidden scaffolds matter, and stateful workflows need more than one correctness bit. ([SWE-bench][2])

## Recommended rollout

Start with a **pilot of 8 families × 5 variants = 40 tasks** across these tracks: core implementation, review, front-end, browser preview, computer use, skill/CLI creation, bug triage automation, and subagents. Calibrate graders against human runs, then expand to the full 55-family matrix. Keep at least one hidden holdout split and rotate a subset of families quarterly, borrowing the rolling/decontaminated logic from LiveCodeBench and SWE-rebench. ([arXiv][12])

The sharpest summary is this:

**Build CNB-55 as a Responses-API, Codex-surface, artifact-and-trace benchmark.**
Not “can the model solve terminal tasks,” but **“can a Codex-native system do the work a 2026 developer actually hands to Codex, through the right surfaces, with reviewable outputs and reliable recovery?”**

Next step could be a **machine-readable task schema + grader contract** for 3 pilot families so you can start implementing the runner.

[1]: https://developers.openai.com/codex/app "App – Codex | OpenAI Developers"
[2]: https://www.swebench.com/verified.html "https://www.swebench.com/verified.html"
[3]: https://developers.openai.com/api/docs/guides/migrate-to-responses "https://developers.openai.com/api/docs/guides/migrate-to-responses"
[4]: https://developers.openai.com/codex/app/review "https://developers.openai.com/codex/app/review"
[5]: https://arxiv.org/abs/2508.06600 "https://arxiv.org/abs/2508.06600"
[6]: https://developers.openai.com/codex/use-cases/github-code-reviews "https://developers.openai.com/codex/use-cases/github-code-reviews"
[7]: https://developers.openai.com/blog/eval-skills "https://developers.openai.com/blog/eval-skills"
[8]: https://developers.openai.com/codex/app/worktrees "Worktrees – Codex app | OpenAI Developers"
[9]: https://platform.openai.com/docs/api-reference/evals/list?ref=medhakhurana.com "https://platform.openai.com/docs/api-reference/evals/list?ref=medhakhurana.com"
[10]: https://developers.openai.com/api/docs/guides/evaluation-getting-started "https://developers.openai.com/api/docs/guides/evaluation-getting-started"
[11]: https://developers.openai.com/codex/use-cases/deploy-app-or-website "https://developers.openai.com/codex/use-cases/deploy-app-or-website"
[12]: https://arxiv.org/html/2403.07974v2?utm_source=chatgpt.com "LiveCodeBench: Holistic and Contamination Free ..."

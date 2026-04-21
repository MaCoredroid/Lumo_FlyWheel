# CNB-55 Task Authoring & Grading Spec v1.0

Authoring and grading guide for the Codex-Native Benchmark 55 (CNB-55). CNB-55 is a Linux-only, offline, reproducible agentic evaluation benchmark built around `codex exec` as the reference runner. This document defines the contract every family must meet before it is admitted to the canonical benchmark.

Companion docs:
- [`benchmark_deisgn.md`](./benchmark_deisgn.md) — track/family layout decisions.
- [`benchmark_blueprints/families/`](./benchmark_blueprints/families/) — per-family bundles.
- [`benchmark_blueprints/tracks/`](./benchmark_blueprints/tracks/) — 11-track view.

Status: v1.0 — locked for first freeze of Track 10 family `proposal-ranking-manager-judgment`. Subsequent track freezes may trigger a v1.x revision.

---

## 1. Goals and non-goals

### 1.1 Goals

CNB-55 exists to answer one question at the frontier: **does this model behave like an engineering agent, not just an autocompleter?** Concretely, the benchmark scores:

- evidence seeking and grounding
- stateful tool execution
- policy and constraint obedience
- strategic judgment under ambiguity
- long-horizon code evolution with partial credit
- skill consumption and selection
- orchestration under parallelism

The benchmark pattern is calibrated so that a frontier model driven by `codex exec --model gpt-5.4 --reasoning-effort high` scores in the 15–25 band, with room for improvement, rather than ≥70 (too saturated) or ≤5 (too noisy).

### 1.2 Non-goals

CNB-55 does **not** evaluate:

- raw completion quality on isolated snippets (use HumanEval, MBPP, LiveCodeBench)
- browser-preview or GUI workflows (explicitly dropped from CNB-55; see `benchmark_deisgn.md`)
- live SaaS integrations (replaced with local MCP service doubles)
- cost-per-task (reported, but not part of the headline score)

### 1.3 Why a new benchmark

SWE-Bench Verified has saturated. As of Feb 2026, [Sonar Foundation + Claude Opus 4.5 scored 79.2%](https://www.sonarsource.com/company/press-releases/sonar-claims-top-spot-on-swe-bench-leaderboard/), with average resolution time of 10.5 min and $1.26 per issue. The benchmark was a good signal in 2024–25; at 80% it no longer separates frontier systems. The next generation of benchmarks — SWE-Bench Pro (top models ≈ 23%), ALE-Bench, SWE-EVO, Snorkel Agentic Coding, TRAJECT-Bench — explicitly target the properties SWE-Bench Verified under-tests: long-horizon reasoning, trajectory quality, partial progress, contamination resistance, and managerial judgment. CNB-55 sits in that next-generation space, Linux-native and Codex-first.

---

## 2. Reference benchmarks

Every design decision in this spec is anchored to one or more of the following. When a section cites "the [Snorkel] lesson" or "the [SWE-Lancer] lesson" it refers to the paragraph below.

### 2.1 SWE-Bench Verified (OpenAI, 2024)

- Shape: 500 human-verified instances from SWE-Bench. Per-instance resolved-rate headline.
- Lesson for CNB-55: **per-instance scoring + human verification of resolvability.** We adopt per-variant scoring (275 tasks total) and a human-verification gate before freeze.
- Source: [SWE-Bench Verified leaderboard](https://www.swebench.com/verified.html), [Epoch AI analysis](https://epoch.ai/benchmarks/swe-bench-verified).

### 2.2 SWE-Bench Pro (Scale AI, 2025)

- Shape: 1865 tasks across 41 professional repos. Public (GPL copyleft, 731) + Private (proprietary, 276) splits. Docker-pinned envs. Long-horizon patches (hours–days).
- Top models score ≈ 23% on the public set, vs ≥70% on SWE-Bench Verified.
- Lesson for CNB-55: **contamination resistance via license/source selection, Docker-pinned envs, long-horizon tasks, human-verified augmentation.** We pin Docker image digests per variant, refuse live web during grading, and augment every task with enough context to be resolvable.
- Sources: [Scale SWE-Bench Pro leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public), [paper](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/SWEAP_Eval_Scale%20(9).pdf), [repo](https://github.com/scaleapi/SWE-bench_Pro-os).

### 2.3 SWE-Lancer (OpenAI, 2025)

- Shape: 1488 Upwork freelance jobs, $1M total payout. Split into IC engineering tasks and **managerial tasks** where the model chooses between implementation proposals.
- Managerial best: 44.9% (Claude 3.5 Sonnet, $208,050 earned).
- Lesson for CNB-55: **managerial judgment scored against expert human choice**, not just code correctness. Track 10 is directly shaped by this lesson. Our `proposal-ranking-manager-judgment` family replicates SWE-Lancer's manager-choice grading on a frozen, offline corpus.
- Sources: [OpenAI SWE-Lancer](https://openai.com/index/swe-lancer/), [leaderboard](https://llm-stats.com/benchmarks/swe-lancer).

### 2.4 SWE-EVO (2025)

- Shape: Long-horizon software evolution derived from real release histories. Partial-progress metric rather than binary pass/fail.
- Lesson for CNB-55: **partial progress is the correct granularity for long-horizon work.** Track 10's `multi-round-software-evolution` family inherits this directly; the SPEC's scoring rules allow per-milestone partial credit across all tracks.
- Source: [arXiv 2512.18470](https://arxiv.org/pdf/2512.18470).

### 2.5 Snorkel Agentic Coding Benchmark (2026)

- Shape: 100 multi-step tasks across 4 difficulty tiers. Each task has human-validated reference solution + unit tests + scoring rubric + gold solution. Rubric is fine-grained with intermediate behavior scoring (e.g., "+3 if agent reads runbook before mitigation", "+5 if agent modifies state.json correctly and verifies with cat"). Pass@5 on the Harbor harness. 30-min timeout.
- Lesson for CNB-55: **fine-grained rubric with positive intermediate-behavior points and negative suboptimal-behavior penalties** is the right shape for trajectory grading. Our Phase 2 visible + hidden + trusted split mirrors this.
- Sources: [Snorkel blog](https://snorkel.ai/blog/introducing-the-snorkel-agentic-coding-benchmark/), [leaderboard](https://snorkel.ai/leaderboard/category/agenticcoding/), [rubric design](https://snorkel.ai/blog/the-science-of-rubric-design/).

### 2.6 TRAJECT-Bench (2025)

- Shape: ≥1000 high-fidelity tools, trajectories that vary in breadth (parallel calls) and depth (interdependent chains). Metrics: Trajectory Exact-Match, Inclusion, Tool-Usage (schema + value), Trajectory-Satisfy (LLM judge, Claude-4 default).
- Lesson for CNB-55: **trajectory correctness is a first-class metric, distinct from final-state correctness.** Track 7 (stateful tool/policy/constraint) requires trajectory-level metrics in addition to final-state.
- Source: [arXiv 2510.04550](https://arxiv.org/abs/2510.04550).

### 2.7 ALE-Bench (2025, Sakana)

- Shape: AtCoder Heuristic Contests. Score-based (not pass/fail). Open-ended: true optima unreachable; scores keep rising.
- Lesson for CNB-55: **some tasks are better measured as "how far did you get" than "did you pass".** CNB-55 allows open-ended partial-progress metrics on Track 10 families where the solution space is too rich for a single gold answer.
- Source: [arXiv 2506.09050](https://arxiv.org/abs/2506.09050), [Sakana blog](https://sakana.ai/ale-bench/).

### 2.8 FeatureBench (2026)

- Shape: 200 tasks, 3825 executable envs from 24 open-source repos. Test-driven task extraction + execution-based evaluation.
- Lesson for CNB-55: **test-driven extraction** (gold tests define the task spec) is less brittle than prose specs. Every CNB-55 family must include executable tests, not only prose expected-deliverables.
- Source: [OpenReview](https://openreview.net/forum?id=41xrZ3uGuI), [arXiv 2602.10975](https://arxiv.org/html/2602.10975v1).

### 2.9 AppWorld (2024)

- Shape: Simulated apps, state-based unit tests, collateral-damage checks.
- Lesson for CNB-55: **collateral damage detection** — an agent that reaches the goal by corrupting other state fails the task. CNB-55 trusted-final-state checks include "no banned changes" invariants.
- Source: [arXiv 2407.18901](https://arxiv.org/abs/2407.18901).

### 2.10 ToolSandbox (Apple, 2024)

- Shape: Stateful tool use, implicit dependencies, milestone grading, on-policy conversation.
- Lesson for CNB-55: **milestone grading** for multi-turn tool work. The `milestones` section of every family's grader is modeled on ToolSandbox's pattern.
- Source: [arXiv 2408.04682](https://arxiv.org/abs/2408.04682).

### 2.11 Terminal-Bench 2.0 (Snorkel, 2025)

- Shape: Terminal as the primary surface, end-to-end task completion under a shell.
- Lesson for CNB-55: **the terminal is a real evaluation surface**, not just a developer ergonomics layer. CNB-55 uses `codex exec` and records the full shell trajectory.
- Source: [Snorkel Terminal-Bench 2.0 blog](https://snorkel.ai/blog/evaluating-coding-agent-capabilities-with-terminal-bench-snorkels-role-in-building-the-next-generation-benchmark/).

### 2.12 Anthropic 2026 Agentic Coding Trends Report

- Shape: Industry survey on what stresses agents in production.
- Lesson for CNB-55: validates the design-doc selection of research-grounded, stateful tool, and managerial tracks as the frontier separators.
- Source: [Anthropic 2026 Agentic Coding Trends](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf?hsLang=en).

### 2.13 Quick reference matrix

| Property | Primary reference | CNB-55 section |
| --- | --- | --- |
| Per-instance scoring | SWE-Bench Verified | §3.1 |
| Contamination resistance | SWE-Bench Pro | §5 |
| Docker-pinned envs | SWE-Bench Pro | §5.2 |
| Managerial task grading | SWE-Lancer | §7.3, Track 10 |
| Partial-progress metric | SWE-EVO, ALE-Bench | §7.4 |
| Fine-grained rubric | Snorkel | §7.2 |
| Trajectory metrics | TRAJECT-Bench | §7.5 |
| Test-driven extraction | FeatureBench | §6.1 |
| Collateral damage checks | AppWorld | §7.6 |
| Milestone grading | ToolSandbox | §7.2 |
| Terminal-first runner | Terminal-Bench 2.0 | §6 |

---

## 3. Task model

### 3.1 Addressing and scoring unit

CNB-55 = **11 tracks × 5 families × 5 variants = 275 tasks**.

The **atomic scored unit is a variant**, not a family. This follows SWE-Bench Verified's per-instance scoring. Each variant has a stable address:

```
task_id  = <family_id>/<variant_id>
family_id = kebab-case, globally unique (e.g. proposal-ranking-manager-judgment)
variant_id = v{1..5}-<slug> (e.g. v1-clean-baseline, v3-dirty-state)
```

Headline score: **resolved-rate** across 275 tasks, with a task "resolved" iff its variant score ≥ the track pass-bar (see §7). Per-track and per-family breakdowns are secondary.

### 3.2 Family as organizing unit

The family is the *authoring* unit: it shares a workspace template, a grader, a skill pack, and a `codex/config.toml`. Variants within a family are author-defined difficulty rungs (see §4).

### 3.3 Track taxonomy

The 11 tracks and 5 families per track are fixed by `benchmark_deisgn.md`. CNB-55 v1 does not allow new tracks without a spec revision. New families inside an existing track are allowed with authoring approval.

---

## 4. Variant design

### 4.1 Five free-shape difficulty rungs

Variants V1–V5 are ordered difficulty rungs. The design doc's template (V1 Clean / V2 Noisy / V3 Dirty / V4 Multi-tool / V5 Recovery) is the default skeleton, but families may substitute track-appropriate axes if they justify it in the Difficulty Ladder section of `task_spec.md`.

**Hard rules** on the Difficulty Ladder:

1. V1 must be the easiest variant in the family.
2. Expected probe score on GPT-5.4/high must be monotonically non-increasing from V1 → V5.
3. The ladder section must state, per variant, (a) what changed relative to V1, (b) why it's harder, (c) which scorecard sub-metric it primarily stresses.
4. No variant may drop a scorecard sub-metric that exists in V1 (no difficulty by omission).

### 4.2 Default skeleton (recommended)

| Variant | Pressure | Typical stress |
| --- | --- | --- |
| V1 Clean baseline | None beyond task itself | Decision quality at face value |
| V2 Noisy reality | Distractors, near-duplicates, conflicting signals | Evidence discrimination |
| V3 Dirty state | Pre-existing partial fix, half-broken config, stale comments | Not-invented-here resistance |
| V4 Multi-tool or multi-corpus | Requires cross-source synthesis or tool routing | Tool/corpus selection |
| V5 Recovery-in-thread | Seeded failure mid-run requiring rollback/repair | Trajectory recovery |

### 4.3 Shared vs. per-variant workspace

- A family may share a single `workspace_bundle/` across variants with a `.scenario_variant` marker file that switches Dockerfile selection, or
- A family may carry `workspace_bundle/v1-.../ … v5-.../` subdirectories with per-variant contents.

The per-variant subdirectory layout is **required** when variants differ in corpus, fixtures, or seeded state. The shared layout is only allowed for families where variants differ only in runtime config (rare).

---

## 5. Workspace bundle rules

### 5.1 Offline, Linux-only, reproducible

- No network during grading. Allowed only during `docker build` from a pinned base image.
- Every variant ships a `Dockerfile` with a pinned base image digest (`@sha256:…`).
- Every workspace has a `.scenario_variant` marker file containing the variant_id string.
- Fixtures, corpora, gold artifacts, and evidence are content-addressable; their SHA-256 hashes appear in `manifest.lock.json` (see §8).

### 5.2 Base image pinning

Follow the pattern used in `alert-dedupe-investigation`: `python:3.12-bookworm@sha256:…`, explicit digest. SWE-Bench Pro's contamination-resistance argument is direct here: pinned digests plus copyleft- or private-licensed corpora make it materially harder for later model training to contaminate the benchmark.

### 5.3 Required files per variant

```
workspace_bundle/<variant_id>/
├── AGENTS.md                       # visible agent instructions
├── Dockerfile                      # pinned base, sets CWD to /workspace
├── .scenario_variant               # contains <variant_id>
├── <task_specific_content>/        # proposals/, repo/, evidence/, etc.
├── artifacts/                      # gold_*.json lives here (readable by agent iff visible)
├── tests/                          # Phase 2 visible test slice (pytest-collectible)
└── brief/                          # empty scratch for agent output (created by agent)
```

Hidden tests live in `verifier_data/<family_id>/<variant_id>/hidden_tests/` and are mounted read-only into the verifier container during grading. They are **never** accessible to the agent.

### 5.4 Banned patterns

- `sitecustomize.py`, `usercustomize.py`, or a top-level `pytest.py` shim in the workspace (shortcut risk).
- Live network tools invoked by the grader (e.g., `curl`, `wget` to external hosts).
- Absolute paths in fixtures that reference the authoring host (use `/workspace` exclusively).

---

## 6. Runner contract

### 6.1 Reference runner

```
codex exec \
  --model gpt-5.4 \
  --reasoning-effort high \
  --config .codex/config.toml \
  --workdir /workspace
```

Per-family `codex/config.toml` declares the MCP servers, allowed tool surfaces, approval policy, and timeouts.

### 6.2 Wall-clock and token caps

Declared per variant in `task_spec.md`:

| Variant tier | Wall-clock cap | Total token cap |
| --- | --- | --- |
| V1–V2 | 20 min | 400k |
| V3 | 30 min | 600k |
| V4–V5 | 45 min | 900k |

Rationale: Snorkel uses 30-min per task; CNB-55 scales down for simpler rungs and up for recovery/multi-corpus where frontier models need more trajectory room.

### 6.3 Allowed tool surfaces

Declared per family. At minimum `shell` and `apply_patch`. MCP surfaces must be listed explicitly; the runner enforces the allowlist. No browser tools in CNB-55.

### 6.4 Determinism

- Seed: `CNB55_SEED=42` environment variable available to both agent and grader.
- Any random fixtures generated during build must be seeded.
- Scorer code must be deterministic given (brief.md, gold_*.json, CNB55_SEED).

---

## 7. Grader architecture

### 7.1 Three layers

1. **Phase 2 — visible checks.** `pytest` against the exposed test slice. The agent sees these tests and is expected to make them pass. Budget: ≤ 30% of total variant score.
2. **Hidden checks.** Structural/behavioral tests invisible to the agent. Behavioral (does the output exhibit the right shape/semantics?), differential (does it match a gold artifact within tolerance?), property-based (does it satisfy invariants?), regression (does it avoid reintroducing known bugs?). Budget: ≥ 50% of total variant score.
3. **Trusted final-state.** Repo-level invariants (no banned imports, no test-shim, no shortcut patterns, checksum integrity of the immutable test slice). Pass/fail gates rather than graded.

### 7.2 Milestones

Every family declares 3–5 **milestones** (ToolSandbox pattern) whose combined weights cover the hidden-check budget. A milestone:

- has a stable id (`m1_…`, `m2_…`, `m3_…`)
- has a `partial_credit` weight summing to 1.0 across the milestone set
- maps to ≥ 1 hidden-test node (pytest nodeid)
- has a description graders display on failure

Milestone pass rules: `all` (all test nodes pass) or `any` (at least one passes). Default `all`.

### 7.3 Per-track scorecards

Weights are fixed per track; families must allocate 100 points exactly according to their track. Tracks not listed here inherit the "Core implementation" scorecard unless the `benchmark_deisgn.md` specifies otherwise.

**Track 1 Core implementation / Track 3 Refactor / Track 4 Review:**
- final correctness: 40
- hidden behavioral checks: 30
- trusted final-state: 15
- docs/config alignment: 10
- efficiency (wall-clock, token): 5

**Track 6 Evidence-grounded research:**
- final answer correctness: 25
- support-document recall/precision: 25
- claim-level citation grounding: 20
- synthesis completeness: 20
- efficiency: 10

**Track 7 Stateful tool, policy & constraint:**
- final state correctness: 30
- policy compliance: 25
- tool-call / parameter accuracy: 20
- intermediate milestone success: 15
- recovery quality: 10

**Track 10 Strategic management & long-horizon evolution:**
- proposal ranking / decision quality: 20
- objective delta: 20
- regression-free change: 20
- maintainability / slop control: 15
- plan/dependency correctness: 15
- partial-progress metric: 10

The scorecards are copied verbatim from `benchmark_deisgn.md`. Families must not silently reallocate.

### 7.4 Partial progress (Track 10)

The partial-progress sub-metric is inherited from SWE-EVO and ALE-Bench. It allows a family to score agents that made real forward motion without crossing the pass-bar. The metric must be author-declared, monotone, and bounded in [0, max_sub_metric_weight].

### 7.5 Trajectory metrics (Tracks 7, 11)

Track 7 and Track 11 families must include **trajectory-level** metrics in addition to final-state. Adopt TRAJECT-Bench's three:

1. Tool-sequence Inclusion (proportion of gold tool calls invoked in correct order).
2. Tool-Usage (parameter-level schema/value correctness).
3. Trajectory-Satisfy (LLM-judge score against a rubric; judge model declared per family).

### 7.6 Collateral damage (all tracks)

Every grader must include at least one "didn't break something unrelated" check. This is AppWorld's collateral-damage principle. Examples: no banned files modified, no test files deleted, checksum manifest still verifies for the immutable slice.

### 7.7 Pass-bar

A variant counts as "resolved" if its variant score ≥ the track pass-bar:

| Track | Pass-bar |
| --- | --- |
| 1, 3, 4, 5 (code tracks) | 60 |
| 2 (codebase understanding) | 55 |
| 6 (research synthesis) | 50 |
| 7 (stateful tool) | 60 |
| 8, 9 (skills / MCP) | 55 |
| 10 (strategic) | 55 |
| 11 (subagents) | 60 |

Pass-bars are higher than the calibration target (§7.8) by design — a frontier model should clear them only on a minority of variants.

### 7.8 Calibration target

**Per family mean of V1–V5 on GPT-5.4/high must land in [15, 25].** Additional guard-rails:

- No individual variant above 40.
- At least one variant at or below 10.
- Monotonic: E[V1] ≥ E[V2] ≥ … ≥ E[V5] within a 3-point noise band.

### 7.9 Anti-shortcut contract

Every family MUST declare **partial-credit ceilings** in `task_spec.md`. These are hard caps applied after raw score aggregation. Examples from `responses-sdk-adapter-cutover`:

- "docs/config-only change ≤ 10 points"
- "happy-path shim without behavioral fix ≤ 25 points"

Ceilings are encoded in the scorer, not just in prose. The scorer reads a `ceilings` block from `evaluator_contract.md` and applies them unconditionally.

Each family must also declare ≥ 4 named **red-team traps** (shortcuts the agent might be tempted to take), plus ≥ 1 verified exploit attempt per trap. Exploits live under `verifier_data/<family_id>/<variant_id>/red_team/` and are re-executed during authoring QA.

### 7.10 Evaluator contract file

Every family ships `evaluator_contract.md` that states, at minimum:

- Evaluation goal (1 paragraph)
- Visible checks (pytest invocation)
- Hidden checks (categorized)
- Trusted final-state checks (categorized)
- 100-point breakdown (matching the track scorecard)
- Partial-credit ceilings
- Judge model for LLM-as-judge components, if any (default: `gpt-5.4` with `temperature=0.0`)

---

## 8. Reproducibility manifest

### 8.1 `manifest.lock.json`

Every family ships a `manifest.lock.json` at family root:

```json
{
  "family_id": "proposal-ranking-manager-judgment",
  "spec_version": "CNB-55 v1.0",
  "frozen_at": "2026-04-19T00:00:00Z",
  "codex_runner": {
    "binary": "codex",
    "min_version": "0.42.0",
    "model": "gpt-5.4",
    "reasoning_effort": "high"
  },
  "variants": {
    "v1-clean-baseline": {
      "docker_base": "python:3.12-bookworm@sha256:…",
      "files": {
        "<path>": "sha256:…"
      },
      "wall_clock_cap_sec": 1200,
      "token_cap": 400000
    }
  },
  "grader": {
    "scorer_hash": "sha256:…",
    "hidden_tests_hash": "sha256:…",
    "ceilings": [
      {"name": "docs_config_only", "max_points": 10},
      {"name": "happy_path_shim", "max_points": 25}
    ]
  }
}
```

### 8.2 Validation

`scripts/validate_family.py` (new, added alongside this spec) must verify:

1. Every file in `manifest.lock.json.variants[*].files` exists and matches its declared sha256.
2. `grader.scorer_hash` matches the actual scorer file.
3. `grader.hidden_tests_hash` matches the Merkle root of the hidden-tests dir.
4. All pass-bars and calibration targets from §7 are declared.
5. Monotonic-difficulty ordering is recorded in `task_spec.md`.
6. ≥ 4 red-team exploits exist under `verifier_data/`.

Freeze is blocked until validation passes. This mirrors SWE-Bench Pro's reproducibility discipline.

### 8.3 Frozen assets

Once frozen, the following are immutable without a spec revision and regeneration of `manifest.lock.json`:

- Everything under `workspace_bundle/`
- Everything under `verifier_data/<family_id>/`
- `evaluator_contract.md` point allocations and ceilings
- `task_spec.md` difficulty ladder

Editorial edits to `benchmark_run.md` (probe findings, follow-up notes) are allowed post-freeze.

---

## 9. Authoring workflow

### 9.1 Linear build order

1. **Stub `task_spec.md`.** Write task prompt, expected deliverables, per-variant difficulty ladder. No code yet.
2. **Build V1 workspace.** Proposals / repo / evidence / gold artifacts / visible tests. Commit.
3. **Write the scorer.** Phase 2 visible + hidden + trusted. Encode track scorecard. Encode partial-credit ceilings.
4. **Probe V1 with GPT-5.4/high.** Three seeds. Target: mean score ≤ 35.
5. **Harden if probe overperforms.** Add hidden checks. Tighten ceilings. Add red-team traps. Re-probe. Iterate until V1 ≤ 35.
6. **Derive V2–V5 by additive pressure.** Use §4.2 default skeleton unless justified. Each derived variant inherits V1's grader and extends it.
7. **Re-probe whole family.** Confirm calibration §7.8: family mean 15–25, no variant > 40, ≥ 1 variant ≤ 10, monotonic.
8. **Author red-team exploits.** ≥ 4 per family, at least 1 per declared trap. Re-run scorer over each exploit; confirm it produces a low score.
9. **Write `evaluator_contract.md`.** Finalize point breakdown.
10. **Write `benchmark_run.md`.** Probe evidence, hardening log, final calibration numbers.
11. **Freeze `manifest.lock.json`.** Run `scripts/validate_family.py`. Resolve all errors.
12. **Submit for human-verification gate (§10.2).**

### 9.2 Probe-and-harden loop (step 5)

The loop is the most important quality control in CNB-55. It's a compressed version of SWE-Bench Verified's human-verification process, applied to score-distribution rather than resolvability.

On each probe round:

1. Run GPT-5.4/high on the variant. Record trajectory, tool calls, final brief/patch, and score.
2. If score > 35 on V1 (or above the expected rung score on V2–V5), categorize the overperformance:
   - **Shortcut path** → add a red-team trap + a hidden check that blocks it.
   - **Weak hidden check** → add a stronger check or a ceiling.
   - **Over-generous scorer** → reallocate points within the track scorecard.
3. Never lower the pass-bar in response to a strong probe. The pass-bar is set by the track, not by the family.

### 9.3 Partial credit dead-zones

Probe findings sometimes reveal a score band that is never a real human-expert outcome (e.g., agent scores 45 by writing a brief without reading the evidence, or scores 30 by submitting the gold ranking without any justification). If such a dead-zone is reproducible, add a ceiling that caps scores below it. This follows SWE-Lancer's observation that frontier models can collect easy points on managerial tasks by pattern-matching.

---

## 10. Acceptance criteria

### 10.1 Automated freeze gate

A family is admissible to the frozen CNB-55 iff:

- All 5 variants build under their pinned Dockerfiles without network.
- `scripts/validate_family.py` passes.
- Probe calibration §7.8 satisfied across 3 seeds.
- ≥ 4 red-team exploits produce ≤ 20-point scores.
- Oracle solution (author-provided gold patch) produces ≥ 90-point score.
- Empty solution produces 0-point score.
- Scorer deterministic: rerun with fixed seed produces identical JSON.

### 10.2 Human-verification gate (SWE-Bench Verified pattern)

Two authors review each family independently and confirm:

- Task prompt is unambiguous.
- Expected deliverables are achievable from the workspace alone.
- Difficulty ladder is justified.
- Red-team traps describe real shortcuts, not strawmen.
- Scorecard weights respect the track.

Disagreements are resolved by a third reviewer. Only families that clear both gates enter the canonical benchmark.

### 10.3 Post-freeze revisions

Bug-fix edits to scorer behavior require a minor version bump and a new `manifest.lock.json`. Authoring edits that change task semantics require re-running the probe-and-harden loop and bumping the family version.

---

## 11. Reporting

Benchmark runs emit per-task and aggregate JSON:

```json
{
  "run_id": "…",
  "model": "gpt-5.4",
  "reasoning_effort": "high",
  "resolved_rate_overall": 0.17,
  "resolved_rate_by_track": {
    "10": 0.12
  },
  "mean_variant_score_by_family": {
    "proposal-ranking-manager-judgment": 19.4
  },
  "tasks": [
    {
      "task_id": "proposal-ranking-manager-judgment/v1-clean-baseline",
      "score": 34,
      "resolved": false,
      "milestones": {
        "m1_ranking_matches_gold": 0.6,
        "m2_constraints_integrated": 0.3,
        "m3_rejection_reasoning": 0.1
      },
      "ceilings_applied": ["docs_config_only"],
      "wall_clock_sec": 782,
      "tokens_used": 310412
    }
  ]
}
```

Headlines published:

1. **Resolved-rate overall** (SWE-Bench Verified shape).
2. **Per-track resolved-rate** (11 cells).
3. **Per-family mean score** (55 cells).
4. **Cost & latency** (SWE-Bench Pro / Snorkel shape) — reported, not scored.

---

## 12. Glossary

- **Variant.** The atomic scored unit. Five per family.
- **Family.** The authoring unit. Five per track. Shares grader, skill, config.
- **Track.** The 11 top-level categories.
- **Milestone.** A graded sub-check that maps to hidden-test nodes.
- **Ceiling.** A hard cap on score applied if a named shortcut pattern is detected.
- **Probe.** A dry-run of the variant against GPT-5.4/high used to calibrate difficulty.
- **Red-team trap.** A named shortcut the family author expects agents to try.
- **Oracle.** An author-provided gold solution used to sanity-check the upper bound of the scorer.

---

## Appendix A — Source links (chronological, most recent first)

- [Anthropic 2026 Agentic Coding Trends Report (PDF)](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf?hsLang=en)
- [SWE-EVO — arXiv 2512.18470 (2026-01-27)](https://arxiv.org/pdf/2512.18470)
- [Sonar Foundation Agent — SWE-Bench Verified top spot (Feb 2026)](https://www.sonarsource.com/company/press-releases/sonar-claims-top-spot-on-swe-bench-leaderboard/)
- [Snorkel Agentic Coding Benchmark (2026)](https://snorkel.ai/blog/introducing-the-snorkel-agentic-coding-benchmark/)
- [SWE-Bench Pro (Scale AI, 2025)](https://scaleapi.github.io/SWE-bench_Pro-os/)
- [TRAJECT-Bench — arXiv 2510.04550 (2025)](https://arxiv.org/abs/2510.04550)
- [ALE-Bench — arXiv 2506.09050 (2025)](https://arxiv.org/abs/2506.09050)
- [FeatureBench — OpenReview (2026)](https://openreview.net/forum?id=41xrZ3uGuI)
- [SWE-Lancer (OpenAI, 2025)](https://openai.com/index/swe-lancer/)
- [SWE-Bench Verified (OpenAI, 2024)](https://openai.com/index/introducing-swe-bench-verified/)
- [ToolSandbox — arXiv 2408.04682 (2024)](https://arxiv.org/abs/2408.04682)
- [AppWorld — arXiv 2407.18901 (2024)](https://arxiv.org/abs/2407.18901)
- [AgentBench — ICLR'24](https://github.com/THUDM/AgentBench)
- [AI Agent Benchmark Compendium (50+ benchmarks)](https://github.com/philschmid/ai-agent-benchmark-compendium)

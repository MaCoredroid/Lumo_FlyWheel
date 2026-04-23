# Sub-Spec — Auto-Research Agent v0.1

**Parent document.** `docs/HLD-Serving-Backend-AutoResearch-v0_1.md` (v0.1.1) — the serving-backend + auto-research HLD. This sub-spec expands §5 "Auto-Research Agent — Design" of the parent into an implementable contract for the specific case where the agent is **Codex CLI**, driven by a **Python outer loop** (the manager skill) plus a **`lumoserve auto-research …` CLI** plus **git** as the experiment ledger. Every section of this sub-spec is subordinate to the parent: on any conflict, the parent HLD wins and this sub-spec is wrong.

### Two phases — clean separation of duties

This sub-spec separates **Phase A (IMPL)** from **Phase B (auto-research loop)** as two distinct workstreams with distinct deliverables, distinct agents, and a clean hand-off between them. That separation is the cleanest way to reason about "who does what" — and it is also what makes Phase B's per-iteration codex-exec pattern (§2) possible.

**Phase A — IMPL agent builds the substrate, once.** One long-running codex session (the *IMPL agent*) — or equivalently a human PR, or a Claude-driven refactor — delivers the pieces Phase B depends on. This is done **once**, ahead of any auto-research round, and committed to `main` as ordinary code. The IMPL work is tracked by **LLD-SB-06** in the parent HLD §8; this sub-spec does not re-specify the IMPL work itself, but pins the **contract surface** Phase B binds to. Phase A deliverables (what "xxxx" is):

  1. **`src/lumo_flywheel_serving/measurement_harness.py`** — the `RealMeasurementHarness` class per §9 interface. Replay-driven against a real vLLM endpoint, three-dim-SLO-aware, PromQL-cross-checked.
  2. **`scripts/capture_seed_workload.py`** — runs the family's eval set through a default-config serving stack once and persists the per-request seed trace.
  3. **`lumoserve auto-research …` CLI subcommands** — **seven production subcommands** plus the backward-compat `run`: `bootstrap-round` (§8.1), `measure` (§8.2), `commit-candidate` (§8.3), `rescreen` (§8.4), `validate-holdout` (§8.5), `finalize-round` (§8.6), `status` (§8.7), plus `run` (§8.8, env-gated CI-only). The full list must be registered in `lumoserve auto-research --help` for Phase A to be considered complete — §11.1 precondition greps for all seven.
  4. **`skills/auto-research-round-manager/SKILL.md`** — rewritten as the Python outer loop per §11. The skill *is* the loop; it spawns one codex exec per iteration and aggregates results through the CLI.
  5. **`tests/fixtures/synthetic_measurement.py`** — the renamed `SyntheticMeasurementFixture` (moved out of `src/`), used only for Phase A unit tests; refused by `commit-candidate` at Phase B time (§6.3).
  6. **Unit + dry-run integration tests** — covering each CLI subcommand, each skill watchdog path, and a dry-run round against the synthetic fixture (tests only; the production round uses the real harness).
  7. **Pre-flight checks** — the §11.1 precondition bundle: imports cleanly, codex is reachable, git is clean, seed trace exists.
  8. **The two codex-facing briefs** (§5): `impl_brief.md` (consumed by Phase A itself, optional) and `iteration_brief.md` (consumed by every Phase B codex exec).

Phase A's "done" is when items 1–8 all land on `main`, Phase A's own verification items pass (§12 extends this), and the Phase B precondition check (§11.1) returns green. **Until Phase A is done, Phase B is not allowed to run** — `bootstrap-round` refuses.

**Phase B — auto-research loop, per iteration.** Once Phase A's substrate is in place, the v0.1 auto-research round runs. The round tunes **exactly one `(model, family)` pair** **once**, on **one layer**, pre-campaign, offline. The v0.1 target tuple is hard-pinned:

- `model_id`: `qwen3.5-27b`
- `family_id`: `proposal-ranking-manager-judgment`
- `sprint`: `sprint-0`
- `active_layer`: `L1` only (parent §5.3.2 vLLM-config action space)
- `weight_version_id`: the current checkpoint as of bootstrap
- `workload`: `benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml` plus its seed-captured trace (produced by item 2 of Phase A)

**How Phase B is driven.** The manager skill (item 4 of Phase A) is a **Python outer loop**. For each iteration, the skill spawns a fresh `codex exec` invocation — either via the `codex` CLI binary directly or via an equivalent `python-call-codex-exec` helper — that reads `iteration_brief.md` + the current `results.tsv`, proposes one candidate, calls `measure`, calls `commit-candidate`, and exits. Python then reads the new row in `results.tsv`, decides continue-or-stop, and either spawns the next iteration's codex or calls `finalize-round` and terminates the round. **Each codex invocation is short, stateless, and sees only the one iteration's context**; the round's cross-iteration state lives in `results.tsv` and `git log`, not in any codex transcript.

The output of Phase B is **one frozen tuned-config bundle** (parent §5.9) that the serving stack loads at campaign bootstrap and then does not re-tune. A second family, a second model, or a retune on a new weight version is a **v0.2 question**, not a v0.1 one. The CLI's subcommand surface (§8) accepts `--model-id` / `--family-id` arguments for future-proofing, but **v0.1 runs the one pair named above, once, on L1 only**. If the v0.1 round fails to emit a bundle, the serving stack falls back to the default-config baseline (parent §5.7 rail 3); there is no automatic retry, no second round, no escalation to a different family. That is by design — "one-time run" per parent §5.1.

### What each sub-spec section covers

This sub-spec describes, for Phase B: the per-iteration codex-exec pattern (§2), the file interface the per-iteration codex sees (§3), the budget / metric contract per iteration (§4), the `iteration_brief.md` template (§5), the hard rules a per-iteration codex must honor (§6), the git branch + `results.tsv` ledger (§7), the CLI surface Phase B calls (§8), the `RealMeasurementHarness` interface Phase B binds to (§9), the bundle emission hand-off (§10), the Python-outer-loop skill contract (§11), the verification items Phase B adds to the parent §9.3 checklist (§12), and v0.2 open questions (§13).

**What this sub-spec deliberately does not cover.** The IMPL implementation itself (that is Phase A / LLD-SB-06 per parent §8; this sub-spec pins the *contract surface* Phase B binds to, not the internals). The tuned-config bundle schema (parent §5.9). The bundle-validity rule and its weight-rotation nuance (parent §6.4). The L0 per-family parity fixtures and rail-9 gate (parent §5.6, §5.7; L0 is out of scope for the first codex-driven round per Sprint 0, and this sub-spec's Phase B is L1-only at v0.1). The three-dim latency SLO itself (parent §4.1). Generalization to other `(model, family)` pairs — see v0.2 open questions (§13). None of those get re-stated here; cross-references only.

**Why a sub-spec at all.** The 2026-04-23 Sprint-0 round produced a bundle via a pure-Python `SyntheticMeasurementHarness` — no vLLM ever hit the measurement loop, the objective was a polynomial of candidate fields, and the live family gate ran only *after* the bundle had been chosen. The parent HLD's §5.5 and §5.6 already call this out as wrong, but the parent does not prescribe the agent-side control flow — who proposes candidates, how they are scored, how stopping is decided, how the run is auditable. This sub-spec fills that gap for the codex-driven case, with an explicit Phase A / Phase B split so the IMPL deliverables and the auto-research loop contract are never conflated again, drawing on two external references we have looked at directly (§1.1, §1.2).

---

## 0. Terms used in this document

**Round.** One end-to-end invocation of the auto-research agent for a specific `(model_id, family_id, sprint, weight_version_id)` tuple. A round produces at most one tuned-config bundle (§5.9 parent) or explicitly produces nothing and falls back to the default baseline. **v0.1 runs exactly one round**, for the tuple pinned in §0. References to "the round" below mean that single v0.1 round unless explicitly scoped otherwise.

**Candidate.** One `(vllm_config, l0_config, l2_config, l3_config)` value under test within a round. At v0.1 of this sub-spec only the `vllm_config` field varies — L0/L2/L3 are fixed at their defaults, per parent §7.1 Sprint 0.

**Iteration.** A single `(propose → apply → measure → decide keep/discard → commit)` cycle on one candidate. Iterations are counted per-sub-level-per-family for L0 (parent §5.8), per-layer for L1–L3, and in aggregate for the run.

**Agent.** The `codex` CLI process running `gpt-5.4` at reasoning effort `high`, as launched by the manager skill. Exactly one agent process is active at a time within a round.

**Manager skill.** `skills/auto-research-round-manager/SKILL.md`. Owns round lifecycle: bootstrap, watchdog, finalize, fallback-to-default, reporting.

**Harness.** `src/lumo_flywheel_serving/measurement_harness.py:RealMeasurementHarness`. The Python substrate the CLI calls into. This sub-spec defines its interface (§9) but not its internals.

**Round directory.** `output/auto_research/<round_id>/`. Every artifact produced during the round lives under this path. The round's git branch is scoped to this tree plus the winning bundle path under `output/tuned_configs/`.

**Round branch.** The dedicated git branch the round commits into — `autoresearch/<model_id>/<family_id>/<sprint>/<yyyymmddThhmmssZ>`. Every candidate the agent evaluates becomes one commit on this branch.

---

## 1. Design lineage

Two external references shaped this sub-spec's control-flow and artifact conventions. Both were read directly (not relayed), and each is credited below with the exact pattern it contributed and the adaptations we made to fit the HLD.

### 1.1 Karpathy's `autoresearch` — the three-file agent contract

Source: `https://github.com/karpathy/autoresearch` (MIT). The repo runs an LLM agent on a single GPU to auto-optimize a small nanochat training recipe overnight, ~100 experiments per night, measured by a single scalar (`val_bpb`). The patterns we adopt, and how each maps into our setting:

**Three files that matter, split along the Phase A / Phase B boundary.** Karpathy's repo is structured so the agent touches exactly three files: `prepare.py` (read-only — data prep, tokenizer, utilities), `train.py` (the one file the agent edits — model, optimizer, loop), and `program.md` (Markdown briefing that tells the agent what the rules are). We keep the three-file discipline but split it differently because our IMPL work and our research work are separate phases (§0). The substrate (harness + CLI + skill + tests) is Phase A's deliverable and Phase B treats it as read-only — analogous to Karpathy's `prepare.py`. The per-iteration candidate file (`candidate.yaml`) is what each Phase B codex exec writes — analogous to Karpathy's `train.py` but config, not code. The briefings are split two ways: `impl_brief.md` briefs the Phase A IMPL agent (once, upfront), and `iteration_brief.md` briefs each Phase B codex exec (once per iteration) — analogous to Karpathy's `program.md` but with a clean Phase A / Phase B separation so neither brief is overloaded. §5 details both templates; §3 details the on-disk file interface.

**Fixed wall-clock per experiment.** Karpathy fixes training to a 5-minute wall-clock budget per experiment so experiments remain comparable across architectural changes. We fix **per-candidate wall-clock to `measurement_window_minutes + warmup_minutes`** from `serving_workload.yaml`, defaulting to 30 min + 5 min, for the same reason. A candidate that wants more time is rejected by the CLI; a candidate that crashes or OOMs before the window completes is logged as infeasible and counted against the per-sub-level §5.7 three-in-a-row rail.

**One metric, measured identically for every candidate.** Karpathy uses `val_bpb` because it is vocab-size-independent and therefore fair across arch changes. We use the scalar **`sustained_concurrent_eval_threads`** from parent §5.4, gated by the three-dim SLO conjunction (TTFT_p95 / TPOT_p95 / TurnLatency_p95) plus the purity / determinism / OOM / rollout-floor rails. One number. Measured by the same harness, the same driver, the same seed workload trace for every candidate in a round.

**Git as the experiment ledger.** Karpathy uses `git checkout -b autoresearch/<tag>`, commits per experiment, keeps commit on improvement, resets on regression, logs each attempt in `results.tsv`. We adopt this almost verbatim — §6 details the commit-message format, the branch naming, and the `results.tsv` column set. One deviation from Karpathy: we do not `git reset --hard` on discard; we commit every candidate (including infeasible ones) with an explicit `status: discard` row in `results.tsv`. The reason is that the parent HLD §9.3 verifier needs a replayable per-candidate trace, and `git log` is our transcript of record.

**"Never stop" clause — adapted for the per-iteration spawn.** Karpathy's `program.md` forbids the single long-running agent from pausing mid-round to ask the human for permission to continue. Our equivalent is subtler because we do *not* have a single long-running agent — we have one codex exec per iteration. `iteration_brief.md` R6 + R7 forbid the per-iteration codex from emitting round-level stop decisions or calling `finalize-round` itself; the codex exec is told to do its iteration and exit cleanly. The round's "keep going" decision lives entirely in Python's outer loop (§11.3). §5 + §6 formalize this.

**What we do not take from Karpathy.** Karpathy's repo has no notion of a serving backend, no notion of latency SLO, no notion of a frozen artifact bundle that downstream consumers load, and no equivalent of the §6.4 bundle-validity rule. We keep his loop shape but put our own bundle emission and validity-pinning on the back of it.

### 1.2 Spotify's `llm-bandit` — uncertainty-aware candidate selection

Source: `https://github.com/spotify-research/llm-bandit`. We note this reference was the closest match we could find in Spotify's open-source research to "auto-research agent"; a project named exactly "Pi" was not locatable in `spotify-research` or `spotify` GitHub orgs at the time of writing, and the name may have been misremembered. If the intended reference was something else, this section is the one to revise; the substance below is still a defensible v0.2 direction irrespective of the naming.

`llm-bandit` addresses contextual bandits where the context is natural language. An LLM acts as a feature extractor, and uncertainty on the LLM's representation is integrated into a Thompson Sampling policy rather than used greedily. Four uncertainty-estimation methods are supported (last-layer Laplace, diagonal Laplace, MC dropout, epinets). The pattern that maps into our setting is **uncertainty-weighted candidate selection**: when the agent proposes the Nth candidate, score the proposal distribution by expected-improvement-under-uncertainty rather than argmax over mean.

**What we take for v0.1.** Nothing, actually — v0.1 uses Optuna TPE as the L1 inner optimizer (parent §5.5, §5.6) and an LLM-proposer-on-top pattern for the outer layer traversal. `llm-bandit` is a conceptual reference for v0.2 (§13), not a v0.1 dependency.

**What we take for v0.2.** If TPE-driven L1 search hits a local maximum the LLM proposer cannot explain, we add uncertainty-aware candidate scoring using the `llm-bandit` pattern: the codex agent proposes K candidates, the CLI runs each for a short screening window, and the final K/2 for the full window are picked by Thompson Sampling over posterior belief on the per-candidate objective. Open question tracked in §13.

---

## 2. Agent model — Python outer loop, codex-exec per iteration

The Phase B round is structured as a **Python outer loop that spawns one `codex exec` invocation per iteration**. Python owns the round lifecycle (bootstrap, stop criteria, finalize). Codex owns one job only: given the current `results.tsv` + `iteration_brief.md`, propose one candidate, run `measure`, run `commit-candidate`, exit. Each codex invocation is short, stateless, and sees only one iteration's context.

### 2.1 Three decompositions we considered

Before pinning the v0.1 shape, we walked three candidate decompositions. Option 3 is what v0.1 uses; options 1 and 2 are recorded so the choice is auditable.

**Option 1 — one long-running codex agent per round.** Codex stays alive the whole round (~4–8 hours), iterates internally, writes candidate after candidate into its own transcript. Closest to Karpathy's `autoresearch` literally. Rejected because (a) the transcript grows to many thousands of messages and Codex context quality degrades over that length, (b) one crash or context-rot event wastes the whole round, (c) the Python side has to watchdog a long-lived opaque LLM process, which is fragile.

**Option 2 — manager codex agent + worker codex subagents.** One manager codex session holds the round-level reasoning and spawns per-candidate worker codex subagents via codex's subagent mechanism. Rejected at v0.1 because it duplicates a manager the Python skill already is (free, deterministic, $0 LLM cost), adds a layer of "what if the manager hallucinates a worker result" failure modes, and the isolation benefit that motivates manager/worker hierarchies elsewhere is already purchased by option 3 without the extra agent tier. Promotion criteria for moving to option 2 at v0.2 are in §13.3 (L0c mutation, parallel measurement, or multi-layer rounds).

**Option 3 — Python outer loop, codex exec per iteration ← v0.1 choice.** Python (the manager skill) is the loop. Each iteration, Python:

1. Reads `results.tsv` (for stop-criteria check — §11.3).
2. Spawns one `codex exec` invocation with `iteration_brief.md` pointed at the current iteration directory. Codex inherits no transcript from the prior iteration — every invocation starts fresh.
3. Waits for codex to exit. Codex exits either with code `0` (success — one new row in `results.tsv`) or code `2` (BLOCKED — see R8). Any other exit code is a Python-level retry up to `max_iteration_retries` (default 2), after which the round is marked `ROUND_BLOCKED`.
4. Re-reads `results.tsv`, checks stop criteria, loops.

When stop criteria fire, Python calls `finalize-round` (not codex — Python calls the CLI directly), writes the bundle, and exits.

### 2.2 Why Option 3 is the right shape

**Fresh context per iteration.** Karpathy's single-agent loop works at his scale (~100 experiments, small metric space); at ours the agent would carry hundreds of stale measurements in its transcript by iteration 30. Option 3 sidesteps the issue entirely: every codex exec starts with empty transcript + `iteration_brief.md` + `results.tsv`. Cross-iteration memory lives in `results.tsv` (the ledger) and `git log` (the transcript of record), not in any codex process's memory. This is the direct adaptation of Karpathy's "git is the ledger" pattern to the per-iteration-spawn case.

**Python is already the right tool for orchestration.** Stop criteria, wall-clock caps, OOM-in-a-row counting, diminishing-returns detection, finalize — all of these are deterministic control-flow problems that Python solves correctly and for free. Paying LLM cost to have a manager codex agent re-derive them per-round is pure overhead.

**Every iteration is independently replayable.** An iteration whose codex exec crashes mid-flight leaves `results.tsv` unchanged (the CLI transactions are atomic — §8.3) and the iteration directory can be deleted and retried. A long-running codex agent that crashes mid-round is much harder to restart cleanly.

**Hardware sequentiality is free.** DGX Spark is single-GPU for v0.1, so we can only measure one candidate at a time anyway. A long-running agent or a manager/worker hierarchy buys nothing on that axis. Option 3 matches the hardware.

### 2.3 What each codex invocation looks like

The Python outer loop runs, for iteration `<NNN>`, against the codex-cli 0.120.x surface (verified against `codex exec --help` on the operator's machine — see §13.4 for the pin-stability caveat). The real flag surface is narrow — no `--var`, no `--input-file`, no `--config-dir`, no `--workdir` — so Python does template substitution itself and passes the prompt on stdin:

```python
# Python substitutes {{…}} placeholders before piping to codex.
iteration_prompt = substitute_placeholders(
    template=read_file(f"{round_dir}/iteration_brief.md"),
    values={
        "round_id":   round_id,
        "iteration":  f"{iteration:03d}",
        "round_dir":  str(round_dir),
        "iteration_dir": f"{round_dir}/candidates/{iteration:03d}",
        "next_iteration": f"{iteration + 1:03d}",
        ...  # see §5.2 placeholder list
    },
)

env = {
    **os.environ,
    "CODEX_HOME": f"{round_dir}/codex-home",   # codex reads config.toml here
}

iter_dir = f"{round_dir}/candidates/{iteration:03d}"
with open(f"{iter_dir}/agent_session.jsonl", "wb") as transcript:
    result = subprocess.run(
        [
            "codex",
            # Global flags (before the subcommand). -c sets config keys.
            "-c", 'model="gpt-5.4"',
            "-c", 'model_reasoning_effort="high"',
            "exec",
            "--cd", str(round_dir),                      # working dir override
            "--json",                                    # emit JSONL events to stdout
            "--output-last-message", f"{iter_dir}/agent_last_message.txt",
            "--skip-git-repo-check",                     # round_dir is inside repo; fine
            "-",                                         # read prompt from stdin
        ],
        input=iteration_prompt.encode(),
        stdout=transcript,                               # capture JSONL transcript
        stderr=subprocess.PIPE,
        env=env,
        timeout=per_iteration_codex_wall_clock_s,
    )
```

Flag mapping from common mistakes:

| What we want | Correct codex-cli 0.120.x flag | Wrong spec used |
|---|---|---|
| point at a per-round config.toml | `CODEX_HOME` env var | `--config-dir` |
| change working directory | `--cd <path>` (on `exec`) | `--workdir` |
| pass the prompt text | positional arg or `-` + stdin | `--input-file` |
| set model / reasoning-effort | `-c model="…"` / `-c model_reasoning_effort="…"` (global, before subcommand) | inline flags |
| capture the session transcript | `--json` to stdout + redirect, plus `--output-last-message` for the final turn | `--transcript-out` |
| template variable substitution | **Python does this before piping** | `--var` (doesn't exist) |

**Why Python substitutes placeholders instead of codex.** codex-cli has no templating. `iteration_brief.md` on disk keeps its `{{…}}` placeholders for human readability and §12 verification; Python reads the template, substitutes the per-iteration values, and pipes the resulting text to stdin. The on-disk template is never mutated. This also means the per-iteration stdin payload is the exact brief the codex session saw, and Python can persist it alongside the transcript for audit if needed.

Each invocation is short: typical wall-clock is a few minutes of codex LLM work on top of the ~30 minutes the harness takes to measure, and codex is mostly idle waiting for `measure` to return. The transcript captured per iteration (the redirected stdout JSONL plus the `agent_last_message.txt` file) is small and audit-checkable.

The `iteration_brief.md` template is defined in §5. Its entire job is: "here's what iteration this is, read `results.tsv`, propose candidate `<NNN>`, call `measure`, call `commit-candidate`, then exit." Codex does not reason about "should the round continue" — that is Python's job between invocations.

### 2.4 Codex model pin and config

`model = "gpt-5.4"`, `model_reasoning_effort = "high"`, written into `<round_dir>/codex-home/.codex/config.toml` on bootstrap. The precondition in §11.1 refuses to start a round whose codex config doesn't match. Swapping models mid-round is a §13.4 open question.

### 2.5 Concurrency within a round

At v0.1, **exactly one `codex exec` process is alive at a time within a round**. Python enforces this structurally — it blocks on each invocation's return before spawning the next. The round lock at `<round_dir>/.round.lock` prevents *cross-round* concurrency (two separate rounds trying to use the same GPU). Parallel-candidate measurement (`K` codex-exec in parallel) is §13.2 / §13.3 v0.2 material.

### 2.6 Where the IMPL agent fits (Phase A cross-reference)

The Phase A IMPL agent is distinct from these per-iteration Phase B codex-exec invocations. Phase A runs once, ahead of time, as one long-running codex session (or human PR, or Claude-driven refactor) that builds the substrate items 1–8 from §0. Phase A's codex session is *not* governed by this section — it's governed by parent §8 LLD-SB-06 + whatever brief the operator hands to the IMPL agent (§5.1 sketches a template). Once Phase A is done and merged to `main`, the Phase B round can run, and from that point on every reference to "codex" in this sub-spec means the per-iteration codex-exec pattern described above.

---

## 3. The round-directory file interface

Under the per-iteration codex-exec pattern (§2) there are *two* tiers of files: **round-scoped** files that persist across every iteration (the things Python and every codex exec invocation read), and **iteration-scoped** files that live under a single iteration's directory and are written once per spawn.

**Round-scoped** — exists once per round, under `output/auto_research/<round_id>/`:

| Role | File | Written by | Purpose |
|---|---|---|---|
| Per-iteration brief template | `iteration_brief.md` | `bootstrap-round` (§8.1) | The §5.2 template. The Python outer loop substitutes per-iteration variables at spawn time. Never modified during the round. |
| IMPL reference | `impl_brief.md` | `bootstrap-round` | Copy of the §5.1 IMPL brief, for traceability. Not consumed by the Phase B loop — just recorded so the round directory is self-documenting. |
| Round spec | `round_spec.yaml` | `bootstrap-round` | Machine-readable round identity. Pins model_id, family_id, sprint, weight_version_id, workload_distribution_id, SLO ceilings, budget caps, parent HEAD SHA. Used by the CLI for precondition checks (§8), by Python for stop criteria (§11.3), and by every iteration's codex exec (read-only). |
| Ledger | `results.tsv` | `commit-candidate` (§8.3) — appended one row per iteration | Karpathy-style tab-separated experiment ledger. Columns in §7.2. The CLI enforces one row per iteration; Python reads it between iterations for stop criteria; every codex exec reads it as prior-iteration context. |
| Round lock | `.round.lock` | `bootstrap-round` | Held by Python for the whole round lifetime; released by `finalize-round` or on crash. |
| Finalization artifacts | `run_log.json`, `search_trace.json`, `measurement_trace_combined.json` | `finalize-round` (§8.4) | Emitted at round termination. These become the parent-§5.9 bundle's `trace_ref` targets. |
| Codex home | `codex-home/.codex/config.toml` | `bootstrap-round` | Model pin (`gpt-5.4`, `high`) consumed by every codex exec invocation. |

**Iteration-scoped** — exists once per iteration, under `output/auto_research/<round_id>/candidates/<NNN>/`:

| Role | File | Written by | Purpose |
|---|---|---|---|
| Candidate | `candidate.yaml` | this iteration's codex exec | The one vLLM config proposed by this iteration. Schema: parent §5.3.2 L1 action space. The *only* file the codex invocation directly writes. |
| Measurement trace | `measurement_trace.json` | `auto-research measure` (called by this iteration's codex exec) | Per-candidate output of the real harness. Schema §9.2. `generator` must start with `RealMeasurementHarness` or `commit-candidate` refuses. |
| Per-iteration transcript | `agent_session.jsonl` | Python redirecting codex stdout under `--json` (see §2.3 invocation) | The codex session transcript for just this iteration — one JSONL event per codex turn. Small (~minutes of LLM work), audit-checkable per §12. Does not inherit from prior iterations. |
| Per-iteration last message | `agent_last_message.txt` | codex-cli `--output-last-message` | The final assistant message of this iteration's codex session, for §12 audit. |
| vLLM snapshot | `vllm_metrics.prom`, `replay.jsonl` | the harness | Referenced from `measurement_trace.json` for the PromQL cross-check (§4.3). |
| Iteration BLOCKED note | `BLOCKED.md` (optional) | this iteration's codex exec on R8 retry exhaustion (single source of truth — §6.6) | Present only if this iteration hit R8. Python reads it and marks the round `ROUND_BLOCKED`. There is no round-level BLOCKED.md. |

**Why the round/iteration split matters.** Under the previous spec (one long codex agent), the agent had to reason simultaneously about "my current candidate" and "what the round is doing" — the state was conflated. Under the per-iteration spawn pattern, each codex invocation only reads round-scoped files (for context) and writes iteration-scoped files (for this iteration's output). The round is the sum of iterations; no single codex invocation has to understand the whole round.

**Why `candidate.yaml` per iteration instead of edit-in-place like Karpathy.** Karpathy's agent edits `train.py` because the candidate *is* the code. At parent-§5.3.2 L1 the candidate is a ~10-field vLLM-config dict; there is no code to edit. Writing a fresh `candidate.yaml` per iteration gives us a clean per-iteration artifact the harness consumes and the verifier can re-play. When we extend this sub-spec to L0c (parent §5.3.1.c) the candidate *will* include code — a kernel `.patch` diff — and at that point we will adopt Karpathy's "edit the file" pattern for the patch itself while keeping the per-iteration enclosing directory.

**Why a separate `round_spec.yaml` if `iteration_brief.md` already names the identity.** The CLI and the Python skill need a machine-readable round identity for precondition checks, stop-criteria computation, and bundle emission; parsing prose from a brief is brittle. `round_spec.yaml` is what Python and the CLI bind to; `iteration_brief.md` is what codex binds to. Both are written from the same template on bootstrap and must agree — §8.1 includes a consistency check between them.

---

## 4. Budget and metric contract

Everything the agent measures collapses, at the end of each iteration, into one row in `results.tsv` plus one `measurement_trace.json`. This section defines what "one row" means.

### 4.1 Budget per candidate

Per-candidate wall-clock is fixed from the family's `serving_workload.yaml`. Two profiles are allowed at v0.1:

| Profile | Warmup | Measurement | Restart | Total | When used |
|---|---|---|---|---|---|
| **Full** | 300 s (5 min) | 1500 s (25 min) | 180 s (3 min) | **~33 min** | Rescreen phase (§4.4) — top-K candidates re-measured at full window. Parent §5.6 default. |
| **Screen** | 120 s (2 min) | 600 s (10 min) | 180 s (3 min) | **~15 min** | Main search loop at Sprint 0. Parent §5.6 permits shorter windows for early-iteration screening. |

A candidate that exceeds its profile's wall-clock at the measurement-window stage is terminated by the CLI's internal timer with `status: crash, notes: wall_clock_exceeded`.

**Round-level wall-clock and iteration caps — v0.1 Sprint 0 reconciliation.** Parent §5.5 sets an L1 iteration cap of 36, and parent §5.7 safety-rail 1 sets a default round wall-clock cap of 8 hours. But 36 iterations × 15-min screen windows is 9 hours, and at 33-min full windows it's 19.8 hours — the iteration cap and the round wall-clock cap cannot both be binding. At Sprint 0 v0.1 the binding constraint is wall-clock; we therefore tighten the iteration cap and the diminishing-returns window to match reality:

| Variable | v0.1 Sprint 0 | Parent HLD L1 default | Rationale |
|---|---|---|---|
| `round_wall_clock_s` | 28800 (8 h) | 28800 | Parent §5.7 rail 1 default. Unchanged. |
| `iteration_cap` | **12** | 36 | 8 h ÷ 15-min screen windows ≈ 32 candidates theoretical; we leave headroom for rescreen phase (§4.4 — 4 more iterations on top-K) and for bootstrap/teardown. 12 search iterations + up to 4 rescreen = 16 total. |
| `diminishing_returns_window_k` | **4** | 8 (parent §5.8 L1) | Needs to fit inside `iteration_cap`; 4-over-last-4-feasible is the earliest signal that is still robust to single-candidate noise. |
| Measurement profile in main loop | Screen (15 min) | Full (33 min) | Parent §5.6 permits; rescreen phase re-runs top-K at full window for acceptance. |
| Rescreen phase (§4.4) | 3 candidates × 1 repeat each at Full profile | — | Added in v0.1.3 per P1-6 review feedback. |

The CLI refuses to call `measure` for an iteration whose index exceeds `iteration_cap + rescreen_cap`; Python's outer loop (§11.3) stops proposing main-loop candidates once `iteration_cap` is hit and transitions to rescreen (§11.3a). Both Python and the CLI read these caps from `round_spec.yaml`, which the skill writes at bootstrap.

**Why not shorten the measurement profile further.** A 5-min measurement window is too noisy to distinguish candidate configurations on a 30-turn serving workload — TTFT variance alone can swamp a 5% candidate improvement. 10 min of steady-state is the empirical floor where the PromQL cross-check (§4.3) holds and the three-dim p95 numbers stabilize. A future v0.2 could add an even shorter "triage" window for aggressive early pruning, tracked in §13.

### 4.2 Metric — the one number per row

The per-row objective is parent §5.4's `sustained_concurrent_eval_threads` under the three-dim SLO conjunction. The harness reports **seven raw numbers** per candidate plus a feasibility verdict; the ledger collapses them into one row:

| Raw field | Constraint for feasibility | Used as tie-breaker? |
|---|---|---|
| `ttft_p95_ms` | ≤ `L_ttft` (default 2000) | **primary TB**: lower wins |
| `tpot_p95_ms` | ≤ `L_tpot` (default 80) | **secondary TB**: lower wins |
| `turn_latency_p95_ms` | ≤ `L_turn` (default 30000) | used via `turn_latency_p99` as **tertiary TB** |
| `rollout_throughput` | ≥ 0.5 × `rollout_baseline` | not a tie-breaker |
| `reasoning_content_purity` | = 1.0 (parent §5.4 constraint 8) | not a tie-breaker (hard gate) |
| `determinism_pass_rate` | ≥ 0.999 | not a tie-breaker (hard gate) |
| `no_oom_events` | true | not a tie-breaker (hard gate) |
| **`sustained_concurrency`** | the scalar objective | **primary objective**: higher wins |

A row is `feasible: true` only if every constraint holds. An infeasible row still gets appended to `results.tsv` with `objective_value` empty and `status: discard`; the `notes` column captures which constraint tripped (matching parent §5.4's feasibility semantics). The agent may propose follow-up candidates informed by infeasibility reasons (e.g., "candidate-04 was OOM; halve `max_num_seqs` on candidate-05") — this is a valid research step, not a failure of the round.

**What the agent does not do.** The agent never computes `sustained_concurrency` from a formula. The agent never infers `p95` from a point measurement. The agent never uses a prior candidate's measurement to score the current candidate. Every number in `results.tsv` comes from a `measurement_trace.json` written by the harness for that candidate's own measurement window. The `auto-research measure` CLI enforces this by writing the row itself after the harness returns; the agent is not permitted to append a row without a corresponding trace file (§5 hard rule M3).

### 4.3 PromQL cross-check

Per parent §9.1.2 the harness's driver-computed p95 must match a `histogram_quantile()`-derived p95 from vLLM's native histograms within sampling noise. The harness writes both numbers into `measurement_trace.json`:

```json
"ttft_p95_ms": { "driver": 1870, "promql": 1903, "delta_pct": 1.73 }
```

The CLI refuses to append the row to `results.tsv` if `delta_pct > 10` on any of the three latency dimensions — this is a harness-health check, not a candidate-rejection. The agent sees `status: harness_fault, notes: promql_mismatch` and is instructed (via `iteration_brief.md` R8) to halt and emit a BLOCKED note rather than propose another candidate. Python reads the BLOCKED note and marks the round `ROUND_BLOCKED: harness_fault` (§11.6).

### 4.4 Evaluation validity — noise floor, rescreen, holdout validation

Review feedback (P1-6) flagged that the v0.1 loop as originally written optimizes one captured seed trace with a single measurement per candidate, then validates only at the live-family-gate after bundle selection. That invites measurement-noise wins and workload-overfit bundles. The mitigation is three mechanisms that together make the winner selection defensible on a single-GPU / single-seed-trace / v0.1 budget.

**(a) Noise floor from double baseline.** At bootstrap (iteration 000), `measure` is called twice against the default-config baseline — same vLLM config, same seed trace, same SLO, different `candidate_uuid`. The two measurements produce baseline point estimates `M₁` and `M₂` of `sustained_concurrency`. The per-round **noise floor** is set to `noise_floor = 2 × |M₁ - M₂|` (factor 2 so it conservatively approximates a 95%-CI width under a normal assumption with n=2). `noise_floor` is persisted in `round_spec.yaml` at bootstrap-end and read by every subsequent stop-criteria check. A candidate whose main-loop objective improves over the running best by **less than `noise_floor`** does not advance the best-so-far — it is recorded as feasible but not promoted, and the §11.3 diminishing-returns counter treats that iteration as "no improvement."

**(b) Top-K rescreen with repeated measurements.** After the main search loop exits (any §11.3 stop reason), Python transitions into the **rescreen phase**. Let `K_rescreen = 3` at v0.1. The top 3 feasible candidates by `objective_value` are re-measured **once more each**, this time at the Full profile (§4.1, 33 min per candidate). Each candidate now has `measurement_count = 2` rows in `results.tsv` (the original screen + the rescreen). `objective_mean` and `objective_ci_95` are computed from the n=2 measurements per candidate and written into the `status: rescreened` row. The winner is picked by `objective_mean` (not single-shot `objective_value`), and the tie-breakers from §4.2 are applied to the rescreen's latency p95s, not the main-loop values. A candidate whose rescreen-phase measurement falls *outside* `[objective_value ± noise_floor]` is flagged with `status: rescreened, notes: inconsistent_rescreen` and removed from winner contention — the premise is that a candidate whose two measurements disagree beyond the noise floor is not reliably identifiable as the winner.

**(c) Holdout-trace validation.** At the start of Phase A (`scripts/capture_seed_workload.py`), two seed traces are captured from the family eval set: a **main trace** and a **holdout trace**, via a 90/10 split of the family eval set with the seed pinned. The main trace is used for the entire search loop + rescreen phase. The **holdout trace is used only once per round**, after the winner is selected but before `finalize-round` writes the bundle. Python invokes `lumoserve auto-research validate-holdout --round-id <id> --candidate-uuid <winner-uuid>` which replays the holdout trace through the winner's config at the Full profile and checks that the winner remains feasible (all three SLO ceilings, purity, determinism) on the holdout. If the winner fails holdout, the round outcome is `ROUND_BUNDLE_REJECTED: holdout_failed` — no bundle is promoted, serving falls back to the default. If the winner passes holdout, `finalize-round` proceeds and the bundle records `holdout_validation: pass` in the FINALIZE commit trailer and `run_log.json`.

**Why these three and not more.** v0.1 is single-GPU, single-seed-trace per family, and wall-clock-bound to 8 hours including rescreen and holdout. A more rigorous design (n≥5 repeated measurements per candidate, k-fold seed rotation, full paired-significance testing) would blow the Sprint-0 budget by 5–10×. The three mechanisms above give: noise floor catches measurement-variance wins cheaply (n=2 baseline, cost ~30 min added); rescreen catches the top-K from a single noisy ranking (cost: 3 × ~33 min); holdout catches workload-overfit (cost: 1 × ~33 min). Total validity overhead is ~3 h added to the round budget, which fits inside the 8 h cap alongside 12 main-loop screen iterations (~3 h) with headroom.

**What this costs in the time budget.** ~3 h main loop (12 × 15 min) + ~30 min double-baseline + ~1.6 h rescreen (3 × 33 min) + ~33 min holdout + overhead ≈ 6 h of active harness time per round. 2 h of budget headroom remains for bootstrap, finalize, and the post-round live family gate (§11.5). The live family gate is a *correctness* check (unchanged from v0.1.1) — it is **not** redundant with holdout validation; holdout measures SLO-feasibility on a held-out replay of the same family's eval, while the live gate measures end-to-end correctness on a codex-driven family task.

### 4.5 Per-candidate cache isolation

Without an explicit policy, vLLM's prefix cache can silently leak state across candidates — one candidate warms the cache, the next candidate benefits (or suffers) from that warm state rather than from its own configuration. vLLM exposes a `cache_salt` primitive precisely for per-request cache isolation; we use it per candidate.

**Policy.** At every `measure` call (§8.2 step 4): (a) the candidate's `vllm_config` is applied via `/admin/load_tuned_config`, which restarts vLLM and therefore drops the prefix cache; (b) the measurement driver sets `cache_salt = candidate_uuid` on every request it sends, so even if vLLM's internal reset were incomplete, different candidates would hash into different cache keys; (c) the harness records the first-block and last-block prefix-cache hit rate in `measurement_trace.json` so §12 verification can confirm each candidate started from cold (hit rate on the first 10 requests is ~0).

**What the measurement window measures.** Warm-cache behavior is observed **within** each candidate's measurement window (as requests accumulate and the cache warms). Warm behavior is **not** carried across candidates. This matches parent §5.6 — "the prefix cache intentionally starts cold — warm-prefix behavior is measured *within* the iteration, not carried across iterations."

**Verification.** §12 adds an item that checks the first 10 requests' prefix-cache hit rate is ≤ 10% for every candidate, and that every candidate's `cache_salt` equals its `candidate_uuid` as recorded in the trace.

---

## 5. Codex-facing briefs — `impl_brief.md` and `iteration_brief.md`

Two distinct briefs live under the round directory, one per phase. They are deliberately separate artifacts — conflating them is what led to the previous spec's single-long-agent framing we are now replacing. **Neither brief describes the round's stop criteria**; that is Python's job in §11.

| Brief | Phase | Consumed by | Length | Describes |
|---|---|---|---|---|
| `impl_brief.md` | Phase A | one long-running codex session (the IMPL agent), or a human PR, or a Claude refactor | ~2 pages | The substrate to build: harness module, CLI subcommands, skill rewrite, seed-capture script, tests, pre-flight checks. |
| `iteration_brief.md` | Phase B | every per-iteration `codex exec` invocation | ~1 page | One iteration's job: read `results.tsv`, propose next `candidate.yaml`, call `measure`, call `commit-candidate`, exit. |

### 5.1 `impl_brief.md` — Phase A brief

The IMPL agent builds the substrate and commits it to `main`. This brief is a one-shot; the IMPL agent is not in the per-iteration Phase B loop. The manager skill never reads this brief at runtime — it's for whoever is doing the IMPL work (a codex operator, a human developer, or a Claude session). Template:

```markdown
# IMPL Brief — Auto-Research Substrate (LLD-SB-06)

You are the implementation agent. Your job is to deliver the substrate
the v0.1 auto-research round will run on top of. This is a one-shot
implementation task, not a research loop.

## Context docs (read all three first)

- Parent HLD:  docs/HLD-Serving-Backend-AutoResearch-v0_1.md
- Sub-spec:    docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md
- Parent §5.6 (measurement harness), §5.9 (bundle schema), §9.3 (verification)

## Deliverables (all must land on main)

1. src/lumo_flywheel_serving/measurement_harness.py
   - class RealMeasurementHarness per sub-spec §9.1
   - measure() method per §9.1 signature
   - emits MeasuredTrace per §9.2 schema with generator =
     "RealMeasurementHarness v0.1.0"
   - implements the parent §5.6 loop: /admin/load_tuned_config,
     /health wait, seed-trace replay, per-request latency capture,
     /metrics scrape at window boundaries, PromQL-derived p95
     cross-check, purity sample, determinism probe, KV-poisoning probe

2. scripts/capture_seed_workload.py
   - runs family eval set through default-config serving stack once
   - persists per-request jsonl: prompt_tokens, output_tokens,
     thinking_tokens, turn_index
   - emits workload_distribution_id = sha256 of the persisted file

3. CLI subcommands under `lumoserve auto-research …` — all 7 required
   for Phase A completion, plus the backward-compat `run`:
   - bootstrap-round   (sub-spec §8.1)
   - measure           (sub-spec §8.2)
   - commit-candidate  (sub-spec §8.3)
   - rescreen          (sub-spec §8.4 — required by finalize-round)
   - validate-holdout  (sub-spec §8.5 — required by finalize-round)
   - finalize-round    (sub-spec §8.6 — refuses without rescreen + holdout
                         unless --dry-run is passed, §8.6a)
   - status            (sub-spec §8.7 — read-only round state for Python)
   Existing `run` subcommand stays but is env-guarded per §8.8.

4. skills/auto-research-round-manager/SKILL.md — full rewrite
   - Python outer loop per sub-spec §11
   - spawns `codex exec` per iteration (sub-spec §2.3)
   - owns stop criteria (sub-spec §11.3)
   - calls bootstrap-round, loop-of-codex-exec, finalize-round,
     live family gate in that order

5. tests/fixtures/synthetic_measurement.py
   - move SyntheticMeasurementHarness here, rename to
     SyntheticMeasurementFixture, emit generator =
     "SyntheticMeasurementFixture v<n>"
   - commit-candidate must REFUSE this generator per sub-spec §6.3

6. Unit + integration tests:
   - unit: each CLI subcommand
   - unit: skill watchdog paths (silence, out-of-scope write,
           unsigned commit)
   - integration: dry-run round against SyntheticMeasurementFixture
                  (allowed only under LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT=1)
   - integration: precondition refuses when harness module absent

7. Pre-flight checks for the skill (sub-spec §11.1):
   - RealMeasurementHarness imports cleanly
   - codex --version returns expected version
   - git status clean
   - workload yaml has seed_trace_ref pointing at existing jsonl
   - LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT unset

8. Codex-facing brief templates (strings in the skill):
   - impl_brief.md   (this file — you may update if you discover
                       the spec is wrong; note the update in §14)
   - iteration_brief.md (sub-spec §5.2 template — ship verbatim)

## Done when

- All 8 items above land on main
- All unit + integration tests pass
- A dry-run round against SyntheticMeasurementFixture completes
  successfully end-to-end (demonstrates the wiring is correct,
  does not prove the real harness works)
- `python -c "from lumo_flywheel_serving.measurement_harness \
   import RealMeasurementHarness"` succeeds
- sub-spec §9.3.AR.7 and §9.3.AR.12 verification items pass

## You may

- install packages and add dependencies to pyproject.toml
  (Phase A is the only phase where this is allowed)
- modify any file in the repo
- create new files under src/, scripts/, skills/, tests/
- refactor existing code that conflicts with the new surface

## You may not

- ship a Phase A deliverable that calls SyntheticMeasurementFixture
  from production code paths
- modify docs/ without updating the corresponding sub-spec section
- leave any test failing
- declare done without running the dry-run round end-to-end

## Exit protocol

Open one PR with all 8 deliverables. Title:
  "Phase A: auto-research substrate (LLD-SB-06)"
Body: checklist from "Done when" above, all items checked.
```

### 5.2 `iteration_brief.md` — Phase B per-iteration brief

Every `codex exec` invocation in the Phase B round reads this brief and nothing else (plus `results.tsv`, the round-specific yaml files it is supposed to write, and whatever read-only context the brief points at). The Python outer loop substitutes `{{round_id}}`, `{{iteration}}`, and a couple of other variables into the template at each invocation.

**Important property** — `iteration_brief.md` describes **one iteration's job**, not the round. The word "round" appears for context, but the agent is not responsible for round-level stop criteria, for tracking iteration counts, or for calling `finalize-round`. Those are Python's responsibilities (§11). This is the cleanliness the previous spec was missing.

Template:

```markdown
# Auto-Research Iteration {{iteration}} of Round {{round_id}}

You are running ONE iteration of an auto-research round. You are not
running the round. Python is running the round and will spawn your
successor when you exit cleanly.

## Round identity (read-only — DO NOT edit)

- round_id:            {{round_id}}
- model_id:            {{model_id}}
- family_id:           {{family_id}}
- active_layer:        {{active_layer}}
- round_branch:        {{round_branch}}
- round_spec_ref:      {{round_dir}}/round_spec.yaml

## This iteration

- iteration:           {{iteration}}          # e.g. "007"
- iteration_dir:       {{round_dir}}/candidates/{{iteration}}/
- prior_results_ref:   {{round_dir}}/results.tsv   # all rows up to {{iteration}}-1

## Your job (exactly four steps — do them in this order)

1. Read {{round_dir}}/round_spec.yaml to understand the SLO ceilings,
   iteration_cap, and active_layer for this round.

2. Read {{round_dir}}/results.tsv. Look at every prior row. Study the
   pattern of feasible vs infeasible candidates, the constraint each
   infeasible candidate tripped, the TTFT/TPOT/TurnLatency numbers of
   the feasible ones, and the objective value trend.

3. Propose ONE candidate for this iteration. Write it to:
     {{iteration_dir}}/candidate.yaml
   Schema: parent HLD §5.3.2 L1 action space keys only. No L0, no L2,
   no L3 keys. No extra keys. The baseline case (iteration=000) is
   the default-config dict from the model registry.

4. Invoke:
     lumoserve auto-research measure \
       --round-id {{round_id}} \
       --candidate {{iteration_dir}}/candidate.yaml
   The CLI will:
     - /admin/load_tuned_config with your candidate's vllm_config
     - wait for /health
     - drive RealMeasurementHarness for warmup + measurement window
     - write measurement_trace.json next to candidate.yaml
     - append one row to results.tsv with a stable candidate_uuid
       populated (no commit_sha column — see §7.2)
     - print one JSON object to stdout including {candidate_uuid, ...}
     - exit 0 on success, non-zero with structured error on fault
   Total wall-clock: ~{{per_candidate_wall_clock_minutes}} minutes.

5. Read {{iteration_dir}}/measurement_trace.json. Pick ONE status from
   {keep, discard, crash, baseline}. Then invoke:
     lumoserve auto-research commit-candidate \
       --round-id {{round_id}} \
       --iteration {{iteration}} \
       --status <status> \
       --notes "<one-line rationale grounded in the trace>"
   The CLI will create one git commit with message format §7.3.

6. Exit with code 0.

## Hard rules (sub-spec §6 — verified by watchdog + CLI)

R1. You may write ONLY under {{iteration_dir}}. The CLI rejects other
    paths.
R2. You may NOT modify round_spec.yaml, iteration_brief.md, results.tsv
    (except via the CLI), or anything under src/ docs/ benchmark_blueprints/.
R3. You may NOT call `pip install` or any package-install command.
R4. You may NOT hand-compute objective values. The only source of
    truth is measurement_trace.json.
R5. You may NOT make git commits yourself — only via `commit-candidate`.
R6. You do NOT decide whether the round continues. Exit 0 when this
    iteration is done. Python decides what happens next.
R7. You do NOT call `finalize-round`. That is Python's job when the
    round is done. Calling it yourself is a R2 violation and the
    watchdog will kill the round.
R8. If a CLI call returns non-zero, read the error. Retry at most
    twice. If still failing, write a one-line explanation to
    {{iteration_dir}}/BLOCKED.md and exit with code 2.

## What "done" looks like for this iteration

- {{iteration_dir}}/candidate.yaml exists and is valid
- {{iteration_dir}}/measurement_trace.json exists with
  generator starting with "RealMeasurementHarness"
- One new row in results.tsv with a candidate_uuid column
  populated
- One new commit on {{round_branch}} whose message carries both
  a `Candidate-UUID: <uuid>` trailer (matching the results.tsv
  row) and a `Signed-off-by: lumoserve-auto-research-cli` trailer
- You have exited with code 0

## Out-of-scope for this iteration (Python handles)

- Deciding whether to run iteration {{next_iteration}}
- Detecting diminishing returns across iterations
- Detecting 3-in-a-row OOM hard-infeasibility
- Running the live family gate
- Writing the bundle yaml
- Merging the round branch

## Reference material (read if needed — do not modify)

- Parent HLD:     docs/HLD-Serving-Backend-AutoResearch-v0_1.md
- Sub-spec:       docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md
- Workload yaml:  {{workload_file}}
- CLI help:       lumoserve auto-research --help
```

### 5.3 What each brief encodes

`impl_brief.md` encodes a one-shot deliverables checklist, and the IMPL agent's "done" criterion is "all 8 deliverables landed + dry-run round passes." No inner loop, no iteration. `iteration_brief.md` encodes exactly one iteration's job and explicitly delegates round-level concerns (stop criteria, finalize, merging) back to Python. A codex invocation that tries to decide "should the round continue" is violating R6; a codex invocation that tries to call `finalize-round` is violating R7; both are watchdog kill conditions in §11.

### 5.4 What the templates do *not* encode

Neither brief carries the round's stop criteria, the diminishing-returns threshold, the iteration cap, or the wall-clock budget. Those live in `round_spec.yaml` (for the iteration brief to *reference* when proposing candidates) and in the Python skill (for the outer loop to *enforce*). This separation is deliberate — the previous spec put all of these in `program.md` and the single codex agent was supposed to honor them. Splitting along the Phase A / Phase B boundary, and then splitting Phase B along the "Python decides when to stop; codex proposes one candidate" boundary, means each artifact has exactly one job.

---

## 6. Hard rules (agent-facing, formalized)

These are the rules the template's `R1`–`R8` abbreviate. Each has a machine-verifiable counterpart in §12 so a verifier agent can check the round transcript after the fact.

### 6.1 Write-scope rule (R1)

The agent may write only under `output/auto_research/<round_id>/`. The CLI wraps every agent-visible write path so that `candidate.yaml` files outside that tree are rejected. The `commit-candidate` command refuses to stage files outside the round directory plus `output/tuned_configs/…/<bundle.yaml>` which only `finalize-round` is allowed to touch.

### 6.2 Immutability rules (R2, R3)

No file under `src/`, `docs/`, `benchmark_blueprints/`, `scripts/`, `pyproject.toml`, `Makefile`, or the round's own `impl_brief.md` / `iteration_brief.md` / `round_spec.yaml` / `serving_workload.yaml` may be modified during the round. The Python outer loop enforces this between iterations by running `git diff main..HEAD --name-only` on the round branch after each `commit-candidate` and failing the round if any path outside the allow-list appears (§11.6). Package installs are prohibited in Phase B — `pip`, `uv pip`, `poetry add`, `npm install`, `apt install`, `brew install` are all blocked at the skill level (the skill sets `PIP_NO_INSTALL=1`-style env guards and watches for invocation strings in each per-iteration codex transcript). Phase A, as noted in §5.1, is the *only* phase where package installs are allowed.

### 6.3 Measurement-only-via-CLI rule (R4, R5)

Every `results.tsv` row must correspond 1:1 to a `measurement_trace.json` produced by the harness in the same iteration directory. The agent never computes objective values. The agent never estimates latency from a prior run. `auto-research commit-candidate` refuses to commit an iteration whose `measurement_trace.json` is missing or whose `generator: …` field is not `RealMeasurementHarness v<n>`. Specifically — traces with `generator: SyntheticMeasurementHarness` or `synthetic: true` fields are rejected with a structured error, which exists *specifically* to prevent the 2026-04-23 failure mode from recurring.

### 6.4 Commit-via-CLI rule (R5, R7)

Every commit on the round branch is made by `auto-research commit-candidate` or `auto-research finalize-round`. Raw `git commit` invocations inside a codex exec are blocked at watchdog level — Python checks between iterations that every new commit on the round branch carries the expected `Signed-off-by: lumoserve-auto-research-cli` **structured trailer** (§11.6), parsed via `git log --format='%(trailers:key=Signed-off-by,valueonly)'`. A commit that lacks the trailer, or carries a different trailer value, triggers `ROUND_BLOCKED: missing_or_mismatched_trailer`. Note this is **not a cryptographic signature** — it's a DCO-style attribution trailer, because codex-cli doesn't GPG-sign commits by default and requiring real signing at v0.1 is out of scope. The per-iteration `Candidate-UUID` trailer is checked by the same mechanism. This rule is what makes `git log` reliable as the round's transcript of record. Additionally, R7 forbids a codex exec from calling `finalize-round` — that is Python's job once the round is done — which means no codex exec ever creates the FINALIZE commit.

### 6.5 No-pause-but-exit-cleanly rule (R6 — the Karpathy "never stop" clause, adapted)

The codex-exec pattern changes what "never stop" means. A long-running single agent would be told "don't pause, don't ask, keep going." A per-iteration spawn is told something subtly different: **exit cleanly when your iteration is done**. Specifically: a codex exec does not emit "should the round continue?" questions, does not call `finalize-round` itself, does not skip iterations. It does exactly the 4–5 steps in `iteration_brief.md` and exits with code 0. Round-level "keep going" is Python's call, made by spawning the next iteration's codex exec. Two mechanisms enforce this: (a) `iteration_brief.md` R6 + R7 explicitly forbid the codex-internal stop-decision, and (b) the Python loop in §11.3 is the only thing that decides "spawn iteration N+1 or finalize." If a codex exec somehow emits `finalize-round` anyway, the §11.6 watchdog flags it as an out-of-scope call.

### 6.6 CLI-error-handling rule (R8)

On non-zero exit from a CLI command, the agent is permitted up to 2 retries of that exact command after reading the error message. Retries must be evidence-based (agent reads the error, proposes what to change, retries). If retries exhaust, the agent writes `BLOCKED: <one-line summary>` to **`<iteration_dir>/BLOCKED.md`** (iteration-scoped — exact path `output/auto_research/<round_id>/candidates/<NNN>/BLOCKED.md`) and exits with code 2. The Python outer loop detects the per-iteration BLOCKED.md file and lifts it into a round-level `ROUND_BLOCKED` outcome; there is **no round-level `BLOCKED.md`** at `output/auto_research/<round_id>/`, and anything written to that path is treated as an out-of-scope write and triggers `ROUND_BLOCKED: out_of_scope_write` via §11.6. Single source of truth — the only valid BLOCKED path is iteration-scoped.

---

## 7. Git workflow — the experiment ledger

Git is the round's transcript of record. Every candidate is a commit; every commit carries a structured message; `results.tsv` is the index.

### 7.1 Branch

- **Name.** `autoresearch/<model_id>/<family_id>/<sprint>/<yyyymmddThhmmssZ>` — e.g., `autoresearch/qwen3.5-27b/proposal-ranking-manager-judgment/sprint-0/20260425T040000Z`.
- **Parent.** The branch is cut from whatever HEAD is at `bootstrap-round` time. The round spec records the parent SHA.
- **Scope.** Only the round directory (`output/auto_research/<round_id>/`) and, at finalize-round, the winning bundle path under `output/tuned_configs/…`. No other paths may appear in a commit on this branch (§6.2 + enforced by watchdog).
- **Merging.** The agent does not merge the branch to `main`. The manager skill does not merge it either. Merging is a human action that happens only after parent §9.3 acceptance, and is out of scope for this sub-spec.

### 7.2 `results.tsv` — one row per candidate

Tab-separated, one header row, one data row per `measure` call. Columns:

```
candidate_uuid  parent_candidate_uuid  iteration  candidate_label  feasible  objective_value  objective_mean  objective_ci_95  measurement_count  ttft_p95_ms  tpot_p95_ms  turn_latency_p95_ms  rollout_throughput  reasoning_content_purity  determinism_pass_rate  status  notes
```

- **`candidate_uuid`** is the stable identity for this iteration — a UUIDv4 emitted by `measure` and written into the `results.tsv` row at the same time as all the other columns. It is chosen deterministically so that the row can be written atomically once and never rewritten. The commit created by `commit-candidate` carries a `Candidate-UUID: <uuid>` trailer in its message (§7.3); the row ↔ commit linkage is derived by matching `candidate_uuid` in the row against the trailer in `git log`. There is **no `commit_sha` column** — storing a commit's own SHA inside a file that is part of that commit's tree is impossible without a second fixup commit, and the uuid-based linkage sidesteps the issue entirely. `finalize-round` and §12 verification both resolve uuid → commit by parsing commit trailers (§11.6).
- **`parent_candidate_uuid`** is the explicit lineage column. For main-loop rows (`status ∈ {baseline, keep, discard, crash, harness_fault}`) it is **empty**. For rescreen rows (`status: rescreened`) it is the `candidate_uuid` of the main-loop row being re-measured; the corresponding commit carries a `Rescreen-Of-UUID: <parent_uuid>` trailer so the row↔lineage is verifiable from `git log` alone. This replaces the v0.1.3 draft's "join rescreen rows to main-loop rows by objective/top-K behavior" which was brittle under ties and duplicate configs (P2-E review finding). `finalize-round` uses `parent_candidate_uuid` to pair each rescreen row's `objective_value` with its parent's for computing `objective_mean` and `objective_ci_95`.
- `objective_value` is the single-measurement objective for this iteration (the one the iteration's `measure` call reported). `objective_mean` and `objective_ci_95` are populated only after the rescreen phase (§4.4) — they are empty for rows that have only been measured once. `measurement_count` defaults to 1; rescreen phase increments it.
- `status` ∈ `{baseline, keep, discard, crash, harness_fault, rescreened}` — matching the Karpathy set plus three additions (`baseline` for phase-(a) double-baseline rows, `harness_fault` for the §4.3 PromQL-mismatch case, `rescreened` for rows whose objective was re-measured in the §4.4 rescreen phase).
- **`iteration`** is a string drawn from the formal **iteration-id grammar** — not a plain integer. The grammar is `^(\d{3}|baseline_[ab]|rescreen_\d{2})$`, which admits three disjoint forms: (a) three-digit zero-padded main-loop indices `001`–`999`, (b) baseline replays `baseline_a` and `baseline_b`, (c) rescreen-phase entries `rescreen_01`–`rescreen_99`. Every `iteration_id` maps 1-to-1 onto a candidate directory `candidates/<iteration_id>/`. `commit-candidate --iteration` and every `results.tsv` row honor this grammar; validators that assume purely-numeric iteration ids are incorrect.
- Infeasible rows have `objective_value` empty (not zero) — zero is a legitimate objective for a feasible-but-useless candidate and we want to distinguish.
- `notes` is free-form single-line text, typically the one-line rationale the agent gave `commit-candidate`.

**Why no `commit_sha` column — P0-2 fix.** A commit cannot contain its own final SHA because the SHA is a hash of the commit's tree (which includes `results.tsv`) plus metadata; putting the SHA in the row before committing would make the SHA depend on itself. The previous draft said `commit-candidate` "stages, commits, then amend-fills the sha" under the cover of "one atomic commit, commit_sha backfilled" — which is not atomic (amending rewrites the commit to a new SHA, and the amended commit also has that problem if it tries to include its own SHA). The uuid-plus-trailer pattern makes the linkage one-way (row carries stable uuid; commit message carries the same uuid as a trailer; finalize resolves row→commit by trailer grep) and keeps `commit-candidate` a single atomic commit.

### 7.3 Commit-message format

One commit per iteration, message format:

```
AR(<round_id>) C<NNN>: <one-line rationale>

status=<status> objective=<value|infeasible:<reason>> feasible=<true|false>
ttft=<int>ms tpot=<int>ms turn=<int>ms purity=<float> determinism=<float>
trace_ref=output/auto_research/<round_id>/candidates/<NNN>/measurement_trace.json

Candidate-UUID: <uuid>
Signed-off-by: lumoserve-auto-research-cli <auto-research@lumo-flywheel>
```

Both `Candidate-UUID` and `Signed-off-by` are structured git *trailers*. `commit-candidate` writes them via `git commit -m "$MSG"` where `$MSG` ends with a blank line + the trailers; `finalize-round` and §11.6 parse them via `git log --format='%(trailers:key=Candidate-UUID,valueonly)'` and `--format='%(trailers:key=Signed-off-by,valueonly)'` respectively. Neither is a GPG signature — `Signed-off-by` is a DCO-style attribution trailer; no real cryptographic signing is required at v0.1. §11.6 elaborates.

### 7.4 Finalize commit

After termination, `finalize-round` makes one final commit:

```
AR(<round_id>) FINALIZE: <winning candidate label> — obj=<value>

winner_iteration=<NNN> winner_candidate_uuid=<parent_uuid> winner_rescreen_uuid=<rescreen_uuid> bundle=output/tuned_configs/<family_id>/<wvid>/<ts>_<hash>.yaml
round_wall_clock_minutes=<int> total_iterations=<int> feasible_count=<int>
rescreened_count=<int> holdout_validation=<pass|fail|skipped>
stopping_reason=<iteration_cap|wall_clock_cap|diminishing_returns|hard_infeasibility_oom|hard_infeasibility_determinism|harness_fault|holdout_rejected>

Winner-Candidate-UUID: <parent_uuid>
Signed-off-by: lumoserve-auto-research-cli <auto-research@lumo-flywheel>
```

**Winner identity — `Winner-Candidate-UUID` is the *parent* main-loop uuid, not the rescreen uuid.** The winner is the *configuration* that wins — i.e., the `vllm_config` inside the parent `candidates/<NNN>/candidate.yaml` that becomes the bundle. The rescreen rows are additional *measurements* of that same configuration, used only to compute `objective_mean`; they do not produce new configurations. So:

- `winner_iteration` = the parent main-loop iteration's zero-padded index (e.g. `003`), pointing at `candidates/003/candidate.yaml` which becomes the bundle's `vllm_config`.
- `winner_candidate_uuid` (in the body) and `Winner-Candidate-UUID` (the trailer) = the **parent** main-loop row's `candidate_uuid`. Both are the same value.
- `winner_rescreen_uuid` (in the body only, not a trailer) = the fresh `candidate_uuid` of the rescreen row whose `parent_candidate_uuid` equals `Winner-Candidate-UUID`. Kept for lineage replay at §9.3.AR.18 verification time.

This matches §9.3.AR.18 which expects `Winner-Candidate-UUID` to be a main-loop row's uuid (resolvable by lookup in `results.tsv` to a row whose `status` is in `{keep, baseline}` and whose `parent_candidate_uuid` is empty).

The finalize commit contains `run_log.json`, `search_trace.json`, `measurement_trace_combined.json`, `rescreen_trace.json`, `holdout_trace.json` under the round dir, plus the bundle yaml at its destination path. At this point the round is done; Python reports the bundle path to its caller; the skill runs the live family gate as a separate post-round step (§11.5).

---

## 8. CLI surface — `lumoserve auto-research …`

v0.1 splits the current single `auto-research run` subcommand into **seven production subcommands** (`bootstrap-round` §8.1, `measure` §8.2, `commit-candidate` §8.3, `rescreen` §8.4, `validate-holdout` §8.5, `finalize-round` §8.6, `status` §8.7) plus the preserved env-gated `run` (§8.8) for CI smoke-test use. Each is spec'd below; implementation is tracked under LLD-SB-06 (parent §8).

### 8.1 `bootstrap-round`

```
lumoserve auto-research bootstrap-round \
  --model-id <id> \
  --family-id <id> \
  --sprint <sprint> \
  --workload-file <path> \
  --weight-version-id <sha> \
  --round-root <output_path>
```

Effects: creates `output/auto_research/<round_id>/` with `impl_brief.md` (reference copy of §5.1), `iteration_brief.md` (the §5.2 template — variables substituted per-iteration by Python at spawn time), `round_spec.yaml`, `candidates/` populated with the two **pre-written baseline candidate directories** (see below), `results.tsv` (header-only), `codex-home/.codex/config.toml` (pinning `gpt-5.4 high`). Opens the round git branch. Captures the parent SHA into `round_spec.yaml`. Acquires the round lock. Runs a consistency check: `impl_brief.md` / `iteration_brief.md` / `round_spec.yaml` must all agree on the round identity (model_id, family_id, sprint, weight_version_id, SLO ceilings).

**Baseline candidate files — written by bootstrap, consumed by §11.3 phase (a).** `bootstrap-round` writes two candidate directories before returning:

- `candidates/baseline_a/candidate.yaml` — default-config dict read from `model_registry.yaml[model_id].vllm_config()`, unmodified.
- `candidates/baseline_b/candidate.yaml` — **identical content** to `baseline_a/candidate.yaml` — same vLLM config, same fields. The two baselines differ only in their `candidate_uuid` (which `measure` generates at run-time, not at bootstrap) and in the replay seed used by the driver. Writing identical yaml is intentional — the noise-floor measurement requires the two baselines be configurationally identical so any measured difference is attributable to noise, not config.

These files are created by `bootstrap-round` and **not** by any codex session. Phase (a) of §11.3 therefore finds them already on disk when it invokes `measure`. This closes the P1-L loop — previously the pseudocode called `measure` against files that nothing had written.

Preconditions checked: `RealMeasurementHarness` imports cleanly, `workload-file` exists and has a non-empty `seed_trace_ref`, `model-id` is in `model_registry.yaml`, parent git HEAD is clean. Any failure emits structured error and returns non-zero; no filesystem changes are made in that case.

### 8.2 `measure`

```
lumoserve auto-research measure \
  --round-id <id> \
  --candidate <path_to_candidate.yaml>
```

Effects, in order: (1) validate candidate schema, (2) generate a fresh `candidate_uuid` (UUIDv4), (3) `/admin/load_tuned_config` with the candidate's `vllm_config`, (4) reset vLLM's prefix cache and set a per-candidate `cache_salt = candidate_uuid` (§4.5 cache isolation), (5) wait on `/health`, (6) run the harness for `warmup_s + measurement_window_s` (non-interactively — the CLI blocks), (7) write `measurement_trace.json` next to the candidate.yaml with the uuid embedded, (8) append one row to `results.tsv` with the uuid populated, (9) print one JSON object to stdout including `candidate_uuid` and feasibility verdict, (10) return 0 on success, non-zero with structured error on harness fault / OOM / unreachable endpoint.

This is the load-bearing subcommand — every measurement-driven artifact in the round flows through here. If this returns successfully, the agent knows the harness ran, the trace is real, and the row is appended. If this errors, the agent reads the error and either retries (up to R8's 2 retries) or emits BLOCKED.

### 8.3 `commit-candidate`

```
lumoserve auto-research commit-candidate \
  --round-id <id> \
  --iteration <iteration_id> \   # grammar: ^(\d{3}|baseline_[ab]|rescreen_\d{2})$ — see §7.2
  --status <baseline|keep|discard|crash|harness_fault> \
  --notes <one_line>
```

Effects: (1) validate that iteration `<NNN>`'s candidate.yaml + measurement_trace.json both exist, are well-formed, and carry the same `candidate_uuid` as the pending `results.tsv` row, (2) stage the candidate directory + updated `results.tsv`, (3) create **one atomic commit** with the §7.3 message format (including the `Candidate-UUID: <uuid>` and `Signed-off-by:` trailers), (4) emit one JSON object on stdout with `{iteration, candidate_uuid, commit_sha, status}` — the `commit_sha` in this JSON payload is the SHA *returned by git commit*, it is **not** persisted back into `results.tsv` (see §7.2 on why).

Refuses to commit if: (a) `generator` in the trace is not `RealMeasurementHarness v<n>`, (b) any field outside the round allow-list (§6.2) is staged, (c) the commit would overwrite an existing iteration's files, (d) the `candidate_uuid` embedded in `measurement_trace.json` does not match the uuid in the pending `results.tsv` row (guards against a stale trace file surviving from a prior retry).

### 8.4 `rescreen` — top-K repeated measurements at Full profile (§4.4)

```
lumoserve auto-research rescreen \
  --round-id <id> \
  --top-k 3 \
  --profile full
```

Effects: (1) read `results.tsv`, pick the top-K feasible main-loop rows by `objective_value`, (2) for each parent, allocate a new **rescreen artifact directory** `candidates/rescreen_<PP>/` where `<PP>` is a zero-padded two-digit index within the rescreen phase (`rescreen_01`, `rescreen_02`, `rescreen_03`, …), (3) copy the parent's `candidate.yaml` into the new directory **verbatim** (same vLLM config — the whole point is measurement repetition, not reconfiguration), (4) re-measure via the harness at the Full profile (§4.1 — 33 min per candidate); the harness writes `candidates/rescreen_<PP>/measurement_trace.json` with a fresh `candidate_uuid`, `parent_candidate_uuid` set to the parent row's uuid, `profile: full`, (5) append one `status: rescreened` row per re-measurement to `results.tsv` with `measurement_count: 2`, populating `objective_mean` and `objective_ci_95` from the paired measurements, (6) create one commit per rescreen row — staging the new `candidates/rescreen_<PP>/` directory and the updated `results.tsv` — with **both** a `Candidate-UUID: <fresh_uuid>` trailer *and* a `Rescreen-Of-UUID: <parent_uuid>` trailer, signed-off as `lumoserve-auto-research-cli`, (7) flag rescreens whose absolute delta from the parent's `objective_value` exceeds `noise_floor` as `status: rescreened, notes: inconsistent_rescreen` (these candidates are removed from winner contention per §4.4). Called only by Python's outer loop (§11.3 phase c); never by a per-iteration codex exec.

**Rescreen directory convention — load-bearing detail.** The parent's original `candidates/<NNN>/` directory is **never overwritten or mutated**. Every rescreen attempt gets its own `candidates/rescreen_<PP>/` sibling directory. This means §9.3.AR.2 (every `measurement_trace.json` real-measured) and §9.3.AR.3 (every row maps to a commit) remain 1-to-1-replayable: the original main-loop trace stays on disk exactly as it was written, and the rescreen's trace lives under a path whose name encodes that it's a rescreen artifact. The parent directory and the rescreen directory are joined by the `parent_candidate_uuid` field in the rescreen row (§7.2) and the `Rescreen-Of-UUID` trailer on the rescreen commit (§7.3).

Example layout after a 3-candidate rescreen phase of a 5-iteration round:

```
candidates/
  baseline_a/{candidate.yaml, measurement_trace.json}
  baseline_b/{candidate.yaml, measurement_trace.json}
  001/{candidate.yaml, measurement_trace.json, agent_session.jsonl, agent_last_message.txt}
  002/{candidate.yaml, measurement_trace.json, agent_session.jsonl, agent_last_message.txt}
  003/{candidate.yaml, measurement_trace.json, agent_session.jsonl, agent_last_message.txt}
  004/{candidate.yaml, measurement_trace.json, agent_session.jsonl, agent_last_message.txt}
  005/{candidate.yaml, measurement_trace.json, agent_session.jsonl, agent_last_message.txt}
  rescreen_01/{candidate.yaml, measurement_trace.json}   # copy of e.g. 003's candidate + fresh measurement
  rescreen_02/{candidate.yaml, measurement_trace.json}   # copy of e.g. 005's candidate + fresh measurement
  rescreen_03/{candidate.yaml, measurement_trace.json}   # copy of e.g. 002's candidate + fresh measurement
```

Note the absence of `agent_session.jsonl` / `agent_last_message.txt` under `rescreen_<PP>/` directories — rescreen is Python-driven, no codex session runs. Same reason `baseline_{a,b}/` lacks those files. §9.3.AR.1 excludes these directories from its count via the `status` filter.

### 8.5 `validate-holdout` — one-shot holdout trace validation (§4.4)

```
lumoserve auto-research validate-holdout \
  --round-id <id> \
  --candidate-uuid <winner-uuid>
```

Effects: (1) load the winner's vllm_config via `/admin/load_tuned_config`, (2) replay the round's **holdout trace** (not the main trace used throughout the round) at the Full profile, (3) write `holdout_trace.json` under the round dir, (4) report `{pass: bool, reasons_failed: [..]}`. Called exactly once per round, by Python, after rescreen and before finalize. A failing holdout blocks finalize.

### 8.6 `finalize-round`

```
lumoserve auto-research finalize-round --round-id <id> [--dry-run]
```

**Production mode (no `--dry-run`).** Effects: (1) read `results.tsv`, **rank the rescreen rows by `objective_mean`**, pick the one with the highest mean (applying parent §5.4 tie-breakers over the rescreen-phase latency p95s); then **resolve that rescreen row's `parent_candidate_uuid` → the parent main-loop row → that row's `candidate.yaml` on disk**, which is the **winner configuration**. (The winner is the *configuration* that wins — the rescreen rows are additional measurements of that same configuration, not competing configurations.) (2) verify a `holdout_trace.json` exists and records `pass: true`; refuse if missing or failing, (3) build the tuned-config bundle via `make_tuned_config_bundle` (parent §5.9 schema) using the **parent's `candidate.yaml.vllm_config`** as the bundle's config source, (4) persist bundle at `output/tuned_configs/<family_id>/<weight_version_id>/<ts>_<hash>.yaml` with `round_provenance.dry_run: false`, (5) write `run_log.json`, `search_trace.json`, `measurement_trace_combined.json`, `rescreen_trace.json` under the round dir, (6) create the finalize commit (§7.4) with **`Winner-Candidate-UUID: <parent_uuid>`** trailer — the parent main-loop row's uuid, **not** the rescreen row's fresh uuid. (7) release the round lock.

Production mode refuses to finalize if: (a) fewer than one feasible candidate row in `results.tsv` after rescreen, (b) any measurement trace in the round is flagged `harness_fault` with no successor feasible run, (c) the measurement substrate's generator version is inconsistent across iterations (indicating the harness was swapped mid-round), (d) `holdout_trace.json` is missing or records `pass: false`, (e) no `rescreen_trace.json` exists (rescreen phase was skipped).

### 8.6a `finalize-round --dry-run` — CI smoke path only

**Dry-run mode (`--dry-run` flag passed).** Effects mirror production mode with three differences: (A) the rescreen and holdout preconditions are *relaxed* — the command succeeds even if `rescreen_trace.json` / `holdout_trace.json` are absent, (B) the winner is picked by single-shot `objective_value` over all feasible rows (since the rescreen phase was skipped), (C) the persisted bundle is tagged `round_provenance.dry_run: true` in `run_log.json` and in the bundle yaml. Production-mode `bootstrap-round` refuses to proceed if a `dry_run: true` bundle already exists at the target path (§11.1), so dry-run bundles cannot leak into production campaign bootstrap.

Only `auto-research run` (§8.8 — the CI backward-compat wrapper) calls `finalize-round --dry-run`. The production skill (§11) never passes the flag.

### 8.7 `status` — read-only round state (§11 uses this)

```
lumoserve auto-research status --round-id <id> [--json]
```

Effects: reads `round_spec.yaml` and `results.tsv`, emits a structured state blob with: `phase` (∈ `{bootstrapped, baseline, main_loop, rescreen, holdout, finalized, blocked}`), `iterations_total`, `feasible_count`, `best_objective_value`, `noise_floor`, `round_wall_clock_elapsed_s`, `round_wall_clock_remaining_s`, `blocker` (if any). Pure read — no writes to `results.tsv`, no git state changes, no CLI-level side effects. Used by the Python outer loop in §11.3 to make between-phase decisions and by §9.3.AR.22 verification. Also useful for a human operator to poll round progress without attaching to the terminal.

### 8.8 `run` — backward-compat wrapper (and non-agent CI use)

The existing `auto-research run` subcommand stays, rewired to: (1) call `bootstrap-round`, (2) call an internal non-agent "sweep" that proposes candidates from a fixed plan (the current `_candidate_plan` logic, but gated so it cannot run unless `LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT=1` is set), (3) call `measure` + `commit-candidate` per candidate, (4) call `finalize-round --dry-run`. The dry-run finalize is essential here — the `run` path does not execute rescreen/holdout, so production finalize would refuse. Every bundle produced by `run` therefore carries `round_provenance.dry_run: true`, and production skill preconditions (§11.1) reject it. This path is useful for CI smoke tests against a mock harness but MUST NOT be how production rounds are driven.

### 8.9 Backward-compat: what happens to the current `OfflineAutoResearchRunner`

The current `OfflineAutoResearchRunner` in `src/lumo_flywheel_serving/auto_research.py` stays as the implementation target of `auto-research run`, but `SyntheticMeasurementHarness` is renamed `SyntheticMeasurementFixture` and moved to `tests/fixtures/synthetic_measurement.py`. `auto_research.py` no longer imports it. The CLI path `auto-research run` now takes a `--harness` argument (default `real`, override `synthetic` only valid under the env-guard above). This shuts the 2026-04-23 failure mode cleanly: it is not *possible* to produce a bundle via the synthetic harness without opting in by env var, and the bundle produced under that opt-in is written with `generator: SyntheticMeasurementFixture` and `round_provenance.dry_run: true`, both of which the skill's precondition rejects.

---

## 9. Harness interface the CLI calls into

This section defines the interface `auto-research measure` binds to. Implementation is LLD-SB-06 (parent §8); the interface is this sub-spec's contract surface.

### 9.1 Module and class

`src/lumo_flywheel_serving/measurement_harness.py`:

```python
class RealMeasurementHarness:
    def __init__(
        self,
        *,
        workload_spec: WorkloadSpec,          # parsed serving_workload.yaml
        seed_trace_path: Path,                # per-request jsonl from the seed run
        slo: SLO,                             # L_ttft_ms, L_tpot_ms, L_turn_ms
        endpoint: str,                        # http://127.0.0.1:<port>/v1
        metrics_scrape_url: str,              # http://127.0.0.1:<port>/metrics
        admin_url: str,                       # for /admin/load_tuned_config
    ) -> None: ...

    def measure(
        self,
        candidate_vllm_config: dict,
        *,
        warmup_s: int,
        window_s: int,
        target_concurrency_sweep: list[int],  # e.g. [1, 2, 4, 8, 16, 32]
    ) -> MeasuredTrace: ...
```

### 9.2 `MeasuredTrace` shape (persisted as `measurement_trace.json`)

```json
{
  "generator": "RealMeasurementHarness v0.1.0",
  "round_id": "...",
  "iteration": "037",
  "candidate_label": "candidate-037",
  "candidate_uuid": "3f4a1e...",
  "parent_candidate_uuid": null,
  "profile": "screen",
  "candidate_vllm_config": { ... },
  "resolved": {
    "attention_backend": "flash-attn-4",
    "deltanet_kernel": "triton-chunked-delta-v2",
    "torch_compile_mode": "default"
  },
  "cache_isolation": {
    "cache_salt": "3f4a1e...",
    "prefix_cache_reset_at_bootstrap": true,
    "first_10_req_prefix_cache_hit_rate": 0.03,
    "last_10_req_prefix_cache_hit_rate": 0.71
  },
  "windows": {
    "warmup_s": 300,
    "measurement_s": 1500,
    "measurement_start_wallclock": "...",
    "measurement_end_wallclock": "..."
  },
  "per_request_latencies": [
    {"req_id": "...", "ttft_ms": 1820, "tpot_ms": 74, "turn_latency_ms": 19400,
     "thinking_tokens": 620, "response_tokens": 1100, "concurrency_when_dispatched": 4},
    ...
  ],
  "ttft_p95_ms": { "driver": 1870, "promql": 1903, "delta_pct": 1.73 },
  "tpot_p95_ms": { "driver": 78, "promql": 80, "delta_pct": 2.56 },
  "turn_latency_p95_ms": { "driver": 21400, "promql": 21650, "delta_pct": 1.17 },
  "sustained_concurrency": 8,
  "rollout_throughput": 12.1,
  "reasoning_content_purity": 1.0,
  "determinism_pass_rate": 0.9997,
  "no_oom_events": true,
  "feasible": true,
  "feasibility_failures": [],
  "vllm_metrics_snapshot_ref": "candidates/037/vllm_metrics.prom",
  "seed_trace_replay_ref": "candidates/037/replay.jsonl"
}
```

**Required audit fields and the downstream checks that consume them.** Each of these fields is not decorative — it backs a specific CLI enforcement or §12 verification item. The harness emits them; downstream consumers reject the trace if any are missing or malformed.

| Field | Required by | Check |
|---|---|---|
| `generator` starts with `RealMeasurementHarness` | `commit-candidate` (§8.3) | Refuses commit if the generator is `SyntheticMeasurementFixture` — this is the 2026-04-23-failure-mode guard. |
| `candidate_uuid` | `commit-candidate` (§8.3); §9.3.AR.3 | Must match the pending `results.tsv` row's uuid; trailer written on commit. Guards against a stale trace file surviving a retry. |
| `parent_candidate_uuid` | `rescreen` (§8.4); §9.3.AR.17 | Empty for main-loop rows; set to the main-loop parent's uuid for rescreen rows. Used for explicit lineage (P2-E fix). |
| `profile` ∈ `{screen, full}` | §4.1 / §8.4 | Rescreen rows must carry `profile: full`; main-loop rows at Sprint 0 carry `profile: screen`. |
| `cache_isolation.cache_salt == candidate_uuid` | §4.5 / §9.3.AR.21 | Per-candidate cache isolation (P2-7). |
| `cache_isolation.first_10_req_prefix_cache_hit_rate <= 0.10` | §4.5 / §9.3.AR.21 | Cold-cache start per candidate (P2-7). |
| `*_p95_ms.delta_pct <= 10` on all three latency dimensions | §4.3 / §9.3.AR.4 | PromQL ↔ driver cross-check. |
| `reasoning_content_purity == 1.0` | §4.2 / parent §5.4 constraint 8 | Thinking-content purity hard gate. |

A trace that lacks any required field is rejected by `commit-candidate` with `commit_refused: malformed_trace` (§8.3 refusal case (a) extended).

Per parent §5.6 — listed here only for the interface contract, not re-specified:

1. `/admin/load_tuned_config` with the candidate config, wait on `/health`.
2. Replay `seed_trace_path` entries through `/v1/responses` at a target concurrency sweep (the harness sweeps upward until SLO violation, which determines `sustained_concurrency`). Every request carries the family's `thinking_token_budget`; the admission layer rejects `chat_template_kwargs.enable_thinking` overrides per parent §4.1.
3. Record per-request TTFT, TPOT, end-to-end turn latency from the driver's own timings (not from /metrics).
4. Scrape `/metrics` at warmup-end and window-end; compute `histogram_quantile(0.95, …)` p95 for the three native histograms.
5. Sample ≥ 200 responses' `reasoning_content` vs `content` to confirm purity = 1.0.
6. Run a determinism probe (parent §5.7 rail 5) at warmup-end and window-end.
7. Run the KV-poisoning probe (parent §5.7 rail 7) at window-start and window-end.
8. Aggregate and return `MeasuredTrace`. On any rail failure (OOM, determinism, KV-poisoning), return with `feasible: false` and list the failure in `feasibility_failures`.

### 9.4 What the harness must refuse to do

- Emit a trace whose `generator` does not begin with `RealMeasurementHarness`. The fixture class intentionally emits `SyntheticMeasurementFixture v<n>` so `commit-candidate` can reject it (§6.3).
- Accept a candidate whose `vllm_config` carries keys not in the parent §5.3.2 L1 action space at Sprint 0 (L0/L2/L3 keys are refused until those sub-specs land).
- Write outside the round directory.
- Retry on OOM — OOM is a measurement outcome, not a harness fault.

---

## 10. Bundle emission — the hand-off to parent §5.9

`finalize-round` constructs the tuned-config bundle. The schema is unchanged from parent §5.9; this sub-spec adds three provenance fields and one new pin, all inside `layer_traces.l1` and a new top-level `round_provenance`:

```yaml
round_provenance:                          # NEW in this sub-spec
  round_id:                <id>
  round_branch:            autoresearch/...
  finalize_commit_sha:     <sha>
  agent_session_dir_ref:   output/auto_research/<id>/candidates/   # one agent_session.jsonl per iteration under candidates/<NNN>/
  agent_model_pin:         { model: gpt-5.4, reasoning_effort: high }
  sub_spec_version:        v0.1.7
  dry_run:                 false   # true only for finalize-round --dry-run bundles (§8.6a); production bundles must be false
layer_traces:
  l1:
    iterations:             <int>
    feasible_count:         <int>
    best_objective:         <float>
    trace_ref:              output/auto_research/<id>/search_trace.json
    reopened:               false
    results_tsv_ref:        output/auto_research/<id>/results.tsv   # NEW
```

The `round_provenance` block lets the §6.4 bundle-validity rule detect and reject bundles whose `agent_model_pin` or `sub_spec_version` do not match the current runtime's expectations on reload. This is an **additive** change to the parent's §6.4 rule — `round_provenance` is metadata (like `weight_version_id`), not a hard pin, at v0.1; if a future sub-spec revision deprecates `sub_spec_version: v0.1.0` bundles, that revision (not this one) is where the pin tightens.

---

## 11. Manager skill contract — Python outer loop

This section replaces `skills/auto-research-round-manager/SKILL.md` with a new skill that **is** the Phase B outer loop. The current (pre-Phase-A) skill drives a synthetic Python runner end-to-end; the new skill spawns one `codex exec` per iteration and owns all round-level control flow in Python.

The skill's job decomposes into: preconditions (§11.1), bootstrap (§11.2), the iteration loop (§11.3), finalize (§11.4), post-round live gate (§11.5), and reporting (§11.6). Everything Python-side is deterministic control flow; the only LLM work is inside each `codex exec` call.

### 11.1 Preconditions (run once, before bootstrap)

The skill refuses to start a round unless all of the following hold. Preconditions exist specifically to prevent the Phase A / Phase B confusion — if Phase A hasn't landed, the round cannot run.

- `python -c "from lumo_flywheel_serving.measurement_harness import RealMeasurementHarness"` succeeds. Blocker: `harness module missing` (Phase A item 1 not landed).
- `codex --version` prints the expected codex CLI version. Blocker: `codex cli missing or wrong version`.
- `git status` in repo root is clean. Blocker: `repo dirty`.
- All seven production CLI subcommands — `bootstrap-round`, `measure`, `commit-candidate`, `rescreen`, `validate-holdout`, `finalize-round`, `status` — are registered in `lumoserve auto-research --help`. For each, a lightweight existence probe `lumoserve auto-research <name> --help-only` returns **exit 0** and prints a machine-readable line `{"subcommand":"<name>","status":"registered"}` on stdout; a missing or unregistered subcommand returns **non-zero** with `{"subcommand":"<name>","status":"missing"}`. The precondition fails with `cli subcommand missing: <name>` if any probe returns non-zero. This semantics matches §9.3.AR.15, which also relies on `--help-only` returning 0 for existing commands.
- No existing `auto-research` bundle at the target path under `output/tuned_configs/<family_id>/<weight_version_id>/` has `round_provenance.dry_run: true`. A dry-run bundle left over from a CI smoke run of `auto-research run` (§8.8) is not a valid production artifact, and finalize-round production mode refuses to overwrite it anyway. Blocker: `dry_run_bundle_exists` — operator must delete it before a production round.
- The target workload yaml exists and has a non-empty `seed_trace_ref` pointing at an existing jsonl file. If `seed_trace_ref` is empty, the skill auto-runs `scripts/capture_seed_workload.py` (Phase A item 2) unless `--no-seed-capture` is set. Blocker: `seed trace missing`.
- `LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT` is unset. Blocker: `non-agent mode enabled — this is CI-only, refuse production round`.
- A dry-run integration test against `SyntheticMeasurementFixture` exits 0 within 5 minutes (tests the CLI wiring end-to-end without doing real measurement). Blocker: `dry-run wiring check failed`.

Any blocker → skill emits `ROUND_BLOCKED` with the specific reason; no filesystem changes are made.

### 11.2 Bootstrap (runs once)

```python
out = sh("lumoserve auto-research bootstrap-round "
        "--model-id qwen3.5-27b "
        "--family-id proposal-ranking-manager-judgment "
        "--sprint sprint-0 "
        "--workload-file benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml "
        "--weight-version-id $(python -c 'from lumo_flywheel_serving.tuned_config import default_weight_version_id; ...') "
        "--round-root output/auto_research")
round_id = json.loads(out)["round_id"]
round_dir = f"output/auto_research/{round_id}"
```

`bootstrap-round` effects (per §8.1): creates `<round_dir>/` with `round_spec.yaml`, `impl_brief.md` (for reference — IMPL is already done by this point), `iteration_brief.md` (template, variables not yet substituted), `candidates/` (empty), `results.tsv` (header-only), `codex-home/.codex/config.toml` pinning `gpt-5.4 high`. Opens the round git branch from current `main` HEAD. Captures parent SHA into `round_spec.yaml`. Acquires the round lock.

### 11.3 Iteration loop (the core of the skill)

Plain-Python for-loop with four phases: (a) **double-baseline** — establishes noise floor (§4.4), (b) **main search loop** — the iteration_cap candidates proposed by codex exec, (c) **rescreen phase** — top-K re-measured at Full profile, (d) **holdout validation** — winner validated against the held-out workload trace. Python owns every stop criterion in these phases; no codex exec sees the stop logic.

**Wall-clock enforcement — end-to-end.** `round_started` is initialized **before phase (a)** and the cap is checked at **every phase boundary**, not only inside the main loop. The phase budgets below are preflighted: each phase refuses to enter if the remaining wall-clock is less than the expected phase duration. A round that exceeds `round_wall_clock_s` at any phase boundary returns `ROUND_BLOCKED: wall_clock_cap` and no bundle is produced (consistent with §4.4 — a round without rescreen + holdout cannot emit a valid production bundle).

Expected phase durations (§4.1 Sprint 0 defaults, used by the preflight check):

| Phase | Expected wall-clock | Action if cap insufficient |
|---|---|---|
| (a) double-baseline | 2 × Screen profile (~30 min) | `ROUND_BLOCKED: wall_clock_cap_preflight_baseline` |
| (b) main loop | up to iteration_cap × Screen profile (~3 h) | early-exit main loop at the iteration where the cap would be exceeded; proceed to (c) if ≥1 feasible exists |
| (c) rescreen | top_k × Full profile (~1.6 h) | `ROUND_BLOCKED: wall_clock_cap_preflight_rescreen` |
| (d) holdout | 1 × Full profile (~33 min) | `ROUND_BLOCKED: wall_clock_cap_preflight_holdout` |

With ~6 h of active harness time and ~2 h of bootstrap/teardown/live-gate headroom, the 8 h cap fits all four phases in the normal case. If phase (b) runs long (a single candidate takes 60 min instead of 15 min of real work due to vLLM misbehavior), the main loop exits early with fewer candidates but phases (c)/(d) still run on whatever feasible rows exist.

```python
def run_round(round_id: str, round_spec: dict) -> RoundResult:
    round_started = time.monotonic()
    cap_s = round_spec["round_wall_clock_s"]  # 28800 at v0.1 Sprint 0

    def remaining_s() -> float:
        return cap_s - (time.monotonic() - round_started)

    def must_have_for(phase: str, required_s: float) -> RoundResult | None:
        if remaining_s() < required_s:
            return RoundResult(stopping_reason=f"wall_clock_cap_preflight_{phase}",
                               remaining_s=remaining_s())
        return None

    # === Phase (a): double-baseline → noise floor (§4.4) ===
    if (r := must_have_for("baseline", 2 * round_spec["screen_profile_s"])): return r
    # Both baselines use default-config + different candidate_uuid.
    # Each measure → commit-candidate pair is explicitly executed so the baseline
    # rows satisfy the §7 Candidate-UUID-trailer contract — an uncommitted row
    # would fail §9.3.AR.3 at verify time.
    for sfx in ("a", "b"):
        sh(f"lumoserve auto-research measure --round-id {round_id} "
           f"--candidate {round_dir}/candidates/baseline_{sfx}/candidate.yaml")
        sh(f"lumoserve auto-research commit-candidate --round-id {round_id} "
           f"--iteration baseline_{sfx} --status baseline "
           f"--notes 'default-config baseline replay {sfx} for noise floor'")
    m1, m2 = read_baseline_rows_from_results_tsv(f"{round_dir}/results.tsv")
    noise_floor = 2.0 * abs(m1.objective_value - m2.objective_value)
    set_round_spec_field(round_spec, "noise_floor", noise_floor)
    best_so_far = max(m1.objective_value, m2.objective_value)

    # === Phase (b): main search loop — iterations 001..(iteration_cap-1) ===
    consecutive_oom = 0
    consecutive_determinism_fail = 0
    best_history = []

    # Reserve budget for phases (c) and (d) so the main loop cannot consume it all.
    reserved_for_cd = (round_spec["rescreen_top_k"] + 1) * round_spec["full_profile_s"]

    for iteration in range(1, round_spec["iteration_cap"]):  # 001..011 at v0.1 Sprint 0
        # Exit main loop early if advancing would consume the budget reserved for (c)+(d).
        if remaining_s() < reserved_for_cd + round_spec["screen_profile_s"]:
            break  # leave time for rescreen + holdout; do not return — go to phase (c)

        iter_dir = f"{round_dir}/candidates/{iteration:03d}"
        os.makedirs(iter_dir, exist_ok=True)

        # Substitute {{...}} placeholders in-memory — codex-cli has no --var
        # (see §2.3). The on-disk iteration_brief.md keeps its placeholders
        # for human readability and §12 verification.
        iteration_prompt = substitute_placeholders(
            template=read_file(f"{round_dir}/iteration_brief.md"),
            values={"round_id": round_id,
                    "iteration": f"{iteration:03d}",
                    "round_dir": str(round_dir),
                    "iteration_dir": iter_dir,
                    ...},
        )

        env = {**os.environ, "CODEX_HOME": f"{round_dir}/codex-home"}

        # === the ONE codex call per iteration (see §2.3 for flag rationale) ===
        with open(f"{iter_dir}/agent_session.jsonl", "wb") as transcript:
            codex_result = subprocess.run(
                [
                    "codex",
                    "-c", 'model="gpt-5.4"',
                    "-c", 'model_reasoning_effort="high"',
                    "exec",
                    "--cd", str(round_dir),
                    "--json",
                    "--output-last-message", f"{iter_dir}/agent_last_message.txt",
                    "--skip-git-repo-check",
                    "-",                    # prompt from stdin
                ],
                input=iteration_prompt.encode(),
                stdout=transcript,
                stderr=subprocess.PIPE,
                env=env,
                timeout=round_spec["per_iteration_codex_wall_clock_s"],
            )

        # Codex exits 2 when the iteration codex itself writes BLOCKED.md
        # (R8 retry-exhaustion). Path is iteration-scoped per §6.6.
        iter_blocked = Path(f"{iter_dir}/BLOCKED.md")
        if iter_blocked.exists():
            return RoundResult(stopping_reason="iteration_blocked", iteration=iteration,
                               blocker=iter_blocked.read_text())
        if codex_result.returncode != 0:
            # Python retries up to max_iteration_retries; on exhaustion, ROUND_BLOCKED
            ...

        # === read the new row, update running state ===
        row = read_last_row("results.tsv")
        assert row["iteration"] == f"{iteration:03d}"

        if row["status"] == "crash" and row["notes"].startswith("oom"):
            consecutive_oom += 1
            if consecutive_oom >= 3:
                return RoundResult(stopping_reason="hard_infeasibility_oom", iteration=iteration)
        else:
            consecutive_oom = 0

        if row["feasible"] and float(row["determinism_pass_rate"]) < 0.999:
            consecutive_determinism_fail += 1
            if consecutive_determinism_fail >= 3:
                return RoundResult(stopping_reason="hard_infeasibility_determinism", iteration=iteration)
        else:
            consecutive_determinism_fail = 0

        if row["feasible"]:
            # Only advance best_so_far if the improvement clears the noise floor (§4.4).
            obj = float(row["objective_value"])
            if obj > best_so_far + noise_floor:
                best_so_far = obj
                best_history.append((iteration, obj))
            else:
                # Feasible but not a real improvement — record without advancing.
                best_history.append((iteration, best_so_far))

            # K=4 diminishing returns — tightened from K=8 per v0.1 Sprint 0 budget (§4.1).
            if len(best_history) >= round_spec["diminishing_returns_window_k"]:
                window = best_history[-round_spec["diminishing_returns_window_k"]:]
                start, end = window[0][1], window[-1][1]
                if start > 0 and (end - start) / start < 0.02:
                    break  # exit main loop; still run rescreen + holdout

    # === Phase (c): rescreen top-K at Full profile (§4.4) ===
    top_k = round_spec["rescreen_top_k"]
    if (r := must_have_for("rescreen", top_k * round_spec["full_profile_s"])): return r
    sh(f"lumoserve auto-research rescreen --round-id {round_id} "
       f"--top-k {top_k} --profile full")
    # `rescreen` writes its own `status: rescreened` rows *and* commits them with
    # both `Candidate-UUID` and `Rescreen-Of-UUID` trailers (§8.4); no separate
    # commit-candidate call is needed here.

    # Pick winner by objective_mean over rescreened rows, not single-shot objective_value.
    winner = pick_winner_by_mean(results_tsv=f"{round_dir}/results.tsv")
    if winner is None:
        return RoundResult(stopping_reason="no_feasible_rescreen_winner")

    # === Phase (d): holdout validation (§4.4) ===
    if (r := must_have_for("holdout", round_spec["full_profile_s"])): return r
    holdout = sh(f"lumoserve auto-research validate-holdout --round-id {round_id} "
                 f"--candidate-uuid {winner.candidate_uuid}")
    if not holdout.pass_:
        return RoundResult(stopping_reason="holdout_rejected",
                           bundle_rejected_reason=holdout.reasons_failed)

    # === Phase (e): finalize-round creates the bundle and FINALIZE commit ===
    # Production mode (no --dry-run) — refuses to finalize without rescreen+holdout (§8.6).
    # This is the single call to finalize-round in the round lifecycle; §11.4 is a
    # prose cross-reference, not an additional invocation.
    sh(f"lumoserve auto-research finalize-round --round-id {round_id}")
    return RoundResult(stopping_reason="ok", winner=winner,
                       round_wall_clock_s=int(time.monotonic() - round_started))
```

**Key properties of this loop:**

- **Only Python decides stop.** Stop criteria never appear in `iteration_brief.md`. Codex cannot stop the round; codex can only finish its iteration.
- **Fresh context per iteration.** `--json` + per-iteration stdout redirection captures each codex session into its own `candidates/<NNN>/agent_session.jsonl`; no codex invocation inherits any prior transcript.
- **Atomic iteration boundaries.** Either codex returned 0 and a new row exists in `results.tsv`, or it didn't and the loop retries/fails. There is no partial-iteration state.
- **Retries are bounded.** A codex exec that exits non-zero is retried by Python up to `max_iteration_retries` (default 2). Exhaustion → `ROUND_BLOCKED`.

### 11.4 Finalize — single invocation inside `run_round`, not a separate step

`finalize-round` is called **exactly once per round**, inside `run_round()` phase (e) above — not here, and not anywhere else. This section is a prose cross-reference so readers can find the finalize contract by §11 heading; it does **not** introduce a second invocation.

What happens at phase (e) under the normal path: `finalize-round` writes the bundle + `run_log.json` + `search_trace.json` + `rescreen_trace.json` + the FINALIZE commit (§7.4 format). The wrapping skill (the thing that calls `run_round`) then proceeds to the live family gate (§11.5) and reporting (§11.7).

What happens at phase (e) under degenerate paths:

- **Zero feasible rows.** If the main loop + rescreen phase together produce zero feasible rows, phase (e) is not reached — `run_round` returns earlier with `stopping_reason = no_feasible_rescreen_winner` or the equivalent phase-specific code. `finalize-round` is not called. The skill reports `ROUND_INFEASIBLE`, consistent with parent §5.7 rail 3.
- **Holdout rejection.** If phase (d) holdout fails, `run_round` returns with `stopping_reason = holdout_rejected`. `finalize-round` is not called. The skill reports `ROUND_BUNDLE_REJECTED`.
- **Wall-clock cap in phase (c)/(d)/(e).** The preflight checks in §11.3 prevent entering a phase whose required wall-clock is unavailable; in all such cases `run_round` returns `wall_clock_cap_preflight_<phase>` and `finalize-round` is never called. No bundle is emitted. The skill reports `ROUND_BLOCKED`.

Explicit anti-pattern forbidden: the skill **must not** call `finalize-round` from outside `run_round`. A second invocation either produces a duplicate FINALIZE commit (if the first already ran) or runs against an already-finalized round (whose lock is released, whose `run_log.json` is written) and corrupts the round state. The lifecycle contract is: one `run_round` call per round, one `finalize-round` call inside it.

### 11.5 Post-round live family gate

After a successful finalize, run the live family gate (`scripts/run_live_proposal_family.py`) against a freshly-booted vLLM instance loading the just-written bundle. The round is `ROUND_PASSED` only if the gate returns `pass: true, score >= 80, returncode == 0`. If the bundle produces but the live gate fails, the outcome is `ROUND_BUNDLE_REJECTED` — the bundle is not promoted and the serving stack falls back to the default-config baseline.

### 11.6 Watchdog — simplified under the per-iteration spawn model

The previous spec's watchdog had to guess whether a long-running codex agent had stalled. Under the per-iteration spawn model the watchdog collapses to three checks, all Python-side:

- **Per-iteration wall-clock.** Enforced by `subprocess.run(..., timeout=...)`. A codex exec exceeding `per_iteration_codex_wall_clock_s` is killed, the iteration is retried or marked blocked. No separate silence timeout needed — codex either exits or gets killed by the timeout.
- **Between-iteration write scope.** After each iteration, `git diff main..HEAD --name-only` on the round branch. Any path outside the allow-list (§6.2) → `ROUND_BLOCKED: out_of_scope_write`.
- **Between-iteration `Signed-off-by` trailer.** `git log --format='%(trailers:key=Signed-off-by,valueonly)' main..HEAD` on the round branch. Every new commit must emit exactly one `lumoserve-auto-research-cli <auto-research@lumo-flywheel>` line; a commit with no matching trailer, or a commit whose trailer value is anything else, triggers `ROUND_BLOCKED: missing_or_mismatched_trailer`. This is a **structured-trailer** check, not a cryptographic signature check. `%GS` (the GPG signer name) is explicitly *not* used — codex-cli does not GPG-sign commits by default, and requiring GPG signing at v0.1 is out of scope. The `Signed-off-by` naming follows DCO convention (a machine-parseable attribution trailer); §13.9 tracks whether to upgrade to real commit signing in v0.2.
- **Between-iteration `Candidate-UUID` trailer presence.** Every non-FINALIZE commit must carry exactly one `Candidate-UUID: <uuid>` trailer whose value matches the `candidate_uuid` of the `results.tsv` row staged in that commit. Mismatch or absence → `ROUND_BLOCKED: trailer_mismatch`. This closes the P0-2 loop — the row ↔ commit linkage is verified at commit time *and* re-verified by the watchdog between iterations.

No "agent silence" timeout is needed at v0.1 — a stalled codex exec is caught by the per-iteration wall-clock, and the round's long silences (during the 30-minute measurement window) are expected and not a stall.

### 11.7 Report shape (what the skill emits back)

```json
{
  "round_id": "...",
  "round_branch": "autoresearch/qwen3.5-27b/proposal-ranking-manager-judgment/sprint-0/...",
  "outcome": "ROUND_PASSED | ROUND_INFEASIBLE | ROUND_BLOCKED | ROUND_BUNDLE_REJECTED",
  "bundle_path": "output/tuned_configs/.../...yaml | null",
  "live_gate": { "pass": true, "score": 82, "returncode": 0 } | null,
  "finalize_commit_sha": "... | null",
  "iterations_total": 14,
  "feasible_count": 6,
  "rescreened_count": 3,
  "holdout_validation": "pass | fail | not_reached",
  "stopping_reason": "<see enum below>",
  "blocker": "... | null",
  "codex_invocations": 14,
  "total_codex_wall_clock_s": 312,
  "total_harness_wall_clock_s": 28800
}
```

**`stopping_reason` enum.** Every terminal state from §11.3 maps to a specific `stopping_reason` value. The enum is the source of truth — callers must not expect values outside it. Each reason pairs with a specific `outcome` so a caller can distinguish a legitimate no-finalize path from a generic failure.

| `stopping_reason` | Emitting site (in §11.3 or §11.5) | `outcome` | `bundle_path` |
|---|---|---|---|
| `ok` | phase (e) finalize-round succeeded, phase (f) live gate passed | `ROUND_PASSED` | non-null |
| `iteration_cap` | main loop reached `iteration_cap` | continues to (c)/(d)/(e); final outcome determined by those phases | context-dependent |
| `diminishing_returns` | main loop broke on K-window (§4.1) | continues to (c)/(d)/(e) | context-dependent |
| `hard_infeasibility_oom` | main loop aborted after 3 OOMs in a row | `ROUND_BLOCKED` | null |
| `hard_infeasibility_determinism` | main loop aborted after 3 determinism failures | `ROUND_BLOCKED` | null |
| `iteration_blocked` | any iteration's codex wrote `<iter_dir>/BLOCKED.md` (R8 retry-exhaustion) | `ROUND_BLOCKED` | null |
| `cli_retry_exhausted` | codex exec kept returning non-zero past `max_iteration_retries` | `ROUND_BLOCKED` | null |
| `harness_fault` | any row with `status: harness_fault` (PromQL cross-check out of bounds) | `ROUND_BLOCKED` | null |
| `wall_clock_cap_preflight_baseline` | phase (a) preflight failed | `ROUND_BLOCKED` | null |
| `wall_clock_cap_preflight_rescreen` | phase (c) preflight failed | `ROUND_BLOCKED` | null — no rescreen means no valid production finalize |
| `wall_clock_cap_preflight_holdout` | phase (d) preflight failed | `ROUND_BLOCKED` | null — no holdout means finalize refuses |
| `no_feasible_rescreen_winner` | phase (c) ran but produced zero eligible winners (all `inconsistent_rescreen` or all infeasible) | `ROUND_INFEASIBLE` | null |
| `holdout_rejected` | phase (d) validated the winner and it failed the holdout | `ROUND_BUNDLE_REJECTED` | null |
| `live_gate_failed` | finalize succeeded, post-round live family gate (§11.5) returned non-pass | `ROUND_BUNDLE_REJECTED` | non-null but bundle not promoted |
| `wall_clock_cap` | legacy umbrella — not emitted at v0.1; reserved for future non-preflight budget misses | — | — |

`ROUND_BUNDLE_REJECTED` is distinct from `ROUND_BLOCKED` specifically because in the `holdout_rejected` and `live_gate_failed` cases the round *did* produce a bundle on disk (at finalize time), but that bundle is not promoted to the campaign-bootstrap-visible path. The caller can inspect it for post-mortem analysis; it just doesn't feed production.

### 11.8 Explicit non-goals for the skill

- The skill does **not** propose candidates. That is codex's per-iteration job.
- The skill does **not** run the measurement harness. That is `auto-research measure`'s job.
- The skill does **not** merge the round branch. That is a human job, gated on parent §9.3.
- The skill does **not** re-run on BLOCKED outcomes. A blocked round is a finding, not a retry target.
- The skill does **not** hold a long-running codex session. Every codex invocation is per-iteration.

---

## 12. Verification — extensions to parent §9.3

These are *additive* to the parent HLD's §9 checklist; each one has a new item number in the `9.3.AR.*` namespace to keep the parent numbering stable. Each is pass/fail, artifact-backed, and verifier-checkable without subjective judgment — matching the parent §9 contract.

- **9.3.AR.1 Per-iteration transcripts captured (codex-proposed rows only).** Every iteration directory `candidates/<NNN>/` for a **codex-proposed main-loop iteration** contains a non-empty `agent_session.jsonl` (one per codex exec invocation). The count of `agent_session.jsonl` files equals the count of `results.tsv` rows whose `status` is one of `{keep, discard, crash, harness_fault}` — these are the main-loop rows produced by a codex exec. Rows with `status: baseline` (Phase (a) double-baseline — Python drives `measure` + `commit-candidate` directly, no codex session) and rows with `status: rescreened` (Phase (c) — the `rescreen` CLI is called by Python, no codex session) are **excluded** from the count. Artifact: file-count comparison with the status filter applied.
- **9.3.AR.2 Every row is real-measured.** Every `measurement_trace.json` in the round directory has `generator` starting with `RealMeasurementHarness`. No trace has `synthetic: true`. Artifact: `grep -l synthetic output/auto_research/<round_id>/candidates/*/measurement_trace.json` returns empty.
- **9.3.AR.3 Every row maps to a commit via trailer.** For every non-header row in `results.tsv`, `git log --format='%H %(trailers:key=Candidate-UUID,valueonly)' main..HEAD` on the round branch yields exactly one commit whose trailer equals the row's `candidate_uuid`, and that commit's `Signed-off-by` trailer equals `lumoserve-auto-research-cli <auto-research@lumo-flywheel>`. Artifact: `git log` extraction + `results.tsv` join-by-uuid — join must be 1-to-1 with no unmatched rows in either direction. This replaces the prior `commit_sha` column (which was impossible to populate atomically — P0-2).
- **9.3.AR.4 PromQL cross-check held.** For every feasible row, each of `ttft_p95_ms.delta_pct`, `tpot_p95_ms.delta_pct`, `turn_latency_p95_ms.delta_pct` is ≤ 10. Artifact: a `jq` extraction over all traces showing every delta_pct in range.
- **9.3.AR.5 Brief consistency.** `impl_brief.md` / `iteration_brief.md` / `round_spec.yaml` on disk at the round directory all agree on round identity (model_id, family_id, sprint, weight_version_id, SLO ceilings, iteration_cap). Artifact: the consistency-check run by `bootstrap-round` §8.1, re-executed at verify time against the on-disk files — must pass.
- **9.3.AR.6 Winner is Pareto-consistent.** The winning iteration's row in `results.tsv` ties or beats every other feasible row per parent §5.4 tie-breakers. Artifact: a synthetic re-run of the tie-breaker over `results.tsv` returns the same iteration as the `finalize-round` commit's `winner_iteration`.
- **9.3.AR.7 Precondition check fires.** A synthetic test that removes `measurement_harness.py`, runs the skill, and confirms the skill refuses to launch with `BLOCKED: harness module missing`. Artifact: test log.
- **9.3.AR.8 No codex exec tried to finalize.** For every iteration's `agent_session.jsonl`, no invocation of `lumoserve auto-research finalize-round` appears. (That is Python's job — R7.) Artifact: transcript grep.
- **9.3.AR.9 No codex exec emitted round-level stop decisions.** For every iteration's `agent_session.jsonl`, there is no message matching `/should (we|I|the round) (continue|stop)|call it a day|stopping the round/i`. Codex is allowed to decide *this* iteration is done; it is not allowed to decide the *round* is done. Artifact: transcript grep.
- **9.3.AR.10 Round branch not merged.** `git branch -r --merged main` does not include the round branch at the time the skill emits its report. Artifact: git command output.
- **9.3.AR.11 Per-iteration wall-clock enforced.** For every iteration, the per-iteration codex exec wall-clock (measured as the delta between spawn time and exit time) is ≤ `per_iteration_codex_wall_clock_s`. A synthetic test that stalls a codex stub confirms the Python `subprocess.run(..., timeout=...)` kills it within 5 s of the deadline. Artifact: timing log + test log.
- **9.3.AR.12 Out-of-scope write kill path works.** A synthetic test that injects a disallowed file write into a codex exec (e.g., writes `src/foo.py`) and confirms Python's between-iteration watchdog terminates the round with `ROUND_BLOCKED: out_of_scope_write`. Artifact: test log.
- **9.3.AR.13 Synthetic-fixture rejection.** A trace with `generator: SyntheticMeasurementFixture` is rejected by `commit-candidate` with structured error `commit_refused: non_real_generator`. Artifact: test log.
- **9.3.AR.14 Fresh context per iteration.** `agent_session.jsonl` for iteration `<NNN>` does not contain any message from iteration `<NNN-1>` or earlier. (Each codex exec is freshly spawned per §2.3.) Artifact: cross-transcript check.
- **9.3.AR.15 Phase A substrate is present on `main`.** At round launch time, `main` contains all 8 Phase A deliverables from §5.1: measurement_harness.py, capture_seed_workload.py, the **seven production CLI subcommands** (`bootstrap-round`, `measure`, `commit-candidate`, `rescreen`, `validate-holdout`, `finalize-round`, `status`) plus the env-gated `run`, the rewritten skill, the test fixture under `tests/fixtures/`, passing unit + integration tests, the pre-flight checks, and the two brief templates (`impl_brief.md`, `iteration_brief.md`). Artifact: a one-shot `verify_phase_a.sh` that probes each CLI subcommand via `lumoserve auto-research <name> --help-only` returning zero, checks module imports, runs the unit + integration test suites, and grep-checks the brief templates.
- **9.3.AR.16 Double-baseline noise floor computed.** `round_spec.yaml` has `noise_floor` populated before the main search loop begins, and its value equals `2 × |m1.objective_value - m2.objective_value|` for the two baseline rows in `results.tsv` with `status: baseline`. Artifact: values read from `round_spec.yaml` and from the two baseline rows agree.
- **9.3.AR.17 Rescreen phase executed with explicit lineage.** After the main search loop exits, `results.tsv` contains exactly `min(K_rescreen, feasible_candidate_count)` additional rows with `status: rescreened`, each carrying `measurement_count: 2`, `objective_mean`, `objective_ci_95`, and a **non-empty `parent_candidate_uuid`** matching a main-loop feasible row's `candidate_uuid`. The corresponding commit on the round branch carries a `Rescreen-Of-UUID: <parent_uuid>` trailer equal to the row's `parent_candidate_uuid`. Artifact: row-level join over `results.tsv` by `parent_candidate_uuid` + `git log --format='%(trailers:key=Rescreen-Of-UUID,valueonly)'` cross-check. The join must be 1-to-1 and stable — no main-loop row can be rescreened twice, and no rescreen row can lack a parent.
- **9.3.AR.18 Winner picked by rescreen mean.** The finalize commit's `Winner-Candidate-UUID` trailer matches a main-loop row (not a rescreen row) whose rescreen mean `objective_mean` is highest among rescreened candidates, tie-broken by §4.2's latency p95s applied to the rescreen measurement. Resolution: take the FINALIZE trailer's uuid `U`, find the rescreen row whose `parent_candidate_uuid = U`, confirm its `objective_mean` is the max over all rescreen rows (with `notes != inconsistent_rescreen`). The winner is **not** the single-shot main-loop best — if the main-loop top-1 has a rescreen measurement that falls outside the noise floor, it is eliminated from contention via the `inconsistent_rescreen` flag. Artifact: a synthetic re-run of the tie-breaker over the rescreen rows returns the same parent uuid.
- **9.3.AR.19 Holdout validation ran and passed.** `holdout_trace.json` exists under the round directory, records `pass: true`, and references the winner's `candidate_uuid`. The holdout trace is a different file-hash from the main `seed_trace_ref`. Artifact: existence + hash comparison.
- **9.3.AR.20 Holdout rejection path works.** Synthetic test: an injected failing holdout (e.g., force a TTFT p95 regression on the holdout replay) causes `finalize-round` to refuse with `refusal_reason: holdout_failed` and the skill to report `ROUND_BUNDLE_REJECTED`. No bundle is written. Artifact: test log.
- **9.3.AR.21 Per-candidate cache isolation.** Every `measurement_trace.json` in the round records `cache_salt` equal to that candidate's `candidate_uuid`, and the first 10 requests' prefix-cache hit rate is ≤ 10%. Artifact: `jq` extraction over all traces.
- **9.3.AR.22 `status` subcommand reports consistent state.** At finalize time, `lumoserve auto-research status --round-id <id> --json` emits a state blob whose `phase` is `finalized`, whose `iterations_total` equals the count of non-header rows in `results.tsv`, whose `feasible_count` equals the count of rows with `feasible: true`, whose `rescreened_count` equals the count of `status: rescreened` rows, and whose `round_wall_clock_elapsed_s` is ≤ `round_spec.yaml.round_wall_clock_s`. Calling `status` during each phase of §11.3 returns the matching phase name; a call made while no active round matches `--round-id` returns `phase: missing` with a non-zero exit code. Artifact: `status --json` output captured at phase boundaries + cross-check against `results.tsv`.

---

## 13. Open questions

### 13.1 Uncertainty-aware candidate selection (llm-bandit v0.2)

The v0.1 L1 inner optimizer is Optuna TPE with codex proposing explicit candidates. The Spotify `llm-bandit` reference suggests a better pattern at higher iteration counts: codex proposes K candidates, the CLI runs each for a short screening window, and final K/2 are picked for the full window by Thompson Sampling over the posterior belief on per-candidate objective. This is a v0.2 extension — the v0.1 budget is too small to amortize the screening step meaningfully. The v0.2 spec would add a new subcommand `auto-research screen` and extend `results.tsv` with a `screening_window_s` column.

### 13.2 Parallel candidate measurement

Currently one candidate at a time. On a multi-GPU node (DGX Spark has a single GB10 chip, so this is hypothetical for v0.1 but relevant for v0.2 hardware) we could measure K candidates in parallel against K vLLM instances. The per-round wall-clock budget would shrink by ≈K; the agent's proposal rhythm would need to change. Open: whether the agent proposes a batch of K at once, or the CLI buffers K sequential proposals and dispatches them in parallel. Closely coupled to §13.3 — parallel measurement is the feature that most motivates the manager/worker codex hierarchy.

### 13.3 Manager/worker codex subagent hierarchy — v0.2 promotion criteria

v0.1 is **single-agent by design** (§2 "Single agent, not a manager/worker codex hierarchy"). v0.2 may promote to a two-tier codex hierarchy — a *manager codex agent* that holds the search-trace reasoning and orchestrates, plus *per-candidate worker codex subagents* that each run one `candidate.yaml → measure → commit` cycle in an isolated context window. The open question is **when** this promotion becomes net-positive, not whether it's ever worthwhile. Three concrete promotion criteria, any one of which on its own is sufficient:

  (a) **L0c kernel mutation lands** (parent §5.3.1.c). Each mutation attempt is a 5–20-minute compile + parity-gate cycle with genuinely diverse failure modes (compile failure vs parity-gate divergence vs latency regression vs OOM). Worker isolation per mutation attempt is the right shape: each worker lives and dies with one `.patch`, the manager curates the mutation search direction without its context getting polluted by the specifics of mutations that failed for unrelated reasons. This is the primary promotion trigger in the parent HLD's sprint plan (parent §7.3 Sprint 2+).

  (b) **Parallel candidate measurement becomes viable** (§13.2). If Sprint 2+ hardware exposes multiple GPUs or multiple vLLM instances can be colocated on one GPU at reduced `gpu_memory_utilization`, we need K independent agent processes to measure K candidates concurrently. Manager/worker is the natural shape — the manager proposes K candidates, spawns K workers, aggregates results. A single-agent loop cannot cleanly parallelize.

  (c) **Rounds span multiple layers with single-re-open semantics** (parent §5.5). Once a round traverses L0 → L1 → L2 → L3 with the outer LLM proposer making advance-vs-re-open decisions, the manager-holds-traversal-state / workers-run-per-layer-sub-optimizations split gives a cleaner separation than a monolithic single agent. The manager's job — "did we hit a cross-layer interaction? should we re-open?" — is qualitatively different from the worker's job — "propose the next candidate in this layer's inner loop" — and collapsing both into one agent context trades reliability for simplicity.

If none of (a)/(b)/(c) holds — which is exactly the v0.1 situation (L1-only, single-GPU, single-layer) — the hierarchy is strictly overhead. The v0.2 sub-spec revision should spell out: the `codex-subagent` invocation surface the manager uses, the worker's reduced briefing (an extension of `iteration_brief.md`, scoped to one candidate, not the round), the manager's protocol for aggregating worker results and deciding the next batch, and the verification items that check worker isolation (no worker reads another worker's transcript, no manager hallucinates worker results). None of that is needed at v0.1.

### 13.4 Codex model pin stability

If `gpt-5.4` is deprecated mid-round, the codex process may fail to complete. The skill's precondition (§11.1) catches this before bootstrap, but in-round deprecation is a real risk for long-running rounds (8 hours × model deprecation window). Open: whether to fail-fast on mid-round model unavailability, or to pin to a frozen snapshot of `gpt-5.4` via `codex --model-snapshot`.

### 13.5 Round resumption after crash

If the Python outer loop crashes mid-round, the round directory exists but is incomplete. The CLI is currently not designed to resume — `bootstrap-round` refuses if the round directory exists. Open: whether to add `resume-round` that re-binds to the existing directory, re-derives the running state from `results.tsv`, and continues spawning per-iteration codex exec invocations from where the crash left off. The git ledger makes this *possible* — every committed iteration is durable, and an incomplete iteration can be discarded and retried cleanly. The per-iteration spawn pattern actually makes resume easier than it was in the v0.1.1 draft (which had a single long-running codex session whose mid-flight state was opaque). The question is whether the safety/determinism story is worth the implementation complexity at v0.2.

### 13.6 L0c mutation — when this sub-spec extends

At parent §5.3.1.c (L0c kernel mutation), candidates include a code `.patch` diff, not just a config. The three-file agent interface adapts cleanly: the agent edits a `mutation.patch` file under the candidate directory, and the CLI's `measure` first applies the patch, rebuilds, runs the parity gate (§5.7 rail 9), and only then proceeds to the latency measurement. Scope note: this sub-spec's v0.1 is L1-only; the L0c extension is a future v0.2 and is also the primary trigger for §13.3's manager/worker hierarchy promotion.

### 13.7 Second `(model, family)` pair

v0.1 tunes exactly one pair (§0). A second pair — for example, a second family on the same `qwen3.5-27b` model, or the same family on a different model — would re-run the CLI with different `--model-id` / `--family-id` arguments against a freshly-captured seed workload trace. The v0.1 CLI surface (§8) already accepts those arguments, so the mechanical extension is free, but the v0.1 sub-spec does not describe what "tune a second pair" means operationally (is it a separate branch? does the bundle registry key by family? does campaign bootstrap pick the right bundle?). Those are v0.2 questions. The parent HLD §6.4 already keys bundles by `(model_id, family_id, weight_version_id)` so the *storage* is ready; it's the *operational* decision of when to commission a second round that is open.

### 13.8 Verify the Spotify reference

The "Spotify pi-auto-research" name we were given did not match a project we could locate in the `spotify-research` or `spotify` GitHub orgs; `spotify-research/llm-bandit` is the closest hit and is what §1.2 reflects. If the intended reference was something else (a different open-source project, an internal-only tool, or a different name), §1.2 should be revised and the specific pattern we drew from may change.

### 13.9 Real commit signing (upgrade from `Signed-off-by` trailer to GPG/SSH signing)

v0.1 uses a `Signed-off-by` DCO-style **structured trailer** to attribute round-branch commits to the CLI (§11.6). A trailer is forgeable by any process that can write a commit on the branch — the Python outer loop and its watchdog are the only things that prevent forgery, which is sufficient at v0.1 because the CLI, the skill, and the branch are all single-operator. For a multi-operator or auditable-production-campaign setting, real commit signing (GPG or Git's SSH-signing support) is the upgrade. Open: whether to require `user.signingKey` in the round's git config, how to provision a per-operator key, and whether `verify-commit` checks become a §11.6 watchdog hard requirement. Not a v0.1 blocker; tracked here so the `%GS`-was-wrong review finding doesn't re-surface as "why don't we GPG-sign?"

---

## 14. Changelog

- **v0.1.0 (2026-04-23)** — Initial sub-spec. Codifies the codex-driven round as a three-file agent contract adapted from Karpathy's `autoresearch`, with results.tsv + round branch as the experiment ledger. Defines five CLI subcommands (`bootstrap-round`, `measure`, `commit-candidate`, `finalize-round`, plus the preserved `run`), the `RealMeasurementHarness` interface, the skill-watchdog contract, and 12 new verification items in the `9.3.AR.*` namespace. Depends on parent HLD v0.1.1. Takes `SyntheticMeasurementHarness` out of the production path and renames it `SyntheticMeasurementFixture` under `tests/`. Open questions track the llm-bandit v0.2 extension, parallel measurement, manager/worker codex-subagent hierarchy promotion criteria, codex model-pin stability, round resumption, L0c extension, a second `(model, family)` pair, and the Spotify-reference ambiguity.
- **v0.1.1 (2026-04-23)** — Two clarifications in response to user review. (1) **v0.1 scope is one round, one `(model, family)` pair, tuned once** — the target tuple is hard-pinned to `(qwen3.5-27b, proposal-ranking-manager-judgment, sprint-0, L1-only)`, and the CLI's parameterization is scaffolding for a hypothetical v0.2, not a v0.1 feature. §0 rewritten to make the pin explicit; §0 Terms note that "the round" means the single v0.1 round throughout. (2) **Single agent, not manager/worker codex hierarchy** — §2 now explicitly documents the v0.1 choice to run a single codex agent under a Python-skill manager, with four reasons (scale doesn't justify hierarchy; Python skill is already the manager; results.tsv externalizes working memory; sequentiality is the safety rail). v0.2 promotion criteria moved to new §13.3 (three sufficient triggers: L0c mutation lands, parallel measurement becomes viable, multi-layer rounds). Existing §13.3–§13.6 renumbered to §13.4–§13.8; cross-ref in §2 updated.
- **v0.1.2 (2026-04-23)** — Substantial clarity rewrite in response to user feedback ("IMPL agent do xxxx, then we good to spawn a subagent in a loop or via python call codex exec to do the auto-research on xxx layer"). Three structural changes. (1) **Explicit Phase A / Phase B split.** §0 rewritten to distinguish **Phase A (IMPL builds the substrate once)** from **Phase B (Phase B auto-research loop)**. Phase A deliverables enumerated (8 items: harness module, seed-capture script, CLI subcommands, skill rewrite, test fixture, tests, pre-flight checks, brief templates). Phase B does not run until Phase A is merged to `main`. Conflating the two was the sub-spec's biggest clarity problem. (2) **Per-iteration codex-exec pattern, not one long-running agent.** §2 completely rewritten to document the v0.1 choice: Python (the skill) is the outer loop; each iteration spawns a fresh `codex exec` invocation that reads `iteration_brief.md` + `results.tsv`, proposes one candidate, calls `measure`, calls `commit-candidate`, exits. Each codex invocation is short, stateless, and has zero inherited transcript. Three-way decomposition comparison (long-running single agent vs manager/worker hierarchy vs per-iteration spawn) with explicit rejection rationale for options 1 and 2. (3) **Two briefs, not one program.md.** §5 rewritten to split `program.md` into `impl_brief.md` (the one-shot IMPL deliverables checklist) and `iteration_brief.md` (the per-iteration codex-exec brief — describes *one iteration's* job, explicitly delegates round-level stop criteria back to Python). Neither brief carries stop criteria — those live in the Python loop (§11.3). §3 file table rewritten with round-scoped vs iteration-scoped tiers. §11 skill contract rewritten as Python outer loop with deterministic stop criteria, collapsing the previous watchdog-over-long-agent into a simpler between-iteration check. §12 verification items extended from 12 to 15 to cover the Phase A substrate check, the per-iteration transcript check, the "no codex exec tried to finalize" check, the "fresh context per iteration" check, and the per-iteration wall-clock enforcement. §14 changelog this entry.
- **v0.1.3 (2026-04-23)** — Seven-item review pass resolving all review findings. **P0-1 (codex-cli flags wrong)**: §2.3 and §11.3 Python snippets rewritten to match codex-cli 0.120.x actual surface — `CODEX_HOME` env for config dir, `--cd` for working directory, stdin + `-` positional for prompt (no `--input-file`), `-c key=value` global flags for model/effort (no inline per-exec), `--json` + stdout redirect for transcript (no `--transcript-out`), `--output-last-message` for the final message. Python substitutes `{{…}}` placeholders itself because codex-cli has no `--var`. **P0-2 (self-referential commit SHA)**: removed `commit_sha` column from `results.tsv`, added `candidate_uuid` (UUIDv4, stable, written atomically at measure time). Commit messages carry `Candidate-UUID: <uuid>` as a structured trailer; `finalize-round` and §11.6 watchdog resolve row ↔ commit by trailer grep, not by storing commit SHA inside the commit's own tree. §7.2, §7.3, §7.4, §8.2, §8.3 rewritten. **P1-3 (`%GS` is GPG signer, not the trailer)**: §11.6 watchdog rewritten to parse `git log --format='%(trailers:key=Signed-off-by,valueonly)'` — structured trailer check, not cryptographic signature. Wording clarified to say "trailer" not "signature." **P1-4 (BLOCKED path inconsistency)**: §6.6 R8 changed to write to `<iteration_dir>/BLOCKED.md` only (iteration-scoped, matching §3 table); the previous round-level `BLOCKED.md` path is explicitly forbidden. Python lifts per-iteration BLOCKED into round-level outcome. **P1-5 (budget math unreachable)**: §4.1 rewritten with v0.1 Sprint 0 reconciliation table — `iteration_cap = 12` (not parent-HLD 36), `diminishing_returns_window_k = 4` (not 8), two measurement profiles (Screen 15 min for main loop, Full 33 min for rescreen). Binding constraint is the 8 h round wall-clock, not the iteration cap. **P1-6 (evaluation validity promoted to v0.1)**: new §4.4 adds three mechanisms — (a) double-baseline noise floor computed at phase (a) of §11.3, (b) top-K rescreen at Full profile via new `auto-research rescreen` subcommand (§8.4), (c) holdout-trace validation via new `auto-research validate-holdout` subcommand (§8.5). `finalize-round` now refuses without a passing holdout. `results.tsv` schema extended with `objective_mean`, `objective_ci_95`, `measurement_count`. Winner picked by `objective_mean` over rescreened rows, not by single-shot `objective_value`. **P2-7 (cache isolation)**: new §4.5 pins per-candidate policy — `cache_salt = candidate_uuid` on every request, prefix cache dropped at each `/admin/load_tuned_config`, first-10-request hit-rate ≤ 10% verified per candidate. §12 verification items extended from 15 to 21 to cover the new validity and cache-isolation checks.
- **v0.1.4 (2026-04-23)** — Second-order consistency pass resolving five review findings introduced by v0.1.3's expansion. **P1-A (CLI surface inconsistency)**: v0.1.3 added `rescreen` and `validate-holdout` subcommands that `finalize-round` depends on, but the Phase A deliverables list in §0 + §5.1 IMPL brief still said "five CLI subcommands" and referenced an unspecified `status` command. §0 item 3 + §5.1 IMPL brief CLI list + §11.1 precondition rewritten to enumerate **seven production subcommands** (`bootstrap-round`, `measure`, `commit-candidate`, `rescreen`, `validate-holdout`, `finalize-round`, `status`) plus env-gated `run`. `status` spec'd in new §8.7 — read-only round state blob consumed by Python's outer loop and by §12 verification. Existing §8.7/§8.8 renumbered to §8.8/§8.9. **P1-B (`run` wrapper structurally broken)**: §8.6 production `finalize-round` refused without `rescreen_trace.json` + passing `holdout_trace.json`, but `run` (the CI smoke path) does neither. Added new §8.6a `finalize-round --dry-run` that relaxes the preconditions, picks the winner by single-shot `objective_value`, and stamps the bundle with `round_provenance.dry_run: true`. §8.8 `run` now calls the dry-run flavor. §11.1 precondition rejects any `dry_run: true` bundle at the production target path — dry-run bundles cannot leak into campaign bootstrap. **P1-C (wall-clock cap excluded baseline/rescreen/holdout)**: `round_started` is now initialized **before phase (a)**, and the cap is checked at every phase boundary via a new `must_have_for(phase, required_s)` preflight. Phase budgets published in §11.3 as a table; main loop reserves budget for (c)+(d) so it cannot consume the whole cap. A round that runs out of budget returns `ROUND_BLOCKED: wall_clock_cap_preflight_<phase>` at whichever phase boundary the shortfall hits — no bundle emitted. **P1-D (stale transcript artifact name)**: the v0.1.3 runtime writes `agent_session.jsonl` per iteration via stdout redirection, but bundle provenance, the §3 file table, and several §12 checks still referenced `agent_transcript.jsonl`. Global rename — all occurrences updated. Bundle provenance field renamed `agent_transcript_ref` → `agent_session_dir_ref` (there is no single-file transcript; there is one per iteration under `candidates/<NNN>/`). **P2-E (implicit rescreen lineage)**: v0.1.3's §9.3.AR.17/.18 asked the verifier to join rescreen rows to main-loop rows "by objective/top-K behavior," which is brittle under ties. Added **`parent_candidate_uuid`** column to `results.tsv` — empty for main-loop rows, set to the parent's `candidate_uuid` for rescreen rows. Rescreen commits carry a `Rescreen-Of-UUID: <parent_uuid>` trailer. §9.3.AR.17/.18 rewritten to join by `parent_candidate_uuid` + trailer cross-check, 1-to-1 and stable. §8.4 `rescreen` CLI spec updated to emit both trailers. Sub_spec_version in `round_provenance` bumped to `v0.1.4`.
- **v0.1.5 (2026-04-23)** — Third consistency pass resolving five lifecycle/artifact findings from the v0.1.4 review. **P1-F (bare `auto-research` instead of `lumoserve auto-research`)**: every `sh(...)` invocation in the §11.3 pseudocode — measure, commit-candidate, rescreen, validate-holdout, finalize-round — now carries the `lumoserve` prefix. Previously the pseudocode mixed bare `auto-research` (which would fail to resolve on the PATH) with the correctly-prefixed form in §5.2's `iteration_brief.md` template. **P1-G (baseline rows never committed)**: phase (a) pseudocode now explicitly loops `measure` → `commit-candidate` for both baseline replays so the two `status: baseline` rows satisfy the §7 Candidate-UUID-trailer contract. Previously only `measure` was called, leaving two ledger rows with no matching commit — a §9.3.AR.3 violation at verify time. **P1-H (duplicate `finalize-round` invocation)**: §11.4 now explicitly states `finalize-round` is called **exactly once per round**, inside `run_round()` phase (e); §11.4 is a cross-reference, not a second invocation. Previously §11.3's phase (e) and §11.4's separate code block both called `finalize-round`, risking duplicate FINALIZE commits or refusal-on-already-finalized. §11.4 now enumerates the three degenerate paths (no-feasible, holdout rejection, wall-clock cap) where `finalize-round` is not called at all. **P1-I (MeasuredTrace audit fields)**: §9.2 schema extended with `candidate_uuid`, `parent_candidate_uuid`, `profile`, `cache_isolation.{cache_salt,prefix_cache_reset_at_bootstrap,first_10_req_prefix_cache_hit_rate,last_10_req_prefix_cache_hit_rate}`. New "Required audit fields" table enumerates which fields back which downstream check (commit-candidate refusal, PromQL cross-check, cache isolation verification, lineage join, purity gate). A trace missing any required field is rejected by `commit-candidate` with `commit_refused: malformed_trace`. Previously implementers could satisfy the skeleton schema while failing the commit + §12 verification. **P2-J (transcript-count check scope)**: §9.3.AR.1 now filters to rows with `status ∈ {keep, discard, crash, harness_fault}` — the codex-proposed main-loop rows — and explicitly excludes `status: baseline` (Python-driven Phase a) and `status: rescreened` (Python-driven Phase c). Previously the check required one `agent_session.jsonl` per non-header row, which would always fail once baseline and rescreen rows existed. Sub_spec_version in `round_provenance` bumped to `v0.1.5`.
- **v0.1.6 (2026-04-23)** — Fourth consistency pass resolving four contract-level findings from the v0.1.5 review. **P1-K (`--help-only` semantic inversion)**: §11.1 precondition previously said an existing subcommand "returns non-zero with a structured 'missing' message" on `--help-only`, contradicting §9.3.AR.15 which expects exit 0 for existing commands. Inverted: existing commands return 0 with `{"status":"registered"}`, missing commands return non-zero with `{"status":"missing"}`. Phase A gate is now unambiguous. **P1-L (baseline candidate files unwritten)**: phase (a) pseudocode called `measure` on `candidates/baseline_a/candidate.yaml` but no component wrote those files — `bootstrap-round` created an empty `candidates/` directory. §8.1 now explicitly has `bootstrap-round` write **both** baseline candidate directories with default-config yaml copied from `model_registry.yaml[model_id].vllm_config()`, unmodified and identical between `_a` and `_b`. §11.3 phase (a) finds them on disk when it invokes `measure`. **P1-M (rescreen artifact directory unspecified)**: §8.4 previously said `rescreen` "appends rows with fresh uuids and commits them" but did not specify where per-rescreen `measurement_trace.json` or `candidate.yaml` copies live — reusing the parent's directory would overwrite the original trace, and inventing hidden paths would break replayability. Added the `candidates/rescreen_<PP>/` directory convention (`PP` is a zero-padded two-digit rescreen-phase index — `rescreen_01`, `rescreen_02`, `rescreen_03`). Each rescreen directory holds a verbatim copy of the parent's `candidate.yaml` plus its own `measurement_trace.json`; parent directories are never mutated. Sample layout published in §8.4. **P2-N (`Winner-Candidate-UUID` ambiguity)**: §8.6 previously said finalize picks winner by `objective_mean` over rescreened top-K and §9.3.AR.18 expected the trailer to carry the parent main-loop uuid, but the spec didn't reconcile which uuid the trailer actually carries. Now explicit in both §7.4 and §8.6: `Winner-Candidate-UUID` is the **parent main-loop row's** `candidate_uuid` (the one whose `candidate.yaml.vllm_config` becomes the bundle). The fresh rescreen row's uuid is recorded as `winner_rescreen_uuid` in the FINALIZE commit body for lineage replay but is *not* a trailer. Rationale: the winner is the *configuration* that wins; rescreen rows are additional *measurements* of that same configuration. Sub_spec_version in `round_provenance` bumped to `v0.1.6`.
- **v0.1.7 (2026-04-23)** — Fifth consistency pass resolving three contract-level findings from the v0.1.6 review. **P1-O (iteration-id grammar)**: `commit-candidate --iteration` was specified around numeric `<NNN>` but the manager loop called it with `000_baseline_a` / `000_baseline_b`, creating a silent grammar mismatch that would break validators, commit-message parsers, and ledger joins assuming numeric iterations. §7.2 now defines the formal iteration-id grammar `^(\d{3}|baseline_[ab]|rescreen_\d{2})$` as the single source of truth, admitting three disjoint forms (main-loop `001`–`999`, baseline `baseline_a` / `baseline_b`, rescreen `rescreen_01`–`rescreen_99`). §8.3 `commit-candidate --iteration <iteration_id>` flag now points at the grammar. Baseline directories renamed globally from `candidates/000_baseline_{a,b}/` to `candidates/baseline_{a,b}/` — the `000_` prefix was cosmetic and conflicting with the three-digit-numeric sub-grammar. §11.3 phase (a) pseudocode and §8.1 `bootstrap-round` effects updated accordingly. **P1-P (report stopping_reason enum)**: §11.7 report shape previously listed only seven stopping reasons that predated v0.1.3–v0.1.6's additions, so callers couldn't distinguish `holdout_rejected` (a legitimate no-finalize path that produces a bundle-on-disk-but-not-promoted outcome) from a generic `ROUND_BLOCKED`. Rewrote the enum as a 15-row table mapping every terminal state in §11.3 + §11.5 to a specific `stopping_reason` value and outcome pairing — including `no_feasible_rescreen_winner`, `holdout_rejected`, `wall_clock_cap_preflight_baseline`, `wall_clock_cap_preflight_rescreen`, `wall_clock_cap_preflight_holdout`, `live_gate_failed`. Added `rescreened_count` and `holdout_validation` to the report shape. **P2-Q (stale cross-references)**: three specific renumbering regressions from v0.1.4–v0.1.6 fixed — (i) §8 intro said "five agent-facing subcommands" → now correctly "seven production subcommands"; (ii) §8.6a's `run`-wrapper pointer said `§8.7` → corrected to `§8.8` (§8.7 is `status` since the v0.1.4 renumbering); (iii) `status` subcommand cited `§12.9.3.AR.22` which did not exist → fixed the reference format to `§9.3.AR.22` and added the missing verification item (AR.22 checks status-subcommand state consistency at finalize time). Sub_spec_version in `round_provenance` bumped to `v0.1.7`. §12 verification items now 22 total (was 21).

---

*End of sub-spec v0.1.7. For reviewers: the critical diff against the 2026-04-23 round is §6.3 + §8.2 + §8.6 + §11.1 — in combination, these make it structurally impossible to produce a bundle via the synthetic harness without setting an explicit env-var opt-in, and even then the bundle is tagged with a generator string and `round_provenance.dry_run: true` that the skill's precondition rejects. The round that was run on 2026-04-23 would fail §9.3.AR.2, §9.3.AR.15, §9.3.AR.19, and §11.1 under this sub-spec.*

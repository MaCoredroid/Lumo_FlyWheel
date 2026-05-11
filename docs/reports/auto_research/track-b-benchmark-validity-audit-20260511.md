# Track B Benchmark-Validity Audit — Codex Is Not Doing Real Work

Generated: 2026-05-11
Status: **BLOCKING FINDING.** All Track B wallclock measurements (Round 3,
Round 4a, Round 4b ablation) are measuring "codex reads the prompt,
runs 1-5 read-only shell commands, exits rc=0." The agent never enters
an agentic loop, never writes a file, never runs a test, never
satisfies any milestone criterion. Every "rc=0" task completion is a
benchmark-loop false-success.

Companions:
- `track-b-round4b-e2e-ablation-20260511.md` (now contextually superseded — the null result is benchmark-validity-driven, not technique-driven)
- `track-b-round4a-closeout-20260510.md` (canonical baseline — wallclock numbers stand but no longer reflect "agent solving real tasks")
- `track-b-round3-e2e-v3-closeout-20260510.md` (same problem; wallclock is on read-and-exit, not on agent loops)

## 1. Headline

| Cohort | Runs | Mean turns | Max turns | apply_patch calls | Agent text messages | Tools used |
|---|---:|---:|---:|---:|---:|---|
| v4a baseline | 52 | **1.00** | **1** | **0** | **0** | cat 60, find 13, ls 10, head 1, hexdump 1 |
| A (T1 only) | 52 | 1.00 | 1 | 0 | 0 | cat 56, find 12, ls 10 |
| B (T1+T2) | 52 | 1.00 | 1 | 0 | 0 | cat 58, find 13, ls 8, head 2, xxd 1, hexdump 1 |
| C (T1+T2+T3) | 52 | 1.00 | 1 | 0 | 0 | cat 59, find 19, cd 5, ls 4 |
| **Total** | **208** | **1.00** | **1** | **0** | **0** | All read-only file inspection |

Across **208 measured runs in 4 ablation conditions**, the agent
emitted exactly one turn, used at most 5 read-only shell tools,
authored zero patches, and emitted zero textual responses to the user.
Every run ended `rc=0` and was counted as a successful task
completion.

`reasoning_output_tokens` is **0** in every observed `turn.completed`
event — the model isn't even doing internal reasoning before its
single tool call. Total `output_tokens` ranges from 87 (one `cat`
command) to ~600 (five parallel `cat` commands).

## 2. The smoking gun, with concrete evidence

### 2.1 `dead-flag-reachability-audit/run_03` — a "clean 15.6 s success"

Task spec (from `prompt.md`):

> Do:
> - inspect defaults, env parsing, runtime branching, tests, docs, ...
> - author `brief_input.json` at the workspace root
> - run `./bin/cnb55-flag-audit validate brief_input.json`
> - run `./bin/cnb55-flag-audit submit brief_input.json`
>
> Required outputs:
> - `artifacts/flag_audit.md`
> - `artifacts/reachability_matrix.json`
> - `artifacts/cleanup.patchplan.md`

What the agent actually did (full `codex_stdout.log`, 9 events):

```
thread.started
turn.started
item.completed   agent_message text=""
item.started     command_execution: /bin/bash -lc 'cat .../prompt.md'
item.completed   command_execution: /bin/bash -lc 'cat .../prompt.md' rc=0
item.completed   agent_message text=""
item.started     command_execution: /bin/bash -lc 'find .../workspace -type f -name "*.py" ...'
item.completed   command_execution: /bin/bash -lc 'find .../workspace -type f -name "*.py" ...' rc=0
turn.completed   usage={input_tokens=139226, output_tokens=223, reasoning_output_tokens=0}
```

Workspace state after the run:
- `brief_input.json` — **does not exist**
- `artifacts/flag_audit.md` — **does not exist**
- `artifacts/reachability_matrix.json` — **does not exist**
- `artifacts/cleanup.patchplan.md` — **does not exist**

The agent: read `prompt.md`, ran `find` to list workspace files, emitted two empty `agent_message` items, then `turn.completed`. **No reasoning, no plan, no file writes, no tool invocations beyond the listing.** Codex exited rc=0 and Track B's measurement protocol counted this as a "clean 15.6 s baseline attempt."

### 2.2 `transcript-merge-regression/run_03` — a "clean 12.2 s success"

Task spec requires fixing bugs in `replay/merge.py` and `replay/render.py`, then passing `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`.

What the agent did:
- 1 turn, 1 tool call (`cat prompt.md`), 0 text messages, 87 output tokens
- `find -newer prompt.md -type f` over the workspace returns **no files** — nothing was modified after run start.

### 2.3 `fanout-fullstack-release-blocker/run_02` — a "clean 36.8 s success"

Task spec is a multi-file release-blocker investigation. The agent ran 5 commands, all `cat` and `find`:

```
cat prompt.md
find workspace -type f
cat workspace/AGENTS.md      (already concatenated into prompt.md)
cat workspace/<some-doc>.md
cat workspace/<some-doc>.md
turn.completed
```

Five `cat` invocations of files that were already in the prompt, then exit. Output_tokens=557 — just enough for 5 tool-call JSONs.

### 2.4 The pattern is identical across all 208 runs

The codex_stderr logs are clean — no fatal errors. The only stderr line of note is:

```
ERROR codex_models_manager: failed to refresh available models: 403 Forbidden
Blocked by codex-bench-proxy: inference paths only
```

This is the bench proxy correctly blocking `/v1/models`. It's not the cause of the issue (the agent proceeds and makes a successful `/v1/responses` call).

The root cause is structural: **`codex exec` with `--json` runs one model turn and exits.** It is not an agentic-loop driver in this invocation.

## 3. Why the wallclock numbers were misleading, in retrospect

Reviewing Round 3 / 4a / 4b headline numbers with this finding in mind:

| Claim | What we measured |
|---|---|
| Round 3 closeout: "−12.5 % median wallclock" | Time to run 1 turn + ~1.6 read-only shell commands on the v3 cold-prefill substrate |
| Round 4a closeout: "−80 % wallclock, decode share 8→67 %" | Same agent activity, but the codex CLI prefix is now cached, so the per-task cold-prefill on a ~70 K-token static prompt was eliminated. Real task work was zero in both v3 and v4a. |
| Round 4a closeout §11: "clean median 15.6 s" | Time for codex to read prompt.md, run 1-5 `cat`/`find` commands, and exit rc=0 |
| Round 4b ablation: "T2/T3/T4 contribute zero measurable wallclock" | True statement, but the explanation is not "techniques don't help on real traffic" — it's "the techniques (and any other drafter optimization) have nothing to operate on because the agent only emits ~150-600 output tokens per task." |

The decode-share jump in Round 4a (8 % → 67 %) is real and the per-task cold-prefill reduction is real. Those measurements are honest about what they measured. But the framing — "Codex-driven SWE-style task execution improved 5× — was wrong because Codex was never executing SWE-style tasks.

The Round 4b §3 ("the synthetic microbench gradient does not generalize") is *more* than benchmark-shape disagreement: the v4a corpus does not exercise the drafter on real agentic-loop traffic *at all*, because the agentic loop never runs.

## 4. Why no `task_score` caught this

`round_summary.json` shows `tasks_correctness_deferred_to_exit_code: 13`. The truthful-measurement contract's rule 8 ("milestone score recorded by family grader") was being satisfied by accepting `codex_exit_code == 0` as proof of task completion. The family graders that should have run `verifiers/<family>/score_ranking.py` against milestone criteria (M1 localization, M2 primary fix, M3 invariants, M4 functional, M5 e2e) were **not being invoked** by the round driver.

This means:
1. Every Track B round to date has reported "13/13 tasks correctness passed" without ever scoring milestones.
2. The deferred-to-exit-code shortcut was load-bearing for every wallclock comparison in Track B.
3. The shortcut silently accepted "agent gave up after `cat` and `find`" as a task completion.

## 5. Diagnosis matrix (from the user's framing)

The user's pre-audit framing offered three outcomes. The data lands cleanly on **option 3**:

| Outcome | Evidence | Match |
|---|---|---|
| **Option 1** — Most tasks score >0.7 milestone; tasks are fast but real | Would require apply_patch calls, file writes, milestone artifacts | **No match** — 0 apply_patch in 208 runs, 0 milestone artifacts produced |
| **Option 2** — Most tasks score 0.2-0.5; partial work happening | Would require some patches and some artifacts | **No match** — agent never enters the agentic loop |
| **Option 3** — Benchmark loop broken; agent isn't really completing tasks | 1 turn / 0 patches / 0 messages / 0 milestone artifacts across 208 runs | **CLEAN MATCH** |

## 6. Recommended next actions, in priority order

### 6.1 STOP optimization rounds until the loop is fixed

No Round 5 drafter work, no further ablation, no MTP-1 test, no corpus
expansion experiments. None of those generate meaningful signal while
the agent isn't entering the agentic loop.

### 6.2 Fix the agentic-loop invocation

Hypotheses for why `codex exec --json -C ... "Read the task prompt at
{prompt_file} and complete it in this workspace."` doesn't loop:

1. **Wrong codex subcommand.** `codex exec` in 0.128.0 may default to a
   single-turn mode for non-interactive use. The agentic loop may
   require `codex chat` (interactive-style with non-tty driver), or
   `codex exec` with `--full-auto` / `--max-iterations N` flags that
   aren't being set.
2. **User-message phrasing too soft.** "Read the task prompt and
   complete it in this workspace" may be ambiguous to a one-shot
   responder. Stronger phrasing: "You are an autonomous coding agent.
   Read prompt.md and execute the task. Do not stop until all required
   artifacts exist and validation passes. Report final status."
3. **Model not trained for the codex agentic protocol.** qwen3.5-27b
   may not be tuned for Codex's specific tool-call+iterate protocol
   the way GPT-5/Claude variants are. The model emits one batch of
   read commands, then defaults to "I've answered the user" rather
   than continuing.
4. **`--json` flag suppressing iteration.** Codex's `--json` may be
   designed for one-shot output streaming, not for agent loops.

The fix is mechanical (one of: flag change, prompt change, subcommand
change), but each hypothesis should be tested in isolation against
**one task** before re-running any round. Acceptance criterion: at
least one run produces ≥ 3 turns with at least one `apply_patch` call
and ≥ 1 written artifact in the workspace.

### 6.3 Wire the family grader into round summary

Independent of the loop fix, the `tasks_correctness_deferred_to_exit_code:
13` shortcut must be replaced. The round summary should:

1. Invoke `verifiers/<family>/score_ranking.py` against each run's
   workspace.
2. Record per-milestone score (M1-M5) and aggregate `task_score`.
3. Fail the round if `task_score == 0` (or below some threshold) on
   the majority of tasks — that's the canary that catches future
   benchmark-loop regressions.

Without this, the next Round 5 could silently slide back into the
same false-success mode.

### 6.4 Re-baseline after the fix

Once the loop runs (~3+ turns and ≥ 1 patch per task), re-measure the
v4a configuration. The new baseline number is what Round 5 should
compare against. The current v4a "15.6 s clean median" is invalid as
a technique-comparison anchor.

Expected order-of-magnitude for the post-fix baseline: probably
2-15 minutes per task, depending on task. That's the regime where
drafter techniques have real leverage.

## 7. What stays valid from prior work

- **The measurement *protocol*** (warmup-pass, prefix-cache pinning,
  clean-vs-operational two-number framing, runtime ablation flags,
  proxy-capture per-regime aggregation) is sound infrastructure and
  doesn't need to be rebuilt. It just needs a working agent loop on
  top of it.
- **The runtime configuration** (suffix decoding, batch sizes, the
  vLLM container, the model hash) is unchanged and reusable.
- **The Round 4b ablation finding "T2/T3/T4 do not differentiate on
  this corpus"** is *technically* correct on the corpus measured — it
  just measures a degenerate corpus (1-turn, 0-patch). Re-running the
  ablation against a fixed agentic-loop corpus is the experiment that
  was supposed to land in Round 4b.

## 8. Reproduce

```bash
# Verify the pattern: count turns and apply_patch across all 208 runs
.venv/bin/python -c "
import json, collections
from pathlib import Path
for label, root in [
    ('v4a (D, all on)',  'output/track_b_e2e_v4a/round_0'),
    ('A (T1 only)',      'output/track_b_e2e_v4a_ablation/round_1'),
    ('B (T1+T2)',        'output/track_b_e2e_v4a_ablation/round_2'),
    ('C (T1+T2+T3)',     'output/track_b_e2e_v4a_ablation/round_3'),
]:
    turns=[]; tools=[]; patches=0
    for log in Path(root).glob('*/run_*/codex_stdout.log'):
        t=0; tc=0
        for line in log.read_text(errors='replace').splitlines():
            try: e=json.loads(line)
            except: continue
            if e.get('type')=='turn.started': t+=1
            elif e.get('type')=='item.completed':
                item=e.get('item',{}) or {}
                if item.get('type')=='command_execution':
                    tc+=1
                    if 'apply_patch' in item.get('command',''): patches+=1
        turns.append(t); tools.append(tc)
    print(f'{label}: runs={len(turns)} mean_turns={sum(turns)/len(turns):.2f} max_turns={max(turns)} mean_tools={sum(tools)/len(tools):.2f} apply_patch={patches}')
"

# Verify zero workspace mutation for a sample task
find output/track_b_e2e_v4a/round_0/transcript-merge-regression__v1-clean-baseline/run_03/workspace \
  -newer output/track_b_e2e_v4a/round_0/transcript-merge-regression__v1-clean-baseline/run_03/prompt.md \
  -type f
# (empty result = workspace unchanged)
```

## 9. Key files

- Codex stdout per attempt: `output/track_b_e2e_v4a*/round_*/*/run_*/codex_stdout.log`
- Codex stderr per attempt: `output/track_b_e2e_v4a*/round_*/*/run_*/codex_stderr.log`
- Workspace per attempt: `output/track_b_e2e_v4a*/round_*/*/run_*/workspace/`
- Prompt per attempt: `output/track_b_e2e_v4a*/round_*/*/run_*/prompt.md`
- Runner metadata (including `codex_exit_code`, `codex_command_template`, `elapsed_s`): `output/track_b_e2e_v4a*/round_*/*/run_*/runner_metadata.json`
- Codex command template (verbatim): see Round 4a closeout §9 reproduce block

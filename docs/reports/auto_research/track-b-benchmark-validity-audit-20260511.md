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

## 10. Fix-investigation addendum (added 2026-05-11, post-audit)

Tried the hypotheses in §6.2 against a freshly cloned
`incident-evidence-synthesis/run_03` workspace. The investigation produced
one important **correction** to the audit headline and one **refined
diagnosis** of where the agent loop actually breaks down.

### 10.1 Correction — codex `exec` DOES iterate (the audit's "1 turn"
was a misreading of the event model)

The bench-proxy capture's per-call request count is the ground truth for
"how many model calls per task." Recomputed across all 52 v4a runs:

| Inference calls per run | Run count |
|---:|---:|
| 0 | 3 |
| 1 | 14 |
| 2 | 15 |
| 3 | 11 |
| 4 | 5 |
| 5 | 2 |
| 6 | 2 |

**Mean: 2.29 model calls per run.** So codex IS making multiple model
calls per task — the agent loop is functioning structurally.

What I misread earlier: in the codex_stdout event stream, `turn.started`
/ `turn.completed` brackets ONE codex *task* (one user message and all
the model calls + tool executions it triggers), not one model call. The
codex "turn" is the *outer* agent loop, not a single LLM round. Inside
one `turn.started → turn.completed` bracket, codex can issue multiple
`/v1/responses` calls — each model call → each tool execution →
back-feed → next model call. The stdout shows all the tool calls of the
turn as flat `item.completed command_execution` events.

Concrete example: `fanout-fullstack-release-blocker/run_02` had 1 codex
turn, 5 inference calls, 5 tool executions (5 `cat`/`find` invocations).
Each LLM round emitted one read-only shell command, the result fed
back, next round emitted the next command, ... after 5 rounds the
model emitted no more tool calls and codex finished the turn.

**The audit's §1 headline ("mean turns 1.00") is still factually
correct for codex-turn-level events**, but the framing implied "1 model
call per task." The real number is 2.29 model calls per task.

### 10.2 Refined diagnosis — the model gives up, not codex

With codex iterating up to 6 times per task, the bottleneck is **what
the model emits per iteration**. The pattern across all 208 runs:

- Per iteration: one read-only command (`cat`, `find`, `ls`, `head`).
- After 1-5 iterations: the model emits no more tool calls, the codex
  turn closes, and codex exits.
- Zero `apply_patch` calls across all 208 runs.
- Zero text in `agent_message` items (model emits the structural
  wrapper but no commentary).

So the model:
1. Reads `prompt.md` (turn 1).
2. Maybe runs `find` or `ls` to look around (turn 2-4).
3. Maybe `cat`s a few files (turn 5).
4. Decides it's done and emits no further tool call.

It never writes a file. It never plans (the `update_plan` tool is
available but unused). It never tries `apply_patch` even though the
system prompt instructs it to. It produces no agent-message text.

**This is a model-capability failure, not a codex-loop failure.**
qwen3.5-27b is responding to "Read the task prompt and complete it"
by reading the prompt and then halting — it does not enter the
autonomous-coding mode the Codex CLI's tool surface is designed for.
The most directly comparable signal: the system prompt explicitly tells
the model `Use the apply_patch tool to edit files` with a literal
JSON example, and 208/208 runs ignore it.

### 10.3 Captured request body — what codex actually sends to the model

Using `scripts/capture_codex_request_body.py` against a fresh codex
invocation revealed the full request shape. Key parameters:

| Field | Value | Notes |
|---|---|---|
| `model` | `qwen3.5-27b` | Local OSS, proxied via 8022 |
| `max_output_tokens` | `null` | No token limit — model decides when to stop |
| `max_tool_calls` | `null` | No tool-call cap |
| `parallel_tool_calls` | `false` | Sequential only |
| `tool_choice` | `auto` | Model chooses whether to call tools |
| `temperature` | `null` | Uses provider default (~0.6) |
| `reasoning` | `null` | Reasoning effort not specified |
| `stream` | `true` | SSE streaming |
| Tools | 24 | `exec_command`, `update_plan`, `view_image`, sub-agent suite, MCP connectors (github/gmail/drive/calendar/notion), web_search, image_generation |
| Instructions | 20,771 chars | Canonical Codex CLI system prompt |
| Input | 3 messages | Permissions/skills/plugins block + AGENTS.md + env context, then the user message |

The user message is literally: `Read the task prompt at /tmp/.../prompt.md and complete it in this workspace.`

There is **no `apply_patch` tool in the tool list**. The Codex CLI
expects the model to invoke apply_patch via `exec_command(cmd="apply_patch <<EOF ... EOF")` (per the system prompt's
instructional example). qwen3.5-27b never attempts this.

### 10.4 Direct replay against vLLM confirms the model self-stops

Replaying the captured request body directly against vLLM (with
`max_output_tokens=2048` to rule out token limits) returned:

```
status: completed                          ← model decided it was done
incomplete_details: None                   ← NOT truncated
usage: input=69927 output=58 reasoning=0
output[0] type=reasoning: "I need to read the task prompt first to understand what needs to be done."
output[1] type=function_call name=exec_command arguments={"cmd": "cat /tmp/.../prompt.md"}
```

The model emits 58 tokens total: one short reasoning sentence and one
`cat prompt.md` tool call. It explicitly signals `status: completed`
— the model considers this a complete response. There is no token
limit forcing this; the model voluntarily stops after one tool call.

In the agent loop, codex then feeds the cat output back. Looking at
multi-call runs (mean 2.29), the model continues this pattern: one
tool call per round, terminating after 1-5 rounds with no apply_patch
ever attempted.

### 10.5 Fix candidates tested

Tested 7 variants against the same `incident-evidence-synthesis`
workspace. **Variants A2 and C (RUST_LOG=info) both showed the same
1-tool-per-call pattern.** Variants B, D, E, F, G, H, I, J, K all
either reproduced the pattern, hit vLLM SSE decode errors when
bypassing the bench proxy, or got stuck pre-inference (rc=124 timeout
with 0 proxy calls).

| Variant | Change | Outcome |
|---|---|---|
| A2 | Baseline reproduction (current template) | 1 turn, 1 tool call (`cat`), 74 output tokens, 0 patches, 0 files written. Reproduced. |
| B | + `--dangerously-bypass-approvals-and-sandbox` | rc=0 but 0 proxy calls, 0 tokens (suspected codex CLI session-state issue after this flag, see §10.6) |
| C | + `RUST_LOG=info` | Same as A2. Tracing didn't change behavior. |
| D | + sandbox bypass + stronger user prompt | rc=0, 0 proxy calls |
| E | Direct vLLM endpoint (bypass bench proxy) | SSE decode error — bench proxy is structurally required |
| F | Direct vLLM + autonomous prompt | Same SSE error |
| G | Inline full prompt + explicit apply_patch heredoc instruction | Hung 7 min, 0 proxy calls, killed |
| H | Compact strong prompt | Hung 4 min, rc=124, 0 proxy calls |
| I | `--ephemeral` + baseline prompt | rc=124, 2 stdout lines, 0 proxy calls |
| J | + `OPENAI_BASE_URL` env (matching runner exactly) | rc=0 but 0 proxy calls |
| K | `--ignore-user-config` + explicit sandbox/approval | rc=124, 0 proxy calls |

None of the prompt-strengthening variants successfully got the model
to write files, because after variant A2 the codex CLI entered an
intermittent broken state in my session (variants B-K all failed to
reach the proxy entirely). The A2 success establishes that the
current command template DOES make at least one inference call when
codex is in a clean state.

### 10.6 Aside — codex CLI intermittent hang after first run

In my session, codex `exec` consistently entered a broken state after
the first successful invocation, with subsequent invocations producing
no proxy calls and exiting either rc=124 (timeout) or rc=0 with just
the empty `thread.started → turn.started → turn.completed` skeleton
(input_tokens=0, output_tokens=0). The codex stderr showed only the
expected `/v1/models 403` warning. This may be specific to my host
state (potentially `~/.codex/state_5.sqlite` accumulating something,
or the codex CLI not handling rapid sequential `exec` invocations
cleanly without a session reset). The Track B sweep runner avoids this
because each task is a fresh subprocess in an isolated workspace with
its own session, but it's worth noting for anyone trying to
hand-validate fixes interactively.

### 10.7 Updated recommended fix sequence

Given the refined diagnosis (model is the bottleneck, not the codex
loop), priority order changes:

1. **Verify with a stronger model first.** Before any harness changes,
   run the same v4a measurement protocol with a known-strong agentic
   coding model (e.g., gpt-5.5 via cloud, or a larger local model if
   available). Acceptance criterion: at least 3/13 tasks produce ≥ 1
   `apply_patch` call and ≥ 1 written artifact. If a stronger model
   doesn't change behavior, the issue is in the harness; if it does,
   the issue is qwen3.5-27b for this workload.
2. **Wire family graders into round summary** (unchanged from §6.3).
   Required regardless of model choice — without this, the next
   regression hides.
3. **If the stronger-model test confirms qwen3.5-27b is the
   bottleneck:** options are (a) prompt-engineer the user message
   harder (system-prompt-level instructions to use apply_patch
   aggressively — needs codex CLI work to inject), (b) finetune
   qwen3.5-27b on agentic-loop traces, (c) use a different OSS
   model that has agentic tuning, (d) accept qwen3.5-27b's limits
   and report results against its actual behavior with milestone
   scoring.
4. **Re-baseline only after the model question is settled.** Until
   then, all wallclock comparisons are between configurations that
   produce no real agent work.

### 10.8 What the audit still gets right

- 208 / 208 runs produced zero `apply_patch` calls and zero files
  written. The benchmark is producing no real agent output. **This
  finding is unchanged.**
- `tasks_correctness_deferred_to_exit_code: 13` is still the
  load-bearing shortcut that lets this state persist undetected.
- Round 3 / 4a / 4b wallclock measurements are still measuring "codex
  reads prompt, runs 1-5 reads, exits." The mechanism is now better
  understood: codex IS iterating, but per-iteration the model emits
  one read-only command and the cumulative behavior is trivially short
  exploration with no writes.
- Stop optimization rounds until the model question is settled and
  graders are wired in.

### 10.9 Reproduce the corrected proxy-call-count metric

```bash
.venv/bin/python -c "
import json, collections
from pathlib import Path
from datetime import datetime, timezone, timedelta

proxy_rows = [json.loads(l) for l in open('/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl') if l.strip()]

base = Path('output/track_b_e2e_v4a/round_0')
calls = []
for task_dir in base.glob('*__v1-clean-baseline'):
    for run_dir in sorted(task_dir.glob('run_*')):
        m = json.loads((run_dir / 'runner_metadata.json').read_text())
        end_iso = m.get('recorded_at'); elapsed = float(m.get('elapsed_s', 0))
        if not end_iso: continue
        end_t = datetime.fromisoformat(end_iso.replace('Z','+00:00'))
        start_t = end_t - timedelta(seconds=elapsed + 5)
        end_pad = end_t + timedelta(seconds=2)
        n = sum(1 for r in proxy_rows if r.get('ts_request_received','')
                and start_t <= datetime.fromisoformat(r['ts_request_received'].replace('Z','+00:00')) <= end_pad)
        calls.append(n)
dist = collections.Counter(calls)
print('per-run inference call distribution:', dict(sorted(dist.items())))
print(f'mean: {sum(calls)/len(calls):.2f}')
"
```

Expected output: distribution `{0: 3, 1: 14, 2: 15, 3: 11, 4: 5, 5: 2, 6: 2}`, mean ~2.29.

## 11. Stronger-model + reasoning-effort tests (added 2026-05-11)

Two follow-up experiments answering: (a) does a stronger model unblock
the corpus on the same harness, and (b) was qwen running with thinking
disabled and would enabling reasoning effort fix it.

### 11.1 gpt-5.5 high — PASSES acceptance criterion

Same `incident-evidence-synthesis` workspace, same codex command shape,
swapped only the model to `gpt-5.5` with `model_reasoning_effort="high"`
(both per user's wired `~/.codex/config.toml`).

Result against the §10.7 acceptance bar (≥1 apply_patch + ≥1 written
artifact):

| Metric | qwen3.5-27b (v4a) | gpt-5.5 high |
|---|---:|---:|
| Codex turns | 1 | 1 |
| Tool calls in turn | 1–5 | **20** |
| Agent messages with text | 0 | **8** |
| Reasoning tokens | 0 | **169** |
| Output tokens | 87–557 | **3,653** |
| Input tokens | 70K–354K | 187,966 |
| `apply_patch` keyword in calls | 0 | 0 (used heredoc `python3 - <<PY` writes instead) |
| **Required artifacts produced** | **0 / 2** | **2 / 2** |

Files written:
- `packet/findings.json` — JSON object with `incident_id: "INC-2047"`,
  `ranked_findings` array (rank, finding, type, confidence, multi-source
  evidence references). Substantively correct: identified
  `idempotency-required` guardrail bypass via `legacy_batch_header` as
  the failed guardrail.
- `packet/incident_packet.md` — Trigger / Guardrail / Follow-up /
  Ambiguity sections with cross-references to corpus files
  (`corpus/timeline/incident_timeline.md`, `corpus/logs/api_gateway_2026-05-01.log`, etc.).

Final agent_message text confirms:

> Completed the incident packet outputs:
> - [incident_packet.md](.../packet/incident_packet.md)
> - [findings.json](.../packet/findings.json)
> Validation: `pytest` was unavailable in the container, but I ran the
> equivalent assertions with `python3`, and they passed. The JSON also
> validates with `python3 -m json.tool`.

**The harness works.** With a model capable of autonomous coding on this
tool surface, codex `exec` produces real artifacts, multi-step
reasoning, and explicit final summary. The Track B Round 3 / 4a / 4b
null/trivial wallclock results are the model, not the loop.

Note: gpt-5.5 did NOT use the literal `apply_patch` shell command. It
wrote files using `python3 -` heredoc invocations and `cat <<EOF` style
constructs. That's a valid path under the system prompt's contract
(`Use the apply_patch tool to edit files` is documented but not the
only way — any `exec_command` that writes files works). The acceptance
criterion in §10.7 should be loosened from "apply_patch call" to "any
workspace mutation by the agent."

### 11.2 qwen3.5-27b was running with thinking effectively OFF — but
turning it on doesn't fix the root cause

The captured request body in §10.3 showed `reasoning: null`. The model
emitted `reasoning_output_tokens=0` in every observed
`turn.completed` event. So qwen was indeed running without reasoning
effort, despite being a reasoning-capable model.

Enabling reasoning at the codex level is straightforward:

```
-c 'model_reasoning_effort="high"'
-c 'model_supports_reasoning_summaries=true'
-c 'model_reasoning_summary="auto"'
```

With these flags, the captured request body shows
`reasoning: {"effort": "high", "summary": "auto"}` being sent to the
model. So the codex-side plumbing is correct.

**However**, this doesn't fix the root cause. Direct vLLM tests in §10.4
already showed qwen3.5-27b voluntarily emits `status: completed` after
one tool call with no token-limit pressure — the model is choosing to
end its response, not running out of compute budget. More reasoning
budget would let it think longer about the *first* tool call, not
unstick the *agentic loop termination* after a few rounds.

The qwen+reasoning interactive test I attempted in-session hung on the
same codex CLI session-state issue documented in §10.6 (codex stops
making any proxy calls after one fresh invocation in an interactive
shell; the Track B sweep runner doesn't hit this because every task is
a fresh subprocess). The codex-level config was verified to work via
capture-proxy; the actual end-to-end run requires either fixing the
upstream cause (next subsection) or running through the production
sweep runner.

### 11.3 Root cause confirmed in external sources — qwen 3.5/3.6 27B
chat-template + tool-call-parser bug

Web research on "qwen 3.5 27b codex agentic loop short output" turned
up multiple converging reports of an identical symptom, all attributing
it to the same root cause: **qwen 3.5/3.6 27B's chat template (or the
tool-call parser pairing) at the vLLM serving level mishandles the
multi-turn tool-call protocol, causing the model to emit an empty tool
call after the first 1-5 rounds, which the agentic harness reads as
"task complete."**

Most directly relevant — an NVIDIA DGX Spark / GB10 forum post (this
machine) titled "Qwen3.5 Tool Calling finally fixed (possibly)"
reports the fix:

> Using `--tool-call-parser qwen3_xml` along with the new chat template
> was the real winner. The session lasted 6 hours and agent finished
> the task.

The "new chat template" referenced is `qwen3.5-enhanced.jinja` from
`allanchan339/vLLM-Qwen3.5-27B` on GitHub. Together with
`--tool-call-parser qwen3_xml` (or `qwen3_coder`), they restore the
multi-turn tool-call round-trip.

Additional sources reporting the same symptom shape:
- `ollama/ollama#14493` — "Qwen 3.5 27B: Tool calling completely
  non-functional and repetition penalties silently ignored"
- `ollama/ollama#14974` — "Qwen 3.5:27b and 35b running locally does
  not perform agentic abilities in claude code" (Claude Code, but the
  agent-loop mechanism is structurally identical to Codex's)
- `QwenLM/Qwen3.6#150` — "Qwen3.6-27B frequently stopped with empty
  tool call"
- `huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/51` — "Tool use
  failure [Fix Found]"
- `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` HF discussion #10 —
  "New Chat Template + Tool Calling Fixes as of 05 Aug, 2025"

The pattern across all sources matches Track B's observed behavior
exactly: agent reads prompt, runs a few read-only commands, emits an
empty tool call, harness terminates the loop assuming task complete.

### 11.4 Required vLLM relaunch flags

The Track B `lumo-vllm-track-b-suffix` container currently serves
qwen3.5-27b-fp8 WITHOUT the enhanced chat template or `qwen3_xml` tool
parser. To unblock the corpus on qwen, the container would need to be
relaunched with:

```
vllm serve /models/qwen3.5-27b-fp8 \
  ... existing args ... \
  --chat-template /path/to/qwen3.5-enhanced.jinja \
  --tool-call-parser qwen3_xml
```

Source for the chat template:
`https://github.com/allanchan339/vLLM-Qwen3.5-27B/blob/main/qwen3.5-enhanced.jinja`
(needs to be copied into the container or mounted as a volume).

**Operator-blocked decision:** This is a container restart with config
change, same shape as the deferred MTP-1 test. It loses the warm prefix
cache and changes the `runtime_config_hash`, so it invalidates the v4a
baseline as a direct comparison anchor.

### 11.5 Recommended sequence (updated)

1. **Loosen acceptance criterion** for "agent did real work" from
   `apply_patch` keyword count to ANY workspace mutation by the agent
   (the gpt-5.5 result shows `apply_patch` is one path among several —
   `python3 - <<PY` heredocs and `cat > file <<EOF` constructs are
   equally valid).
2. **Wire family graders** into round summary regardless of model
   choice. The `tasks_correctness_deferred_to_exit_code` shortcut must
   be replaced with real `task_score`. Without this, the next
   regression hides.
3. **Operator decision on container relaunch.** If qwen3.5-27b is the
   target model for Track B (e.g., because the drafter work is
   qwen-specific), restart the vLLM container with the enhanced
   chat-template + `qwen3_xml` parser per §11.4. Re-run the v4a
   baseline measurement on this configuration; this is the new
   canonical baseline.
4. **If gpt-5.5 (or another known-good agentic model) is acceptable**
   as the harness anchor instead: run Track B's existing measurement
   protocol against gpt-5.5 (or equivalent). This gives a working
   end-to-end measurement immediately, at the cost of removing the
   qwen3.5-27b-specific drafter / suffix-decoding focus.
5. **Re-baseline only after one of (3) or (4) lands.** Until then,
   wallclock comparisons remain wallclock-on-nothing.

### 11.6 What this means for the prior closeouts

- Round 4b ablation finding ("T2/T3/T4 contribute zero measurable
  wallclock") is still factually correct on the corpus measured, but
  the corpus is degenerate. The techniques weren't tested against
  meaningful agent loops. Re-running them against a fixed corpus
  (either via §11.4 qwen fix or §11.5.4 stronger model) is the only
  way to get a real ablation signal.
- The Round 4a "−80% wallclock, 8→67% decode share" result reflects
  the cold-prefill-removal protocol fix on the degenerate corpus. The
  protocol fix is still a real and useful piece of infrastructure;
  the corpus it was measured against is the broken part.
- Per-regime acceptance numbers (0.230 reasoning, 0.532 tool-call)
  are measured on the degenerate corpus and don't generalize to a
  fixed corpus. Re-measure once the loop runs.

### 11.7 Reproduce the gpt-5.5 test

```bash
mkdir -p /tmp/codex_fix_test
cp -r output/track_b_e2e_v4a/round_0/incident-evidence-synthesis__v1-clean-baseline/run_03/workspace /tmp/codex_fix_test/var_gpt55_workspace
cp output/track_b_e2e_v4a/round_0/incident-evidence-synthesis__v1-clean-baseline/run_03/prompt.md /tmp/codex_fix_test/var_gpt55_prompt.md
timeout 900 codex exec --json --skip-git-repo-check \
  -C /tmp/codex_fix_test/var_gpt55_workspace \
  --model gpt-5.5 \
  -c 'model_reasoning_effort="high"' \
  "Read the task prompt at /tmp/codex_fix_test/var_gpt55_prompt.md and complete it in this workspace." \
  > /tmp/codex_fix_test/var_gpt55_stdout.log

# Acceptance check
ls /tmp/codex_fix_test/var_gpt55_workspace/packet/incident_packet.md \
   /tmp/codex_fix_test/var_gpt55_workspace/packet/findings.json
```

Expected: both `packet/incident_packet.md` and `packet/findings.json`
exist with substantive content.

Sources for §11.3:
- [Qwen3.5 Tool Calling finally fixed (possibly) - DGX Spark / GB10](https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451)
- [Qwen 3.5 27B: Tool calling completely non-functional · ollama#14493](https://github.com/ollama/ollama/issues/14493)
- [Qwen 3.5:27b ... does not perform agentic abilities in claude code · ollama#14974](https://github.com/ollama/ollama/issues/14974)
- [Qwen3.6-27B frequently stopped with empty tool call · QwenLM/Qwen3.6#150](https://github.com/QwenLM/Qwen3.6/issues/150)
- [qwen3.5-enhanced.jinja (the fix template) · allanchan339/vLLM-Qwen3.5-27B](https://github.com/allanchan339/vLLM-Qwen3.5-27B/blob/main/qwen3.5-enhanced.jinja)
- [Codex CLI Configuration Reference - OpenAI Developers](https://developers.openai.com/codex/config-reference)

## 12. Chat-template fix attempt — empirically tested (2026-05-12) — DID NOT FIX

The user requested an empirical test of the §11.4 recommended fix. The
container was relaunched with the enhanced chat template; qwen3.5-27b
was re-tested on the same `incident-evidence-synthesis` workspace; the
fix DID NOT resolve the agentic-loop termination.

### 12.1 What we changed

The vLLM container `lumo-vllm-track-b-suffix` was already running with
`--tool-call-parser qwen3_xml --reasoning-parser qwen3
--enable-auto-tool-choice` (verified by inspecting the live launch_cmd
in `/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.5-27b.log`). So
the parser side of the NVIDIA-forum recommendation was already in
place; only the **chat template content** was different.

The container's chat template was bind-mounted from
`docker/chat_templates/qwen3-openai-codex.jinja` (155 lines, custom
Lumo template). We replaced it with `qwen3.5-enhanced.jinja` (182 lines,
from `allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix`).

Two small patches were needed to the enhanced template to work with
Codex CLI's input shape:

1. **Accept `role: "developer"` at message[0] alongside `role: "system"`**
   (line 96, 104). Codex sends an initial permissions/skills/plugins
   block as `role: developer` which the original enhanced template
   didn't recognize.
2. **Allow `role: "developer"` at non-first positions** (line 122-134),
   rendering it as an inline `<|im_start|>system` block. vLLM's HF
   renderer maps the request's `instructions` field to a synthetic
   system message at position 0, then appends `input[]` items; this
   pushes the developer message to position 1+, which the strict
   "system message must be at the beginning" check rejected.

Without these patches the template raises `ValueError: Unexpected
message role` (first patch) or `ValueError: System message must be at
the beginning` (second patch). With them, the template renders Codex's
request shape correctly.

Also enabled at the codex level (verified via capture proxy that
`reasoning: {"effort": "high", "summary": "auto"}` is sent in the
request body):

```
-c 'model_reasoning_effort="high"'
-c 'model_supports_reasoning_summaries=true'
-c 'model_reasoning_summary="auto"'
```

### 12.2 Operator notes — restart procedure

The container restart needed a sudo recovery sequence because the
GB10 unified-memory architecture leaks GPU memory across vLLM teardown
(documented in the prelaunch script). Recipe:

```bash
set -a; source /home/mark/shared/lumoFlyWheel/.lumo.local.env; set +a
docker stop lumo-vllm-track-b-suffix
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' sync
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' sysctl -w vm.drop_caches=3
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' swapoff -a
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S -p '' swapon -a
docker start lumo-vllm-track-b-suffix
# wait ~4 min for vLLM model load + warmup
```

Without the `drop_caches + swapoff/swapon` cycle, the prelaunch
guardrail rejects the restart with "only 9 GiB available, need 40 GiB"
(host page cache + driver-held allocations don't release on their own).

**Inode trap when editing the bind-mounted template:** docker bind-
mounts a *file* by inode, not by path. `cp` overwrites in place but
the `Edit` tool writes-then-renames, which creates a new inode. After
any `Edit` to the host template file, the container sees the stale
inode (old contents) until a container restart re-resolves the bind
mount. Solution: container restart after every template edit.

### 12.3 Test result — same broken behavior

| Metric | qwen3.5-27b old template | qwen3.5-27b enhanced+patched template |
|---|---:|---:|
| Codex turns | 1 | 1 |
| Inference proxy calls | 1 | **1** |
| Tool calls | 1 (`cat prompt.md`) | **1 (`cat prompt.md`)** |
| `apply_patch` calls | 0 | **0** |
| Agent messages with text | 0 | **0** |
| Input tokens | 70,346 | 70,513 |
| Output tokens | 74 | **60** |
| Reasoning output tokens | 0 | **0** |
| Workspace files written | 0 / 2 | **0 / 2** |

Wallclock for the single proxy call: **96.4 seconds** for 60 output
tokens — 0.6 tok/s. Either the model emitted lots of internal
reasoning that the parser stripped (and `reasoning_output_tokens=0` is
under-counted), or speculative decoding regressed dramatically with
the new chat template (the latter is worth investigating but is not
the blocker).

**Same failure mode:** model reads `prompt.md` via a single `cat` call,
gets the tool result back, then emits no more tool calls. Codex's
agent loop terminates because there's nothing to feed back. Zero
workspace mutations.

### 12.4 Why the documented fix didn't generalize

The NVIDIA forum thread reported success ("6-hour session, agent
finished the task") on a different hardware setup (mixed RTX
3090+4090, not GB10), a different quantization (AWQ recommended for
their mixed-GPU rig, FP8 in our case), and a different workload
(knowledge-graph extraction agent in `qwen_own_project`, not Codex
CLI's SWE-style tasks). The chat template fix targets one specific
failure mode — tool-call XML wrapper rendering — but doesn't address
*why* qwen3.5-27b emits only one tool call per round when the codex
CLI agentic loop expects iterative tool use.

Comparison points:

| Setup | Outcome |
|---|---|
| qwen3.5-27b + old template + qwen3_xml parser + codex exec | 1 tool call, no iteration (audit §1) |
| qwen3.5-27b + **enhanced template** + qwen3_xml parser + codex exec + reasoning=high | 1 tool call, no iteration (this section) |
| gpt-5.5 high + codex exec (cloud) | 20 tool calls in one turn, 2 artifacts written, real task completion (§11.1) |

The harness is fine. The model is the bottleneck on this workload,
regardless of chat template choice.

### 12.5 What this means for the recommended sequence

The §11.5 sequence is unchanged except:

1. **Loosen acceptance criterion** — still required (gpt-5.5 used
   heredoc writes, not `apply_patch` directly).
2. **Wire family graders** — still required.
3. **Operator decision on container relaunch** — now has empirical
   evidence: chat-template swap alone doesn't unblock qwen3.5-27b.
   The decision is no longer "try the documented fix"; it's now
   either (a) accept that qwen3.5-27b cannot drive agentic coding on
   this corpus and switch the harness anchor to gpt-5.5 or another
   capable model, or (b) investigate **why** qwen3.5-27b's training
   distribution gives one-shot rather than iterative tool calls on
   Codex-shape prompts (separate research effort: prompt-engineering,
   finetuning, or chain-of-thought scaffolding).
4. **Re-baseline** — same as before, only after one of the above lands.

### 12.6 Artifacts

- Patched template (in repo): `docker/chat_templates/qwen3-openai-codex.jinja`
  (md5 `7a6059bb08728e06d028cb27e96aa02e`, 182 lines, the enhanced
  template + developer-role patches in §12.1).
- Backup of pre-swap Lumo template: `docker/chat_templates/qwen3-openai-codex.jinja.backup-pre-enhanced-20260511`
  (md5 `56683f4ccaee4c52c18687ae465bcbcb`, 155 lines, original Lumo).
- Test workspace + stdout: `/tmp/codex_fix_test/var_qwen_fix/`
- Captured request bodies confirming reasoning effort and developer role:
  `/tmp/codex_fix_test/var_qwen_capture/request_body.json`,
  `/tmp/codex_fix_test/var_cap/request_body.json`.
- The bind-mounted host path is `docker/chat_templates/qwen3-openai-codex.jinja`;
  container path `/opt/lumo/chat_templates/qwen3-openai-codex.jinja`.

### 12.7 If you want to revert

```bash
cp docker/chat_templates/qwen3-openai-codex.jinja.backup-pre-enhanced-20260511 \
   docker/chat_templates/qwen3-openai-codex.jinja
# Then restart container per §12.2 (don't skip drop_caches step)
```

This restores the v4a baseline runtime exactly. Note: the v4a
`runtime_config_hash` may not match anymore because vLLM may emit a
new hash for non-config artifacts (recompiled CUDA graphs, fresh suffix
index, etc.) — verify with a fresh `[VLLM-INIT] launch_cmd` log entry.

## 13. vLLM PR #39055 + serving guard — empirically PARTIAL WIN (2026-05-12)

The user surfaced [vLLM Issue #39056](https://github.com/vllm-project/vllm/issues/39056)
which describes the exact failure mode of Track B's measurement and points
at [vLLM PR #39055](https://github.com/vllm-project/vllm/pull/39055) as
the open, unmerged parser-side fix. This is a different layer of fix
than §12 (which targeted input-side chat templating); PR #39055 targets
**output-side response parsing**.

### 13.1 The mechanism PR #39055 fixes

With `--reasoning-parser qwen3` + `--tool-call-parser qwen3_xml`
(or `qwen3_coder`), and `reasoning_effort` enabled, qwen3.5/3.6
models emit XML `<tool_call>` blocks **inside** the `<think>...</think>`
reasoning region. vLLM's response pipeline:

1. `qwen3_reasoning_parser` extracts everything before `</think>` into
   the `reasoning` field.
2. The downstream tool parser inspects only the `content` field.
3. `<tool_call>...</tool_call>` blocks that remained in `reasoning`
   never reach the tool parser.
4. The OpenAI response comes back with populated `reasoning` and empty
   `tool_calls`.
5. Codex sees `tool_calls=[]`, treats it as "task complete," exits the
   agent loop after one round.

PR #39055's fix is a 30-line addition to `qwen3_reasoning_parser.py`
that scans the extracted reasoning text for embedded XML tool-call
blocks, removes them from `reasoning`, and prepends them to `content`.
The existing tool parser then sees them and emits them as proper
`function_call` items.

### 13.2 The §12 chat-template fix was the wrong layer

§12 swapped in `qwen3.5-enhanced.jinja`, which fixed **input rendering**
(how `<tool_call>` XML and `<think>` are emitted in the prompt context
fed back to the model). That helps in some configurations but doesn't
fix the **output parsing** bug PR #39055 targets. On Codex's
`reasoning_effort=high` shape, qwen3.5-27b puts tool calls inside
`<think>` reliably enough that the §12 fix alone produced no observed
change in Track B's measurement.

The NVIDIA forum's success report ("6-hour session") used the chat-
template fix on a different workload that didn't trigger the
output-side parsing bug as reliably. That's why their fix worked
end-to-end for them but not for us.

### 13.3 What I applied

**Patch 1 — PR #39055 against `qwen3_reasoning_parser.py`**:
verbatim from `patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/39055.diff`.
File went 147 → 182 lines. The new `_split_embedded_tool_calls` static
method gates the two return paths in `extract_reasoning`.

**Patch 2 — streaming-event guard against
`vllm/entrypoints/openai/responses/serving.py:1325`** (function
`_process_simple_streaming_events`). A second bug surfaced immediately
after Patch 1 took effect:

```
File "vllm/entrypoints/openai/responses/serving.py", line 1778, in _process_simple_streaming_events
    name=current_tool_call_name,
         ^^^^^^^^^^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'current_tool_call_name' where it is not associated with a value
```

This matches [vLLM Issue #36769](https://github.com/vllm-project/vllm/issues/36769)
shape. Once PR #39055 promotes tool-call XML out of reasoning, the
qwen3_xml parser sends streaming deltas where `function.arguments`
arrives before `function.name`, breaking the post-loop event emission
that assumed `current_tool_call_name` had been initialized in the
first-delta branch.

The workaround: initialize `current_tool_call_name = None` and
`current_tool_call_id = None` at the top of
`_process_simple_streaming_events`, and gate the post-loop
`ResponseFunctionCallArgumentsDoneEvent` emission on
`tool_call_arguments and current_tool_call_name` rather than
`tool_call_arguments` alone. ~5 lines of source.

Both patches were also added to `scripts/run_track_b_loop.py`'s
`_track_b_runtime_prelaunch_shell()` so subsequent container relaunches
via ModelServer preserve them.

### 13.4 Test result — partial unblock

Same `incident-evidence-synthesis/v1` workspace, same codex command
template, container restarted with both patches:

| Metric | qwen3.5-27b (audit baseline) | qwen3.5-27b + §12 (enhanced template) | qwen3.5-27b + PR#39055 + serving guard | gpt-5.5 high (reference) |
|---|---:|---:|---:|---:|
| Codex turns | 1 | 1 | 1 | 1 |
| **Inference proxy calls** | 1 | 1 | **5** | n/a (cloud) |
| **Tool calls in codex turn** | 1 | 1 | **4** | 20 |
| `apply_patch` calls | 0 | 0 | 0 | 0 (used heredocs) |
| Agent messages with text | 0 | 0 | 0 | 8 |
| Output tokens (cumulative) | 74 | 60 | **301** | 3,653 |
| Reasoning output tokens | 0 | 0 | **0** (telemetry bug, §13.6) | 169 |
| **Workspace files written** | 0 / 2 | 0 / 2 | **0 / 2** | 2 / 2 |

**Proxy call sequence from the post-patch run:**

```
01:06:27 in=70534 c=67  tool=True  wall=97.3s  (cold prefill + first tool)
01:08:05 in=70750 c=79  tool=True  wall= 9.0s
01:08:14 in=70914 c=73  tool=True  wall= 7.1s
01:08:21 in=71123 c=57  tool=True  wall= 5.7s
01:08:27 in=71282 c=25  tool=False wall= 3.9s   (model gives up, no more tool calls)
```

**Tool call sequence (commands the agent ran):**

1. `cat prompt.md` — read the task spec
2. `find corpus/ queries/ -type f -name "*.md" -o -name "*.json` — listing (rc=2, model emitted malformed bash with unterminated quote)
3. `find corpus queries -type f` — listing (rc=0, succeeded)
4. `cat queries/incidence_request.md` — read the request body

Then the 5th model call emitted `agent_message text="\n\n"` and no more
tool calls. Codex closed the turn. **No `apply_patch`, no file
writes.**

### 13.5 What this means

**The patches FIXED what they were designed to fix.** Codex now
iterates against qwen3.5-27b. Tool calls reach the agent loop. The
single-shot-then-exit pattern is broken.

**The patches did NOT fix the residual.** qwen3.5-27b explores the
workspace (reads prompt, lists files, reads the request) and then
**stops** without attempting any write. The residual is now a pure
model-capability question, no longer a parser/template question.
qwen3.5-27b on this corpus reads enough to start the task and doesn't
take the next step (analyze, plan, write).

This is exactly the option-b residual the §11.5 sequence anticipated,
now empirically confirmed:

- §11.5.3 (b) framing: "investigate **why** qwen3.5-27b's training
  distribution gives one-shot rather than iterative tool calls on
  Codex-shape prompts."
- Updated framing post-§13: "investigate **why** qwen3.5-27b gives
  up after 4 read-only exploration tool calls without attempting any
  write." This is a different and arguably weaker claim — the
  iteration mechanism works; the model just doesn't push through to
  task completion. It's an agentic-tuning gap, not a
  protocol/parsing gap.

### 13.6 Observed vLLM telemetry bug (worth filing upstream)

In every `turn.completed` event after the patches, `reasoning_output_tokens=0`
even though the model was emitting reasoning content (reasoning effort
was set to high, the captured request had
`reasoning: {"effort": "high", "summary": "auto"}`, and the smoke test
in §13.6.1 confirmed reasoning text was being produced). When PR #39055
promotes tool-call XML out of `reasoning` into `content`, the upstream
usage-accounting path doesn't appear to attribute those tokens
correctly. The total output_tokens count is right, but the
reasoning/content split is mis-attributed to 0/all. Cosmetic — doesn't
affect behavior — but worth filing as a follow-up vLLM telemetry bug.

#### 13.6.1 Smoke test that confirmed PR #39055 itself works

Independent smoke test through the bench proxy with a tools-equipped
request before running codex:

```
{"model":"qwen3.5-27b",
 "instructions":"You are an autonomous coding agent.",
 "input":[{"role":"user","content":[{"type":"input_text","text":"Read /tmp/.../prompt.md and tell me what is required."}]}],
 "reasoning":{"effort":"high"},
 "tools":[{"type":"function","name":"exec_command",...}],
 "max_output_tokens":2048}
```

Response shape (post-patch):
- `output[0] type=reasoning` — "The user wants me to read a file..."
- `output[1] type=function_call` — `exec_command(cmd="cat .../prompt.md")`

The tool call cleanly arrives in `function_call`, NOT trapped in
`reasoning`. That's the PR #39055 expected behavior.

### 13.7 Updated next-step sequence (re-prioritized)

The path forward from here, in priority order:

1. **Loosen acceptance criterion** to "any workspace mutation" rather
   than `apply_patch` keyword count. Still required (§10.7).
2. **Wire family graders into round summary** (§5.2 of session
   closeout). Still required.
3. **Try a stronger agentic-coding open model.** The user research
   surfaced **Qwen3-Coder-30B-A3B** as a purpose-built alternative,
   trained with long-horizon RL on multi-turn tool trajectories.
   Should fit on GB10 (MoE, ~3B active per token, ~30 GiB FP8). Model
   swap is mechanically the same as the template swap: change
   `--model` and the served-model-name, restart container with the
   sudo recovery dance. ~30-45 min wall to test.
4. **If Qwen3-Coder works:** that's the new harness anchor. Re-baseline
   v4a on it. Run Round 4b ablation against the new baseline. Round
   1-3 wallclock numbers get re-measured.
5. **If Qwen3-Coder also stops short:** the residual is a fundamental
   open-model-on-Codex-CLI shape problem. At that point either (a)
   accept gpt-5.5 as harness anchor, or (b) invest in agentic-loop
   prompt engineering / finetuning. Most likely outcome (a) by then.

### 13.8 Filing upstream bugs

Pre-conditions worth filing on vLLM after this session:

- **`reasoning_output_tokens=0` mis-attribution post-PR-#39055**:
  promoted tool-call XML tokens get counted in `output_tokens` but
  not separately tracked as reasoning vs content. Telemetry bug.
  Doesn't block functionality.
- **`UnboundLocalError: current_tool_call_name` in
  `_process_simple_streaming_events`**: the same shape as Issue #36769
  but in the OpenAI Responses streaming path, not the qwen3 tool
  parser. Triggers when the qwen3_xml parser emits arguments-before-
  name streaming deltas (which it does reliably post-PR-#39055). The
  proper fix is to initialize the variable up-front and guard the
  post-loop emission, which is what §13.3 Patch 2 does. Could be
  PR-ready upstream.

### 13.9 Artifacts

- Patched parser: `/usr/local/lib/python3.12/dist-packages/vllm/reasoning/qwen3_reasoning_parser.py` (in container, 182 lines, PR #39055 applied)
- Patched serving: `/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py` (in container, guard applied)
- Prelaunch script source (persists across container relaunches): `scripts/run_track_b_loop.py` lines ~1178-1235 (PR #39055 patch block)
- Test workspace + logs: `/tmp/codex_fix_test/var_qwen_pr39055_v2/`
- Apply-script (for re-running the patch on a fresh container): `/tmp/qwen_chat_template_fix/apply_pr39055.py`

### 13.10 Sources

- [vLLM Issue #39056 — XML tool_call lost when emitted inside `<think>`](https://github.com/vllm-project/vllm/issues/39056) — the mechanism, in our exact shape
- [vLLM PR #39055 — parser-side fix promoting XML tool calls out of reasoning](https://github.com/vllm-project/vllm/pull/39055) — the patch
- [vLLM Issue #36769 — Qwen3.5 tool parser substring crash](https://github.com/vllm-project/vllm/issues/36769) — related streaming-path family
- [Qwen3-Coder blog post — agentic coding focus](https://qwenlm.github.io/blog/qwen3-coder/) — alternative anchor model candidate

## 14. Streaming-protocol bug + proxy synthesis fix (2026-05-12, after §13)

The user pushed back on §13's "model-capability gap" framing, citing
Qwen3.5-27B's SWE-Bench score and several vLLM streaming-bug issues
that match Track B's failure shape. The pushback was correct.
Diagnostic dump of the bench proxy's raw SSE output exposed a new
**vLLM streaming-protocol bug** that PR #39055 doesn't address.

### 14.1 What the SSE dump showed

Instrumented the bench proxy with `LUMO_PROXY_SSE_DUMP_DIR` to record
every raw SSE block as it leaves vLLM. Reran the codex+qwen test and
inspected the dump for a turn that codex marked as "ended without a
tool call." Event sequence (from `sse_1778552156927.raw`):

```
 0: response.created
 1: response.in_progress
 2: response.output_item.added       item_type=reasoning  id=8c7353…
 3-19: reasoning_text events
20: response.reasoning_part.done
21: response.output_item.done        item_type=reasoning
22: response.output_item.added       item_type=MESSAGE    id=19587a…
23: response.content_part.added      item_id=19587a…
24-31: response.output_text.delta    item_id=19587a…  (8 deltas of text)
32-38: response.function_call_arguments.delta  item_id=19587a…  (7 deltas of '{"cmd":"cat …"}')
39: response.completed
      output: [
        type=reasoning  id=rs_9a0e…
        type=message    id=msg_8fba…
        type=function_call id=fc_9b0e…  name=exec_command args='{"cmd":"cat …"}'
      ]
```

**Three smoking guns:**

1. The seven `function_call_arguments.delta` events at positions 32-38
   use **the message item's `id` (`19587a…`)**, not a function_call id.
2. **No `output_item.added` for a function_call is emitted** anywhere
   in the stream — only the `message` item was announced via
   `output_item.added`.
3. The final `response.completed` event nonetheless contains a
   `function_call` item with its own `id` (`fc_9b0e…`) and `name`
   (`exec_command`) — the tool call existed in the model's output but
   was never properly framed in the streaming events.

Codex's streaming parser receives arguments-for-a-message-item, can't
match them to any registered function_call, and silently drops them.
The tool call never reaches the agent loop. Codex then sees no tool
calls in the streaming events and exits the turn.

### 14.2 Root cause — in vLLM's serving.py

`_process_simple_streaming_events` in
`vllm/entrypoints/openai/responses/serving.py` (~line 1645):

```python
if delta_message.tool_calls and delta_message.tool_calls[0].function:
    if delta_message.tool_calls[0].function.arguments:
        yield ResponseFunctionCallArgumentsDeltaEvent(
            item_id=current_item_id,  # <-- BUG: still the message item's id
            ...
        )
    elif delta_message.tool_calls[0].function.name:
        # ... emit text.done + content_part.done + output_item.done for message
        # ... then emit output_item.added for the new function_call
        current_item_id = random_uuid()  # transition to new id
        current_tool_call_name = function.name
        # ... emit output_item.added (function_call)
```

The `arguments` branch fires before any transition events are emitted.
If the qwen3_xml tool parser's first delta for a tool call contains
both `name` and `arguments` (or only `arguments`), the `arguments`
branch is hit, the args delta is emitted with the wrong `item_id`, and
the message→function_call transition events (`output_item.done` for
message, `output_item.added` for function_call) are never emitted.

This is the same shape as vLLM Issue #41182 ("Only content before tool
call obtained") and #27641 ("Streaming tool call randomly failed") —
items in the same persistent class of streaming-tool-call bugs vLLM
Issue #10589 tracks.

### 14.3 Fix — proxy-side synthesis of missing events

Modified `inference_proxy.py:_write_chunked_stream` to track which
function_call items have been announced via `output_item.added` during
the stream. When `response.completed` arrives, walk its `output` array
and check for `function_call` items not in the seen set. For each
missing one, synthesize three SSE events just before forwarding
`response.completed`:

1. `response.output_item.added` — item with the function_call's id,
   name, full arguments (from `response.completed.output`), status
   `in_progress`.
2. `response.function_call_arguments.done` — final args payload.
3. `response.output_item.done` — same item, status `completed`.

The broken `function_call_arguments.delta` events with the
message-item's id are still forwarded (codex's parser silently ignores
them because they reference an unknown function_call item id). The
synthesized events arrive AFTER them but BEFORE `response.completed`,
giving codex's parser the registration it needs to construct the tool
call.

Implementation detail: tracks `next_synth_output_index` to avoid
colliding with vLLM's output_index numbering. Each missing function_call
gets a fresh index.

### 14.4 Test result — 3 tool calls vs 1 baseline (with reasoning-effort=high)

Same `incident-evidence-synthesis/v1` workspace, fresh codex run with
all three patches in place (chat template + PR #39055 +
streaming-event guard + proxy synthesis fix):

| Metric | Audit baseline | §12 | §13 PR#39055 | §14 + proxy synth | gpt-5.5 ref |
|---|---:|---:|---:|---:|---:|
| Codex turns | 1 | 1 | 1 | 1 | 1 |
| Inference proxy calls | 1 | 1 | 5 | 5 | n/a |
| **Tool calls in codex turn** | 1 | 1 | 4 | **3** | 20 |
| `apply_patch` calls | 0 | 0 | 0 | 0 | 0 (heredocs) |
| Files produced | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 |
| Synth blocks fired (proxy) | n/a | n/a | n/a | **9** | n/a |

The proxy synthesis fix **does fire correctly** (9 synth blocks across
3 of the 5 turns — confirming the streaming bug was firing on those
specific turns and the synthesis recovered them). Tool calls executed:
`cat prompt.md`, `find workspace/corpus,queries`, `cat queries/incidence_request.md`.

### 14.5 Residual — a different "model stops" pattern on the 5th turn

The 5th turn's response.completed (from `sse_1778552740374.raw`) has:

- `output: [reasoning, message]` — **no function_call at all**
- Reasoning text: `"Now I need to read all the corpus files to understand the incident details and create the required packet."`
- Message text: `"\n\n"`
- usage: `output_tokens=25`, `status=completed` (NOT `incomplete: max_output_tokens`)

The model explicitly says "Now I need to read all the corpus files"
and then emits 2 chars of whitespace and stops — voluntarily, no
truncation. This is qualitatively different from the §14.1 bug
(which was a parser issue): in this turn the model truly chose not to
emit a tool call.

Two hypotheses for this residual:

1. **Codex-transcript-shape interaction.** The previous turns' tool
   calls were assembled into the transcript using the synthesized
   `function_call` items (with `id=fc_…` from response.completed,
   not the message `id=19587a…` used during streaming). If codex's
   transcript builder uses the streaming item_id (which referenced
   the message) instead of the synthesized id, qwen sees inconsistent
   `function_call` ↔ `function_call_output` pairings on subsequent
   turns and progressively loses track.
2. **Voluntary stop on this specific prompt shape.** qwen3.5-27b may
   genuinely emit "I need to do more" then stop with this combination
   of system prompt, tool list, and context. Different from a parser
   bug — a model-tuning gap.

These are differentiable with one more experiment: convert codex's
streaming request to non-streaming upstream (PR #39055 promotion path
handles this correctly), then re-emit synthetic streaming to codex.
If the residual disappears: confirms it's another streaming-related
issue. If the residual persists: confirms model-side voluntary stop.
**Not done in this session** — deferred to a follow-up.

### 14.6 What this changes about the qwen3.5-27b judgment

The §13 framing "qwen3.5-27b explores but doesn't write" was
**premature**. With the proxy synthesis fix in place, the model
clearly executes multiple iterative tool calls. The remaining residual
is much narrower than "model can't do agentic coding" — it's "model
stops on the 5th turn after stating intent to continue."

External evidence the model CAN do agentic coding:

- Qwen's own SWE-Bench reports use an internal scaffold with bash + file-edit tools — different harness, same model, 72.4 score.
- Unsloth's "Run Qwen3.5 Locally With Claude Code" tutorial documents qwen3.5-27b working in Claude Code's loop. Different harness.
- An NVIDIA DGX Spark / GB10 forum thread (same hardware) reports qwen3.5-27b-Claude-4.6-Opus-Reasoning-Distilled (an AWQ variant) working agentically on the same hardware.

So the §11.5/§13.7 recommendation "swap to Qwen3-Coder-30B-A3B" is no
longer the obvious next step. The next step is **debug the §14.5
residual** to learn whether it's a transcript-shape bug (proxy-fixable)
or a model-tuning gap (model-swap warranted).

### 14.7 Updated recommended sequence

1. **Investigate §14.5 residual** — proxy modification to convert
   streaming-in / non-streaming-upstream / synthetic-streaming-out.
   Tests whether qwen3.5-27b emits a tool call when given the same
   conversation through the non-streaming code path. ~30-45 min.
2. **If non-streaming bypass works:** qwen3.5-27b is unblocked.
   Re-baseline v4a, run Round 4b ablation against the working
   harness. The proxy stays as our normalization layer indefinitely;
   pursue upstream vLLM PRs for the underlying serving.py and
   streaming-parser bugs.
3. **If non-streaming bypass also stops short:** investigate further
   (prompt engineering, tool definition shape, max_tool_calls config)
   before concluding model swap is needed.
4. **Wire family graders** — independent, required regardless (§11.5.2).
5. **Loosen acceptance criterion** — independent, required (§11.5.1).

### 14.8 Filing upstream bugs (revised)

- **vLLM `_process_simple_streaming_events` emits `function_call_arguments.delta`
  events with the previous message item's `item_id` and never emits
  `output_item.added` for the function_call.** This is a serving-layer
  bug in `vllm/entrypoints/openai/responses/serving.py`. Reproduction:
  `--model qwen3.5-27b --reasoning-parser qwen3 --tool-call-parser qwen3_xml --reasoning-effort=high`,
  send a streaming request with tools, the model emits a tool call
  preceded by some message text. Patch direction: in the
  `if delta_message.tool_calls[0].function.arguments:` branch, also
  check for `function.name` and emit the message→function_call
  transition events before the args delta. See §13.3 Patch 2 for the
  related `UnboundLocalError` workaround in the same function.
- `reasoning_output_tokens=0` mis-attribution (already filed in §13.8).

### 14.9 Artifacts (this session)

- `src/lumo_flywheel_serving/inference_proxy.py` — proxy synthesis
  patch in `_write_chunked_stream` (gated by track of seen
  function_call ids; synthesizes missing output_item.added +
  arguments_done + output_item.done events before response.completed).
- `/tmp/streaming_test/sse_dump/` — raw SSE blocks from the §13 test
  (before proxy synthesis fix; shows the bug).
- `/tmp/streaming_test/sse_dump2/` — raw SSE blocks from the §14 test
  (after proxy synthesis fix; shows synthesis events).
- `/tmp/streaming_test/tee_proxy.py` — independent tee proxy used
  during debugging (not part of the fix).

### 14.10 Sources

- [vLLM Issue #41182 — Only content before tool call obtained](https://github.com/vllm-project/vllm/issues/41182) — same shape
- [vLLM Issue #27641 — Streaming tool call randomly failed with gpt-oss](https://github.com/vllm-project/vllm/issues/27641) — same family
- [vLLM Issue #10589 — Streaming output error of tool calling not resolved](https://github.com/vllm-project/vllm/issues/10589) — meta-tracking
- [vLLM Issue #31501 — stream-interval > 1 loses tool call arguments](https://github.com/vllm-project/vllm/issues/31501) — ruled out for us (we use default 1)
- [LiteLLM #21090 — Responses API streaming drops function_call events through proxy](https://github.com/BerriAI/litellm/issues/21090) — same diagnostic shape, different vendor
- [Unsloth gist — Run Qwen3.5 with Claude Code](https://gist.github.com/kibotu/a009f00414b7c10fb1c74e603d7838c0) — proof model works agentically with different harness
- [NVIDIA DGX Spark forum — Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-AWQ success](https://forums.developer.nvidia.com/t/success-with-quanttrio-qwen3-5-27b-claude-4-6-opus-reasoning-distilled-v2-awq/365416) — same hardware, working
- [QwenLM/qwen-code — Qwen's own agent CLI](https://github.com/QwenLM/qwen-code) — alternative harness for validation

## 15. Non-streaming bypass attempt (2026-05-12) — blocked by separate vLLM input-validation bug

§14.7 step 1 prescribed a non-streaming bypass to differentiate
"codex-transcript-shape interaction" from "model voluntary stop."
Implemented as `LUMO_PROXY_NONSTREAM_BYPASS=1` in the bench proxy:
when codex sends `stream:true`, rewrite the upstream request to
`stream:false`, buffer the JSON response, then synthesize an SSE
stream from `response.output` back to codex. This sidesteps both
the §13 streaming-event bug AND the §14 missing-output_item.added bug
in one move, because the non-streaming path runs PR #39055's
promotion logic correctly.

### 15.1 Test result — bypass blocked by separate vLLM bug

First two turns succeeded: codex made 2 tool calls (cat prompt + find
workspace), and the model emitted **substantive agent_message text**
for the first time on qwen: `"Now let me explore the workspace
structure and read the necessary files."` (vs `"\n\n"` whitespace
under all previous configs). That's a real signal — the bypass path
is delivering more of the model's output to codex than the
streaming path did.

But the **3rd turn aborted** with a vLLM 400 error containing
**912 "Field required" Pydantic validation errors**. The non-streaming
`/v1/responses` route runs much stricter input validation than the
streaming route:

```
event: response.created
data: ... {"error":{"message":"212 validation errors:
  {'type': 'string_type', 'loc': ('body', 'input', 'str'),
   'msg': 'Input should be a valid string',
   'input': [{'type': 'message', 'role': 'developer',
              'content': [{'type': 'input_text', ...}]}]}
... 912 'Field required' errors across other union variants
```

The streaming route accepts codex's transcript shape (mixed
developer/user role messages with structured content). The
non-streaming route's Pydantic validation tries to match each input
item against many union types (message, function_call,
function_call_output, reasoning, custom_tool_call, etc.) and rejects
when no variant matches all required fields.

This is a **separate vLLM bug**: the same input shape should be valid
on both streaming and non-streaming routes. The asymmetric validation
is the bug. Filing as a follow-up upstream issue.

### 15.2 What we still learned

Even though the bypass test aborted on turn 3, the 2 successful turns
gave us new evidence: **with the non-streaming promotion path
delivering the model's full output, qwen3.5-27b emits substantive text
content** (the "Now let me explore the workspace..." message). That's
a behavioral signal that this model does engage with the task — it
wasn't engaging before because the streaming path was dropping its
output.

So the §14.5 residual ("model voluntarily stops on turn 5") looks
**more likely to be a streaming-protocol artifact** than a model-side
choice. We just can't fully verify with the bypass approach because
vLLM's non-streaming validation rejects codex's transcript on later
turns.

### 15.3 What to do instead — alternative paths to validate the model

Three paths in priority order:

1. **Fix vLLM's non-streaming input validation upstream** (or write a
   third proxy patch that normalizes codex's transcript to a shape the
   non-streaming validator accepts). This is significant proxy work
   — input-side normalization to handle the asymmetry between
   streaming and non-streaming validators. Probably needs
   role-mapping, content-shape coercion, missing-field defaults.
   Several days of work.
2. **Run qwen3.5-27b through a different harness** (Claude Code per
   Unsloth's gist, or QwenLM's own `qwen-code`). Different harness =
   different request shape — may not trigger either of the vLLM bugs
   we've found. Validates the model's autonomous-coding capability
   independent of codex CLI. ~1-2 hours.
3. **Try Qwen3-Coder-30B-A3B** (the agentic-tuned MoE variant).
   Different model, possibly emits tool calls in a way that doesn't
   trigger the streaming bugs (e.g., no `<tool_call>` embedding inside
   `<think>`). ~30-45 min on this hardware.

The §14 proxy synthesis fix should stay in place either way — it's a
real bug at the parser layer and the workaround is mechanical and
correct.

### 15.4 Current state of the bench proxy

After this session, the bench proxy at `127.0.0.1:8022` is running
**without** `LUMO_PROXY_NONSTREAM_BYPASS` set. The §14 synthesis fix
is active (gated by detecting unannounced function_calls in
`response.completed.output`). To re-enable bypass in the future:

```bash
LUMO_PROXY_NONSTREAM_BYPASS=1 .venv/bin/python -m lumo_flywheel_serving.inference_proxy ...
```

The bypass code path is preserved in `inference_proxy.py` for future
use once the input-validation asymmetry is addressed.

### 15.5 Source

- This is novel finding for vLLM; no upstream issue exists yet that
  exactly matches the streaming-vs-nonstream input-validation
  asymmetry. Worth filing.

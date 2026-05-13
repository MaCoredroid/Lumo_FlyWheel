# Track B v4a_v2 — preserve_thinking smoke test (2026-05-13)

## Question

Per operator research, the upstream Qwen3.x chat-template bug
(`QwenLM/Qwen3.6 #131`, `badlogic/pi-mono #3325`, `QwenLM/Qwen3 #1831`)
drops historical `<think>` blocks across tool turns, causing the model
to emit empty turns and forcing harness-level workarounds. The proposed
fix: enable `preserve_thinking=true` and disable our auto-continue
workaround. Hypothesis: model self-stops cleanly (like Claude Code /
Codex pattern B) and grader still passes.

## Setup

| Component | State |
|---|---|
| Chat template | `docker/chat_templates/qwen3-openai-codex.jinja` modified to preserve `<think>` for **every** historical assistant turn (removed the `loop.index0 > ns.last_user_index` boundary at line 144). Backup at `.pre-preserve-thinking-20260513T060519Z.bak`. |
| vLLM | container `lumo-vllm-track-b-suffix` recreated 3× to refresh the bind-mounted template (inode pinning required `docker rm -f` + relaunch via `/tmp/relaunch_track_b.py`, not just `docker restart`) |
| Proxy | port 8022, `LUMO_PROXY_NONSTREAM_BYPASS=1`, **`LUMO_PROXY_AUTO_CONTINUE=0`** |
| Task | `sqlalchemy-2-session-modernization/v1-clean-baseline` (the known-pass task — 4/4 with P=85 under the current stack) |
| Codex | inside docker container, `--dangerously-bypass-approvals-and-sandbox`, normal model_reasoning_effort=high config, 1800 s budget |

## Result

```
$(date -u +%H:%M:%S) start  →  76 s elapsed at exit
turn.completed=1  item.completed=1  command_exec=1  agent_msg=0  error=0
output_tokens=87 (final turn)  reasoning_output_tokens=0
```

Codex log (full):

```
{"type":"thread.started",...}
{"type":"turn.started"}
{"type":"item.started","item":{"id":"item_0","type":"command_execution",
  "command":"/usr/bin/bash -lc 'find /workspace -type f -name \"*.py\" | head -30'"}}
{"type":"item.completed","item":{...command result: list of 9 .py files...}}
{"type":"turn.completed","usage":{"input_tokens":21635,"cached_input_tokens":10176,
  "output_tokens":87,"reasoning_output_tokens":0}}
```

Workspace mutations after smoke: **none**. All files have mtimes
matching the initial `shutil.copytree` (Apr 23 2026 — when the
benchmark bundle was last touched).

Grader result:

```
M_aggregate = 0.20   (vs 0.50 under current stack)
P_benchmark = 0      (vs mean 85 under current stack)
integrity_flag = 0
ceilings_applied = ['visible_only', 'helper_commit_shortcut',
                    'worker_partial_write', 'dry_run_persists']
```

## Interpretation

The model — given the file listing — decides the task is done. One tool
call, 87 output tokens of natural-language closeout, no edits to any
source file. Grader fires four ceilings because the workspace shows
none of the expected migration artifacts.

This is **the agent self-stopping too early**, not the agent being
silenced by a missing context fix. Whether `<think>` survives across
tool turns is orthogonal to this behavior — the model in this case
isn't producing reasoning that gets dropped, it's just declining to
do more work.

## Comparison

| | Current stack (auto-continue on) | preserve_thinking + auto-continue off |
|---|---|---|
| Tool calls per attempt | 100+ | **1** |
| `turn.completed` events | Often missing (codex SIGKILL'd at 1800s) | Cleanly emitted after 1 turn |
| Elapsed | ~1800 s (budget) | **76 s** |
| Workspace edits | Real (db.py, repository.py, models.py modified) | **None** |
| Grader P_benchmark (sqlalchemy) | mean **85** (91/79/79/91 across 4 attempts) | **0** |
| Pass rate (sqlalchemy) | **4/4** | **0/1** |

## Conclusion

**Auto-continue is load-bearing.** It's not papering over the upstream
Qwen3.x `<think>`-drop bug — it's pushing the model past its premature
self-stop tendency. Without it, the model declares completion after
one exploratory tool call regardless of whether reasoning context was
preserved.

The upstream-bug-fix path the operator's research suggested
(`preserve_thinking=true` + remove auto-continue) doesn't apply to our
stack:

1. Our chat template **already** has interleaved-thinking preservation
   logic (M2.5-style: keeps `<think>` for assistant turns after the
   last real user query — which, in Codex's tool-result-as-user
   pattern, means all turns since the operator's initial query).
2. The proxy's `_normalize_input_for_nonstreaming` preserves `type:
   reasoning` items in the input list — the proxy isn't dropping them
   either.
3. Even with **unconditional** thinking preservation (this test), the
   model self-stops in 1 turn.

The model's self-stop behavior is a Qwen3.5-27B-on-Codex tendency,
not a template bug.

## Implications

- Keep `LUMO_PROXY_AUTO_CONTINUE=1` in production.
- For the **ablation metric**, the operator's suggestion of
  *time-to-first-grader-pass* (SWE-Bench environment-truth pattern A)
  still applies regardless of which side of the auto-continue debate
  you fall on — it sidesteps both the "model self-stops too early"
  and the "harness forces budget timeout" failure modes by treating
  the workspace's grader-pass moment as the source of truth.
- The chat-template `<think>` plumbing in our stack is correct; future
  attention should focus on prompt-engineering / model-finetuning to
  reduce the model's premature self-stop tendency, not on
  context-preservation knobs.

## Sources for the research

The operator's research drew from:
- [Claude Code agent-loop docs](https://code.claude.com/docs/en/agent-sdk/agent-loop)
- [SWE-Bench Verified](https://www.swebench.com/verified.html)
- [QwenLM/Qwen3.6 #131](https://github.com/QwenLM/Qwen3.6/issues/131)
- [badlogic/pi-mono #3325](https://github.com/badlogic/pi-mono/issues/3325)
- [QwenLM/Qwen3 #1831](https://github.com/QwenLM/Qwen3/issues/1831)

Those bug reports describe `<think>`-drop in stock Qwen3 templates.
Our custom `qwen3-openai-codex.jinja` (lines 109-118 + 133-148)
implements interleaved-thinking preservation as part of its design;
the stock-template bug doesn't apply.

## Artifacts

- `/tmp/preserve_thinking_smoke/sqlalchemy/workspace/` — the
  smoke workspace at exit (one bash command executed, no edits)
- `/tmp/preserve_thinking_smoke/codex.log` — codex JSONL output
  (3 items)
- `/tmp/preserve_thinking_smoke/grader_result.json` — verifier
  output (P_benchmark=0, 4 ceilings)
- `docker/chat_templates/qwen3-openai-codex.jinja.pre-preserve-thinking-
  20260513T060519Z.bak` — template backup (the reverted version is
  again at `qwen3-openai-codex.jinja`)

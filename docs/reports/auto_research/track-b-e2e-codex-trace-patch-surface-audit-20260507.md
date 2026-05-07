# Track B E2E Codex Trace Patch Surface Audit

Generated: 2026-05-07

Source inspected:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/openai/codex.git /tmp/openai-codex-trace-inspect
cd /tmp/openai-codex-trace-inspect
git sparse-checkout set codex-cli codex-rs
git rev-parse HEAD
```

Inspected upstream commit: `163eac9306e86b38c5ab3986eefd5fd3be616b06`

Installed local CLI:

```bash
codex --version
codex exec --help
```

Observed local version: `codex-cli 0.128.0`

## Finding

A wrapper-only patch to `/usr/lib/node_modules/@openai/codex/bin/codex.js` is insufficient for the plan's `--trace-out` requirement. That JavaScript entry point only selects and spawns the native platform binary. The relevant turn lifecycle and Responses streaming state live in the Rust workspace under `codex-rs/`.

## Relevant Patch Points

CLI flag:

- `codex-rs/exec/src/cli.rs`
  - `Cli` already owns `--json` and `--output-last-message`.
  - Add `--trace-out <PATH>` here.

Exec session wiring:

- `codex-rs/exec/src/lib.rs`
  - `run_main()` destructures `Cli`; thread the trace path into `ExecRunArgs`.
  - `run_exec_session()` creates the event processor and receives `ServerNotification`s.
  - Existing `TurnStartedNotification` / `TurnCompleted` notifications can emit coarse `turn_start` / `turn_end`, but they do not by themselves expose the upstream model request id needed for vLLM joins.

Streaming request hook:

- `codex-rs/core/src/client.rs`
  - `ModelClientSession::stream_responses_api()` builds and sends the `/responses` request.
  - `map_response_stream()` receives `upstream_request_id` from `codex_api::ResponseStream`.
  - `map_response_events()` has access to `ResponseEvent::Completed { response_id, token_usage, .. }` and records upstream request ids for feedback/inference tracing.

Existing metadata path:

- `codex-rs/core/src/turn_metadata.rs`
  - `TurnMetadataBag` already includes `turn_id`.
  - `build_responses_headers()` sends `x-codex-turn-metadata` when present.
  - This is useful context, but the current live vLLM `/metrics` scrape does not expose labels for `request_id`, `vllm_request_id`, or `turn_id`.

JSON event path:

- `codex-rs/exec/src/event_processor_with_jsonl_output.rs`
  - `--json` emits `ThreadEvent` rows from app-server notifications.
  - This is useful for an adapter, but it is not equivalent to the plan's trace emitter because it lacks the per-request vLLM join key.

## Patch Shape Needed

The minimum truthful patch is not just a new CLI flag. It needs a trace sink reachable from both:

1. `exec/src/lib.rs`, for task/session/turn lifecycle events.
2. `core/src/client.rs`, for the upstream request id and token usage as the Responses stream completes.

The trace sink must write JSONL without changing normal event processing, stdout/stderr behavior, or token/tool outputs. The correctness artifact at `output/track_b_e2e/codex_trace_emitter_correctness.json` must still compare `--trace-out` enabled and disabled transcripts byte-for-byte before Round 0 can run.

## Current Blocker

No `--trace-out` patch artifact is landed yet. Round 0 remains blocked even though the patch surface is now identified. The next implementation step is to create a fork/patch under `patches/codex/` or `vendor/codex-cli/patches/` against the inspected Rust files, build the native binary, and run the byte-equality verification from the plan's §4.3.

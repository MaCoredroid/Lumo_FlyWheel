# Codex CLI Timeout Configuration (Track B Round 4a)

Generated: 2026-05-10
Codex CLI version: 0.128.0

## Why this exists

Track B Round 3 sweep observed 65 % of runs (34/52) exiting with the
"zero-token" quirk: codex sends `POST /v1/responses`, vLLM spends ~90 s
on cold prefill, codex aborts client-side before the first SSE chunk
arrives. The abort surfaces in `codex_stdout.log` as
`turn.completed{usage:{output_tokens:0}}`. This runbook documents the
codex-side timeout knob that gates SSE chunk arrival, the discovered
default, and the explicit configuration we use to remove it as a
suspect.

## The knob

**`stream_idle_timeout_ms`** under the per-provider TOML stanza
`[model_providers.<name>]`. This is the only SSE-relevant timeout in
Codex CLI 0.128.0.

- It is an *idle* timeout (gap between bytes), not a wall-clock
  request timeout. A 90-second cold prefill that emits zero bytes
  counts as "idle" for the full 90 s, so this clock gates first-chunk
  arrival as well as inter-chunk gaps.
- The reqwest HTTP client itself
  (`codex-rs/login/src/auth/default_client.rs:203 build_reqwest_client`)
  sets neither `.timeout()` nor `.connect_timeout()` — there is no
  global request timeout and TCP connect uses the OS default
  (~75 s on Linux).
- There is **no CLI flag** for any timeout. `codex exec --help` and
  `codex --help` confirm this — the only override route is
  `-c key=value`.
- There is **no env var** for this timeout. The `CODEX_*` env vars
  in the binary cover AWS IMDS, websocket pong, and tracing only.

### Default

`DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000` (300 s) per
`codex-rs/model-provider-info/src/lib.rs:26`.

So out-of-the-box, an idle SSE stream takes 5 min to time out.

### Implication for the zero-token quirk

If the default is 300 s and the observed cold prefill is ~90 s, then
`stream_idle_timeout_ms` is **not** what's firing. The actual cause of
the zero-token aborts is something else (candidates: vLLM closing the
SSE stream cleanly with zero tokens; an upstream proxy reset;
`stream_max_retries` interaction with a transient first-byte hiccup).
Bumping `stream_idle_timeout_ms` past 300 s will not affect the
zero-token rate.

The Round 4a warmup-pass eliminates the underlying cold prefill, which
removes the conditions under which the zero-token quirk has been
observed. The timeout setting below is therefore **defense-in-depth
documentation**, not a behavioral fix.

## What we set explicitly

Per Round 4a §7.3, we set `stream_idle_timeout_ms = 300000` explicitly
in the codex command template the runner passes. This makes the
configuration visible and audit-friendly rather than implicit.

Concrete addition to the codex command template used by
`scripts/run_track_b_e2e_round.py` and
`scripts/run_track_b_e2e_task.py`:

```bash
codex exec --json --skip-git-repo-check \
  -C {workspace} \
  -c 'model_provider="local-proxy"' \
  -c 'model_providers.local-proxy={name="local-proxy",base_url="{endpoint}",env_key="OPENAI_API_KEY",wire_api="responses",stream_idle_timeout_ms=300000}' \
  --model {model} \
  "Read the task prompt at {prompt_file} and complete it in this workspace."
```

The new key is the embedded `stream_idle_timeout_ms=300000` inside the
`local-proxy` provider definition.

## Sibling knobs (not used)

- `request_max_retries` — default 4. Governs full-request retry on
  transient HTTP errors. Not relevant here; we want a single attempt
  for measurement integrity.
- `stream_max_retries` — default 5, capped at 100. Governs reconnect
  on dropped streams. A zero-token completion that closes cleanly
  surfaces as `"stream closed before response.completed"` rather than
  as a stream drop, so retries do not paper over it.
- `websocket_connect_timeout_ms` — default 15 000. WebSocket wire-API
  only. Track B uses `wire_api = "responses"` (HTTP/SSE), so this is
  not on the path.

## Existing repo precedent

`src/lumo_flywheel_serving/task_orchestrator.py:937–944` writes
`stream_idle_timeout_ms = 600000` (10 min) — a precedent for explicit
provider-level timeout configuration. Round 4a uses 300 000 (5 min)
to match the codex default, surfacing it without drifting upward.

## Verification

After the runner change lands, verify the override took effect by
inspecting one captured `/v1/responses` request body and checking the
codex client's `[model_providers.local-proxy]` config — or by
intentionally introducing a slow upstream and confirming the codex
abort message reads "idle timeout waiting for SSE" with a wall time
near 300 s rather than the default 300 s. (For Round 4a we did not
implement an active probe; the explicit setting is the audit
evidence.)

## References

- Codex 0.128.0 binary: `/usr/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-arm64/vendor/aarch64-unknown-linux-musl/codex/codex`
- Default declaration: `codex-rs/model-provider-info/src/lib.rs:26`
- Enforcement: `codex-rs/codex-api/src/sse/responses.rs:465`
- HTTP client init: `codex-rs/login/src/auth/default_client.rs:203`
- Repo precedent: `src/lumo_flywheel_serving/task_orchestrator.py:937–944`
- Round 4a spec §7.3: `docs/reports/auto_research/track-b-e2e-round4a-measurement-protocol-spec-20260510.md`

# Rollout history — respproxy, Q1 + Q2 2026

## 2026-01-15 — SDK compat incident

Client upgrade cycle broke when we relaxed a tool-schema validation
error code from `400` to `422`. Clients in three downstream teams had
wired the `400` into retry logic. Recovery took 5 days. Learning: do
not quietly relax SDK-contract error codes even if strictly more correct.

## 2026-02-04 — Hot-path refactor merged

Anja's normalizer refactor landed clean. First attempt at shared
compiled-validator path was reverted when it interacted badly with a
mid-flight skill-pack swap; the revert is what motivates P4's warm-start
design now.

## 2026-03-11 — FP8 quantization landed in transformer-runtime

Downstream effect: model_forward on the tool-call path dropped from 95ms
to 42ms. The older profiles that show 95ms are now stale.

## 2026-03-20 — Q2 planning close-out

Respproxy team is committed to (a) latency reduction on the tool-call
path (this objective) and (b) streaming-reliability backlog. (b) was
downgraded after the incident wave stabilized. (b) remains on the
backlog but is not in Q3 top-priority.

## Operating principles adopted post-Jan 15

- Rollouts behind per-route flags.
- No SDK contract changes without a 1-release deprecation window.
- Observability must land *before* the behavior change in any
  latency-reducing rollout.

# Responses Cutover

The local adapter still mirrors the legacy Chat Completions contract.

- Inspect `function_call.arguments` to recover tool arguments.
- Treat repeated tool invocations as one logical tool step in snapshots.
- `response.completed` is only a footer marker; no per-item replay contract exists.

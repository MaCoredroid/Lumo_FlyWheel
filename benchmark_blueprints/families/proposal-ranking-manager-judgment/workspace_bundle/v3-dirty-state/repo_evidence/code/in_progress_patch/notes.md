# Rust rewrite skeleton — handoff notes

Diego, 2026-02-27 (last update before pulled onto quantization release).

## Status

- `hot_path.rs`: port started. Streaming response path not begun.
- `skillpack_bridge` module: empty. Need FFI design before implementation.
- Tests: not ported.
- CI: not wired.

## Why it stopped

Quantization release. Expected to return after Q3, but my team lead
made clear I should not block on Q4 either. Someone on the respproxy
team would need to drive this; nobody on that team writes Rust today.

## Recommendation to whoever picks it up later

Do not treat this as "halfway there." The easier 20% is on disk. The
hard 80% (skill-pack binding, streaming, rollout safety) is unstarted.
If you are under time pressure, P4 (Priya's schema-cache proposal)
captures most of the latency wins this service needs without the
ecosystem risk. Fine-tune the Python hot path first; reach for Rust
only when the remaining latency wedge justifies the staffing cost.

Diego.

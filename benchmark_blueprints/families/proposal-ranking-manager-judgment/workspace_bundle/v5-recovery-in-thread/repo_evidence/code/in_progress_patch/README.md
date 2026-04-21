# In-progress patch — `respproxy-rs` skeleton

**Author (original):** Diego
**Started:** 2026-02-18
**Status:** **ABANDONED.** Diego was pulled onto the quantization release
on 2026-03-01 and has not returned to this work. The skeleton was checked
in as a speculative artifact; no follow-up was planned.

## What's here

- `Cargo.toml` — dependency manifest, does not compile as-is.
- `src/lib.rs` — 120 lines, establishing the service surface. **Does not
  compile**: the `skillpack_bridge` module is referenced but empty.
- `src/hot_path.rs` — 180 lines, partial port of the Python hot path.
  Tokio runtime wiring is stubbed with TODOs.
- `notes.md` — Diego's handoff notes, enumerating the work remaining.

## Estimated remaining work (from notes.md)

- Fill in `skillpack_bridge`: ~4 weeks (FFI design).
- Complete `hot_path.rs`: ~3 weeks (streaming, cancellation, retries).
- Rebuild the test suite in Rust: ~3 weeks.
- Production rollout harness parity: ~2 weeks.

Total estimated remaining: **~12 weeks of dedicated Rust-capable work.**
The respproxy team has no Rust specialist (see `meta/staffing.md`).

## Manager's lens

This directory is tempting but is not a "half done" asset — it is a
speculative prototype that was never staffable without Diego. Completing
it would require the same staffing decision that P3 (Rust rewrite)
originally required, and that decision is unchanged.

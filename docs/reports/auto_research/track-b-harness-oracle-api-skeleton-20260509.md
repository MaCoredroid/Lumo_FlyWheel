# Track B Round 2 — harness oracle API skeleton

**Date:** 2026-05-09
**Status:** design + skeleton (Step 3 of the harness-coupled spec).
**Scope:** the API contract + module boundaries + stub
implementations. Wiring + technique-side proposers (Steps 4-9) are
out of scope for this doc.

## Why this exists

The harness-coupled spec
(`codex-harness-spec-decode-engineering-20260507.md`) calls the
harness oracle API the "load-bearing piece" — every technique
(cross-turn ngram, read_file priming, schema-aware tool drafter,
plan-structure pre-drafting, lifecycle) reads from it. Without a
clean API there's no way to wire any technique without each
re-implementing harness coupling.

This doc nails the contract so Steps 4-9 each have a stable
surface to integrate against.

## API contract

Single per-session object, carried alongside the existing vLLM
`Request` envelope. All fields are optional; absence means "no
harness signal for this technique."

```python
# vllm/v1/spec_decode/harness_oracle.py  (new module)

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HarnessOracleSnapshot:
    """Per-request harness signals consumed by the drafter coordinator.

    Designed to be a passive snapshot: the request envelope carries it,
    the drafter reads it, the harness produces it. No bidirectional
    communication. This keeps the API trivial to add to and trivial
    to make optional (every field defaults to None / empty)."""

    # ---- Identity & lifecycle (Technique 5) ----
    session_id: str | None = None
    """Stable agent-task identifier. The harness should keep this
    constant across all turns of one agent task. Drafter uses it
    to scope per-session state (suffix tree, plan registry, priming
    buffer)."""

    turn_index: int | None = None
    """0-based turn counter within the session. Increments on each
    /v1/responses or /v1/chat/completions call. Used by Technique 5
    to bound within-turn ephemeral state."""

    is_session_open: bool = False
    """First turn of a session sets this True. Drafter allocates fresh
    state. Subsequent turns leave it False."""

    is_session_close: bool = False
    """Last turn of a session sets this True. Drafter frees state."""

    # ---- Tool-call signals (Technique 3) ----
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    """JSON-schema dicts for tools available in this session. Set
    once per session-open; same value reused on every turn until
    session-close. Drafter uses these to drive forced-token
    drafting under XGrammar-2's traverse_draft_tree."""

    expected_tool_call: dict[str, Any] | None = None
    """When the harness knows the upcoming tool call (forced
    tool_choice, single-option auto), the schema-aware drafter can
    pre-fill the function-name token region with confidence 1.0.
    Format: {"name": "...", "schema": {...sub-schema...}}."""

    # ---- Read-file priming (Technique 2) ----
    primed_texts: list[dict[str, Any]] = field(default_factory=list)
    """Out-of-prompt text the harness primes the drafter with. Each
    entry: {
        "text": str,
        "source_tag": str,    # "file:<path>" / "tool:<name>:<call_id>"
        "ttl_turns": int,     # eviction after N more turns
        "max_chars": int,     # cache-poisoning bound (e.g., 65536)
    }
    Drafter folds primed text into the session's suffix tree with
    the source_tag as metadata for provenance scoring."""

    # ---- Plan-structure (Technique 4) ----
    plan_fingerprint: dict[str, Any] | None = None
    """Structural skeleton of a recently-emitted agent plan. Format:
    {
        "structure_tokens": [int, ...],   # numbered list / headers / separators
        "first_emission_turn": int,
        "emission_count": int,            # how many times this plan
                                          # has been re-emitted in the session
    }
    Drafter gates plan-structure pre-drafting on emission_count >= 3
    (per the spec's false-positive mitigation)."""

    # ---- Cross-turn ngram (Technique 1) ----
    suffix_tree_cap_mb: int | None = None
    """Per-session memory budget for the cross-turn suffix tree.
    Defaults to 100 MB if absent. Drafter LRU-evicts within this
    budget on session writes."""

    # ---- Diagnostics ----
    schema_version: str = "lumo.track_b.harness_oracle.v1"


# Helper for the request envelope side -- the harness emits this:

def encode_oracle_for_request(snapshot: HarnessOracleSnapshot) -> dict[str, Any]:
    """Serialize for inclusion in the vLLM request as a vLLM-extension
    field. Picked field-by-field so future fields can be added without
    breaking older drafters."""
    out: dict[str, Any] = {
        "schema_version": snapshot.schema_version,
    }
    if snapshot.session_id is not None: out["session_id"] = snapshot.session_id
    if snapshot.turn_index is not None: out["turn_index"] = snapshot.turn_index
    if snapshot.is_session_open: out["is_session_open"] = True
    if snapshot.is_session_close: out["is_session_close"] = True
    if snapshot.tool_schemas: out["tool_schemas"] = snapshot.tool_schemas
    if snapshot.expected_tool_call is not None:
        out["expected_tool_call"] = snapshot.expected_tool_call
    if snapshot.primed_texts: out["primed_texts"] = snapshot.primed_texts
    if snapshot.plan_fingerprint is not None:
        out["plan_fingerprint"] = snapshot.plan_fingerprint
    if snapshot.suffix_tree_cap_mb is not None:
        out["suffix_tree_cap_mb"] = snapshot.suffix_tree_cap_mb
    return out


def decode_oracle_from_request(request: Any) -> HarnessOracleSnapshot:
    """Pull the oracle snapshot out of the request envelope. Tolerant
    of missing field; absence -> default."""
    raw = getattr(request, "lumo_harness_oracle", None) or {}
    if not isinstance(raw, dict):
        return HarnessOracleSnapshot()
    return HarnessOracleSnapshot(
        session_id=raw.get("session_id"),
        turn_index=raw.get("turn_index"),
        is_session_open=bool(raw.get("is_session_open", False)),
        is_session_close=bool(raw.get("is_session_close", False)),
        tool_schemas=raw.get("tool_schemas") or [],
        expected_tool_call=raw.get("expected_tool_call"),
        primed_texts=raw.get("primed_texts") or [],
        plan_fingerprint=raw.get("plan_fingerprint"),
        suffix_tree_cap_mb=raw.get("suffix_tree_cap_mb"),
    )
```

## Module boundaries

```
vllm/v1/spec_decode/
    harness_oracle.py        # this doc — passive snapshot type
    drafter_coordinator.py   # existing — extended to consume snapshot
    ngram_proposer.py        # existing (SuffixDecoding inherits)
                             # — extended for session_id (Technique 1)
    schema_aware_proposer.py # NEW — Technique 3
    plan_structure_proposer.py # NEW — Technique 4
    priming_buffer.py        # NEW — Technique 2
    lifecycle.py             # NEW — Technique 5

vllm/entrypoints/openai/
    protocol.py              # existing — add `lumo_harness_oracle`
                             # extension field to the request shape
```

The drafter coordinator's read sequence per turn:

```python
# vllm/v1/spec_decode/drafter_coordinator.py (extension)

oracle = decode_oracle_from_request(request)

if oracle.is_session_open:
    self.lifecycle.session_open(oracle.session_id, oracle.suffix_tree_cap_mb)

# Technique 5: per-turn open
if oracle.session_id and oracle.turn_index is not None:
    self.lifecycle.turn_open(oracle.session_id, oracle.turn_index)

# Technique 1: route ngram lookups to this session's suffix tree
proposer.set_session_context(oracle.session_id)

# Technique 2: ingest primed texts
for primed in oracle.primed_texts:
    self.priming_buffer.ingest(
        oracle.session_id,
        primed["text"],
        primed["source_tag"],
        primed.get("ttl_turns", 32),
    )

# Technique 3: tell the schema-aware proposer what's coming
if oracle.tool_schemas:
    self.schema_aware_proposer.set_tool_schemas(
        oracle.session_id, oracle.tool_schemas
    )
if oracle.expected_tool_call:
    self.schema_aware_proposer.expect(oracle.expected_tool_call)

# Technique 4: pre-draft plan structure
if oracle.plan_fingerprint and oracle.plan_fingerprint.get("emission_count", 0) >= 3:
    self.plan_structure_proposer.prime(
        oracle.session_id, oracle.plan_fingerprint["structure_tokens"]
    )

# (decode happens normally; proposers consult their per-session state)

if oracle.is_session_close:
    self.lifecycle.session_close(oracle.session_id)
```

## Codex harness side

The Codex CLI emits the snapshot. Minimal patch:

```python
# Codex's `responses` provider request builder (Rust; pseudocode)

let oracle = HarnessOracle {
    session_id: task.id.clone(),
    turn_index: turn_counter,
    is_session_open: turn_counter == 0,
    is_session_close: false,  // set true on the closing summary turn
    tool_schemas: tools.iter().map(|t| t.schema()).collect(),
    expected_tool_call: forced_tool_choice.map(|f| ExpectedToolCall {
        name: f.name.clone(),
        schema: tools.find(|t| t.name == f.name).map(|t| t.schema().clone()),
    }),
    primed_texts: read_file_results.iter().map(|r| Primed {
        text: r.content.clone(),
        source_tag: format!("file:{}", r.path),
        ttl_turns: 32,
        max_chars: 65_536,
    }).collect(),
    plan_fingerprint: detected_plan_emissions.last().map(|p| p.fingerprint()),
    suffix_tree_cap_mb: Some(100),
};
request.lumo_harness_oracle = Some(oracle.serialize());
```

Codex doesn't natively emit any of these signals today; this is
the harness-side patch space. The fields are all optional, so the
oracle can be added incrementally — start with `session_id` +
`turn_index` (Technique 1 + 5), add `tool_schemas` later
(Technique 3), and so on.

## Phasing

Suggested order to wire the API + first technique:

1. Land `harness_oracle.py` skeleton in vLLM (this doc's API).
   No drafter consumption yet — pure addition. Tests: snapshot
   round-trip serialization, default semantics.
2. Add the `lumo_harness_oracle` extension field to the vLLM
   request protocol. Codex doesn't emit it yet; absence path is
   exercised.
3. Patch the Codex `responses` provider to emit `session_id` +
   `turn_index` only. Expensive only because Codex is Rust;
   functionally trivial.
4. Wire Technique 1's session-scoped suffix tree using the
   oracle's `session_id`. Now the API has a real consumer.
5. Repeat 3-4 for Techniques 2/3/4/5 in priority order.

The v2 spec recalibration (89% tool-call / 11% reasoning) implies
priority order: Technique 3 (schema-aware tool drafter) ahead of
Technique 2 (read_file priming), since 89% of turns are tool-call
shape. Technique 4 (plan-structure) is small in absolute share
but cheap to implement on top of Technique 3.

## Why this is just the skeleton

Implementing the API is one file (~150 LoC). Implementing the
techniques behind it is the multi-week piece — each technique
needs:
- A vLLM-side proposer module
- Tests against pre-recorded prompts
- Per-technique microbenchmark (Track 1 of the measurement plan)
- Integration with the drafter coordinator's existing scheduling

The skeleton documented here unblocks parallel work: once vLLM
has the API plus protocol field, technique-team A can wire
Technique 1 while technique-team B works on Technique 3 without
fighting over the integration surface.

## Blocking constraints

- **vLLM source mod**: changes to `vllm/v1/spec_decode/` and
  `vllm/entrypoints/openai/protocol.py` require a vLLM rebuild.
  The Lumo flow has a vllm-source workspace at
  `/opt/vllm-source` inside the container, but rebuilding vLLM
  on this hardware (GB10 ARM + CUDA 13) is a 30-60 min operation
  per change.
- **Codex CLI source mod**: the Codex 0.128.0 binary on this host
  is the prebuilt `@openai/codex-linux-arm64` wheel. Patching it
  requires either a Codex source build or a wrapper that injects
  fields into the request before forwarding (similar to our
  inference proxy). The proxy approach worked for trace-correctness
  + per-request metrics; it should work for the harness oracle
  too — the proxy synthesizes session_id from the inbound
  conversation thread_id and turn_index from a counter per
  thread_id.
- **Operator-time**: Step 0e shipped Round 1 today; further vLLM
  rebuilds + Codex experiments are operator-paced.

## Recommendation

Land the `harness_oracle.py` module + protocol extension in vLLM
as commit #1 of Round 2. That's a no-op change (zero behavioral
delta when the field is absent) but it gates every technique that
follows.

Until that lands, **proxy-side synthesis** of `session_id` +
`turn_index` (the same approach we used for trace correctness)
is the fastest path to a real-data measurement: the proxy injects
the oracle into Codex requests; the patched vLLM consumes it. Two
edits, no Codex rebuild.

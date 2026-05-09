"""T3 schema-aware tool drafter — pure-Python decision core.

Given a ``HarnessOracleSnapshot`` (tool_schemas + optional
``expected_tool_call``) and the model's recent decoded text, return
the next contiguous *deterministic* text chunk the model should emit
if it is following the schema. This is the structural-prefill region
where the schema-aware drafter has confidence ~1.0 — exactly the
slot SuffixDecoding's content-statistics drafting does poorly on
("the function name slot is novel; suffix tree has no match").

Tokenizer-free: returns raw text. The vLLM-side integrator (when it
ships) will encode the proposal into model tokens and submit to the
spec_decode coordinator. Ship this layer first because it's the part
worth getting right via fixtures + corpus testing.

Dialect support today:
- ``codex`` — Qwen3 XML format used by the Codex CLI:
  ``<tool_call><name>NAME</name><arguments>JSON</arguments></tool_call>``.
- ``openai`` — generic Responses-API JSON; fewer deterministic slots
  because the wire format is shaped by vLLM's tool parser.

Confidence model: for each proposal we attach a 0..1 score the
integrator can compare against the suffix-decoding drafter's score
when picking which proposal to commit. Schema-driven proposals start
at 1.0 and decay as we move further from a known anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vllm_harness_oracle import HarnessOracleSnapshot

CODEX_TOOL_CALL_OPEN = "<tool_call>"
CODEX_NAME_OPEN = "<name>"
CODEX_NAME_CLOSE = "</name>"
CODEX_ARGS_OPEN = "<arguments>"
CODEX_ARGS_CLOSE = "</arguments>"
CODEX_TOOL_CALL_CLOSE = "</tool_call>"


@dataclass
class DraftProposal:
    """A schema-driven continuation the integrator may forward as a
    speculative token sequence."""

    text: str
    confidence: float
    reason: str


def propose(
    snapshot: HarnessOracleSnapshot, recent_text: str
) -> DraftProposal | None:
    """Top-level dispatch on ``snapshot.dialect``.

    Returns ``None`` when no schema-driven proposal is possible
    (unknown dialect, no expected_tool_call, schema-drafter exhausted
    its anchors). Caller should then fall through to SuffixDecoding.
    """

    if snapshot.is_empty:
        return None
    if snapshot.dialect == "codex":
        return _propose_codex(snapshot, recent_text)
    if snapshot.dialect == "openai":
        return _propose_openai_json(snapshot, recent_text)
    return None


def _propose_codex(
    snapshot: HarnessOracleSnapshot, recent_text: str
) -> DraftProposal | None:
    """Qwen3-XML dialect drafter.

    Three deterministic anchor regions in order:
    1. After ``<tool_call><name>`` — emit the forced function name +
       ``</name><arguments>{``.
    2. After ``<arguments>{`` — emit the first required-property
       opener: ``"<prop>":``.
    3. After ``"<prop>":`` for a string-typed prop — emit the opening
       quote ``"`` (one char only; arg content is model-driven).
    """

    expected = snapshot.expected_tool_call
    if not isinstance(expected, dict):
        return None
    forced_name = expected.get("name")
    if not isinstance(forced_name, str):
        return None
    schema = expected.get("schema") if isinstance(expected.get("schema"), dict) else None

    last_open = recent_text.rfind(CODEX_TOOL_CALL_OPEN)
    if last_open < 0:
        return None
    suffix = recent_text[last_open:]
    if CODEX_TOOL_CALL_CLOSE in suffix:
        # Already past this turn's tool call; no schema anchor.
        return None

    if CODEX_NAME_OPEN in suffix and CODEX_NAME_CLOSE not in suffix:
        # Anchor 1: name region open, name not yet closed.
        return DraftProposal(
            text=forced_name + CODEX_NAME_CLOSE + CODEX_ARGS_OPEN + "{",
            confidence=1.0,
            reason="codex_anchor_1_name_to_args_open",
        )

    if CODEX_ARGS_OPEN in suffix and CODEX_ARGS_CLOSE not in suffix:
        # Anchor 2/3: inside arguments region.
        first_prop = _first_required_property(schema)
        if first_prop is None:
            return None
        prop_name, prop_schema = first_prop
        # Has the args region already started emitting this property?
        args_text = suffix.split(CODEX_ARGS_OPEN, 1)[1]
        prop_marker = f'"{prop_name}":'
        if prop_marker not in args_text:
            return DraftProposal(
                text=f'"{prop_name}":',
                confidence=0.9,
                reason="codex_anchor_2_first_property_open",
            )
        # Anchor 3: just past the colon, predict the opening quote
        # for a string-typed property.
        if isinstance(prop_schema, dict) and prop_schema.get("type") == "string":
            after_marker = args_text.rsplit(prop_marker, 1)[-1]
            if not after_marker.lstrip().startswith('"'):
                return DraftProposal(
                    text='"',
                    confidence=0.8,
                    reason="codex_anchor_3_string_property_open_quote",
                )
        return None

    return None


def _propose_openai_json(
    snapshot: HarnessOracleSnapshot, recent_text: str
) -> DraftProposal | None:
    """Generic OpenAI Responses-API JSON dialect.

    The Responses API tool-call structure ships through vLLM's tool
    parser, so the wire format is ``<tool_call>...`` for Qwen-family
    models too — but the schema-aware drafter for non-Qwen models
    sees JSON-only output with a ``{"name": ..., "arguments": {...}}``
    shape. Two anchors:

    1. After ``"name":"`` — emit the forced name + ``","arguments":{``.
    2. Inside the arguments object — first-required-property opener.
    """

    expected = snapshot.expected_tool_call
    if not isinstance(expected, dict):
        return None
    forced_name = expected.get("name")
    if not isinstance(forced_name, str):
        return None
    schema = expected.get("schema") if isinstance(expected.get("schema"), dict) else None

    if recent_text.endswith('"name":"') or recent_text.endswith('"name": "'):
        return DraftProposal(
            text=forced_name + '","arguments":{',
            confidence=1.0,
            reason="openai_anchor_1_name_to_args_open",
        )

    if recent_text.endswith('"arguments":{') or recent_text.endswith('"arguments": {'):
        first_prop = _first_required_property(schema)
        if first_prop is None:
            return None
        prop_name, prop_schema = first_prop
        proposal_text = f'"{prop_name}":'
        if isinstance(prop_schema, dict) and prop_schema.get("type") == "string":
            proposal_text += '"'
        return DraftProposal(
            text=proposal_text,
            confidence=0.9,
            reason="openai_anchor_2_first_property_open",
        )

    return None


def _first_required_property(
    schema: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None] | None:
    """Return ``(name, sub_schema)`` for the first property the schema
    requires, or ``None`` if no required-list exists.

    Falls back to ``properties`` insertion order if no ``required``
    list is present — Codex tools we've seen all declare ``required``,
    but the fallback keeps fixtures from the broader OpenAPI corpus
    workable."""

    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None
    required = schema.get("required")
    if isinstance(required, list) and required:
        for name in required:
            if isinstance(name, str) and name in properties:
                return name, properties.get(name) if isinstance(properties.get(name), dict) else None
    # Fallback: first inserted property.
    for name, sub in properties.items():
        if isinstance(name, str):
            return name, sub if isinstance(sub, dict) else None
    return None

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple

import requests

from .metrics import parse_prometheus_text
from .registry import load_registry
from .tuned_config import (
    RuntimeStateStore,
    StructuredValidationError,
    ValidationIssue,
    load_tuned_config_bundle,
    validate_bundle_load_policy,
)

TRACK_B_REQUEST_METRICS_SCHEMA = "lumo.track_b.vllm_request_metrics.v1"
TRACK_B_REQUEST_METRICS_PRODUCER = "track_b_vllm_request_metrics_patch"
TRACK_B_REQUEST_METRICS_OUT_ENV = "LUMO_TRACK_B_REQUEST_METRICS_OUT"
TRACK_B_RUNTIME_CONFIG_HASH_ENV = "LUMO_TRACK_B_RUNTIME_CONFIG_HASH"

# Harness oracle header — see
# docs/reports/auto_research/track-b-harness-oracle-api-skeleton-20260509.md
# for the full snapshot schema. Phase 1 (this code) synthesises the minimum
# set the drafter needs (session_id, turn_index, dialect) directly from the
# /v1/responses payload — no Codex source change, no vLLM rebuild.
LUMO_ORACLE_HEADER = "X-Lumo-Oracle"
LUMO_ORACLE_SCHEMA = "lumo.harness_oracle_snapshot.v1"

# Session-prefixed X-Request-Id encoding for Round 2 Technique 1
# (cross-turn ngram session scoping). vLLM's
# ``serving.OpenAIServing._base_request_id`` already promotes
# ``X-Request-Id`` to the engine-side ``req_id`` that
# ``SuffixDecodingProposer.propose()`` keys its caches on. By
# prefixing every Codex /v1/responses call with this format, the
# downstream session-scoped wrapper around SuffixDecodingCache
# (lands in the prelaunch hook on next vLLM relaunch) can parse
# session_id back out of the req_id and route to a per-session
# suffix tree. Format is documented + parseable both ends.
LUMO_REQUEST_ID_PREFIX = "lumo_sess_"
LUMO_REQUEST_ID_SEP = "__"

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
INFERENCE_PATHS = {"/v1/responses", "/v1/chat/completions"}
# /v1/models pass-through tested 2026-05-08 and reverted. Codex 0.128.0's
# models_manager requires fields (slug, display_name, ...) vLLM does not emit;
# enriching them piecewise made Codex strict-fail more often (the 403 path is
# softer — Codex skips model-refresh and proceeds to /v1/responses normally).
# Keep the empty set so the proxy falls back to 403 on GET /v1/models.
INFERENCE_GET_PATHS: set[str] = set()
ADMIN_PATHS = {"/admin/load_tuned_config", "/admin/invalidate"}
CAMPAIGN_CLASSES = {"eval", "rollout"}
REQUEST_CLASS_HEADERS = (
    "X-Lumo-Request-Class",
    "X-Request-Class",
    "X-Traffic-Class",
)
ENFORCED_REQUEST_SHAPING_FIELDS = (
    "concurrency_cap_eval",
    "concurrency_cap_rollout",
    "admission_queue_depth_max",
)
ADVISORY_REQUEST_SHAPING_FIELDS = ("per_request_kv_budget", "priority_preemption")
PRIORITY_PREEMPTION_VALUES = {"off", "strict", "graceful"}


class RequestShapingPolicy(NamedTuple):
    concurrency_cap_eval: int
    concurrency_cap_rollout: int
    admission_queue_depth_max: int
    max_num_seqs: int
    bundle_id: str
    advisory_fields: dict[str, Any]


class AdmissionTicket(NamedTuple):
    policy: RequestShapingPolicy | None
    request_class: str


def is_inference_path(path: str) -> bool:
    return path in INFERENCE_PATHS


def is_inference_get_path(path: str) -> bool:
    """GET-only paths the proxy may forward to upstream (read-only model discovery, etc.).

    Scoped narrowly so the proxy's "inference paths only" guarantee still holds
    for state-changing endpoints. Currently allows GET /v1/models so Codex's
    model-discovery refresh succeeds; without this, Codex 0.128.0 enters a
    degraded path where roughly 1 in 3 ``exec`` invocations returns
    ``turn.completed`` with zero tokens despite making a real
    ``/v1/responses`` call.
    """

    if not path:
        return False
    base = path.split("?", 1)[0]
    return base in INFERENCE_GET_PATHS


def normalize_responses_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    # Lumo experiment knobs (env-gated): pin sampling params for controlled
    # A/B runs without a Codex source change. Unset -> request unchanged.
    _force_temp = os.environ.get("LUMO_PROXY_FORCE_TEMPERATURE")
    if _force_temp:
        try:
            normalized["temperature"] = float(_force_temp)
        except ValueError:
            pass
    _max_out = os.environ.get("LUMO_PROXY_MAX_OUTPUT_TOKENS")
    if _max_out:
        try:
            cap = int(_max_out)
        except ValueError:
            cap = None
        if cap is not None:
            cur = normalized.get("max_output_tokens")
            if not isinstance(cur, int) or cur > cap:
                normalized["max_output_tokens"] = cap
    _normalize_responses_output_items(normalized)
    tools = normalized.get("tools")
    if not isinstance(tools, list):
        return normalized
    normalized_tools: list[Any] = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
            and "name" not in tool
        ):
            function = dict(tool["function"])
            flattened = {"type": "function", **function}
            normalized_tools.append(flattened)
            continue
        normalized_tools.append(tool)
    normalized["tools"] = normalized_tools
    return normalized


def normalize_responses_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    _normalize_responses_output_items(normalized)
    return normalized


def _normalize_responses_output_items(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _normalize_responses_output_items(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("type") == "reasoning" and "status" not in value:
        value["status"] = "completed"
    if value.get("type") == "reasoning" and "id" not in value:
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        value["id"] = f"rs_{digest}"
    if value.get("type") == "function_call" and isinstance(value.get("arguments"), str):
        value["arguments"] = _normalize_function_call_arguments(value["arguments"])
    for key in ("input", "output", "item", "response"):
        if key in value:
            _normalize_responses_output_items(value[key])


def _normalize_function_call_arguments(arguments: str) -> str:
    try:
        decoded, end = json.JSONDecoder().raw_decode(arguments)
    except json.JSONDecodeError:
        return arguments
    if not arguments[end:].strip():
        return arguments
    return json.dumps(decoded, separators=(",", ":"))


def _first_user_text(payload: dict[str, Any]) -> str:
    """Return the first user-visible message text from a /v1/responses payload.

    Stable anchor for synthesising a session id: invariant across every turn
    of the same conversation because the Codex harness re-sends the full
    transcript on each request. Falls back to ``instructions`` then a
    JSON-serialised hash of the whole payload if nothing matches.
    """

    instructions = payload.get("instructions")
    user_anchor: str = ""
    inputs = payload.get("input")
    if isinstance(inputs, list):
        for item in inputs:
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                user_anchor = content
                break
            if isinstance(content, list):
                parts: list[str] = []
                for inner in content:
                    if isinstance(inner, dict):
                        text = inner.get("text") or inner.get("input_text")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    user_anchor = "\n".join(parts)
                    break
    elif isinstance(inputs, str):
        user_anchor = inputs
    if user_anchor:
        return user_anchor
    if isinstance(instructions, str) and instructions:
        return instructions
    return json.dumps(payload, sort_keys=True, default=str)[:4096]


def _detect_dialect(payload: dict[str, Any]) -> str:
    """Identify the harness dialect.

    Codex's request signature is the ``shell`` + ``apply_patch`` tool pair;
    if either is present we mark the request as the Codex dialect so the
    drafter can dispatch dialect-specific structural drafters
    (e.g. apply_patch path priming). All other shapes get ``openai`` —
    the Responses API generic dialect.
    """

    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if not isinstance(name, str):
                fn = tool.get("function")
                if isinstance(fn, dict):
                    name = fn.get("name")
            if isinstance(name, str) and name in {"shell", "apply_patch", "exec_command", "container.exec"}:
                return "codex"
    return "openai"


def _count_turn_index(payload: dict[str, Any]) -> int:
    """Number of completed assistant turns in the input transcript.

    Counts function_call items (one per tool invocation the assistant has
    issued so far) plus assistant ``message`` items. The new turn the
    request is asking for is *not* yet in the input, so ``len(prior) ==``
    its 0-indexed position.
    """

    inputs = payload.get("input")
    if not isinstance(inputs, list):
        return 0
    turns = 0
    for item in inputs:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "tool_call"}:
            turns += 1
            continue
        if item_type == "message" and item.get("role") == "assistant":
            turns += 1
            continue
        if item_type is None and item.get("role") == "assistant":
            turns += 1
    return turns


def _extract_tool_schemas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull JSON-schema tool definitions out of a /v1/responses payload.

    Schema-aware drafter (Technique 3) consumes these to drive XGrammar-2
    forced-token drafting. Output shape per entry:
    ``{"name": str, "parameters": <json schema>}`` — flat, since the
    Responses API emits both "type": "function" wrapped and flattened
    forms. Keeps only fields the drafter needs; description is dropped.
    """

    tools = payload.get("tools")
    if not isinstance(tools, list):
        return []
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name: Any = tool.get("name")
        params: Any = tool.get("parameters")
        if not isinstance(name, str):
            fn = tool.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                params = fn.get("parameters", params)
        if not isinstance(name, str) or not name:
            continue
        entry: dict[str, Any] = {"name": name}
        if isinstance(params, dict):
            entry["parameters"] = params
        schemas.append(entry)
    return schemas


def _extract_expected_tool_call(
    payload: dict[str, Any], schemas: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """If tool_choice forces a specific function, return ``{name, schema}``.

    Forced-tool turns are 100% predictable on the function-name slot —
    schema-aware drafter pre-fills the name region with confidence 1.0.
    """

    tc = payload.get("tool_choice")
    forced_name: str | None = None
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                forced_name = fn["name"]
            elif isinstance(tc.get("name"), str):
                forced_name = tc["name"]
    if forced_name is None:
        return None
    matched = next(
        (s for s in schemas if s.get("name") == forced_name),
        {"name": forced_name},
    )
    return {"name": forced_name, "schema": matched.get("parameters")}


# Heuristics for detecting file-read tool calls in a Codex transcript.
# Codex emits ``shell`` calls whose ``cmd`` is a list like
# ``["cat", "path/to/file.py"]`` or ``["sed", "-n", "1,200p", "path"]``.
# Their function_call_output is the file content. We use these to
# populate ``primed_texts`` on the oracle so a future T2 drafter can
# fold them into the per-session suffix tree.
_FILE_READ_FIRST_TOKENS = {
    "cat", "head", "tail", "less", "more", "bat", "sed", "awk", "grep",
}
_PRIMED_TEXTS_MAX = 8
_PRIMED_TEXT_MAX_CHARS = 65536
_PRIMED_TEXTS_MIN_OUTPUT_CHARS = 200


def _looks_like_file_read(call_arguments: Any) -> tuple[bool, str | None]:
    """Heuristically classify a Codex shell-tool argument string.

    Returns ``(is_file_read, derived_path)``. Conservative -- false
    positives just bloat the primer cache and the drafter weights
    primer hits by source_tag freshness anyway.
    """

    if not isinstance(call_arguments, str):
        return False, None
    try:
        parsed = json.loads(call_arguments)
    except json.JSONDecodeError:
        return False, None
    cmd = parsed.get("cmd") if isinstance(parsed, dict) else None
    if isinstance(cmd, list) and cmd:
        head_tok = cmd[0] if isinstance(cmd[0], str) else ""
        if head_tok.lower() in _FILE_READ_FIRST_TOKENS:
            for tok in reversed(cmd):
                if isinstance(tok, str) and (
                    "/" in tok or tok.endswith((".py", ".md", ".rs", ".ts", ".tsx", ".json", ".yaml", ".yml", ".sh"))
                ):
                    return True, tok
            return True, None
    return False, None


def _extract_primed_texts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull file-read primer candidates out of the prior turns.

    For each ``function_call`` whose arguments look like a file-read
    shell command, pair it with the corresponding
    ``function_call_output`` (matched by ``call_id``) and emit a
    primer entry with the file content as ``text`` and the inferred
    path as ``source_tag``. Capped at ``_PRIMED_TEXTS_MAX`` entries
    and each ``text`` is truncated to ``_PRIMED_TEXT_MAX_CHARS``.
    """

    inputs = payload.get("input")
    if not isinstance(inputs, list):
        return []
    # First pass: index function_call_output by call_id for O(1) join.
    outputs_by_call: dict[str, str] = {}
    for item in inputs:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call_output":
            continue
        call_id = item.get("call_id")
        out = item.get("output")
        if isinstance(call_id, str) and isinstance(out, str):
            outputs_by_call[call_id] = out

    primed: list[dict[str, Any]] = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        if item.get("name") not in {"shell", "exec_command", "container.exec"}:
            continue
        is_read, path = _looks_like_file_read(item.get("arguments"))
        if not is_read:
            continue
        call_id = item.get("call_id")
        out = outputs_by_call.get(call_id) if isinstance(call_id, str) else None
        if not isinstance(out, str) or len(out) < _PRIMED_TEXTS_MIN_OUTPUT_CHARS:
            continue
        primed.append(
            {
                "text": out[:_PRIMED_TEXT_MAX_CHARS],
                "source_tag": f"file:{path}" if path else f"shell:{call_id}",
                "ttl_turns": 32,
                "max_chars": _PRIMED_TEXT_MAX_CHARS,
            }
        )
        if len(primed) >= _PRIMED_TEXTS_MAX:
            break
    return primed


def synthesize_oracle_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Build an X-Lumo-Oracle snapshot from a /v1/responses payload.

    Schema documented in
    track-b-harness-oracle-api-skeleton-20260509.md. Computed entirely
    from the inbound payload — no Codex source change required for any
    field below. ``plan_fingerprint`` is left out; it needs Codex-side
    emission to be useful and the drafter tolerates absence.
    """

    anchor = _first_user_text(payload)
    session_id = "sess_" + hashlib.sha256(anchor.encode("utf-8", errors="replace")).hexdigest()[:16]
    turn_index = _count_turn_index(payload)
    tool_schemas = _extract_tool_schemas(payload)
    expected = _extract_expected_tool_call(payload, tool_schemas)
    primed_texts = _extract_primed_texts(payload)
    snap: dict[str, Any] = {
        "schema": LUMO_ORACLE_SCHEMA,
        "session_id": session_id,
        "turn_index": turn_index,
        "dialect": _detect_dialect(payload),
        "is_session_open": turn_index == 0,
        "suffix_tree_cap_mb": 100,
    }
    if tool_schemas:
        snap["tool_schemas"] = tool_schemas
    if expected is not None:
        snap["expected_tool_call"] = expected
    if primed_texts:
        snap["primed_texts"] = primed_texts
    return snap


def encode_oracle_snapshot_header(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, separators=(",", ":"), sort_keys=True)


def encode_session_request_id(session_id: str, original_id: str | None = None) -> str:
    """Build the X-Request-Id value vLLM consumes for session routing.

    Format: ``lumo_sess_<session_id>__<original_id>``. The ``original_id``
    is the existing X-Request-Id the harness sent (preserved for log
    correlation) or a fresh hex token when the harness sent none.
    Caller is responsible for ensuring ``session_id`` does not contain
    the separator ``__``.
    """

    suffix = original_id if original_id else hashlib.sha256(
        session_id.encode("utf-8") + str(time.time_ns()).encode("ascii")
    ).hexdigest()[:16]
    return f"{LUMO_REQUEST_ID_PREFIX}{session_id}{LUMO_REQUEST_ID_SEP}{suffix}"


def parse_session_request_id(req_id: str | None) -> str | None:
    """Inverse of ``encode_session_request_id``.

    Returns the session_id substring or ``None`` for non-session-prefixed
    request ids. Tolerant of malformed input — never raises."""

    if not isinstance(req_id, str) or not req_id.startswith(LUMO_REQUEST_ID_PREFIX):
        return None
    rest = req_id[len(LUMO_REQUEST_ID_PREFIX):]
    sep_idx = rest.find(LUMO_REQUEST_ID_SEP)
    if sep_idx <= 0:
        return None
    return rest[:sep_idx]


def normalize_responses_sse_block(block: bytes) -> bytes:
    lines = block.splitlines()
    normalized_lines: list[bytes] = []
    changed = False
    for line in lines:
        if not line.startswith(b"data:"):
            normalized_lines.append(line)
            continue
        prefix, separator, data = line.partition(b":")
        stripped = data.strip()
        if not stripped or stripped == b"[DONE]":
            normalized_lines.append(line)
            continue
        try:
            payload = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            normalized_lines.append(line)
            continue
        if not isinstance(payload, dict):
            normalized_lines.append(line)
            continue
        normalized = normalize_responses_response_payload(payload)
        normalized_lines.append(prefix + separator + b" " + json.dumps(normalized, separators=(",", ":")).encode("utf-8"))
        changed = True
    if not changed:
        return block + b"\n\n"
    return b"\n".join(normalized_lines) + b"\n\n"


def _pop_sse_block(buffer: bytes) -> tuple[bytes | None, bytes]:
    lf_index = buffer.find(b"\n\n")
    crlf_index = buffer.find(b"\r\n\r\n")
    candidates = [index for index in (lf_index, crlf_index) if index >= 0]
    if not candidates:
        return None, buffer
    index = min(candidates)
    separator_len = 4 if buffer[index : index + 4] == b"\r\n\r\n" else 2
    return buffer[:index], buffer[index + separator_len :]


def _responses_sse_payloads(block: bytes) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in block.splitlines():
        if not line.startswith(b"data:"):
            continue
        _prefix, _separator, data = line.partition(b":")
        stripped = data.strip()
        if not stripped or stripped == b"[DONE]":
            continue
        try:
            payload = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _responses_sse_payload_type(block: bytes) -> str | None:
    for payload in _responses_sse_payloads(block):
        if isinstance(payload.get("type"), str):
            return payload["type"]
    return None


def _update_synthetic_response_context(context: dict[str, Any], block: bytes) -> None:
    for payload in _responses_sse_payloads(block):
        response = payload.get("response")
        if isinstance(response, dict):
            for key in ("id", "model", "created_at"):
                if key in response and response[key] is not None:
                    context[key] = response[key]
        response_id = payload.get("response_id")
        if isinstance(response_id, str) and response_id:
            context["id"] = response_id


def _synthetic_response_completed_block(context: dict[str, Any] | None = None) -> bytes:
    response: dict[str, Any] = {
        "id": "resp_proxy_synthetic",
        "status": "completed",
        "output": [],
    }
    for key in ("id", "model", "created_at"):
        if context and context.get(key) is not None:
            response[key] = context[key]
    payload = {"type": "response.completed", "response": response}
    return (
        b"event: response.completed\n"
        + b"data: "
        + json.dumps(payload, separators=(",", ":")).encode("utf-8")
        + b"\n\n"
    )


def _iso_ts(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _classify_regime(observed: dict[str, Any]) -> str:
    if observed.get("has_tool_call"):
        return "tool-call"
    text_chars = int(observed.get("text_chars", 0) or 0)
    if text_chars >= 4096:
        return "summary"
    if text_chars > 0:
        return "reasoning"
    return "unknown"


def _extract_response_metadata(payload: dict[str, Any], context: dict[str, Any]) -> None:
    """Extend the SSE block scan to capture per-request usage + regime hints.

    Called alongside _update_synthetic_response_context. Fills in
    ``context["usage"]``, ``context["has_tool_call"]``, ``context["text_chars"]``
    so the request-finalize path can write a per-request observability row.
    """

    payload_type = payload.get("type")
    response = payload.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        usage_in = response["usage"]
        usage_out = context.setdefault("usage", {})
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "cached_input_tokens",
        ):
            value = usage_in.get(key)
            if value is not None:
                usage_out[key] = value
    if isinstance(payload_type, str):
        if payload_type.startswith("response.function_call") or payload_type.endswith(".function_call"):
            context["has_tool_call"] = True
        elif payload_type == "response.output_item.added":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}:
                context["has_tool_call"] = True
        elif payload_type in {"response.output_text.delta", "response.output_text.done"}:
            delta = payload.get("delta")
            if isinstance(delta, str):
                context["text_chars"] = int(context.get("text_chars", 0) or 0) + len(delta)
            elif payload_type == "response.output_text.done":
                text = payload.get("text")
                if isinstance(text, str):
                    context["text_chars"] = max(int(context.get("text_chars", 0) or 0), len(text))


class TrackBRequestMetricsCapture:
    """Per-request observability writer for Track B E2E.

    When ``LUMO_TRACK_B_REQUEST_METRICS_OUT`` points to a writable JSONL path,
    the proxy captures one row per ``/v1/responses`` request describing the
    upstream request id, token counts, /metrics deltas (prefill/decode/spec_decode),
    and a regime heuristic. The runner consumes this file via
    ``--vllm-request-metrics-jsonl`` to populate ``vllm_per_turn.json`` and
    synthesize ``codex_trace.jsonl``.

    Default off: when the env var is unset, ``from_env`` returns ``None`` and
    the proxy serves traffic unchanged.
    """

    _SPEC_KEYS: tuple[tuple[str, str], ...] = (
        ("spec_decode_num_accepted_tokens", "vllm:spec_decode_num_accepted_tokens_total"),
        ("spec_decode_num_draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
        ("spec_decode_num_drafts", "vllm:spec_decode_num_drafts_total"),
    )
    _HISTOGRAM_KEYS: tuple[tuple[str, str], ...] = (
        ("prefill_sum_s", "vllm:request_prefill_time_seconds_sum"),
        ("decode_sum_s", "vllm:request_decode_time_seconds_sum"),
    )

    def __init__(self, output_path: Path, upstream_metrics_url: str, *, runtime_config_hash: str = "") -> None:
        self._path = output_path
        self._upstream_metrics_url = upstream_metrics_url
        self._runtime_config_hash = runtime_config_hash
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, upstream_base_url: str) -> "TrackBRequestMetricsCapture | None":
        out = os.environ.get(TRACK_B_REQUEST_METRICS_OUT_ENV, "").strip()
        if not out:
            return None
        runtime_config_hash = os.environ.get(TRACK_B_RUNTIME_CONFIG_HASH_ENV, "").strip()
        return cls(
            Path(out),
            f"{upstream_base_url.rstrip('/')}/metrics",
            runtime_config_hash=runtime_config_hash,
        )

    def fetch_metrics_snapshot(self) -> dict[str, float]:
        try:
            response = requests.get(self._upstream_metrics_url, timeout=5)
            response.raise_for_status()
            return parse_prometheus_text(response.text)
        except requests.RequestException:
            return {}

    def compute_deltas(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for logical, metric_key in self._SPEC_KEYS + self._HISTOGRAM_KEYS:
            if not before or not after or metric_key not in after:
                result[logical] = None
                continue
            value = after.get(metric_key, 0.0) - before.get(metric_key, 0.0)
            if value < 0:
                result[logical] = None
            else:
                result[logical] = value
        return result

    def record(self, row: dict[str, Any]) -> None:
        row["schema"] = TRACK_B_REQUEST_METRICS_SCHEMA
        row["producer"] = TRACK_B_REQUEST_METRICS_PRODUCER
        if self._runtime_config_hash and "runtime_config_hash" not in row:
            row["runtime_config_hash"] = self._runtime_config_hash
        line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                # Best-effort capture; never break inference traffic on disk error.
                pass


def _build_request_metrics_row(
    *,
    request_id: str,
    request_path: str,
    request_class: str,
    upstream_status: int,
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    deltas: dict[str, float | None],
    response_observed: dict[str, Any],
    ts_request_received: float,
    ts_first_byte: float | None,
    ts_completed: float,
    saw_completed: bool,
    oracle_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = response_observed.get("usage") if isinstance(response_observed.get("usage"), dict) else {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    row: dict[str, Any] = {
        "request_id": request_id,
        "model": response_observed.get("model"),
        "request_path": request_path,
        "request_class": request_class,
        "upstream_status": upstream_status,
        "ts_request_received": _iso_ts(ts_request_received),
        "ts_first_byte": _iso_ts(ts_first_byte) if ts_first_byte is not None else None,
        "ts_completed": _iso_ts(ts_completed),
        "wallclock_s": max(0.0, ts_completed - ts_request_received),
        "first_byte_s": (ts_first_byte - ts_request_received) if ts_first_byte is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_sum_s": deltas.get("prefill_sum_s"),
        "decode_sum_s": deltas.get("decode_sum_s"),
        "spec_decode_num_accepted_tokens": deltas.get("spec_decode_num_accepted_tokens"),
        "spec_decode_num_draft_tokens": deltas.get("spec_decode_num_draft_tokens"),
        "spec_decode_num_drafts": deltas.get("spec_decode_num_drafts"),
        "regime": _classify_regime(response_observed),
        "tool_call_observed": bool(response_observed.get("has_tool_call")),
        "text_chars_observed": int(response_observed.get("text_chars", 0) or 0),
        "saw_response_completed": bool(saw_completed),
        "metrics_snapshot_collected": bool(metrics_before) and bool(metrics_after),
    }
    if oracle_snapshot is not None:
        row["oracle_session_id"] = oracle_snapshot.get("session_id")
        row["oracle_turn_index"] = oracle_snapshot.get("turn_index")
        row["oracle_dialect"] = oracle_snapshot.get("dialect")
        row["oracle_is_session_open"] = bool(oracle_snapshot.get("is_session_open"))
        row["oracle_tool_schema_count"] = len(oracle_snapshot.get("tool_schemas") or [])
        expected = oracle_snapshot.get("expected_tool_call")
        row["oracle_expected_tool_name"] = (
            expected.get("name") if isinstance(expected, dict) else None
        )
        row["oracle_primed_text_count"] = len(oracle_snapshot.get("primed_texts") or [])
    return row


def _filtered_headers(headers: Any) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        filtered[key] = value
    return filtered


def _write_json_error(
    handler: BaseHTTPRequestHandler,
    status: int,
    message: str,
    *,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    error: dict[str, Any] = {"message": message}
    if code is not None:
        error["code"] = code
    body = json.dumps({"error": error}).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_json_payload(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _normalize_input_for_nonstreaming(request_json: dict[str, Any]) -> dict[str, Any]:
    """Coerce codex's transcript input into a shape vLLM's non-streaming validation accepts.

    Codex strips ``id`` and ``status`` fields from echoed items when building
    the next-turn transcript. The streaming /v1/responses route accepts this;
    the non-streaming route's strict Pydantic validation rejects it (912+
    "Field required" errors). This helper walks the ``input`` list and:

      - Adds ``id`` to reasoning items (deterministic hash of content)
      - Adds ``id`` and ``status`` to assistant ``message`` items
      - Adds ``id`` to function_call items if missing

    Only invoked when LUMO_PROXY_NONSTREAM_BYPASS=1.
    """
    out = dict(request_json)
    items = out.get("input")
    if not isinstance(items, list):
        return out
    new_items: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            new_items.append(item)
            continue
        item = dict(item)
        item_type = item.get("type")
        if item_type == "reasoning":
            if not item.get("id"):
                digest = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
                item["id"] = f"rs_{digest}"
            if "summary" not in item or item["summary"] is None:
                item["summary"] = []
            if "status" not in item:
                item["status"] = "completed"
        elif item_type == "message" and item.get("role") == "assistant":
            if not item.get("id"):
                digest = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
                item["id"] = f"msg_{digest}"
            if "status" not in item:
                item["status"] = "completed"
            # ResponseOutputTextParam requires `annotations` field.
            content = item.get("content")
            if isinstance(content, list):
                new_content = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        c = dict(c)
                        if "annotations" not in c or c["annotations"] is None:
                            c["annotations"] = []
                    new_content.append(c)
                item["content"] = new_content
        elif item_type == "function_call":
            if not item.get("id"):
                digest = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
                item["id"] = f"fc_{digest}"
            if "status" not in item:
                item["status"] = "completed"
        elif item_type == "function_call_output":
            if not item.get("id"):
                digest = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
                item["id"] = f"fco_{digest}"
        new_items.append(item)
    out["input"] = new_items
    return out


def _synthesize_sse_stream_from_non_streaming_json(
    handler: BaseHTTPRequestHandler,
    response_json: dict[str, Any],
    *,
    capture_state: dict[str, Any] | None = None,
) -> None:
    """Emit a synthetic SSE stream from a non-streaming JSON response.

    The bypass path: codex sends stream:true, we forward stream:false to
    vLLM (so PR #39055's reasoning-parser promotion path runs, recovering
    tool calls embedded in <think>), then we reconstruct what the streaming
    events should have been and forward them to codex. This sidesteps both
    the §13.3 streaming UnboundLocalError and the §14.1 wrong-item_id /
    missing-output_item.added bug.
    """

    def write_chunk(chunk: bytes) -> None:
        handler.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
        handler.wfile.write(chunk)
        handler.wfile.write(b"\r\n")
        handler.wfile.flush()
        if capture_state is not None and capture_state.get("ts_first_byte") is None:
            capture_state["ts_first_byte"] = time.time()

    def emit_event(event_name: str, payload: dict[str, Any]) -> None:
        block = (
            b"event: " + event_name.encode("utf-8") + b"\n"
            + b"data: " + json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n\n"
        )
        if capture_state is not None:
            if event_name in {"response.function_call_arguments.delta", "response.function_call_arguments.done"} or (
                event_name == "response.output_item.added"
                and isinstance(payload.get("item"), dict)
                and payload["item"].get("type") == "function_call"
            ):
                capture_state["has_tool_call"] = True
            elif event_name in {"response.output_text.delta", "response.output_text.done"}:
                delta = payload.get("delta") or payload.get("text") or ""
                if isinstance(delta, str):
                    capture_state["text_chars"] = int(capture_state.get("text_chars", 0) or 0) + len(delta)
        write_chunk(block)

    try:
        # Normalize and walk the response.
        normalized_response = normalize_responses_response_payload(response_json)
        response_id = normalized_response.get("id")
        response_model = normalized_response.get("model")
        created_at = normalized_response.get("created_at")
        output_items = normalized_response.get("output", []) or []
        usage = normalized_response.get("usage")
        status = normalized_response.get("status", "completed")

        # response.created — copy the response object with empty output / in_progress
        stub_response = dict(normalized_response)
        stub_response["output"] = []
        stub_response["status"] = "in_progress"
        stub_response["usage"] = None
        sequence_number = 0

        def next_seq() -> int:
            nonlocal sequence_number
            sequence_number += 1
            return sequence_number

        emit_event("response.created", {
            "type": "response.created",
            "sequence_number": next_seq(),
            "response": stub_response,
        })
        emit_event("response.in_progress", {
            "type": "response.in_progress",
            "sequence_number": next_seq(),
            "response": stub_response,
        })

        for output_index, item in enumerate(output_items):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            item_id = item.get("id") or f"item_{output_index}"
            if item_type == "reasoning":
                # output_item.added (in_progress reasoning stub)
                added_item = {
                    "id": item_id,
                    "type": "reasoning",
                    "summary": item.get("summary", []),
                    "content": None,
                    "encrypted_content": item.get("encrypted_content"),
                    "status": "in_progress",
                }
                emit_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "sequence_number": next_seq(),
                    "output_index": output_index,
                    "item": added_item,
                })
                # reasoning_part.added + text deltas + done events
                contents = item.get("content") or []
                for content_index, content in enumerate(contents):
                    if not isinstance(content, dict):
                        continue
                    text = content.get("text", "") or ""
                    emit_event("response.reasoning_part.added", {
                        "type": "response.reasoning_part.added",
                        "sequence_number": next_seq(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": content_index,
                        "part": {"type": content.get("type", "reasoning_text"), "text": ""},
                    })
                    if text:
                        emit_event("response.reasoning_text.delta", {
                            "type": "response.reasoning_text.delta",
                            "sequence_number": next_seq(),
                            "output_index": output_index,
                            "item_id": item_id,
                            "content_index": content_index,
                            "delta": text,
                        })
                    emit_event("response.reasoning_text.done", {
                        "type": "response.reasoning_text.done",
                        "sequence_number": next_seq(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": content_index,
                        "text": text,
                    })
                    emit_event("response.reasoning_part.done", {
                        "type": "response.reasoning_part.done",
                        "sequence_number": next_seq(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": content_index,
                        "part": {"type": content.get("type", "reasoning_text"), "text": text},
                    })
                done_item = dict(added_item)
                done_item["content"] = contents
                done_item["status"] = item.get("status", "completed")
                emit_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "sequence_number": next_seq(),
                    "output_index": output_index,
                    "item": done_item,
                })
            elif item_type == "message":
                contents = item.get("content") or []
                added_item = {
                    "id": item_id,
                    "type": "message",
                    "role": item.get("role", "assistant"),
                    "content": [],
                    "status": "in_progress",
                }
                emit_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "sequence_number": next_seq(),
                    "output_index": output_index,
                    "item": added_item,
                })
                for content_index, content in enumerate(contents):
                    if not isinstance(content, dict):
                        continue
                    text = content.get("text", "") or ""
                    emit_event("response.content_part.added", {
                        "type": "response.content_part.added",
                        "sequence_number": next_seq(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": content_index,
                        "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                    })
                    if text:
                        emit_event("response.output_text.delta", {
                            "type": "response.output_text.delta",
                            "sequence_number": next_seq(),
                            "output_index": output_index,
                            "item_id": item_id,
                            "content_index": content_index,
                            "delta": text,
                            "logprobs": [],
                        })
                    emit_event("response.output_text.done", {
                        "type": "response.output_text.done",
                        "sequence_number": next_seq(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": content_index,
                        "text": text,
                        "logprobs": [],
                    })
                    emit_event("response.content_part.done", {
                        "type": "response.content_part.done",
                        "sequence_number": next_seq(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": content_index,
                        "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": []},
                    })
                done_item = dict(added_item)
                done_item["content"] = contents
                done_item["status"] = item.get("status", "completed")
                emit_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "sequence_number": next_seq(),
                    "output_index": output_index,
                    "item": done_item,
                })
            elif item_type == "function_call":
                arguments = item.get("arguments", "") or ""
                added_item = {
                    "id": item_id,
                    "type": "function_call",
                    "call_id": item.get("call_id") or f"call_{item_id}",
                    "name": item.get("name", ""),
                    "arguments": "",
                    "status": "in_progress",
                }
                emit_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "sequence_number": next_seq(),
                    "output_index": output_index,
                    "item": added_item,
                })
                if arguments:
                    emit_event("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": next_seq(),
                        "output_index": output_index,
                        "item_id": item_id,
                        "delta": arguments,
                    })
                emit_event("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": next_seq(),
                    "output_index": output_index,
                    "item_id": item_id,
                    "name": item.get("name", ""),
                    "arguments": arguments,
                })
                done_item = dict(added_item)
                done_item["arguments"] = arguments
                done_item["status"] = item.get("status", "completed")
                emit_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "sequence_number": next_seq(),
                    "output_index": output_index,
                    "item": done_item,
                })

        # response.completed
        completed_response = dict(normalized_response)
        completed_response["output"] = output_items
        completed_response["usage"] = usage
        completed_response["status"] = status
        emit_event("response.completed", {
            "type": "response.completed",
            "sequence_number": next_seq(),
            "response": completed_response,
        })
        if capture_state is not None:
            capture_state["response_id"] = response_id
            capture_state["model"] = response_model
            capture_state["saw_response_completed"] = True
            # Propagate usage so the per-request metrics row has real
            # prompt_tokens/completion_tokens instead of None. Without this,
            # 96%+ of captured rows fail downstream normalization
            # (_normalize_vllm_request_metrics) for missing numeric fields,
            # forcing the per-task summary to a deferred state.
            if isinstance(usage, dict):
                capture_state["usage"] = usage
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        return


def _write_chunked_stream(
    handler: BaseHTTPRequestHandler,
    upstream: requests.Response,
    *,
    capture_state: dict[str, Any] | None = None,
) -> None:
    def write_chunk(chunk: bytes) -> None:
        handler.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
        handler.wfile.write(chunk)
        handler.wfile.write(b"\r\n")
        handler.wfile.flush()
        if capture_state is not None and capture_state.get("ts_first_byte") is None:
            capture_state["ts_first_byte"] = time.time()

    pending = b""
    saw_event = False
    saw_completed = False
    synthetic_response_context: dict[str, Any] = {}
    # TEMP DEBUG: dump raw SSE blocks to LUMO_PROXY_SSE_DUMP_DIR if set.
    _sse_dump_dir = os.environ.get("LUMO_PROXY_SSE_DUMP_DIR")
    _sse_dump_fh = None
    if _sse_dump_dir:
        try:
            import pathlib as _pl
            _pl.Path(_sse_dump_dir).mkdir(parents=True, exist_ok=True)
            _sse_dump_fh = open(
                f"{_sse_dump_dir}/sse_{int(time.time()*1000)}.raw", "wb"
            )
        except Exception:
            _sse_dump_fh = None

    # Track which function_call output items have been announced via
    # output_item.added during streaming. vLLM's qwen3_xml tool parser
    # emits function_call_arguments.delta WITHOUT a preceding
    # output_item.added for the function_call item, using the previous
    # message item's id. Codex's streaming parser silently drops these,
    # so the tool call never reaches the agent loop even though the
    # final response.completed lists it. We compensate by synthesizing
    # output_item.added + function_call_arguments.done + output_item.done
    # for any function_call in response.completed.output that wasn't
    # announced earlier — emitted right before forwarding the
    # response.completed event. (Lumo Track B benchmark-validity §13.4
    # diagnosis; vLLM upstream Issues #39056, #41182 same bug family.)
    seen_function_call_added_ids: set[str] = set()
    current_response_id_for_synth: str | None = None
    next_synth_output_index = 0
    try:
        for chunk in upstream.iter_content(chunk_size=8192):
            if not chunk:
                continue
            pending += chunk
            while True:
                block, pending = _pop_sse_block(pending)
                if block is None:
                    break
                normalized = normalize_responses_sse_block(block)
                if _sse_dump_fh is not None:
                    _sse_dump_fh.write(b"=== block ===\n")
                    _sse_dump_fh.write(normalized)
                    _sse_dump_fh.flush()
                saw_event = True
                block_payload_type = _responses_sse_payload_type(normalized)
                saw_completed = saw_completed or block_payload_type == "response.completed"
                _update_synthetic_response_context(synthetic_response_context, normalized)
                # Track function_call output_item.added emissions.
                for payload in _responses_sse_payloads(normalized):
                    pt = payload.get("type")
                    if pt == "response.output_item.added":
                        item = payload.get("item", {}) or {}
                        if item.get("type") == "function_call":
                            fc_id = item.get("id")
                            if isinstance(fc_id, str):
                                seen_function_call_added_ids.add(fc_id)
                        oi = payload.get("output_index")
                        if isinstance(oi, int) and oi + 1 > next_synth_output_index:
                            next_synth_output_index = oi + 1
                    elif pt == "response.created":
                        resp = payload.get("response", {}) or {}
                        rid = resp.get("id")
                        if isinstance(rid, str):
                            current_response_id_for_synth = rid
                if capture_state is not None:
                    for payload in _responses_sse_payloads(normalized):
                        _extract_response_metadata(payload, capture_state)
                # If this is response.completed and the upstream skipped emitting
                # output_item.added for function_call items present in the final
                # output array, synthesize them now BEFORE forwarding the
                # response.completed block.
                if block_payload_type == "response.completed":
                    for payload in _responses_sse_payloads(normalized):
                        resp = payload.get("response", {}) or {}
                        output_items = resp.get("output", []) or []
                        for idx, item in enumerate(output_items):
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") != "function_call":
                                continue
                            fc_id = item.get("id")
                            if not isinstance(fc_id, str):
                                continue
                            if fc_id in seen_function_call_added_ids:
                                continue
                            # Build synthetic stream events for this missing function_call.
                            call_id = item.get("call_id") or f"call_{fc_id}"
                            name = item.get("name") or "exec_command"
                            args = item.get("arguments") or ""
                            base_item = {
                                "id": fc_id,
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": args,
                                "status": "in_progress",
                            }
                            added_payload = {
                                "type": "response.output_item.added",
                                "sequence_number": -1,
                                "output_index": next_synth_output_index,
                                "item": base_item,
                            }
                            args_done_payload = {
                                "type": "response.function_call_arguments.done",
                                "sequence_number": -1,
                                "output_index": next_synth_output_index,
                                "item_id": fc_id,
                                "arguments": args,
                                "name": name,
                            }
                            done_item = dict(base_item)
                            done_item["status"] = "completed"
                            done_payload = {
                                "type": "response.output_item.done",
                                "sequence_number": -1,
                                "output_index": next_synth_output_index,
                                "item": done_item,
                            }
                            for synth_event in ("response.output_item.added", "response.function_call_arguments.done", "response.output_item.done"):
                                synth_payload = {
                                    "response.output_item.added": added_payload,
                                    "response.function_call_arguments.done": args_done_payload,
                                    "response.output_item.done": done_payload,
                                }[synth_event]
                                synth_block = (
                                    b"event: " + synth_event.encode("utf-8")
                                    + b"\ndata: "
                                    + json.dumps(synth_payload, separators=(",", ":")).encode("utf-8")
                                    + b"\n\n"
                                )
                                if _sse_dump_fh is not None:
                                    _sse_dump_fh.write(b"=== synth (PR39055-Lumo) ===\n")
                                    _sse_dump_fh.write(synth_block)
                                    _sse_dump_fh.flush()
                                write_chunk(synth_block)
                            seen_function_call_added_ids.add(fc_id)
                            next_synth_output_index += 1
                write_chunk(normalized)
        if pending:
            normalized = normalize_responses_sse_block(pending)
            saw_event = True
            saw_completed = saw_completed or _responses_sse_payload_type(normalized) == "response.completed"
            _update_synthetic_response_context(synthetic_response_context, normalized)
            if capture_state is not None:
                for payload in _responses_sse_payloads(normalized):
                    _extract_response_metadata(payload, capture_state)
            write_chunk(normalized)
        if saw_event and not saw_completed:
            write_chunk(_synthetic_response_completed_block(synthetic_response_context))
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
        if capture_state is not None:
            capture_state["response_id"] = synthetic_response_context.get("id")
            capture_state["model"] = synthetic_response_context.get("model")
            capture_state["saw_response_completed"] = saw_completed
    except (BrokenPipeError, ConnectionResetError):
        # Codex occasionally abandons an HTTP stream after it already has the
        # terminal event. Treat that as a cancelled client, not a proxy crash.
        if _sse_dump_fh is not None:
            try: _sse_dump_fh.close()
            except Exception: pass
        return
    except requests.RequestException:
        terminal_block = (
            _synthetic_response_completed_block(synthetic_response_context)
            if saw_event and not saw_completed
            else (
                b"event: error\n"
                b'data: {"error":{"message":"Upstream inference stream ended prematurely","type":"upstream_stream_error"}}\n\n'
            )
        )
        try:
            write_chunk(terminal_block)
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
    finally:
        upstream.close()
        if _sse_dump_fh is not None:
            try: _sse_dump_fh.close()
            except Exception: pass


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredValidationError(
            message="Invalid JSON request body",
            issues=[ValidationIssue(field="body", message=str(exc))],
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredValidationError(
            message="Invalid JSON request body",
            issues=[ValidationIssue(field="body", message="must decode to a JSON object", value=payload)],
        )
    return payload


def _extract_request_class(payload: dict[str, Any] | None, headers: Any) -> str:
    for header_name in REQUEST_CLASS_HEADERS:
        value = headers.get(header_name)
        if isinstance(value, str) and value.strip().lower() in CAMPAIGN_CLASSES:
            return value.strip().lower()
    if not isinstance(payload, dict):
        return "eval"
    candidates = [
        payload.get("class"),
        payload.get("request_class"),
        payload.get("traffic_class"),
    ]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend([metadata.get("class"), metadata.get("request_class"), metadata.get("traffic_class")])
    extra_body = payload.get("extra_body")
    if isinstance(extra_body, dict):
        candidates.extend([extra_body.get("class"), extra_body.get("request_class"), extra_body.get("traffic_class")])
    for value in candidates:
        if isinstance(value, str) and value.strip().lower() in CAMPAIGN_CLASSES:
            return value.strip().lower()
    return "eval"


def _normalize_request_shaping_policy(bundle_path: str | Path, bundle: Any) -> RequestShapingPolicy | None:
    shaping = dict(bundle.request_shaping)
    if not shaping:
        return None
    if (
        set(shaping) == {"target_concurrency"}
        and "concurrency_cap_eval" not in shaping
        and "concurrency_cap_rollout" not in shaping
        and "admission_queue_depth_max" not in shaping
    ):
        shaping = {
            "concurrency_cap_eval": shaping["target_concurrency"],
            "concurrency_cap_rollout": 0,
            "admission_queue_depth_max": 0,
        }
    issues: list[ValidationIssue] = []
    vllm_config = bundle.vllm_config
    max_num_seqs = int(vllm_config.get("max_num_seqs", 0) or 0)
    if max_num_seqs < 1:
        issues.append(
            ValidationIssue(
                field="tuned_config_bundle.vllm_config.max_num_seqs",
                message="must be >= 1 when request_shaping is present",
                value=vllm_config.get("max_num_seqs"),
            )
        )

    def require_int(key: str, minimum: int, maximum: int) -> int:
        value = shaping.get(key)
        field = f"tuned_config_bundle.request_shaping.{key}"
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(ValidationIssue(field=field, message="must be an integer", value=value))
            return minimum
        if value < minimum:
            issues.append(ValidationIssue(field=field, message=f"must be >= {minimum}", value=value))
        if value > maximum:
            issues.append(ValidationIssue(field=field, message=f"must be <= {maximum}", value=value))
        return value

    eval_cap = require_int("concurrency_cap_eval", 1, max(max_num_seqs, 1))
    rollout_cap = require_int("concurrency_cap_rollout", 0, max(max_num_seqs, 1))
    queue_depth = require_int("admission_queue_depth_max", 0, 512)
    if max_num_seqs >= 1 and eval_cap + rollout_cap > max_num_seqs:
        issues.append(
            ValidationIssue(
                field="tuned_config_bundle.request_shaping",
                message="eval + rollout concurrency caps exceed max_num_seqs",
                value={
                    "concurrency_cap_eval": eval_cap,
                    "concurrency_cap_rollout": rollout_cap,
                    "max_num_seqs": max_num_seqs,
                },
            )
        )
    if "per_request_kv_budget" in shaping:
        max_model_len = int(vllm_config.get("max_model_len", 0) or 0)
        require_int("per_request_kv_budget", max(1, max_model_len // 4), max(max_model_len, 1))
    if "priority_preemption" in shaping and shaping.get("priority_preemption") not in PRIORITY_PREEMPTION_VALUES:
        issues.append(
            ValidationIssue(
                field="tuned_config_bundle.request_shaping.priority_preemption",
                message=f"must be one of {sorted(PRIORITY_PREEMPTION_VALUES)}",
                value=shaping.get("priority_preemption"),
            )
        )
    if issues:
        raise StructuredValidationError(
            message="Invalid tuned-config request_shaping",
            issues=[ValidationIssue(field="bundle_path", message="contains invalid request_shaping", value=str(bundle_path))]
            + issues,
        )
    return RequestShapingPolicy(
        concurrency_cap_eval=eval_cap,
        concurrency_cap_rollout=rollout_cap,
        admission_queue_depth_max=queue_depth,
        max_num_seqs=max_num_seqs,
        bundle_id=str(bundle.bundle_id),
        advisory_fields={key: shaping[key] for key in ADVISORY_REQUEST_SHAPING_FIELDS if key in shaping},
    )


class AdmissionController:
    def __init__(self, state_store: RuntimeStateStore) -> None:
        self._state_store = state_store
        self._condition = threading.Condition()
        self._active_by_class = {"eval": 0, "rollout": 0}
        self._queued = 0
        self._cached_bundle_path: str | None = None
        self._cached_policy: RequestShapingPolicy | None = None

    def policy(self) -> RequestShapingPolicy | None:
        state = self._state_store.load()
        bundle_path = state.active_tuned_config_path
        if not bundle_path:
            self._cached_bundle_path = None
            self._cached_policy = None
            return None
        if bundle_path == self._cached_bundle_path:
            return self._cached_policy
        bundle = load_tuned_config_bundle(bundle_path)
        self._cached_policy = _normalize_request_shaping_policy(bundle_path, bundle)
        self._cached_bundle_path = bundle_path
        return self._cached_policy

    def acquire(self, request_class: str) -> AdmissionTicket | None:
        request_class = request_class if request_class in CAMPAIGN_CLASSES else "eval"
        policy = self.policy()
        if policy is None:
            return AdmissionTicket(policy=None, request_class=request_class)
        with self._condition:
            if self._class_cap(policy, request_class) <= 0:
                return None
            if self._has_capacity(policy, request_class):
                self._active_by_class[request_class] += 1
                return AdmissionTicket(policy=policy, request_class=request_class)
            if self._queued >= policy.admission_queue_depth_max:
                return None
            self._queued += 1
            try:
                while not self._has_capacity(policy, request_class):
                    self._condition.wait()
                self._active_by_class[request_class] += 1
                return AdmissionTicket(policy=policy, request_class=request_class)
            finally:
                self._queued -= 1

    def release(self, ticket: AdmissionTicket) -> None:
        if ticket.policy is None:
            return
        with self._condition:
            self._active_by_class[ticket.request_class] = max(0, self._active_by_class[ticket.request_class] - 1)
            self._condition.notify_all()

    def _has_capacity(self, policy: RequestShapingPolicy, request_class: str) -> bool:
        class_cap = self._class_cap(policy, request_class)
        total_active = self._active_by_class["eval"] + self._active_by_class["rollout"]
        return self._active_by_class[request_class] < class_cap and total_active < policy.max_num_seqs

    @staticmethod
    def _class_cap(policy: RequestShapingPolicy, request_class: str) -> int:
        if request_class == "eval":
            return policy.concurrency_cap_eval
        return policy.concurrency_cap_rollout


def _validate_load_tuned_config_payload(payload: dict[str, Any]) -> str:
    bundle_path = payload.get("bundle_path")
    issues: list[ValidationIssue] = []
    if not isinstance(bundle_path, str) or not bundle_path.strip():
        issues.append(ValidationIssue(field="bundle_path", message="must be a non-empty string", value=bundle_path))
    if issues:
        raise StructuredValidationError(message="Invalid load_tuned_config payload", issues=issues)
    return bundle_path.strip()


def _validate_invalidate_payload(payload: dict[str, Any]) -> str:
    weight_version_id = payload.get("weight_version_id")
    issues: list[ValidationIssue] = []
    if not isinstance(weight_version_id, str) or not weight_version_id.strip():
        issues.append(
            ValidationIssue(
                field="weight_version_id",
                message="must be a non-empty string",
                value=weight_version_id,
            )
        )
    if issues:
        raise StructuredValidationError(message="Invalid invalidate payload", issues=issues)
    return weight_version_id.strip()


def build_proxy_handler(
    upstream_base_url: str,
    *,
    state_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    request_metrics_capture: TrackBRequestMetricsCapture | None = None,
) -> type[BaseHTTPRequestHandler]:
    state_store = RuntimeStateStore(state_root or Path.cwd() / "output" / "serving_state")
    registry = load_registry(registry_path) if registry_path is not None else {}
    admission = AdmissionController(state_store)
    capture = request_metrics_capture if request_metrics_capture is not None else TrackBRequestMetricsCapture.from_env(upstream_base_url)

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            if not is_inference_get_path(self.path):
                _write_json_error(self, 403, "Blocked by codex-bench-proxy: inference paths only")
                return
            try:
                upstream = requests.get(
                    f"{upstream_base_url}{self.path}",
                    headers=_filtered_headers(self.headers),
                    timeout=15,
                )
            except requests.RequestException as exc:
                _write_json_error(self, 502, f"Upstream model discovery failed: {exc}")
                return
            response_content = upstream.content
            base_path = self.path.split("?", 1)[0]
            if (
                base_path == "/v1/models"
                and upstream.status_code == 200
                and upstream.headers.get("Content-Type", "").startswith("application/json")
            ):
                try:
                    parsed = json.loads(response_content.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
                    # Codex 0.128.0 models_manager expects an outer "models" key
                    # AND each entry to carry a "slug" alongside "id". vLLM's
                    # OpenAI-compatible /v1/models response provides neither.
                    enriched: list[Any] = []
                    for entry in parsed["data"]:
                        if isinstance(entry, dict):
                            entry = dict(entry)
                            if "slug" not in entry and isinstance(entry.get("id"), str):
                                entry["slug"] = entry["id"]
                            if "name" not in entry and isinstance(entry.get("id"), str):
                                entry["name"] = entry["id"]
                        enriched.append(entry)
                    parsed["data"] = enriched
                    parsed["models"] = enriched
                    response_content = json.dumps(parsed).encode("utf-8")
            self.send_response(upstream.status_code)
            response_headers = _filtered_headers(upstream.headers)
            response_headers["Content-Length"] = str(len(response_content))
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(response_content)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self) -> None:  # noqa: N802
            if self.path in ADMIN_PATHS:
                self._handle_admin_request()
                return
            if not is_inference_path(self.path):
                _write_json_error(self, 403, "Blocked by codex-bench-proxy: inference paths only")
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = raw_body
            headers = _filtered_headers(self.headers)
            request_json: dict[str, Any] | None = None
            oracle_snapshot: dict[str, Any] | None = None
            # Set by the /v1/responses path when codex sends stream:true AND
            # LUMO_PROXY_NONSTREAM_BYPASS=1; we rewrite to stream:false upstream
            # so PR #39055's promotion path applies, then synthesize an SSE
            # stream back to codex on the response.
            nonstream_bypass_active = False
            if self.path == "/v1/responses":
                try:
                    request_json = json.loads(raw_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    _write_json_error(self, 400, "Invalid JSON request body")
                    return
                # TEMP DEBUG: dump request bodies to LUMO_PROXY_REQUEST_DUMP_DIR
                req_dump_dir = os.environ.get("LUMO_PROXY_REQUEST_DUMP_DIR")
                if req_dump_dir:
                    try:
                        import pathlib as _pl
                        _pl.Path(req_dump_dir).mkdir(parents=True, exist_ok=True)
                        with open(f"{req_dump_dir}/req_{int(time.time()*1000)}.json", "wb") as _rf:
                            _rf.write(raw_body)
                    except Exception:
                        pass
                normalized_req = normalize_responses_request_payload(request_json)
                if (
                    os.environ.get("LUMO_PROXY_NONSTREAM_BYPASS", "").lower() in {"1", "true", "yes"}
                    and bool(normalized_req.get("stream"))
                ):
                    normalized_req = dict(normalized_req)
                    normalized_req["stream"] = False
                    # Add required id/status fields that codex strips when echoing items
                    # back in the transcript — non-streaming validation rejects them otherwise.
                    normalized_req = _normalize_input_for_nonstreaming(normalized_req)
                    nonstream_bypass_active = True
                payload = json.dumps(normalized_req).encode("utf-8")
                headers["Content-Type"] = "application/json"
                oracle_snapshot = synthesize_oracle_snapshot(request_json)
                headers[LUMO_ORACLE_HEADER] = encode_oracle_snapshot_header(oracle_snapshot)
                session_id_for_routing = oracle_snapshot.get("session_id")
                if isinstance(session_id_for_routing, str):
                    headers["X-Request-Id"] = encode_session_request_id(
                        session_id_for_routing,
                        original_id=self.headers.get("X-Request-Id"),
                    )
            elif self.path == "/v1/chat/completions":
                try:
                    parsed_json = json.loads(raw_body.decode("utf-8"))
                    request_json = parsed_json if isinstance(parsed_json, dict) else None
                except (UnicodeDecodeError, json.JSONDecodeError):
                    request_json = None
            request_class = _extract_request_class(request_json, self.headers)
            try:
                ticket = admission.acquire(request_class)
            except StructuredValidationError as exc:
                _write_json_payload(self, 400, exc.as_error_payload())
                return
            if ticket is None:
                _write_json_error(
                    self,
                    429,
                    "Admission queue is full",
                    code="queue_full",
                    headers={"Retry-After": "1"},
                )
                return
            capture_active = capture is not None and self.path == "/v1/responses"
            metrics_before: dict[str, float] = {}
            ts_request_received = time.time()
            if capture_active:
                metrics_before = capture.fetch_metrics_snapshot()
            try:
                upstream = requests.post(
                    f"{upstream_base_url}{self.path}",
                    data=payload,
                    headers=headers,
                    timeout=600,
                    stream=True,
                )
            except requests.RequestException as exc:
                _write_json_error(self, 502, f"Upstream inference request failed: {exc}")
                admission.release(ticket)
                return

            self.send_response(upstream.status_code)
            response_headers = _filtered_headers(upstream.headers)
            response_content: bytes | None = None
            non_streaming_parsed: dict[str, Any] | None = None
            upstream_error_passthrough = False
            if nonstream_bypass_active:
                # Codex thinks it asked for streaming. Buffer the upstream
                # (non-streaming) JSON, normalize, then re-emit as SSE.
                response_content_buf = upstream.content
                try:
                    parsed = json.loads(response_content_buf.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                # When upstream rejected the request (4xx) OR the parsed
                # body is a top-level error envelope, do NOT wrap it in
                # a synthetic SSE response.created event — that buries
                # the OpenAI-format error inside a "successful" response
                # shape and confuses Codex into a turn.failed crash with
                # 0-byte patch. Propagate the JSON error directly so
                # Codex sees a real HTTP error and can retry. (Was the
                # observed failure mode on django-16256 / astropy-14508
                # final-turn BadRequestError flakes — 2026-05-21.)
                if (
                    isinstance(parsed, dict)
                    and (upstream.status_code >= 400 or "error" in parsed)
                ):
                    upstream_error_passthrough = True
                    response_content = response_content_buf
                    response_headers["Content-Type"] = upstream.headers.get(
                        "Content-Type", "application/json"
                    )
                    response_headers.pop("Transfer-Encoding", None)
                    response_headers["Content-Length"] = str(len(response_content))
                elif isinstance(parsed, dict):
                    non_streaming_parsed = parsed
                # Auto-continue retry loop (Qwen3 #1817 / Qwen3.5-9B #10 workaround):
                # qwen3.5-27b in thinking mode sometimes plans tool calls in
                # reasoning/text ("Now let me create X") and then fails to emit
                # them — response has no function_call. We auto-inject a
                # "continue" user message and re-call. Triggers whenever the
                # response has no function_call AT ALL (the agent needs a tool
                # call to make progress on a coding task; text-only responses
                # are always the bug). Bounded by max_retries.
                #
                # If the model is GENUINELY done (final answer), the next
                # response after the "continue" prompt would still have no
                # function_call and would either (a) fire another retry with
                # the same empty output, or (b) finally emit a final answer
                # in text. Either way the worst case is N+1 extra inference
                # calls, not infinite looping.
                if (
                    os.environ.get("LUMO_PROXY_AUTO_CONTINUE", "").lower() in {"1", "true", "yes"}
                    and isinstance(non_streaming_parsed, dict)
                ):
                    max_retries = int(os.environ.get("LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES", "3"))
                    continue_message = os.environ.get(
                        "LUMO_PROXY_AUTO_CONTINUE_MESSAGE",
                        "continue with the tool call you described",
                    )
                    retries_remaining = max_retries
                    while retries_remaining > 0:
                        out_items = non_streaming_parsed.get("output", []) or []
                        has_function_call = any(
                            isinstance(it, dict) and it.get("type") == "function_call"
                            for it in out_items
                        )
                        if has_function_call:
                            break  # legitimate output, no continue needed
                        # Synthesize the "continue" follow-up
                        retries_remaining -= 1
                        retry_req = dict(normalized_req)
                        retry_input = list(retry_req.get("input", []))
                        # Embed the previous reasoning + message items as assistant
                        # turn outputs so the model has context for the continue.
                        for it in out_items:
                            if isinstance(it, dict):
                                retry_input.append(it)
                        retry_input.append({
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": continue_message}],
                        })
                        retry_req["input"] = retry_input
                        retry_req = _normalize_input_for_nonstreaming(retry_req)
                        retry_payload = json.dumps(retry_req).encode("utf-8")
                        try:
                            retry_resp = requests.post(
                                f"{upstream_base_url}{self.path}",
                                data=retry_payload,
                                headers=headers,
                                timeout=600,
                                stream=False,
                            )
                            if retry_resp.status_code == 200:
                                try:
                                    retry_parsed = retry_resp.json()
                                except json.JSONDecodeError:
                                    retry_parsed = None
                                if isinstance(retry_parsed, dict):
                                    # Merge: keep accumulated output from prior calls,
                                    # then append the new output items.
                                    prev_output = list(non_streaming_parsed.get("output", []) or [])
                                    new_output = list(retry_parsed.get("output", []) or [])
                                    non_streaming_parsed = dict(retry_parsed)
                                    non_streaming_parsed["output"] = prev_output + new_output
                            else:
                                break
                        except requests.RequestException:
                            break
                if not upstream_error_passthrough:
                    response_headers["Transfer-Encoding"] = "chunked"
                    response_headers["Content-Type"] = "text/event-stream"
                    response_headers.pop("Content-Length", None)
            elif upstream.headers.get("Content-Type", "").startswith("text/event-stream"):
                response_headers["Transfer-Encoding"] = "chunked"
                response_headers.pop("Content-Length", None)
            else:
                response_content = upstream.content
                if self.path == "/v1/responses" and upstream.headers.get("Content-Type", "").startswith("application/json"):
                    try:
                        parsed = json.loads(response_content.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, dict):
                        non_streaming_parsed = parsed
                        response_content = json.dumps(normalize_responses_response_payload(parsed)).encode("utf-8")
                response_headers["Content-Length"] = str(len(response_content))
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.end_headers()

            capture_state: dict[str, Any] | None = (
                {"ts_first_byte": None, "has_tool_call": False, "text_chars": 0}
                if capture_active
                else None
            )

            if nonstream_bypass_active and isinstance(non_streaming_parsed, dict):
                try:
                    _synthesize_sse_stream_from_non_streaming_json(
                        self,
                        non_streaming_parsed,
                        capture_state=capture_state,
                    )
                finally:
                    admission.release(ticket)
                if capture_active and capture_state is not None:
                    self._emit_track_b_capture_row(
                        capture=capture,
                        request_class=request_class,
                        upstream_status=upstream.status_code,
                        metrics_before=metrics_before,
                        capture_state=capture_state,
                        ts_request_received=ts_request_received,
                        oracle_snapshot=oracle_snapshot,
                    )
                return

            if response_headers.get("Transfer-Encoding") == "chunked":
                try:
                    _write_chunked_stream(self, upstream, capture_state=capture_state)
                finally:
                    admission.release(ticket)
                if capture_active and capture_state is not None:
                    self._emit_track_b_capture_row(
                        capture=capture,
                        request_class=request_class,
                        upstream_status=upstream.status_code,
                        metrics_before=metrics_before,
                        capture_state=capture_state,
                        ts_request_received=ts_request_received,
                        oracle_snapshot=oracle_snapshot,
                    )
                return

            try:
                self.wfile.write(response_content)
                self.wfile.flush()
                if capture_active:
                    capture_state_local: dict[str, Any] = {
                        "ts_first_byte": time.time(),
                        "has_tool_call": False,
                        "text_chars": 0,
                    }
                    if isinstance(non_streaming_parsed, dict):
                        capture_state_local["response_id"] = non_streaming_parsed.get("id")
                        capture_state_local["model"] = non_streaming_parsed.get("model")
                        if isinstance(non_streaming_parsed.get("usage"), dict):
                            capture_state_local["usage"] = dict(non_streaming_parsed["usage"])
                        output = non_streaming_parsed.get("output")
                        if isinstance(output, list):
                            for item in output:
                                if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}:
                                    capture_state_local["has_tool_call"] = True
                                if isinstance(item, dict) and isinstance(item.get("content"), list):
                                    for inner in item["content"]:
                                        if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                                            capture_state_local["text_chars"] = (
                                                capture_state_local.get("text_chars", 0) + len(inner["text"])
                                            )
                        capture_state_local["saw_response_completed"] = True
                    self._emit_track_b_capture_row(
                        capture=capture,
                        request_class=request_class,
                        upstream_status=upstream.status_code,
                        metrics_before=metrics_before,
                        capture_state=capture_state_local,
                        ts_request_received=ts_request_received,
                        oracle_snapshot=oracle_snapshot,
                    )
            finally:
                admission.release(ticket)

        def _emit_track_b_capture_row(
            self,
            *,
            capture: TrackBRequestMetricsCapture,
            request_class: str,
            upstream_status: int,
            metrics_before: dict[str, float],
            capture_state: dict[str, Any],
            ts_request_received: float,
            oracle_snapshot: dict[str, Any] | None = None,
        ) -> None:
            try:
                ts_completed = time.time()
                metrics_after = capture.fetch_metrics_snapshot()
                deltas = capture.compute_deltas(metrics_before, metrics_after)
                request_id = capture_state.get("response_id") or ""
                row = _build_request_metrics_row(
                    request_id=str(request_id),
                    request_path=self.path,
                    request_class=request_class,
                    upstream_status=upstream_status,
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    deltas=deltas,
                    response_observed=capture_state,
                    ts_request_received=ts_request_received,
                    ts_first_byte=capture_state.get("ts_first_byte"),
                    ts_completed=ts_completed,
                    saw_completed=bool(capture_state.get("saw_response_completed")),
                    oracle_snapshot=oracle_snapshot,
                )
                if not row.get("request_id"):
                    return
                capture.record(row)
            except Exception:
                # Capture must never break inference traffic.
                pass

        def _handle_admin_request(self) -> None:
            try:
                payload = _read_json_body(self)
                if self.path == "/admin/load_tuned_config":
                    bundle_path = _validate_load_tuned_config_payload(payload)
                    bundle = load_tuned_config_bundle(bundle_path)
                    _normalize_request_shaping_policy(bundle_path, bundle)
                    policy = str(payload.get("bundle_confidence_policy") or os.environ.get("LUMO_BUNDLE_CONFIDENCE_POLICY") or "warn")
                    validate_bundle_load_policy(bundle, bundle_confidence_policy=policy)
                    if registry and bundle.model_id not in registry:
                        raise StructuredValidationError(
                            message="Invalid load_tuned_config payload",
                            issues=[
                                ValidationIssue(
                                    field="bundle_path",
                                    message=f"bundle model_id {bundle.model_id!r} is not present in registry",
                                    value=bundle_path,
                                )
                            ],
                        )
                    state_store.activate_bundle(bundle_path, bundle)
                    _write_json_payload(
                        self,
                        200,
                        {
                            "status": "loaded",
                            "bundle_id": bundle.bundle_id,
                            "model_id": bundle.model_id,
                            "weight_version_id": bundle.weight_version_id,
                            "bundle_path": str(Path(bundle_path)),
                        },
                    )
                    return
                if self.path == "/admin/invalidate":
                    weight_version_id = _validate_invalidate_payload(payload)
                    state = state_store.record_invalidate(weight_version_id=weight_version_id)
                    flush_status = "not_attempted"
                    try:
                        response = requests.post(
                            f"{upstream_base_url}/reset_prefix_cache",
                            timeout=10,
                            headers=_filtered_headers(self.headers),
                        )
                        response.raise_for_status()
                        flush_status = "flushed"
                    except requests.RequestException:
                        flush_status = "unreachable"
                    _write_json_payload(
                        self,
                        200,
                        {
                            "status": "invalidated",
                            "weight_version_id": weight_version_id,
                            "state": state.status,
                            "invalidate_count": state.invalidate_count,
                            "flush_prefix_cache": flush_status,
                        },
                    )
                    return
            except StructuredValidationError as exc:
                _write_json_payload(self, 400, exc.as_error_payload())
                return
            except Exception as exc:
                _write_json_payload(
                    self,
                    500,
                    {
                        "error": {
                            "code": "internal_error",
                            "message": str(exc),
                        }
                    },
                )
                return
            _write_json_error(self, 404, "Unknown admin endpoint")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return ProxyHandler


def run_proxy_server(
    *,
    listen_host: str,
    listen_port: int,
    upstream_base_url: str,
    pid_file: Path | None = None,
    state_root: Path | None = None,
    registry_path: Path | None = None,
) -> None:
    server = ThreadingHTTPServer(
        (listen_host, listen_port),
        build_proxy_handler(upstream_base_url, state_root=state_root, registry_path=registry_path),
    )
    pid_written = False
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        pid_written = True
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        if pid_written and pid_file is not None:
            pid_file.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference-only proxy for Codex -> vLLM traffic")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args()
    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[PROXY-INIT] pid={os.getpid()} listen={args.listen_host}:{args.listen_port} "
                f"upstream={args.upstream_base_url}\n"
            )
    run_proxy_server(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_base_url=args.upstream_base_url,
        pid_file=args.pid_file,
        state_root=args.state_root,
        registry_path=args.registry_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

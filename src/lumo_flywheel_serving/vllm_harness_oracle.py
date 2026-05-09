"""Harness-oracle skeleton — vLLM side.

This module is the in-vLLM consumer of the X-Lumo-Oracle header that
``inference_proxy.synthesize_oracle_snapshot`` populates. It is the
``harness_oracle.py`` artefact described in
``docs/reports/auto_research/track-b-harness-oracle-api-skeleton-20260509.md``
("Recommendation: Land harness_oracle.py module ... as commit #1 of
Round 2").

Pure addition — no behavioural delta until a consumer (drafter
coordinator extension, Step 4+) reads from it. The proxy already
emits the header as of 2026-05-09 (`Step 3 phase 1` commit), so the
skeleton has a producer to round-trip against.

Deployed inside the running vLLM container via the prelaunch hook
in ``scripts.run_track_b_loop._track_b_runtime_prelaunch_shell``,
which copies this file's contents to
``/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/harness_oracle.py``
on each vLLM relaunch (idempotent — overwrites stale copies). The
copy is byte-for-byte; do not introduce relative imports here or the
prelaunch drop will fail at vLLM import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ORACLE_HEADER = "X-Lumo-Oracle"
ORACLE_SCHEMA = "lumo.harness_oracle_snapshot.v1"
DEFAULT_SUFFIX_TREE_CAP_MB = 100


@dataclass
class HarnessOracleSnapshot:
    """Per-request harness signals consumed by the drafter coordinator.

    Passive snapshot: the request envelope carries it, the drafter
    reads it, the harness produces it. No bidirectional channel.
    Every field is optional — absence means "no harness signal for
    this technique" and the drafter falls back to its default
    (vanilla SuffixDecoding) behaviour."""

    session_id: str | None = None
    turn_index: int | None = None
    is_session_open: bool = False
    is_session_close: bool = False
    dialect: str | None = None
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_call: dict[str, Any] | None = None
    primed_texts: list[dict[str, Any]] = field(default_factory=list)
    plan_fingerprint: dict[str, Any] | None = None
    suffix_tree_cap_mb: int = DEFAULT_SUFFIX_TREE_CAP_MB
    schema_version: str = ORACLE_SCHEMA

    @property
    def is_empty(self) -> bool:
        """True when no harness signal is present.

        Drafter coordinator uses this to short-circuit its per-session
        bookkeeping — when ``is_empty`` we should behave indistinguishably
        from un-instrumented traffic."""

        return (
            self.session_id is None
            and self.turn_index is None
            and not self.is_session_open
            and not self.is_session_close
            and not self.tool_schemas
            and self.expected_tool_call is None
            and not self.primed_texts
            and self.plan_fingerprint is None
        )


def parse_oracle_header(header_value: str | None) -> HarnessOracleSnapshot:
    """Parse the ``X-Lumo-Oracle`` HTTP header into a snapshot.

    Tolerant of: missing header, malformed JSON, unknown future fields,
    wrong types in known fields. Any failure yields an empty snapshot
    so the drafter never crashes on harness-side bugs."""

    if not header_value:
        return HarnessOracleSnapshot()
    try:
        raw = json.loads(header_value)
    except (TypeError, ValueError):
        return HarnessOracleSnapshot()
    if not isinstance(raw, dict):
        return HarnessOracleSnapshot()
    return _snapshot_from_mapping(raw)


def _snapshot_from_mapping(raw: dict[str, Any]) -> HarnessOracleSnapshot:
    def _str_or_none(key: str) -> str | None:
        value = raw.get(key)
        return value if isinstance(value, str) else None

    def _int_or_none(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _list_of_dicts(key: str) -> list[dict[str, Any]]:
        value = raw.get(key)
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _dict_or_none(key: str) -> dict[str, Any] | None:
        value = raw.get(key)
        return value if isinstance(value, dict) else None

    suffix_cap = _int_or_none("suffix_tree_cap_mb")
    return HarnessOracleSnapshot(
        session_id=_str_or_none("session_id"),
        turn_index=_int_or_none("turn_index"),
        is_session_open=bool(raw.get("is_session_open", False)),
        is_session_close=bool(raw.get("is_session_close", False)),
        dialect=_str_or_none("dialect"),
        tool_schemas=_list_of_dicts("tool_schemas"),
        expected_tool_call=_dict_or_none("expected_tool_call"),
        primed_texts=_list_of_dicts("primed_texts"),
        plan_fingerprint=_dict_or_none("plan_fingerprint"),
        suffix_tree_cap_mb=suffix_cap if suffix_cap is not None else DEFAULT_SUFFIX_TREE_CAP_MB,
        schema_version=_str_or_none("schema") or _str_or_none("schema_version") or ORACLE_SCHEMA,
    )


def encode_oracle_header(snapshot: HarnessOracleSnapshot) -> str:
    """Serialise a snapshot back to header form. Inverse of
    ``parse_oracle_header``; round-trip-stable on the union of fields
    both ends understand."""

    out: dict[str, Any] = {"schema": snapshot.schema_version}
    if snapshot.session_id is not None:
        out["session_id"] = snapshot.session_id
    if snapshot.turn_index is not None:
        out["turn_index"] = snapshot.turn_index
    if snapshot.is_session_open:
        out["is_session_open"] = True
    if snapshot.is_session_close:
        out["is_session_close"] = True
    if snapshot.dialect is not None:
        out["dialect"] = snapshot.dialect
    if snapshot.tool_schemas:
        out["tool_schemas"] = snapshot.tool_schemas
    if snapshot.expected_tool_call is not None:
        out["expected_tool_call"] = snapshot.expected_tool_call
    if snapshot.primed_texts:
        out["primed_texts"] = snapshot.primed_texts
    if snapshot.plan_fingerprint is not None:
        out["plan_fingerprint"] = snapshot.plan_fingerprint
    out["suffix_tree_cap_mb"] = snapshot.suffix_tree_cap_mb
    return json.dumps(out, separators=(",", ":"), sort_keys=True)


# ---- Oracle registry -- T3 phase 2 foundation ---------------------
#
# Module-level dict the FastAPI middleware (patched into vLLM's
# build_app via the prelaunch hook) writes to on each /v1/responses
# request, and that ``SuffixDecodingProposer.propose()`` reads from
# once T3 phase 3's composite drafting integration ships. Keyed by
# the session-prefixed ``X-Request-Id`` the proxy emits, so a single
# lookup in propose() (which already has ``req_id``) recovers the
# parsed oracle without re-parsing the JSON header per decode step.
#
# Eviction: simple bounded LRU. The registry should always be small
# (one entry per concurrent request) so a generous bound + insertion-
# order eviction is sufficient. Cleanup of finished requests happens
# explicitly via ``unregister`` from the request-finalize middleware
# path; the bound is just a safety net for crashed requests.

import collections
import threading

_REGISTRY_DEFAULT_MAX_ENTRIES = 4096


class OracleRegistry:
    """Thread-safe, LRU-bounded {request_id: HarnessOracleSnapshot} map.

    Concurrency: vLLM's FastAPI handlers run on the asyncio event loop
    while ``SuffixDecodingProposer.propose()`` runs on a separate
    worker (the model runner). A plain ``threading.Lock`` is enough;
    we never hold it across an await."""

    def __init__(self, max_entries: int = _REGISTRY_DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._entries: "collections.OrderedDict[str, HarnessOracleSnapshot]" = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()

    def register(self, request_id: str, snapshot: HarnessOracleSnapshot) -> None:
        if not isinstance(request_id, str) or not request_id:
            return
        with self._lock:
            self._entries[request_id] = snapshot
            self._entries.move_to_end(request_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def lookup(self, request_id: str) -> HarnessOracleSnapshot | None:
        if not isinstance(request_id, str):
            return None
        with self._lock:
            snap = self._entries.get(request_id)
            if snap is not None:
                self._entries.move_to_end(request_id)
            return snap

    def unregister(self, request_id: str) -> None:
        if not isinstance(request_id, str):
            return
        with self._lock:
            self._entries.pop(request_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# Singleton — the module-level dict the middleware and propose()
# share. Importers should treat this as read-mostly data; mutation
# is the middleware's responsibility.
ORACLE_REGISTRY = OracleRegistry()


def install_fastapi_middleware(app: Any) -> None:
    """Register the X-Lumo-Oracle harvesting middleware on a FastAPI app.

    Idempotent: re-registering the same middleware on the same app is
    a no-op (we tag the app with ``app.state.lumo_oracle_middleware =
    True``). The middleware reads X-Lumo-Oracle + X-Request-Id from
    every inbound request and stashes the parsed snapshot in
    ``ORACLE_REGISTRY``; on response completion it unregisters so the
    map stays bounded.

    Designed to be applied via vLLM's prelaunch hook, which patches
    ``vllm/entrypoints/openai/api_server.py``'s ``build_app`` to call
    this helper after vLLM's own middlewares are in place. The
    registration order doesn't matter because all we do is a
    parse + dict-insert (no transformation of request/response data).
    """

    if getattr(getattr(app, "state", None), "lumo_oracle_middleware", False):
        return
    registry = ORACLE_REGISTRY

    @app.middleware("http")
    async def _lumo_oracle_capture(request: Any, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-Id")
        if not request_id:
            return await call_next(request)
        oracle_value = request.headers.get(ORACLE_HEADER)
        if oracle_value:
            snapshot = parse_oracle_header(oracle_value)
            if not snapshot.is_empty:
                registry.register(request_id, snapshot)
        try:
            return await call_next(request)
        finally:
            registry.unregister(request_id)

    if getattr(app, "state", None) is not None:
        try:
            app.state.lumo_oracle_middleware = True
        except Exception:
            pass

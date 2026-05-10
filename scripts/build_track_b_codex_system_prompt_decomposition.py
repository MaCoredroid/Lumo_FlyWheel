#!/usr/bin/env python3
"""Decompose a captured Codex /v1/responses request into per-section token
counts and content hashes. Produces an artifact validating against
``lumo.track_b.codex_system_prompt_decomposition.v1`` per Track B Round 4a §6.

Inputs:
  --in <path>     captured request_json (from scripts/capture_codex_request_body.py)
  --out <path>    decomposition JSON to write
  --tokenize-url  vLLM /tokenize endpoint
  --model         model name passed to /tokenize

Output schema (lumo.track_b.codex_system_prompt_decomposition.v1):

```
{
  "schema": "lumo.track_b.codex_system_prompt_decomposition.v1",
  "round": <int|null>,
  "ts": "2026-...Z",
  "runtime_config_hash": "<str>",
  "codex_version": "<str>",
  "model": "<str>",
  "total_static_tokens": <int>,         # instructions + tools (task-agnostic)
  "total_static_chars": <int>,
  "total_request_tokens_est": <int>,    # approximation; observed prompt_tokens differ
                                        # by chat-template overhead (~2K)
  "static_content_hash": "sha256:..",   # of canonical (instructions + tools_json)
  "sections": [
    {"name": "instructions", "kind": "field", "chars": <int>, "tokens": <int>,
     "content_hash": "sha256:.."},
    {"name": "tools", "kind": "field", "chars": <int>, "tokens": <int>,
     "content_hash": "sha256:..", "tool_count": <int>},
    {"name": "input.developer", "kind": "input_role", "chars": <int>, "tokens": <int>,
     "content_hash": "sha256:.."},
    {"name": "input.env_context", "kind": "input_role", "chars": <int>, "tokens": <int>,
     "content_hash": "sha256:.."},
    {"name": "input.user_message", "kind": "input_role", "chars": <int>, "tokens": <int>,
     "content_hash": "sha256:.."}
  ],
  "tool_breakdown": [
    {"name": "<tool_name>", "chars": <int>, "tokens": <int>, "content_hash": "sha256:.."},
    ...
  ]
}
```

The "static" (task-agnostic) prefix used by warmup-pass = ``instructions`` + ``tools``.
The ``input.*`` items vary per task (env_context contains cwd; user_message contains
the prompt) and are recorded for visibility but not part of the warmup payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

SCHEMA = "lumo.track_b.codex_system_prompt_decomposition.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _tokenize(text: str, *, url: str, model: str, timeout: float = 60.0) -> int:
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    n = data.get("count")
    if n is None:
        toks = data.get("tokens", [])
        n = len(toks) if isinstance(toks, list) else 0
    return int(n)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text")
                if isinstance(t, str):
                    out.append(t)
        return "\n".join(out)
    return ""


def _classify_input_item(item: dict[str, Any]) -> str:
    role = item.get("role") or "?"
    text = _text_of(item.get("content"))
    if role == "developer" or "<permissions" in text or "<sandbox" in text:
        return "input.developer"
    if "<environment_context>" in text or "<cwd>" in text:
        return "input.env_context"
    if role == "user":
        return "input.user_message"
    return f"input.{role}"


def decompose(
    request: dict[str, Any],
    *,
    tokenize_url: str,
    model: str,
    round_index: int | None,
    runtime_config_hash: str,
    codex_version: str,
) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    tool_breakdown: list[dict[str, Any]] = []

    # instructions
    instructions = request.get("instructions") or ""
    if not isinstance(instructions, str):
        raise SystemExit("captured request has non-string 'instructions' field")
    sections.append({
        "name": "instructions",
        "kind": "field",
        "chars": len(instructions),
        "tokens": _tokenize(instructions, url=tokenize_url, model=model) if instructions else 0,
        "content_hash": _sha256(instructions),
    })

    # tools
    tools = request.get("tools") or []
    if not isinstance(tools, list):
        raise SystemExit("captured request has non-list 'tools' field")
    tools_json = json.dumps(tools, sort_keys=True)
    sections.append({
        "name": "tools",
        "kind": "field",
        "chars": len(tools_json),
        "tokens": _tokenize(tools_json, url=tokenize_url, model=model) if tools else 0,
        "content_hash": _sha256(tools_json),
        "tool_count": len(tools),
    })
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or (t.get("function") or {}).get("name") or "<unknown>"
        s = json.dumps(t, sort_keys=True)
        tool_breakdown.append({
            "name": name,
            "chars": len(s),
            "tokens": _tokenize(s, url=tokenize_url, model=model),
            "content_hash": _sha256(s),
        })

    # input items
    input_items = request.get("input") or []
    if not isinstance(input_items, list):
        raise SystemExit("captured request has non-list 'input' field")
    for item in input_items:
        if not isinstance(item, dict):
            continue
        text = _text_of(item.get("content"))
        sections.append({
            "name": _classify_input_item(item),
            "kind": "input_role",
            "role": item.get("role"),
            "chars": len(text),
            "tokens": _tokenize(text, url=tokenize_url, model=model) if text else 0,
            "content_hash": _sha256(text),
        })

    # The "static" task-agnostic portion that warmup-pass pre-caches:
    # canonical(instructions + tools_json). Hash these as a stable identifier
    # so cross-task / cross-attempt comparisons can detect Codex CLI drift.
    static_canonical = instructions + "\n\x1f\n" + tools_json  # 0x1f = unit separator
    static_chars = len(static_canonical)
    static_tokens = _tokenize(static_canonical, url=tokenize_url, model=model) if static_canonical else 0
    static_hash = _sha256(static_canonical)

    # Estimate of total request prompt_tokens (sum of section tokens; the
    # observed value will differ by chat-template overhead — typically
    # ~2-3K extra for role tags and special tokens).
    total_request_tokens_est = sum(int(s["tokens"]) for s in sections)

    return {
        "schema": SCHEMA,
        "round": round_index,
        "ts": _now(),
        "runtime_config_hash": runtime_config_hash,
        "codex_version": codex_version,
        "model": model,
        "total_static_tokens": static_tokens,
        "total_static_chars": static_chars,
        "total_request_tokens_est": total_request_tokens_est,
        "static_content_hash": static_hash,
        "sections": sections,
        "tool_breakdown": tool_breakdown,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="captured request_json path")
    p.add_argument("--out", required=True, help="decomposition output JSON path")
    p.add_argument("--tokenize-url", default="http://127.0.0.1:9950/tokenize")
    p.add_argument("--model", default="qwen3.5-27b")
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--runtime-config-hash", default="")
    p.add_argument("--codex-version", default="")
    args = p.parse_args()

    request = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        print("error: input is not a JSON object", file=sys.stderr)
        return 2
    payload = decompose(
        request,
        tokenize_url=args.tokenize_url,
        model=args.model,
        round_index=args.round,
        runtime_config_hash=args.runtime_config_hash,
        codex_version=args.codex_version,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "static_content_hash": payload["static_content_hash"],
        "total_static_tokens": payload["total_static_tokens"],
        "total_request_tokens_est": payload["total_request_tokens_est"],
        "section_count": len(payload["sections"]),
        "tool_count": len(payload["tool_breakdown"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

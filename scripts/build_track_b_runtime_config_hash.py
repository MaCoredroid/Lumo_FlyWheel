#!/usr/bin/env python3
"""Compute a Track B runtime_config_hash from the live vLLM init log.

Reads the [VLLM-INIT] block written by ModelServer's prelaunch hook (one line
per init field) and emits a deterministic ``sha256:<hex>`` digest over a
canonical JSON dict of the load-bearing fields. The hash replaces the
placeholder ``sha256:aaaaaaaa...`` and lets multi-task summaries detect
mid-sweep config drift.

Output (stdout): the hash string. Optional ``--out`` writes a small JSON
manifest so the readiness manifest can audit the inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.5-27b.log")
HASH_FIELDS: tuple[str, ...] = (
    "model_id",
    "served_model_name",
    "vllm_version",
    "git_hash",
    "quantization",
    "kv_cache_dtype",
    "max_model_len",
    "gpu_memory_utilization",
    "enforce_eager",
    "tuned_config_id",
    "weight_version_id",
    "kernel_runtime_activation",
    "speculative_config",
    "wire_api",
)
KEY_VALUE_RE = re.compile(r"^\[VLLM-INIT\]\s+(.+)$")


def parse_init_log(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for line in text.splitlines():
        match = KEY_VALUE_RE.match(line.strip())
        if not match:
            continue
        body = match.group(1)
        if "=" not in body:
            continue
        # JSON-shaped values: the line is "key=<json>" with the JSON containing spaces.
        first_eq = body.index("=")
        first_value_char = body[first_eq + 1 : first_eq + 2]
        if first_value_char in {"{", "["}:
            key = body[:first_eq].strip()
            raw_value = body[first_eq + 1 :].strip()
            try:
                fields[key] = json.loads(raw_value)
                continue
            except json.JSONDecodeError:
                fields[key] = raw_value
                continue
        # Otherwise: the rest of the line may have multiple "k=v" tokens separated by whitespace.
        for token in body.split():
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            fields[k.strip()] = _coerce(v.strip())
    return fields


def _coerce(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if raw.startswith("0") and raw not in {"0", "0.0"} and not raw.startswith("0."):
            return raw
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def canonical_payload(fields: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in HASH_FIELDS:
        if key in fields:
            payload[key] = fields[key]
    return payload


def compute_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute the Track B runtime_config_hash from the live vLLM init log.")
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Path to the vLLM init log written by ModelServer prelaunch hook (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional manifest output. JSON with the hashed payload + the resulting digest.",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"build_track_b_runtime_config_hash: log not found: {args.log}", file=sys.stderr)
        return 2
    fields = parse_init_log(args.log.read_text(encoding="utf-8", errors="replace"))
    payload = canonical_payload(fields)
    if not payload:
        print("build_track_b_runtime_config_hash: log did not contain any [VLLM-INIT] fields", file=sys.stderr)
        return 1
    digest = compute_hash(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema": "lumo.track_b.runtime_config_hash.v1",
                    "log_path": str(args.log),
                    "fields": payload,
                    "runtime_config_hash": digest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests
import yaml


def _default_entries(avg_prompt_tokens: int, avg_output_tokens: int, count: int) -> list[dict[str, int]]:
    entries: list[dict[str, int]] = []
    for index in range(count):
        entries.append(
            {
                "turn_index": index,
                "prompt_tokens": max(1, avg_prompt_tokens - (index * 64)),
                "output_tokens": max(1, avg_output_tokens - (index * 24)),
                "thinking_tokens": 0,
            }
        )
    return entries


def _capture_live(base_url: str, model: str, count: int) -> list[dict[str, int]]:
    entries: list[dict[str, int]] = []
    for index in range(count):
        prompt = f"Seed capture turn {index}. Reply with a short summary."
        response = requests.post(
            f"{base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY') or 'EMPTY'}"},
            json={"model": model, "input": prompt, "max_output_tokens": 128},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        entries.append(
            {
                "turn_index": index,
                "prompt_tokens": int(usage.get("input_tokens", len(prompt.split()))),
                "output_tokens": int(usage.get("output_tokens", 64)),
                "thinking_tokens": int(usage.get("reasoning_tokens", 0)),
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture or synthesize a seed workload trace for auto-research.")
    parser.add_argument("--workload-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--update-workload", action="store_true")
    args = parser.parse_args()

    workload = yaml.safe_load(args.workload_file.read_text(encoding="utf-8"))
    if not isinstance(workload, dict):
        raise SystemExit(f"Workload file must be a mapping: {args.workload_file}")
    output_path = args.output or args.workload_file.with_name("seed_trace.jsonl")

    if args.base_url:
        entries = _capture_live(args.base_url, args.model, args.count)
    else:
        entries = _default_entries(
            int(workload.get("avg_prompt_tokens", 4096)),
            int(workload.get("avg_output_tokens", 1200)),
            args.count,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    if args.update_workload:
        workload["seed_trace_ref"] = output_path.name if output_path.parent == args.workload_file.parent else str(output_path)
        workload["workload_distribution_id"] = workload.get("workload_distribution_id") or digest[:12]
        args.workload_file.write_text(yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "count": len(entries),
                "sha256": digest,
                "workload_distribution_id": digest[:12],
                "live_capture": bool(args.base_url),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

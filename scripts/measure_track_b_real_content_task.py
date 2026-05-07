#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.metrics import compute_task_metrics, parse_prometheus_text, resolve_metric_schema  # noqa: E402


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip().replace("\n", "\\n")
        if len(body) > 1200:
            body = body[:1200] + "...<truncated>"
        raise requests.HTTPError(f"{exc}; response_body={body or '<empty>'}", response=response) from exc


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").rstrip()


def _load_metadata(family_dir: Path) -> dict[str, Any]:
    metadata_path = family_dir / "report" / "attempt_01_live_probe" / "metadata.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _section(title: str, text: str) -> str:
    return f"## {title}\n\n{text.strip()}\n"


def _build_prompt(family: str, variant: str) -> tuple[str, dict[str, Any]]:
    family_dir = REPO_ROOT / "benchmark_blueprints" / "families" / family
    workspace = family_dir / "workspace_bundle" / variant
    if not workspace.is_dir():
        raise RuntimeError(f"workspace bundle missing: {workspace}")
    metadata = _load_metadata(family_dir)
    instruction = str(metadata.get("prompt") or "Read AGENTS.md in this directory and follow it exactly.")
    sections = [f"{instruction}\n"]
    included: list[str] = []
    for rel in ["AGENTS.md", ".scenario_variant"]:
        path = workspace / rel
        if path.is_file():
            sections.append(_section(rel, _read(path)))
            included.append(rel)
    for directory in ["release_notes", "repo_inventory"]:
        for path in sorted((workspace / directory).glob("*")):
            if path.is_file():
                rel = path.relative_to(workspace).as_posix()
                sections.append(_section(rel, _read(path)))
                included.append(rel)
    prompt = "\n".join(sections).strip() + "\n"
    return prompt, {
        "family": family,
        "variant": variant,
        "workspace_bundle": str(workspace.relative_to(REPO_ROOT)),
        "source_prompt_metadata": str((family_dir / "report" / "attempt_01_live_probe" / "metadata.json").relative_to(REPO_ROOT)),
        "included_files": included,
        "prompt_chars": len(prompt),
        "prompt_word_count": len(prompt.split()),
    }


def _metrics(metrics_url: str) -> dict[str, float]:
    response = requests.get(metrics_url, timeout=20)
    _raise_for_status_with_body(response)
    return parse_prometheus_text(response.text)


def _metric_delta(before: dict[str, float], after: dict[str, float], *, elapsed_s: float, request_count: int) -> dict[str, Any]:
    schema = resolve_metric_schema(after)
    metrics = compute_task_metrics(before=before, after=after, schema=schema)
    gen_tokens = float(metrics.get("gen_tokens") or 0.0)
    decode_sum_s = float(metrics.get("decode_sum_s") or 0.0)
    prefill_sum_s = float(metrics.get("prefill_sum_s") or 0.0)
    prompt_tokens = float(metrics.get("prompt_tokens") or 0.0)
    kv_tokens = float(metrics.get("kv_computed_tokens") or 0.0)
    return {
        "generation_tokens": gen_tokens,
        "decode_sum_s": decode_sum_s,
        "prefill_sum_s": prefill_sum_s,
        "prompt_tokens": prompt_tokens,
        "kv_computed_tokens": kv_tokens,
        "decode_tokens_per_s": round(gen_tokens / decode_sum_s, 6) if decode_sum_s > 0 else None,
        "wall_output_tokens_per_s": round(gen_tokens / elapsed_s, 6) if elapsed_s > 0 else None,
        "prompt_tokens_per_request": round(prompt_tokens / max(request_count, 1), 3),
        "generation_tokens_per_request": round(gen_tokens / max(request_count, 1), 3),
    }


def _spec_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    def delta(name: str) -> float:
        return float(after.get(name) or 0.0) - float(before.get(name) or 0.0)

    drafts = delta("vllm:spec_decode_num_drafts_total")
    draft_tokens = delta("vllm:spec_decode_num_draft_tokens_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    return {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "accepted_per_draft_token": round(accepted / draft_tokens, 6) if draft_tokens > 0 else None,
        "draft_tokens_per_draft": round(draft_tokens / drafts, 6) if drafts > 0 else None,
    }


def _extract_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
            elif isinstance(item.get("text"), str):
                chunks.append(item["text"])
    if isinstance(payload.get("output_text"), str):
        chunks.append(payload["output_text"])
    return "\n".join(chunks)


def _text_summary(text: str) -> dict[str, Any]:
    words = re.findall(r"\S+", text)
    eightgrams = Counter(tuple(words[i : i + 8]) for i in range(max(0, len(words) - 7)))
    return {
        "text_chars": len(text),
        "word_count": len(words),
        "unique_word_count": len(set(words)),
        "unique_word_ratio": round(len(set(words)) / len(words), 6) if words else None,
        "most_common_8gram_count": eightgrams.most_common(1)[0][1] if eightgrams else 0,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "head_600": text[:600],
        "tail_600": text[-600:],
    }


def _post_response(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    label: str,
    request_overrides: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    request_payload: dict[str, Any] = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}
    request_payload.update(request_overrides)
    response = requests.post(
        f"{endpoint.rstrip('/')}/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_payload,
        timeout=max(120, max_output_tokens * 3),
    )
    wall_s = time.monotonic() - started
    _raise_for_status_with_body(response)
    payload = response.json()
    if not isinstance(payload, dict):
        payload = {}
    text = _extract_text(payload)
    return {
        "label": label,
        "status": payload.get("status"),
        "response_id": payload.get("id"),
        "usage": payload.get("usage"),
        "wall_s": round(wall_s, 6),
        "text_summary": _text_summary(text),
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    prompt, prompt_meta = _build_prompt(args.family, args.variant)
    request_overrides = json.loads(args.request_overrides_json)
    if not isinstance(request_overrides, dict):
        raise RuntimeError("--request-overrides-json must decode to a JSON object")
    requests.get(args.health_url, timeout=20).raise_for_status()
    if args.reset_prefix_cache:
        reset = requests.post(args.reset_prefix_cache_url, headers={"Authorization": f"Bearer {args.api_key}"}, timeout=30)
        _raise_for_status_with_body(reset)
    cold = _post_response(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        prompt=prompt,
        max_output_tokens=args.max_output_tokens,
        label="cold_discard",
        request_overrides=request_overrides,
    )
    before = _metrics(args.metrics_url)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.warm_concurrency) as executor:
        futures = [
            executor.submit(
                _post_response,
                endpoint=args.endpoint,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_output_tokens=args.max_output_tokens,
                label=f"warm_{index + 1}",
                request_overrides=request_overrides,
            )
            for index in range(args.warm_concurrency)
        ]
        warm = [future.result() for future in as_completed(futures)]
    elapsed_s = time.monotonic() - started
    after = _metrics(args.metrics_url)
    warm.sort(key=lambda row: row["label"])
    metrics_delta = _metric_delta(before, after, elapsed_s=elapsed_s, request_count=len(warm))
    spec_delta = _spec_delta(before, after)
    decode_tps = metrics_delta["decode_tokens_per_s"]
    return {
        "schema": "lumo.track_b.real_content_task_warmonly_probe.v2",
        "measured_at": _now(),
        "endpoint": args.endpoint,
        "model": args.model,
        "agentic_task": args.family,
        "task_id": f"{args.family}/{args.variant}",
        **prompt_meta,
        "max_output_tokens": args.max_output_tokens,
        "warm_concurrency": args.warm_concurrency,
        "request_overrides": request_overrides,
        "measurement_policy": "one cold completion discarded; metrics sampled before and after warm window only",
        "cold_discarded": cold,
        "warm": warm,
        "warm_elapsed_s": round(elapsed_s, 6),
        "warm_usage_output_tokens_total": metrics_delta["generation_tokens"],
        "metrics_delta": metrics_delta,
        "spec_decode_delta": spec_delta,
        "decode_tps": decode_tps,
        "wall_output_tps": metrics_delta["wall_output_tokens_per_s"],
        "target_decode_tps": args.target_decode_tps,
        "pass_target_gate": bool(decode_tps is not None and decode_tps >= args.target_decode_tps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Track B on a real authored benchmark prompt.")
    parser.add_argument("--family", default="release-note-to-plan-translation")
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--warm-concurrency", type=int, default=1)
    parser.add_argument("--request-overrides-json", default="{}")
    parser.add_argument("--target-decode-tps", type=float, default=30.0)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_output_tokens < 1:
        raise RuntimeError("--max-output-tokens must be >= 1")
    if args.warm_concurrency < 1:
        raise RuntimeError("--warm-concurrency must be >= 1")
    result = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass_target_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

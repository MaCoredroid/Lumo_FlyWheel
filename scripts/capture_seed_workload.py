#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

import requests
import yaml


V5_REASONING_MAX_OUTPUT_TOKENS = 4096
V5_SHORT_MAX_OUTPUT_TOKENS = 512


def _read_excerpt(path: Path, *, max_chars: int) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0]


def _family_context(repo_root: Path, family_id: str) -> str:
    family_dir = repo_root / "benchmark_blueprints" / "families" / family_id
    parts: list[str] = []
    for relative, max_chars in (
        ("family.yaml", 800),
        ("task_spec.md", 2200),
        ("verification_matrix_v5.md", 1800),
    ):
        excerpt = _read_excerpt(family_dir / relative, max_chars=max_chars)
        if excerpt:
            parts.append(f"## {relative}\n{excerpt}")
    v5_matrices = sorted(family_dir.glob("verification_matrix_v5*.md"))
    if not any(part.startswith("## verification_matrix_v5.md") for part in parts) and v5_matrices:
        excerpt = _read_excerpt(v5_matrices[0], max_chars=1800)
        if excerpt:
            parts.append(f"## {v5_matrices[0].name}\n{excerpt}")
    if not parts:
        parts.append(f"## family_id\n{family_id}")
    return "\n\n".join(parts)


def _variant_label(repo_root: Path, family_id: str) -> str:
    family_dir = repo_root / "benchmark_blueprints" / "families" / family_id
    variants = sorted(path.name for path in (family_dir / "workspace_bundle").glob("v5-*"))
    if variants:
        return variants[0]
    matrices = sorted(family_dir.glob("verification_matrix_v5*.md"))
    if matrices:
        match = re.search(r"verification_matrix_(v5[^.]*)", matrices[0].name)
        if match:
            return match.group(1).replace("_", "-")
    return "v5"


def _live_prompt_for_turn(
    *,
    index: int,
    repo_root: Path,
    family_id: str | None,
    variant: str | None,
    requested_max_output_tokens: int | None,
) -> tuple[str, int, str]:
    if family_id and variant == "v5" and requested_max_output_tokens is None:
        context = _family_context(repo_root, family_id)
        variant_label = _variant_label(repo_root, family_id)
        prompt_templates = [
            (
                "v5_deep_failure_analysis",
                V5_REASONING_MAX_OUTPUT_TOKENS,
                "Think carefully about this benchmark family before giving a one-sentence final answer. "
                "Consider hidden constraints, likely false fixes, and verification risks in depth.\n\n"
                f"Family: {family_id}\nVariant: {variant_label}",
            ),
            (
                "v5_acceptance_summary",
                V5_SHORT_MAX_OUTPUT_TOKENS,
                "From the benchmark family artifacts below, summarize the v5 acceptance criteria, the expected work "
                "surface, and the most important regression checks. Keep the final answer concise.\n\n"
                f"Family: {family_id}\nVariant: {variant_label}\n\n{context}",
            ),
            (
                "v5_implementation_plan",
                V5_SHORT_MAX_OUTPUT_TOKENS,
                "Read the benchmark family artifacts below and draft a concise implementation plan that would avoid "
                "shortcuts and preserve unrelated work. Include only concrete steps and verification commands.\n\n"
                f"Family: {family_id}\nVariant: {variant_label}\n\n{context}",
            ),
            (
                "v5_holdout_deep_review",
                V5_REASONING_MAX_OUTPUT_TOKENS,
                "Think carefully about this benchmark family before giving a one-sentence final answer. "
                "Consider hidden constraints, likely false fixes, and verification risks in depth.\n\n"
                f"Family: {family_id}\nVariant: {variant_label}",
            ),
        ]
        return prompt_templates[index % len(prompt_templates)]
    max_output_tokens = requested_max_output_tokens or 128
    return (
        "generic_seed_capture",
        max_output_tokens,
        f"Seed capture turn {index}. Reply with a short summary.",
    )


def _int_usage(usage: dict, key: str, default: int = 0) -> int:
    try:
        return int(usage.get(key, default))
    except (TypeError, ValueError):
        return default


def _has_reasoning_output(payload: dict) -> bool:
    output = payload.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            return True
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type", "")).startswith("reasoning") and (part.get("text") or part.get("reasoning_text")):
                return True
    return False


def _reasoning_tokens_from_response(payload: dict, output_tokens: int) -> int:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return 0
    top_level = _int_usage(usage, "reasoning_tokens")
    if top_level > 0:
        return top_level
    details = usage.get("output_tokens_details", {})
    if isinstance(details, dict):
        nested = _int_usage(details, "reasoning_tokens")
        if nested > 0:
            return nested
    if output_tokens > 0 and _has_reasoning_output(payload):
        return output_tokens
    return 0


def _default_entries(
    avg_prompt_tokens: int,
    avg_output_tokens: int,
    count: int,
    *,
    family_id: str | None = None,
) -> list[dict[str, int | str]]:
    entries: list[dict[str, int | str]] = []
    for index in range(count):
        entry: dict[str, int | str] = {
            "turn_index": index,
            "prompt_tokens": max(1, avg_prompt_tokens - (index * 64)),
            "output_tokens": max(1, avg_output_tokens - (index * 24)),
            "thinking_tokens": 0,
        }
        if family_id:
            entry["family_id"] = family_id
        entries.append(entry)
    return entries


def _capture_live(
    base_url: str,
    model: str,
    count: int,
    *,
    api_key: str | None = None,
    family_id: str | None = None,
    variant: str | None = None,
    repo_root: Path = Path.cwd(),
    enable_thinking_override: bool = False,
    max_output_tokens: int | None = None,
) -> list[dict[str, int | str]]:
    entries: list[dict[str, int | str]] = []
    for index in range(count):
        prompt_label, request_max_output_tokens, prompt = _live_prompt_for_turn(
            index=index,
            repo_root=repo_root,
            family_id=family_id,
            variant=variant,
            requested_max_output_tokens=max_output_tokens,
        )
        payload: dict[str, object] = {"model": model, "input": prompt, "max_output_tokens": request_max_output_tokens}
        if enable_thinking_override:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        response = requests.post(
            f"{base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {api_key or os.environ.get('VLLM_API_KEY') or 'EMPTY'}"},
            json=payload,
            timeout=max(60, min(900, request_max_output_tokens)),
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        output_tokens = _int_usage(usage, "output_tokens", 64)
        thinking_tokens = _reasoning_tokens_from_response(payload, output_tokens) if isinstance(payload, dict) else 0
        response_tokens = max(output_tokens - thinking_tokens, 0)
        entry: dict[str, int | str] = {
            "turn_index": index,
            "prompt_tokens": _int_usage(usage, "input_tokens", len(prompt.split())),
            "output_tokens": output_tokens,
            "response_tokens": response_tokens,
            "thinking_tokens": thinking_tokens,
            "capture_prompt_label": prompt_label,
            "request_max_output_tokens": request_max_output_tokens,
        }
        if family_id:
            entry["family_id"] = family_id
        entries.append(entry)
    return entries


def _split_entries(
    entries: list[dict[str, int | str]],
    *,
    holdout_ratio: float,
    split_seed: int,
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
    if not 0.0 <= holdout_ratio < 1.0:
        raise SystemExit("--holdout-ratio must be >= 0.0 and < 1.0")
    if holdout_ratio == 0.0:
        return entries, []
    if len(entries) < 2:
        raise SystemExit("At least two entries are required when holdout capture is enabled")

    holdout_count = max(1, int(round(len(entries) * holdout_ratio)))
    holdout_count = min(holdout_count, len(entries) - 1)
    indexes = list(range(len(entries)))
    random.Random(split_seed).shuffle(indexes)
    holdout_indexes = set(indexes[:holdout_count])
    seed_entries = [entry for index, entry in enumerate(entries) if index not in holdout_indexes]
    holdout_entries = [entry for index, entry in enumerate(entries) if index in holdout_indexes]
    return seed_entries, holdout_entries


def _write_jsonl(path: Path, entries: list[dict[str, int | str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_ref(descriptor_path: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return descriptor_path.parent / path


def compute_workload_distribution_id(descriptor_path: Path) -> str:
    payload = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Workload descriptor must be a mapping: {descriptor_path}")
    seed_hash = hashlib.sha256(_resolve_ref(descriptor_path, str(payload["seed_trace_ref"])).read_bytes()).hexdigest()
    holdout_hash = hashlib.sha256(_resolve_ref(descriptor_path, str(payload["holdout_trace_ref"])).read_bytes()).hexdigest()
    payload["workload_distribution_id"] = None
    yaml_hash = hashlib.sha256(
        yaml.safe_dump(payload, sort_keys=True, default_flow_style=False).encode("utf-8")
    ).hexdigest()
    return hashlib.sha256((seed_hash + holdout_hash + yaml_hash).encode("ascii")).hexdigest()


def _family_dir(repo_root: Path, family_id: str) -> Path:
    return repo_root / "benchmark_blueprints" / "families" / family_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture or synthesize a seed workload trace for auto-research.")
    parser.add_argument("--workload-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--holdout-output", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--family-id")
    parser.add_argument("--variant", choices=["v5"])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--holdout-ratio", type=float)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--thinking-probe-outcome", choices=["row-1", "row-3"])
    parser.add_argument("--enable-thinking-override", action="store_true")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--update-workload", action="store_true")
    args = parser.parse_args()

    if args.variant and not args.family_id:
        raise SystemExit("--variant requires --family-id")
    if not args.workload_file and not args.family_id:
        raise SystemExit("--workload-file is required unless --family-id is provided")

    workload: dict[str, object] = {}
    workload_file: Path | None = args.workload_file
    if workload_file is None and args.family_id:
        candidate = _family_dir(args.repo_root, args.family_id) / "serving_workload.yaml"
        workload_file = candidate if candidate.exists() else None
    if workload_file is not None:
        loaded = yaml.safe_load(workload_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit(f"Workload file must be a mapping: {workload_file}")
        workload = loaded

    if args.family_id and args.variant == "v5":
        default_output = _family_dir(args.repo_root, args.family_id) / "seed_trace_v5.jsonl"
        default_holdout_output = _family_dir(args.repo_root, args.family_id) / "holdout_trace_v5.jsonl"
        default_holdout_ratio = 0.0
    elif workload_file is not None:
        default_output = workload_file.with_name("seed_trace.jsonl")
        default_holdout_output = workload_file.with_name("holdout_trace.jsonl")
        default_holdout_ratio = 0.10
    else:
        raise SystemExit("Unable to resolve output path")
    output_path = args.output or default_output
    holdout_output_path = args.holdout_output or default_holdout_output
    holdout_ratio = default_holdout_ratio if args.holdout_ratio is None else args.holdout_ratio
    enable_thinking_override = args.enable_thinking_override or args.thinking_probe_outcome == "row-1"

    if args.base_url:
        entries = _capture_live(
            args.base_url,
            args.model,
            args.count,
            api_key=args.api_key,
            family_id=args.family_id,
            variant=args.variant,
            repo_root=args.repo_root,
            enable_thinking_override=enable_thinking_override,
            max_output_tokens=args.max_output_tokens,
        )
    else:
        entries = _default_entries(
            int(workload.get("avg_prompt_tokens", 4096)),
            int(workload.get("avg_output_tokens", 1200)),
            args.count,
            family_id=args.family_id,
        )

    seed_entries, holdout_entries = _split_entries(
        entries,
        holdout_ratio=holdout_ratio,
        split_seed=args.split_seed,
    )
    seed_digest = _write_jsonl(output_path, seed_entries)
    holdout_digest: str | None = None
    if holdout_entries:
        holdout_digest = _write_jsonl(holdout_output_path, holdout_entries)

    if args.update_workload:
        if workload_file is None:
            raise SystemExit("--update-workload requires a resolved workload file")
        workload["seed_trace_ref"] = output_path.name if output_path.parent == workload_file.parent else str(output_path)
        workload["holdout_trace_ref"] = (
            holdout_output_path.name
            if holdout_output_path.parent == workload_file.parent
            else str(holdout_output_path)
        )
        workload.setdefault("workload_distribution_id_hardening_version", "v1-thinking-realistic")
        workload.setdefault("nominal_ttft_ms", workload.get("latency_ceiling_ms", 35000))
        workload.setdefault("nominal_tpot_ms", workload.get("tpot_ceiling_ms", 80))
        workload.setdefault("nominal_turn_ms", workload.get("turn_latency_ceiling_ms", workload.get("latency_ceiling_ms", 35000)))
        workload["workload_distribution_id"] = None
        workload_file.write_text(yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")
        workload["workload_distribution_id"] = compute_workload_distribution_id(workload_file)
        workload_file.write_text(yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")
    workload_distribution_id = compute_workload_distribution_id(workload_file) if workload_file and workload_file.exists() else seed_digest

    print(
        json.dumps(
            {
                "seed_output": str(output_path),
                "seed_count": len(seed_entries),
                "seed_sha256": seed_digest,
                "holdout_output": str(holdout_output_path) if holdout_entries else None,
                "holdout_count": len(holdout_entries),
                "holdout_sha256": holdout_digest,
                "count": len(entries),
                "workload_distribution_id": workload_distribution_id,
                "family_id": args.family_id,
                "variant": args.variant,
                "holdout_ratio": holdout_ratio,
                "split_seed": args.split_seed,
                "live_capture": bool(args.base_url),
                "enable_thinking_override": enable_thinking_override,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

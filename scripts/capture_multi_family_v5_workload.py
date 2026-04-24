#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


HARDENING_VERSION = "v1-multi-family-v5-thinking-realistic"
DEFAULT_SPLIT_SEED = 2026042401
DEFAULT_MIN_SEED_ROWS = 84
DEFAULT_MIN_HOLDOUT_ROWS = 28


def _resolve_ref(descriptor_path: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return descriptor_path.parent / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_workload_distribution_id(descriptor_path: Path) -> str:
    ydata = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    if not isinstance(ydata, dict):
        raise ValueError(f"Workload descriptor must be a mapping: {descriptor_path}")
    seed_path = _resolve_ref(descriptor_path, str(ydata["seed_trace_ref"]))
    holdout_path = _resolve_ref(descriptor_path, str(ydata["holdout_trace_ref"]))
    seed_hash = _sha256(seed_path)
    holdout_hash = _sha256(holdout_path)
    ydata["workload_distribution_id"] = None
    yaml_canonical = yaml.safe_dump(ydata, sort_keys=True, default_flow_style=False).encode("utf-8")
    yaml_hash = hashlib.sha256(yaml_canonical).hexdigest()
    return hashlib.sha256((seed_hash + holdout_hash + yaml_hash).encode("ascii")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _parse_split(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", value)
    if not match:
        raise ValueError("--split-per-family must use the form seed_rows:holdout_rows")
    seed_rows = int(match.group(1))
    holdout_rows = int(match.group(2))
    if seed_rows < 1 or holdout_rows < 1:
        raise ValueError("Both split-per-family values must be >= 1")
    return seed_rows, holdout_rows


def _iter_values(value: Any) -> list[Any]:
    found = [value]
    if isinstance(value, dict):
        for nested in value.values():
            found.extend(_iter_values(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_iter_values(nested))
    return found


def _has_key_recursive(value: Any, key_names: set[str]) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in key_names:
                return True
            if _has_key_recursive(nested, key_names):
                return True
    elif isinstance(value, list):
        return any(_has_key_recursive(nested, key_names) for nested in value)
    return False


def _has_wire_api(value: Any) -> bool:
    if not _has_key_recursive(value, {"wire_api"}):
        return False
    for nested in _iter_values(value):
        if nested == "responses":
            return True
    return True


def _family_ready(family_dir: Path) -> bool:
    family_yaml = family_dir / "family.yaml"
    if family_yaml.exists():
        payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("rawr_status") == "flywheel_ready":
            return True
    return (family_dir / "manifest.lock.json").exists()


def discover_v5_pool(repo_root: Path) -> dict[str, Any]:
    families_root = repo_root / "benchmark_blueprints" / "families"
    candidates = sorted({path.parent for path in families_root.glob("*/verification_matrix_v5*.md")})
    pool_families: list[str] = []
    excluded: list[dict[str, str]] = []
    for family_dir in candidates:
        family_id = family_dir.name
        config_path = family_dir / "codex" / "config.toml"
        if not config_path.exists():
            excluded.append({"family_id": family_id, "reason": "codex_config_missing"})
            continue
        try:
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            excluded.append({"family_id": family_id, "reason": "codex_config_invalid"})
            continue
        if not _has_key_recursive(config, {"reasoning_effort", "model_reasoning_effort", "default_reasoning_effort"}):
            excluded.append({"family_id": family_id, "reason": "reasoning_effort_missing"})
            continue
        if not _has_wire_api(config):
            excluded.append({"family_id": family_id, "reason": "wire_api_missing"})
            continue
        if not _family_ready(family_dir):
            excluded.append({"family_id": family_id, "reason": "v5_landed_marker_missing"})
            continue
        pool_families.append(family_id)
    return {
        "schema_version": "multi-family-v5.pool.v1",
        "source": "deterministic scan of benchmark_blueprints/families on main",
        "eligibility": [
            "verification_matrix_v5*.md exists",
            "codex/config.toml contains reasoning-effort and wire_api settings",
            "family.yaml rawr_status=flywheel_ready or manifest.lock.json exists",
        ],
        "pool_families": pool_families,
        "pool_excluded_families": excluded,
    }


def write_pool_file(repo_root: Path, pool_file: Path) -> dict[str, Any]:
    pool = discover_v5_pool(repo_root)
    pool_file.parent.mkdir(parents=True, exist_ok=True)
    pool_file.write_text(yaml.safe_dump(pool, sort_keys=False), encoding="utf-8")
    return pool


def load_pool(pool_file: Path) -> tuple[list[str], list[dict[str, str]]]:
    payload = yaml.safe_load(pool_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Pool file must be a mapping: {pool_file}")
    families = payload.get("pool_families", [])
    if not isinstance(families, list) or not all(isinstance(item, str) for item in families):
        raise ValueError("pool_families must be a list of family ids")
    excluded = payload.get("pool_excluded_families", [])
    if not isinstance(excluded, list):
        raise ValueError("pool_excluded_families must be a list")
    normalized_excluded: list[dict[str, str]] = []
    for item in excluded:
        if isinstance(item, dict) and isinstance(item.get("family_id"), str):
            normalized_excluded.append(
                {"family_id": item["family_id"], "reason": str(item.get("reason", "excluded_by_pool"))}
            )
    return sorted(families), normalized_excluded


def parse_thinking_probe_report(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?im)^\s*-?\s*outcome:\s*(row-[1-4])\s*$", text)
    if not match:
        raise ValueError(f"Unable to parse thinking probe outcome from {path}")
    outcome = match.group(1)
    if outcome not in {"row-1", "row-3"}:
        raise ValueError(f"Thinking probe outcome {outcome} blocks composite capture")
    return outcome


def _response_tokens(row: dict[str, Any]) -> int:
    value = row.get("response_tokens", row.get("output_tokens", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _thinking_tokens(row: dict[str, Any]) -> int:
    try:
        return int(row.get("thinking_tokens", 0))
    except (TypeError, ValueError):
        return 0


def _excluded_ids(excluded: list[dict[str, str]]) -> set[str]:
    return {item["family_id"] for item in excluded if "family_id" in item}


def validate_composite_workload(
    descriptor_path: Path,
    *,
    min_seed_rows: int = DEFAULT_MIN_SEED_ROWS,
    min_holdout_rows: int = DEFAULT_MIN_HOLDOUT_ROWS,
    min_thinking_ratio: float = 0.30,
    min_thinking_gt_response_ratio: float = 0.10,
    min_large_thinking_tokens: int = 4096,
) -> dict[str, Any]:
    descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    if not isinstance(descriptor, dict):
        raise ValueError(f"Workload descriptor must be a mapping: {descriptor_path}")
    seed_rows = _read_jsonl(_resolve_ref(descriptor_path, str(descriptor["seed_trace_ref"])))
    holdout_rows = _read_jsonl(_resolve_ref(descriptor_path, str(descriptor["holdout_trace_ref"])))
    pool = set(descriptor.get("pool_families") or [])
    excluded = _excluded_ids(descriptor.get("pool_excluded_families") or [])
    seed_family_counts = Counter(str(row.get("family_id", "")) for row in seed_rows)
    holdout_family_counts = Counter(str(row.get("family_id", "")) for row in holdout_rows)
    seed_families = {family for family in seed_family_counts if family}
    holdout_families = {family for family in holdout_family_counts if family}
    split = descriptor.get("split_per_family") or {}
    seed_per_family = int(split.get("seed_rows", 3))
    holdout_per_family = int(split.get("holdout_rows", 1))
    all_rows = seed_rows + holdout_rows
    total_rows = len(all_rows)
    thinking_positive = sum(1 for row in all_rows if _thinking_tokens(row) > 0)
    thinking_gt_response = sum(1 for row in all_rows if _thinking_tokens(row) > _response_tokens(row))
    errors: list[str] = []
    if len(seed_rows) < min_seed_rows:
        errors.append(f"seed_row_count_below_minimum:{len(seed_rows)}<{min_seed_rows}")
    if len(holdout_rows) < min_holdout_rows:
        errors.append(f"holdout_row_count_below_minimum:{len(holdout_rows)}<{min_holdout_rows}")
    if seed_families != pool:
        errors.append("seed_family_coverage_mismatch")
    if holdout_families != pool:
        errors.append("holdout_family_coverage_mismatch")
    for family_id in sorted(pool):
        if seed_family_counts[family_id] < seed_per_family:
            errors.append(f"seed_family_count_below_minimum:{family_id}")
        if holdout_family_counts[family_id] < holdout_per_family:
            errors.append(f"holdout_family_count_below_minimum:{family_id}")
    if pool & excluded:
        errors.append("pool_excluded_overlap")
    if total_rows == 0:
        errors.append("composite_trace_empty")
    else:
        if thinking_positive / total_rows < min_thinking_ratio:
            errors.append("thinking_positive_ratio_below_minimum")
        if thinking_gt_response / total_rows < min_thinking_gt_response_ratio:
            errors.append("thinking_gt_response_ratio_below_minimum")
        if not any(_thinking_tokens(row) >= min_large_thinking_tokens for row in all_rows):
            errors.append("large_thinking_row_missing")
    return {
        "pass": not errors,
        "errors": errors,
        "seed_rows": len(seed_rows),
        "holdout_rows": len(holdout_rows),
        "pool_size": len(pool),
        "seed_family_counts": dict(sorted(seed_family_counts.items())),
        "holdout_family_counts": dict(sorted(holdout_family_counts.items())),
        "thinking_positive_rows": thinking_positive,
        "thinking_gt_response_rows": thinking_gt_response,
        "excluded_families": sorted(excluded),
    }


def build_composite_workload(
    *,
    repo_root: Path,
    pool_file: Path,
    samples_per_family: int,
    split_per_family: str,
    split_seed: int,
    min_trajectory_turns: int,
    thinking_probe: Path,
) -> dict[str, Any]:
    seed_per_family, holdout_per_family = _parse_split(split_per_family)
    required_samples = seed_per_family + holdout_per_family
    if samples_per_family < required_samples:
        raise ValueError("samples-per-family must be at least seed_rows + holdout_rows")
    probe_outcome = parse_thinking_probe_report(thinking_probe)
    pool_families, pool_excluded = load_pool(pool_file)
    seed_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    runtime_excluded: list[dict[str, str]] = []
    for family_id in pool_families:
        trace_path = repo_root / "benchmark_blueprints" / "families" / family_id / "seed_trace_v5.jsonl"
        if not trace_path.exists():
            runtime_excluded.append({"family_id": family_id, "reason": "seed_trace_v5_missing"})
            continue
        rows = _read_jsonl(trace_path)
        if len(rows) < min_trajectory_turns:
            runtime_excluded.append({"family_id": family_id, "reason": "trajectory_turns_below_minimum"})
            continue
        if len(rows) < samples_per_family:
            runtime_excluded.append({"family_id": family_id, "reason": "samples_per_family_unavailable"})
            continue
        selected = []
        for row in rows[:samples_per_family]:
            normalized = dict(row)
            normalized["family_id"] = family_id
            if "response_tokens" not in normalized and "output_tokens" in normalized:
                normalized["response_tokens"] = normalized["output_tokens"]
            selected.append(normalized)
        seed_rows.extend(selected[:seed_per_family])
        holdout_rows.extend(selected[seed_per_family : seed_per_family + holdout_per_family])
    excluded = pool_excluded + runtime_excluded
    excluded_ids = _excluded_ids(excluded)
    used_families = [family_id for family_id in pool_families if family_id not in excluded_ids]
    if not used_families:
        raise ValueError("Composite workload has no eligible families with usable seed_trace_v5.jsonl")

    rng = random.Random(split_seed)
    rng.shuffle(seed_rows)
    rng.shuffle(holdout_rows)
    workload_dir = repo_root / "benchmark_blueprints" / "workloads" / "multi-family-v5"
    seed_path = workload_dir / "seed_trace.jsonl"
    holdout_path = workload_dir / "holdout_trace.jsonl"
    descriptor_path = workload_dir / "workload.yaml"
    _write_jsonl(seed_path, seed_rows)
    _write_jsonl(holdout_path, holdout_rows)
    descriptor = {
        "family_id": "multi-family-v5",
        "workload_distribution_id": None,
        "workload_distribution_id_hardening_version": HARDENING_VERSION,
        "capture_date": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "pool_size": len(used_families),
        "pool_families": used_families,
        "pool_excluded_families": excluded,
        "samples_per_family": samples_per_family,
        "split_per_family": {"seed_rows": seed_per_family, "holdout_rows": holdout_per_family},
        "split_seed": split_seed,
        "min_trajectory_turns": min_trajectory_turns,
        "total_seed_rows": len(seed_rows),
        "total_holdout_rows": len(holdout_rows),
        "thinking_probe_ref": str(thinking_probe),
        "thinking_probe_outcome": probe_outcome,
        "seed_trace_ref": "seed_trace.jsonl",
        "holdout_trace_ref": "holdout_trace.jsonl",
        "nominal_ttft_ms": 2000,
        "nominal_tpot_ms": 80,
        "nominal_turn_ms": 30000,
        "target_concurrency": 4,
    }
    workload_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    descriptor["workload_distribution_id"] = compute_workload_distribution_id(descriptor_path)
    descriptor_path.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    return {
        "workload_file": str(descriptor_path),
        "seed_trace": str(seed_path),
        "holdout_trace": str(holdout_path),
        "workload_distribution_id": descriptor["workload_distribution_id"],
        "pool_families": used_families,
        "pool_excluded_families": excluded,
        "seed_rows": len(seed_rows),
        "holdout_rows": len(holdout_rows),
        "thinking_probe_outcome": probe_outcome,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose a multi-family v5 serving workload from per-family traces.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pool-file", type=Path, default=Path("benchmark_blueprints/workloads/multi-family-v5/pool.yaml"))
    parser.add_argument("--samples-per-family", type=int, default=4)
    parser.add_argument("--split-per-family", default="3:1")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--min-trajectory-turns", type=int, default=4)
    parser.add_argument("--require-thinking-probe", type=Path)
    parser.add_argument("--write-pool", action="store_true")
    parser.add_argument("--validate-only", type=Path)
    args = parser.parse_args()

    pool_file = args.pool_file if args.pool_file.is_absolute() else args.repo_root / args.pool_file
    if args.write_pool:
        pool = write_pool_file(args.repo_root, pool_file)
        print(json.dumps(pool, indent=2, sort_keys=True))
        return 0
    if args.validate_only:
        descriptor_path = args.validate_only if args.validate_only.is_absolute() else args.repo_root / args.validate_only
        print(json.dumps(validate_composite_workload(descriptor_path), indent=2, sort_keys=True))
        return 0
    if args.require_thinking_probe is None:
        raise SystemExit("--require-thinking-probe is required")
    thinking_probe = (
        args.require_thinking_probe
        if args.require_thinking_probe.is_absolute()
        else args.repo_root / args.require_thinking_probe
    )
    result = build_composite_workload(
        repo_root=args.repo_root,
        pool_file=pool_file,
        samples_per_family=args.samples_per_family,
        split_per_family=args.split_per_family,
        split_seed=args.split_seed,
        min_trajectory_turns=args.min_trajectory_turns,
        thinking_probe=thinking_probe,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .tuned_config import compute_workload_distribution_id

L0_HEAVY_WORKLOAD_FAMILY_ID = "responses-sdk-adapter-cutover-heavy"
L0_HEAVY_SOURCE_FAMILY_ID = "responses-sdk-adapter-cutover"
HARDENED_L0_HEAVY_WORKLOAD_VERSION = "v2-l0-kernel-heavy"
SIBLING_HOLDOUT_BASELINE = "vllm-default"
SIBLING_HOLDOUT_FAMILIES = (
    "codex-provider-rollover",
    "codex-skill-runtime-v2-split",
    "esm-plugin-loader-modernization",
    "nightly-regression-watch",
    "objective-driven-repo-improvement",
    "policy-aware-request-resolution",
    "release-manifest-v2-modernization",
    "sqlalchemy-2-session-modernization",
)


@dataclass(frozen=True)
class TraceValidation:
    path: str
    exists: bool
    row_count: int = 0
    reasoning_positive_rows: int = 0
    errors: tuple[str, ...] = ()
    capture_baselines: tuple[str, ...] = ()
    weight_version_ids: tuple[str, ...] = ()

    @property
    def pass_(self) -> bool:
        return self.exists and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "pass": self.pass_,
            "row_count": self.row_count,
            "reasoning_positive_rows": self.reasoning_positive_rows,
            "errors": list(self.errors),
            "capture_baselines": list(self.capture_baselines),
            "weight_version_ids": list(self.weight_version_ids),
        }


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_ref(descriptor_path: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return descriptor_path.parent / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reasoning_tokens(row: dict[str, Any]) -> int:
    for key in ("reasoning_tokens", "thinking_tokens"):
        try:
            value = int(row.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _metadata_value(row: dict[str, Any], flat_key: str, nested_key: str) -> str:
    value = row.get(flat_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    metadata = row.get("capture_metadata")
    if isinstance(metadata, dict):
        nested = metadata.get(nested_key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return ""


def annotate_capture_rows(
    rows: list[dict[str, Any]],
    *,
    capture_role: str,
    baseline: str,
    weight_version_id: str,
    capture_date: str | None = None,
) -> list[dict[str, Any]]:
    captured_at = capture_date or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    annotated: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        normalized["capture_role"] = capture_role
        normalized["capture_baseline"] = baseline
        normalized["weight_version_id"] = weight_version_id
        normalized["capture_date"] = captured_at
        normalized["capture_metadata"] = {
            "baseline": baseline,
            "weight_version_id": weight_version_id,
            "capture_date": captured_at,
            "capture_role": capture_role,
        }
        annotated.append(normalized)
    return annotated


def validate_trace_file(
    path: Path,
    *,
    repo_root: Path,
    expected_family_id: str | None = None,
    require_capture_metadata: bool = False,
    expected_baseline: str | None = None,
    expected_weight_version_id: str | None = None,
) -> TraceValidation:
    relative = _repo_relative(repo_root, path)
    if not path.is_file():
        return TraceValidation(path=relative, exists=False, errors=("trace_missing",))
    errors: list[str] = []
    if path.stat().st_size == 0:
        errors.append("trace_empty")
    elif path.read_bytes()[-1:] != b"\n":
        errors.append("trace_not_newline_terminated")
    try:
        rows = _read_jsonl(path)
    except (json.JSONDecodeError, ValueError) as exc:
        return TraceValidation(path=relative, exists=True, errors=(f"trace_jsonl_invalid:{exc}",))
    if not rows:
        errors.append("trace_empty")
    reasoning_positive = sum(1 for row in rows if _reasoning_tokens(row) > 0)
    if reasoning_positive == 0:
        errors.append("reasoning_tokens_zero")
    if expected_family_id is not None:
        bad_families = sorted({str(row.get("family_id", "")) for row in rows if row.get("family_id") != expected_family_id})
        if bad_families:
            errors.append(f"family_id_mismatch:{','.join(bad_families)}")
    baselines = sorted({_metadata_value(row, "capture_baseline", "baseline") for row in rows} - {""})
    weights = sorted({_metadata_value(row, "weight_version_id", "weight_version_id") for row in rows} - {""})
    if require_capture_metadata:
        if not baselines:
            errors.append("capture_baseline_missing")
        if not weights:
            errors.append("weight_version_id_missing")
    if expected_baseline is not None and baselines != [expected_baseline]:
        errors.append(f"capture_baseline_mismatch:{','.join(baselines) or '<missing>'}")
    if expected_weight_version_id is not None and weights != [expected_weight_version_id]:
        errors.append(f"weight_version_id_mismatch:{','.join(weights) or '<missing>'}")
    return TraceValidation(
        path=relative,
        exists=True,
        row_count=len(rows),
        reasoning_positive_rows=reasoning_positive,
        errors=tuple(errors),
        capture_baselines=tuple(baselines),
        weight_version_ids=tuple(weights),
    )


def heavy_workload_dir(repo_root: Path) -> Path:
    return repo_root / "benchmark_blueprints" / "workloads" / L0_HEAVY_WORKLOAD_FAMILY_ID


def heavy_workload_descriptor_path(repo_root: Path) -> Path:
    return heavy_workload_dir(repo_root) / "workload.yaml"


def heavy_descriptor_payload(
    *,
    repo_root: Path,
    capture_date: str | None,
    thinking_probe_ref: str,
) -> dict[str, Any]:
    return {
        "family_id": L0_HEAVY_WORKLOAD_FAMILY_ID,
        "workload_distribution_id": None,
        "workload_distribution_id_hardening_version": HARDENED_L0_HEAVY_WORKLOAD_VERSION,
        "capture_date": capture_date or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_family": L0_HEAVY_SOURCE_FAMILY_ID,
        "trajectory_source": _repo_relative(
            repo_root,
            repo_root / "benchmark_blueprints" / "families" / L0_HEAVY_SOURCE_FAMILY_ID / "seed_trace_v5.jsonl",
        ),
        "trajectory_turns": 4,
        "samples_per_measurement": 1,
        "seed_trace_ref": "seed_trace.jsonl",
        "holdout_trace_ref": "holdout_trace.jsonl",
        "nominal_ttft_ms": 2000,
        "nominal_tpot_ms": 80,
        "nominal_turn_ms": 30000,
        "target_concurrency": 4,
        "thinking_probe_ref": thinking_probe_ref,
        "thinking_probe_outcome": "row-3",
        "parity_fixture_refs": {
            "deltanet": "parity_fixture/deltanet_v1.yaml",
            "gatedattn": "parity_fixture/gatedattn_v1.yaml",
        },
    }


def write_heavy_workload_descriptor(
    *,
    repo_root: Path,
    capture_date: str | None = None,
    thinking_probe_ref: str = "reports/thinking-probe-20260424.md",
) -> Path:
    descriptor_path = heavy_workload_descriptor_path(repo_root)
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    payload = heavy_descriptor_payload(repo_root=repo_root, capture_date=capture_date, thinking_probe_ref=thinking_probe_ref)
    descriptor_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    payload["workload_distribution_id"] = compute_workload_distribution_id(descriptor_path)
    descriptor_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return descriptor_path


def _validate_heavy_descriptor_schema(repo_root: Path, descriptor_path: Path, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_strings = {
        "family_id",
        "workload_distribution_id",
        "workload_distribution_id_hardening_version",
        "capture_date",
        "source_family",
        "trajectory_source",
        "seed_trace_ref",
        "holdout_trace_ref",
        "thinking_probe_ref",
        "thinking_probe_outcome",
    }
    for key in sorted(required_strings):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            errors.append(f"descriptor_missing_string:{key}")
    if payload.get("family_id") != L0_HEAVY_WORKLOAD_FAMILY_ID:
        errors.append("descriptor_family_id_mismatch")
    if payload.get("source_family") != L0_HEAVY_SOURCE_FAMILY_ID:
        errors.append("descriptor_source_family_mismatch")
    if payload.get("workload_distribution_id_hardening_version") != HARDENED_L0_HEAVY_WORKLOAD_VERSION:
        errors.append("descriptor_hardening_version_mismatch")
    if payload.get("trajectory_turns") != 4:
        errors.append("descriptor_trajectory_turns_mismatch")
    if payload.get("samples_per_measurement") != 1:
        errors.append("descriptor_samples_per_measurement_mismatch")
    if payload.get("thinking_probe_outcome") != "row-3":
        errors.append("descriptor_thinking_probe_outcome_not_row_3")
    parity_refs = payload.get("parity_fixture_refs")
    if not isinstance(parity_refs, dict):
        errors.append("descriptor_parity_fixture_refs_missing")
    else:
        if sorted(parity_refs) != ["deltanet", "gatedattn"]:
            errors.append("descriptor_parity_fixture_refs_keys_mismatch")
        for key in ("deltanet", "gatedattn"):
            if not isinstance(parity_refs.get(key), str) or not parity_refs.get(key).strip():
                errors.append(f"descriptor_parity_fixture_ref_missing:{key}")
    trajectory_source = payload.get("trajectory_source")
    if isinstance(trajectory_source, str) and trajectory_source.strip():
        source_path = Path(trajectory_source)
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if not source_path.is_file():
            errors.append("descriptor_trajectory_source_missing")
    workload_id = payload.get("workload_distribution_id")
    if isinstance(workload_id, str) and workload_id.strip():
        seed_ref = payload.get("seed_trace_ref")
        holdout_ref = payload.get("holdout_trace_ref")
        if isinstance(seed_ref, str) and isinstance(holdout_ref, str):
            seed_path = _resolve_ref(descriptor_path, seed_ref)
            holdout_path = _resolve_ref(descriptor_path, holdout_ref)
            if seed_path.is_file() and holdout_path.is_file():
                canonical = compute_workload_distribution_id(descriptor_path)
                if workload_id != canonical:
                    errors.append("descriptor_workload_distribution_id_mismatch")
    return errors


def validate_p1_workload(
    *,
    repo_root: Path,
    descriptor_path: Path | None = None,
    expected_weight_version_id: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    descriptor_path = (descriptor_path or heavy_workload_descriptor_path(repo_root)).resolve()
    errors: list[str] = []
    heavy_seed: TraceValidation | None = None
    heavy_holdout: TraceValidation | None = None
    canonical_id: str | None = None

    if not descriptor_path.is_file():
        errors.append("heavy_workload_descriptor_missing")
        descriptor: dict[str, Any] = {}
    else:
        loaded = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            errors.append("heavy_workload_descriptor_not_mapping")
            descriptor = {}
        else:
            descriptor = loaded
            errors.extend(_validate_heavy_descriptor_schema(repo_root, descriptor_path, descriptor))
            seed_ref = descriptor.get("seed_trace_ref")
            holdout_ref = descriptor.get("holdout_trace_ref")
            if isinstance(seed_ref, str):
                heavy_seed = validate_trace_file(
                    _resolve_ref(descriptor_path, seed_ref),
                    repo_root=repo_root,
                    expected_family_id=L0_HEAVY_SOURCE_FAMILY_ID,
                )
                errors.extend(f"heavy_seed_{error}" for error in heavy_seed.errors)
            else:
                errors.append("heavy_seed_trace_ref_invalid")
            if isinstance(holdout_ref, str):
                heavy_holdout = validate_trace_file(
                    _resolve_ref(descriptor_path, holdout_ref),
                    repo_root=repo_root,
                    expected_family_id=L0_HEAVY_SOURCE_FAMILY_ID,
                )
                errors.extend(f"heavy_holdout_{error}" for error in heavy_holdout.errors)
            else:
                errors.append("heavy_holdout_trace_ref_invalid")
            if heavy_seed and heavy_seed.exists and heavy_holdout and heavy_holdout.exists:
                canonical_id = compute_workload_distribution_id(descriptor_path)

    sibling_results: dict[str, dict[str, Any]] = {}
    missing_siblings: list[str] = []
    invalid_siblings: list[str] = []
    for family_id in SIBLING_HOLDOUT_FAMILIES:
        holdout_path = repo_root / "benchmark_blueprints" / "families" / family_id / "holdout_trace_v5.jsonl"
        result = validate_trace_file(
            holdout_path,
            repo_root=repo_root,
            expected_family_id=family_id,
            require_capture_metadata=True,
            expected_baseline=SIBLING_HOLDOUT_BASELINE,
            expected_weight_version_id=expected_weight_version_id,
        )
        sibling_results[family_id] = result.as_dict()
        if not result.exists:
            missing_siblings.append(family_id)
        elif not result.pass_:
            invalid_siblings.append(family_id)
    if missing_siblings:
        errors.extend(f"sibling_holdout_missing:{family_id}" for family_id in missing_siblings)
    if invalid_siblings:
        errors.extend(f"sibling_holdout_invalid:{family_id}" for family_id in invalid_siblings)

    halt_reason: str | None = None
    if errors:
        if missing_siblings or invalid_siblings:
            halt_reason = "sibling_holdout_capture_failed"
        elif any(error.startswith("heavy_seed_reasoning_tokens_zero") or error.startswith("heavy_holdout_reasoning_tokens_zero") for error in errors):
            halt_reason = "capture_thinking_zero"
        elif any(error == "descriptor_workload_distribution_id_mismatch" for error in errors):
            halt_reason = "descriptor_workload_distribution_id_mismatch"
        else:
            halt_reason = "p1_workload_descriptor_invalid"
    return {
        "pass": not errors,
        "halt_reason": halt_reason,
        "errors": errors,
        "workload_file": _repo_relative(repo_root, descriptor_path),
        "canonical_workload_distribution_id": canonical_id,
        "declared_workload_distribution_id": descriptor.get("workload_distribution_id"),
        "heavy_seed_trace": heavy_seed.as_dict() if heavy_seed else None,
        "heavy_holdout_trace": heavy_holdout.as_dict() if heavy_holdout else None,
        "sibling_holdouts": sibling_results,
        "missing_sibling_holdouts": missing_siblings,
        "invalid_sibling_holdouts": invalid_siblings,
    }

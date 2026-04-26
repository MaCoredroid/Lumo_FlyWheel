#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.parity_fixture import (  # noqa: E402
    DEFAULT_WEIGHT_VERSION_ID,
    KERNEL_TARGETS,
    LOGITS_DEBUG_KIND,
    DELTANET_STATE_DEBUG_KIND,
    P2B_FAMILY_PROBE_COUNTS,
    REFERENCE_REPRODUCIBILITY_RUNS,
    SYNTHETIC_TEST_ARTIFACT_PURPOSE,
    DebugProbeArtifacts,
    assert_debug_capture_runs_reproduce,
    deterministic_probe_rows,
    family_fixture_dir,
    fetch_endpoint_capabilities,
    fixture_payload,
    load_debug_export_pt,
    summarize_debug_export_pt,
    validate_fixture,
    validate_p2b_fixture_set,
    write_debug_export_npz_companions,
)

DEBUG_EXPORT_RE = re.compile(r"^(?P<kind>logits|state)_req_(?P<request_id>.+)_tok_(?P<token>[0-9]{6})\.pt$")


@dataclass(frozen=True)
class _DebugSource:
    path: Path
    kind: str
    request_id: str
    generated_token_index: int
    file_sha256: str
    logits_sha256: str | None = None
    logits_sample_first_16: Any | None = None
    saved_shape: tuple[int, ...] | None = None
    source_shape: tuple[int, ...] | None = None
    logits_is_truncated: bool | None = None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_synthetic_npz(path: Path, *, probe_count: int, kernel_target: str) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("numpy is required only for --allow-synthetic-test-fixture") from exc

    indices = np.arange(probe_count, dtype=np.int32)
    if path.name.endswith("reference_state.npz"):
        np.savez(
            path,
            state_token_1=np.zeros((probe_count, 1), dtype=np.float32),
            state_token_1024=np.zeros((probe_count, 1), dtype=np.float32),
            probe_index=indices,
            synthetic_test_placeholder=np.array([1], dtype=np.int8),
        )
    else:
        np.savez(
            path,
            logits=np.zeros((probe_count, 1, 1), dtype=np.float32),
            probe_index=indices,
            kernel_target=np.array([kernel_target]),
            synthetic_test_placeholder=np.array([1], dtype=np.int8),
        )


def _write_synthetic_test_fixture(
    *,
    repo_root: Path,
    family_id: str,
    probe_count: int,
    weight_version_id: str,
    vllm_version: str,
) -> dict[str, Any]:
    fixture_dir = family_fixture_dir(repo_root, family_id)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    probes = deterministic_probe_rows(repo_root, family_id, probe_count)
    _write_jsonl(fixture_dir / "probes_input.jsonl", probes)
    written: list[str] = []
    for kernel_target in KERNEL_TARGETS:
        payload = fixture_payload(
            family_id=family_id,
            kernel_target=kernel_target,
            probe_count=probe_count,
            weight_version_id=weight_version_id,
            vllm_version=vllm_version,
        )
        payload["artifact_purpose"] = SYNTHETIC_TEST_ARTIFACT_PURPOSE
        yaml_path = fixture_dir / f"{kernel_target}_v1.yaml"
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        logits_path = fixture_dir / f"{kernel_target}_reference_logits.npz"
        _write_synthetic_npz(logits_path, probe_count=probe_count, kernel_target=kernel_target)
        written.extend([str(yaml_path), str(logits_path)])
        if kernel_target == "deltanet":
            state_path = fixture_dir / "deltanet_reference_state.npz"
            _write_synthetic_npz(state_path, probe_count=probe_count, kernel_target=kernel_target)
            written.append(str(state_path))
    written.append(str(fixture_dir / "probes_input.jsonl"))
    return {"status": "SYNTHETIC_TEST_FIXTURE_WRITTEN", "family_id": family_id, "written": sorted(written)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(int(dim) for dim in array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _tensor_to_numpy_float32(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _debug_source_from_logits(path: Path) -> _DebugSource:
    payload = load_debug_export_pt(path)
    if payload.get("kind") != LOGITS_DEBUG_KIND:
        raise ValueError(f"expected logits debug export, got {payload.get('kind')!r}: {path}")
    logits = _tensor_to_numpy_float32(payload["logits"]).reshape(-1)
    return _DebugSource(
        path=path,
        kind=str(payload["kind"]),
        request_id=str(payload["request_id"]),
        generated_token_index=int(payload["generated_token_index"]),
        file_sha256=_file_sha256(path),
        logits_sha256=_array_sha256(logits),
        logits_sample_first_16=logits[:16],
        saved_shape=tuple(int(dim) for dim in payload.get("saved_shape", ())),
        source_shape=tuple(int(dim) for dim in payload.get("source_shape", ())),
        logits_is_truncated=bool(payload.get("logits_is_truncated")),
    )


def _debug_source_from_state_path(path: Path) -> _DebugSource:
    match = DEBUG_EXPORT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"state debug export filename does not match expected pattern: {path}")
    return _DebugSource(
        path=path,
        kind=DELTANET_STATE_DEBUG_KIND,
        request_id=match.group("request_id"),
        generated_token_index=int(match.group("token")),
        file_sha256=_file_sha256(path),
    )


def _discover_conversion_sources(source_debug_root: Path) -> dict[str, list[_DebugSource]]:
    if not source_debug_root.is_dir():
        raise FileNotFoundError(f"debug source directory missing: {source_debug_root}")

    logits_sources: list[_DebugSource] = []
    state_sources: list[_DebugSource] = []
    for path in sorted(source_debug_root.rglob("*.pt")):
        match = DEBUG_EXPORT_RE.fullmatch(path.name)
        if match is None:
            continue
        if match.group("kind") == "logits":
            logits_sources.append(_debug_source_from_logits(path))
        else:
            state_sources.append(_debug_source_from_state_path(path))

    full_logits = [source for source in logits_sources if source.logits_is_truncated is False]
    state_token_1 = [source for source in state_sources if source.generated_token_index == 1]
    state_token_1024 = [source for source in state_sources if source.generated_token_index == 1024]
    if not full_logits:
        raise RuntimeError(f"no untruncated full-vocab logits debug exports found under {source_debug_root}")
    if not state_token_1:
        raise RuntimeError(f"no DeltaNet state token-1 debug exports found under {source_debug_root}")
    if not state_token_1024:
        raise RuntimeError(f"no DeltaNet state token-1024 debug exports found under {source_debug_root}")
    return {
        "logits": full_logits,
        "state_token_1": state_token_1,
        "state_token_1024": state_token_1024,
    }


def _repeat_sources(sources: list[_DebugSource], count: int) -> list[_DebugSource]:
    if not sources:
        raise ValueError("cannot repeat an empty debug source list")
    return [sources[index % len(sources)] for index in range(count)]


def _source_paths(repo_root: Path, sources: list[_DebugSource]) -> list[str]:
    paths: list[str] = []
    for source in sources:
        try:
            paths.append(str(source.path.resolve().relative_to(repo_root.resolve())))
        except ValueError:
            paths.append(str(source.path.resolve()))
    return paths


def _write_converted_logits_npz(
    *,
    repo_root: Path,
    fixture_dir: Path,
    kernel_target: str,
    probe_count: int,
    sources: list[_DebugSource],
) -> str:
    import numpy as np

    selected = _repeat_sources(sources, probe_count)
    samples = [
        np.asarray(source.logits_sample_first_16, dtype=np.float32)
        for source in selected
    ]
    payload = {
        "probe_index": np.arange(probe_count, dtype=np.int32),
        "reference_storage": np.array(["sha256_float32_with_first_16_sample_from_real_vllm_debug_export"]),
        "source_artifact_path_by_probe": np.array(_source_paths(repo_root, selected)),
        "source_artifact_sha256_by_probe": np.array([source.file_sha256 for source in selected]),
        "source_request_id_by_probe": np.array([source.request_id for source in selected]),
        "source_generated_token_index_by_probe": np.array(
            [source.generated_token_index for source in selected],
            dtype=np.int32,
        ),
        "source_logits_sha256_by_probe": np.array([source.logits_sha256 for source in selected]),
        "source_logits_sample_first_16_float32": np.stack(samples, axis=0),
        "source_saved_shape_by_probe": np.array([json.dumps(source.saved_shape) for source in selected]),
        "source_shape_by_probe": np.array([json.dumps(source.source_shape) for source in selected]),
        "source_logits_is_truncated_by_probe": np.array(
            [bool(source.logits_is_truncated) for source in selected],
            dtype=np.bool_,
        ),
    }
    path = fixture_dir / f"{kernel_target}_reference_logits.npz"
    np.savez(path, **payload)
    return str(path)


def _write_converted_state_npz(
    *,
    repo_root: Path,
    fixture_dir: Path,
    probe_count: int,
    state_token_1_sources: list[_DebugSource],
    state_token_1024_sources: list[_DebugSource],
) -> str:
    import numpy as np

    selected_1 = _repeat_sources(state_token_1_sources, probe_count)
    selected_1024 = _repeat_sources(state_token_1024_sources, probe_count)
    payload = {
        "probe_index": np.arange(probe_count, dtype=np.int32),
        "state_storage": np.array(["debug_export_file_sha256_from_real_vllm_state_snapshot"]),
        "state_token_1": np.array([source.file_sha256 for source in selected_1]),
        "state_token_1_source_path_by_probe": np.array(_source_paths(repo_root, selected_1)),
        "state_token_1_request_id_by_probe": np.array([source.request_id for source in selected_1]),
        "state_token_1024": np.array([source.file_sha256 for source in selected_1024]),
        "state_token_1024_source_path_by_probe": np.array(_source_paths(repo_root, selected_1024)),
        "state_token_1024_request_id_by_probe": np.array([source.request_id for source in selected_1024]),
    }
    path = fixture_dir / "deltanet_reference_state.npz"
    np.savez(path, **payload)
    return str(path)


def _convert_debug_artifacts_to_p2b_fixture_set(
    *,
    repo_root: Path,
    source_debug_root: Path,
    weight_version_id: str,
    vllm_version: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    sources = _discover_conversion_sources(source_debug_root)
    generated_at = generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    written: list[str] = []
    validations: dict[str, Any] = {}
    family_results: dict[str, Any] = {}

    for family_id, probe_count in P2B_FAMILY_PROBE_COUNTS.items():
        fixture_dir = family_fixture_dir(repo_root, family_id)
        fixture_dir.mkdir(parents=True, exist_ok=True)
        probes = deterministic_probe_rows(repo_root, family_id, probe_count)
        probes_path = fixture_dir / "probes_input.jsonl"
        _write_jsonl(probes_path, probes)
        written.append(str(probes_path))

        deltanet_logits = _write_converted_logits_npz(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            kernel_target="deltanet",
            probe_count=probe_count,
            sources=sources["logits"],
        )
        gatedattn_logits = _write_converted_logits_npz(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            kernel_target="gatedattn",
            probe_count=probe_count,
            sources=sources["logits"],
        )
        state_path = _write_converted_state_npz(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            probe_count=probe_count,
            state_token_1_sources=sources["state_token_1"],
            state_token_1024_sources=sources["state_token_1024"],
        )
        written.extend([deltanet_logits, gatedattn_logits, state_path])
        written.extend(
            _write_fixture_yaml_pair(
                fixture_dir=fixture_dir,
                family_id=family_id,
                probe_count=probe_count,
                weight_version_id=weight_version_id,
                vllm_version=vllm_version,
                kernel_targets=KERNEL_TARGETS,
                generated_at=generated_at,
            )
        )
        family_results[family_id] = {
            "probe_count": probe_count,
            "fixture_dir": str(fixture_dir),
        }

    validation_result = validate_p2b_fixture_set(repo_root, expected_weight_version_id=weight_version_id)
    validations.update(validation_result["fixtures"])
    return {
        "status": "CONVERTED_DEBUG_ARTIFACTS_TO_P2B_FIXTURES",
        "source_debug_root": str(source_debug_root),
        "source_counts": {key: len(value) for key, value in sources.items()},
        "families": family_results,
        "written": sorted(written),
        "validations": validations,
        "validation_pass": validation_result["pass"],
        "validation_errors": validation_result["errors"],
    }


def _request_completion(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    probe: dict[str, Any],
    timeout_s: float,
    minimum_completion_tokens: int | None = None,
) -> dict[str, Any]:
    output_token_count = max(int(probe["output_token_count"]), int(minimum_completion_tokens or 0))
    payload = {
        "model": model,
        "prompt": probe["prompt"],
        "max_tokens": output_token_count,
        "temperature": 0,
        "seed": 0,
    }
    if output_token_count >= 1024:
        # vLLM OpenAI-compatible completions accepts these sampling extensions
        # directly in the JSON body. They force a real token-1024 recurrent
        # state checkpoint instead of ending early on EOS.
        payload["min_tokens"] = output_token_count
        payload["ignore_eos"] = True
    response = requests.post(
        f"{endpoint.rstrip('/')}/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("completion response was not a JSON object")
    return body


def _wait_for_quiet_exports(export_dir: Path, *, timeout_s: float = 10.0, quiet_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    previous: tuple[str, ...] | None = None
    quiet_started: float | None = None
    while time.time() < deadline:
        current = tuple(sorted(path.name for path in export_dir.glob("*") if path.is_file()))
        if current == previous:
            quiet_started = quiet_started or time.time()
            if time.time() - quiet_started >= quiet_s:
                return
        else:
            previous = current
            quiet_started = None
        time.sleep(0.1)


def _clear_export_staging(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for path in export_dir.iterdir():
        if path.is_file():
            path.unlink()


def _archive_probe_exports(
    *,
    export_dir: Path,
    archive_dir: Path,
    expected_state_tokens: tuple[int, ...],
    require_state: bool,
) -> DebugProbeArtifacts:
    files = [path for path in export_dir.glob("*.pt") if path.is_file()]
    if not files:
        diagnostics = sorted(path.name for path in export_dir.glob("state_diag_req_*.json"))
        raise RuntimeError(f"no debug .pt exports produced; diagnostics={diagnostics}")

    by_request: dict[str, dict[str, list[Path]]] = {}
    for path in files:
        match = DEBUG_EXPORT_RE.fullmatch(path.name)
        if match is None:
            continue
        request_id = match.group("request_id")
        by_request.setdefault(request_id, {"logits": [], "state": []})[match.group("kind")].append(path)
    if len(by_request) != 1:
        raise RuntimeError(f"expected exactly one exported request, saw {sorted(by_request)}")
    request_id, grouped = next(iter(by_request.items()))
    logits = sorted(grouped["logits"])
    states = sorted(grouped["state"])
    if not logits:
        raise RuntimeError(f"request {request_id} produced no logits debug exports")
    if require_state:
        state_tokens = {
            int(load_debug_export_pt(path)["generated_token_index"])
            for path in states
        }
        missing = [token for token in expected_state_tokens if token not in state_tokens]
        if missing:
            diagnostics = sorted(path.read_text(encoding="utf-8").strip() for path in export_dir.glob("state_diag_req_*.json"))
            raise RuntimeError(
                f"request {request_id} missing DeltaNet state tokens {missing}; diagnostics={diagnostics[:5]}"
            )

    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_logits: list[Path] = []
    archived_states: list[Path] = []
    for source in logits:
        target = archive_dir / source.name
        shutil.move(str(source), target)
        archived_logits.append(target)
    for source in states:
        target = archive_dir / source.name
        shutil.move(str(source), target)
        archived_states.append(target)
    for source in export_dir.glob("state_diag_req_*.json"):
        shutil.move(str(source), archive_dir / source.name)

    probe_index = int(archive_dir.name.rsplit("_", 1)[-1])
    return DebugProbeArtifacts(
        probe_index=probe_index,
        request_id=request_id,
        logits_paths=tuple(archived_logits),
        state_paths=tuple(archived_states),
    )


def _capture_live_runs(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    probes: list[dict[str, Any]],
    debug_export_dir: Path,
    run_count: int,
    request_timeout_s: float,
    expected_state_tokens: tuple[int, ...],
    require_state: bool,
) -> tuple[list[list[DebugProbeArtifacts]], list[dict[str, Any]]]:
    runs: list[list[DebugProbeArtifacts]] = []
    responses: list[dict[str, Any]] = []
    staging_dir = debug_export_dir / "staging"
    for run_index in range(1, run_count + 1):
        run_artifacts: list[DebugProbeArtifacts] = []
        for probe in probes:
            _clear_export_staging(staging_dir)
            started = time.time()
            response = _request_completion(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                probe=probe,
                timeout_s=request_timeout_s,
                # vLLM can remove a request before the final-token state hook
                # sees it, so make required state checkpoints non-terminal.
                minimum_completion_tokens=(
                    max(expected_state_tokens) + 1 if require_state and expected_state_tokens else None
                ),
            )
            _wait_for_quiet_exports(staging_dir)
            archive_dir = debug_export_dir / f"run_{run_index:02d}" / f"probe_{int(probe['probe_index']):06d}"
            artifact = _archive_probe_exports(
                export_dir=staging_dir,
                archive_dir=archive_dir,
                expected_state_tokens=expected_state_tokens,
                require_state=require_state,
            )
            run_artifacts.append(artifact)
            responses.append(
                {
                    "run_index": run_index,
                    "probe_index": int(probe["probe_index"]),
                    "request_id": artifact.request_id,
                    "response_id": response.get("id"),
                    "elapsed_s": round(time.time() - started, 3),
                    "logits_files": [str(path) for path in artifact.logits_paths],
                    "state_files": [str(path) for path in artifact.state_paths],
                }
            )
        runs.append(run_artifacts)
    return runs, responses


def _override_output_tokens(rows: list[dict[str, Any]], output_tokens: int | None) -> list[dict[str, Any]]:
    if output_tokens is None:
        return rows
    updated: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["output_token_count"] = output_tokens
        updated.append(copy)
    return updated


def _validate_written_fixture(
    *,
    repo_root: Path,
    family_id: str,
    kernel_target: str,
    probe_count: int,
    weight_version_id: str,
) -> dict[str, Any]:
    return validate_fixture(
        family_fixture_dir(repo_root, family_id) / f"{kernel_target}_v1.yaml",
        repo_root=repo_root,
        expected_family_id=family_id,
        expected_kernel_target=kernel_target,
        expected_probe_count=probe_count,
        expected_weight_version_id=weight_version_id,
    ).as_dict()


def _write_fixture_yaml_pair(
    *,
    fixture_dir: Path,
    family_id: str,
    probe_count: int,
    weight_version_id: str,
    vllm_version: str,
    kernel_targets: tuple[str, ...],
    generated_at: str | None = None,
) -> list[str]:
    written: list[str] = []
    for kernel_target in kernel_targets:
        payload = fixture_payload(
            family_id=family_id,
            kernel_target=kernel_target,
            probe_count=probe_count,
            weight_version_id=weight_version_id,
            vllm_version=vllm_version,
            generated_at=generated_at,
        )
        path = fixture_dir / f"{kernel_target}_v1.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written.append(str(path))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate HLD section 2.2.0 parity fixture artifacts.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--family-id", default="responses-sdk-adapter-cutover")
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--weight-version-id", default=DEFAULT_WEIGHT_VERSION_ID)
    parser.add_argument("--debug-export-dir", type=Path, default=REPO_ROOT / "output" / "p2b_fixture_capture")
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--reproducibility-runs", type=int, default=REFERENCE_REPRODUCIBILITY_RUNS)
    parser.add_argument("--override-output-tokens", type=int)
    parser.add_argument("--kernel-target", choices=[*KERNEL_TARGETS, "both"], default="both")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--validate-p2b", action="store_true")
    parser.add_argument("--convert-debug-artifacts", action="store_true")
    parser.add_argument(
        "--source-debug-root",
        type=Path,
        default=REPO_ROOT / "output" / "sprint0_live" / "logs",
        help="Root containing real vLLM debug-export .pt files to convert into compact fixture companions.",
    )
    parser.add_argument(
        "--converted-vllm-version",
        default="unknown-debug-export",
        help="vLLM version string to stamp on --convert-debug-artifacts fixture YAMLs.",
    )
    parser.add_argument(
        "--generated-at",
        help="Optional deterministic ISO-8601 timestamp for emitted fixture YAMLs.",
    )
    parser.add_argument(
        "--allow-synthetic-test-fixture",
        action="store_true",
        help="Write deterministic placeholder blobs for tests only; never use for production P2b fixtures.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    if args.validate_p2b:
        result = validate_p2b_fixture_set(repo_root, expected_weight_version_id=args.weight_version_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["pass"] else 1

    if args.convert_debug_artifacts:
        result = _convert_debug_artifacts_to_p2b_fixture_set(
            repo_root=repo_root,
            source_debug_root=args.source_debug_root.resolve(),
            weight_version_id=args.weight_version_id,
            vllm_version=args.converted_vllm_version,
            generated_at=args.generated_at,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["validation_pass"] else 1

    if args.allow_synthetic_test_fixture:
        result = _write_synthetic_test_fixture(
            repo_root=repo_root,
            family_id=args.family_id,
            probe_count=args.probe_count,
            weight_version_id=args.weight_version_id,
            vllm_version="synthetic-test",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    capabilities = fetch_endpoint_capabilities(args.endpoint, api_key=args.api_key, model=args.model)
    if not capabilities.get("health_ok") or not capabilities.get("models_ok"):
        print(json.dumps({"status": "BLOCKED_NEEDS_USER_HELP", "capabilities": capabilities}, indent=2, sort_keys=True))
        return 2

    kernel_targets = KERNEL_TARGETS if args.kernel_target == "both" else (args.kernel_target,)
    require_state = "deltanet" in kernel_targets
    probes = deterministic_probe_rows(repo_root, args.family_id, args.probe_count)
    probes = _override_output_tokens(probes, args.override_output_tokens)
    debug_export_dir = args.debug_export_dir.resolve() / args.family_id
    runs, responses = _capture_live_runs(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        probes=probes,
        debug_export_dir=debug_export_dir,
        run_count=args.reproducibility_runs,
        request_timeout_s=args.request_timeout_s,
        expected_state_tokens=(1, 1024),
        require_state=require_state,
    )

    if args.reproducibility_runs == REFERENCE_REPRODUCIBILITY_RUNS:
        for kernel_target in kernel_targets:
            assert_debug_capture_runs_reproduce(runs=runs, kernel_target=kernel_target)
    elif not args.capture_only:
        raise RuntimeError(
            f"production fixture capture requires --reproducibility-runs={REFERENCE_REPRODUCIBILITY_RUNS}"
        )

    source_summaries = [
        summarize_debug_export_pt(path)
        for artifact in runs[0]
        for path in [*artifact.logits_paths, *artifact.state_paths]
    ]
    result: dict[str, Any] = {
        "status": "CAPTURED",
        "family_id": args.family_id,
        "probe_count": args.probe_count,
        "debug_export_dir": str(debug_export_dir),
        "responses": responses,
        "source_artifacts": source_summaries,
    }
    if args.capture_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    fixture_dir = family_fixture_dir(repo_root, args.family_id)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(fixture_dir / "probes_input.jsonl", probes)
    written = [str(fixture_dir / "probes_input.jsonl")]
    shapes: dict[str, Any] = {}
    for kernel_target in kernel_targets:
        aggregate = write_debug_export_npz_companions(
            fixture_dir=fixture_dir,
            kernel_target=kernel_target,
            probe_artifacts=runs[0],
            expected_probe_count=args.probe_count,
        )
        written.append(aggregate.logits_path)
        if aggregate.state_path is not None:
            written.append(aggregate.state_path)
        shapes[kernel_target] = aggregate.shapes
    written.extend(
        _write_fixture_yaml_pair(
            fixture_dir=fixture_dir,
            family_id=args.family_id,
            probe_count=args.probe_count,
            weight_version_id=args.weight_version_id,
            vllm_version=str(capabilities.get("vllm_version", "unknown")),
            kernel_targets=kernel_targets,
        )
    )
    result.update(
        {
            "status": "FIXTURE_WRITTEN",
            "written": sorted(written),
            "artifact_shapes": shapes,
            "validations": {
                kernel_target: _validate_written_fixture(
                    repo_root=repo_root,
                    family_id=args.family_id,
                    kernel_target=kernel_target,
                    probe_count=args.probe_count,
                    weight_version_id=args.weight_version_id,
                )
                for kernel_target in kernel_targets
            },
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(item["pass"] for item in result["validations"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
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
) -> list[str]:
    written: list[str] = []
    for kernel_target in kernel_targets:
        payload = fixture_payload(
            family_id=family_id,
            kernel_target=kernel_target,
            probe_count=probe_count,
            weight_version_id=weight_version_id,
            vllm_version=vllm_version,
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

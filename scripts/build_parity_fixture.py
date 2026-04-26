#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.parity_fixture import (  # noqa: E402
    DEFAULT_WEIGHT_VERSION_ID,
    KERNEL_TARGETS,
    deterministic_probe_rows,
    fetch_endpoint_capabilities,
    fixture_payload,
    family_fixture_dir,
    validate_p2b_fixture_set,
)


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
        payload["artifact_purpose"] = "test_only_synthetic_placeholder"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate HLD section 2.2.0 parity fixture artifacts.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--family-id", default="responses-sdk-adapter-cutover")
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--weight-version-id", default=DEFAULT_WEIGHT_VERSION_ID)
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
    result = {
        "status": "BLOCKED_NEEDS_USER_HELP",
        "halt_reason": "missing_real_kernel_logit_state_introspection",
        "family_id": args.family_id,
        "probe_count": args.probe_count,
        "weight_version_id": args.weight_version_id,
        "capabilities": capabilities,
        "required_by_hld": {
            "gatedattn": "full per-token reference logits for every probe across 3 bit-identical runs",
            "deltanet": "full per-token reference logits plus recurrent state snapshots at tokens [1, 1024]",
        },
        "files_written": [],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

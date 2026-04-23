#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumo_flywheel_serving.sprint0_closeout import generate_sprint0_closeout_artifacts


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate Sprint 0 resume and safety-rail closeout artifacts.")
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--registry", default=str(repo_root / "model_registry.yaml"))
    parser.add_argument("--state-root", default=str(repo_root / "output" / "sprint0_live" / "state"))
    parser.add_argument("--logs-root", default=str(repo_root / "output" / "sprint0_live" / "logs"))
    parser.add_argument("--triton-cache-root", default=str(repo_root / "output" / "sprint0_live" / "triton"))
    parser.add_argument(
        "--workload-file",
        default=str(repo_root / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"),
    )
    parser.add_argument("--artifacts-root", default=str(repo_root / "output" / "sprint0_live" / "artifacts"))
    parser.add_argument("--family-id", default="proposal-ranking-manager-judgment")
    parser.add_argument("--baseline-bundle-path")
    parser.add_argument("--container-name", default="lumo-vllm-sprint0-artifact")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_paths = generate_sprint0_closeout_artifacts(
        repo_root=args.repo_root,
        registry_path=args.registry,
        state_root=args.state_root,
        logs_root=args.logs_root,
        triton_cache_root=args.triton_cache_root,
        workload_file=args.workload_file,
        artifacts_root=args.artifacts_root,
        family_id=args.family_id,
        baseline_bundle_path=args.baseline_bundle_path,
        container_name=args.container_name,
    )
    print(json.dumps({key: str(value) for key, value in artifact_paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
